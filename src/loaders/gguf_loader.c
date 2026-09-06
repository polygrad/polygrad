/*
 * gguf_loader.c -- GGUF model loader (thin auto-dispatch wrapper)
 *
 * Decodes GGUF binary, looks up model by general.architecture,
 * dispatches to model-owned import function. Zero model-specific logic.
 */

#define _POSIX_C_SOURCE 200809L
#include "gguf_decode.h"
#include "import_desc.h"
#include "import_error.h"
#include "../model.h"
#include <stdio.h>

PolyModel *poly_gguf_load(
    const uint8_t *data,
    int64_t len,
    int max_batch,
    int max_seq_len,
    PolyDevice device
) {
  poly_import_error_clear();

  PolyGgufDecoded *gguf = NULL;
  if (poly_gguf_decode(data, len, &gguf) != 0 || !gguf) return NULL;

  const PolyImportDesc *desc = poly_import_desc_find(gguf->arch);
  if (!desc || !desc->from_gguf_decoded) {
    poly_import_error_set(
        POLY_IMPORT_ERR_UNSUPPORTED_MODEL, "unsupported GGUF architecture '%s'",
        gguf->arch ? gguf->arch : ""
    );
    poly_gguf_decoded_free(gguf);
    return NULL;
  }

  PolyGenericImportOpts opts = {
      .max_batch = max_batch,
      .max_seq_len = max_seq_len,
      .device = device,
  };
  PolyModel *inst = desc->from_gguf_decoded(gguf, &opts);

  poly_gguf_decoded_free(gguf);
  return inst;
}
