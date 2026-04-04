/*
 * import_desc.h -- Private static import registry
 *
 * Maps model_type strings to model-owned import functions.
 * Used only inside poly_hf_load() and poly_gguf_load() for auto-dispatch.
 */

#ifndef POLY_IMPORT_DESC_H
#define POLY_IMPORT_DESC_H

#include "hf_decode.h"
#include "gguf_decode.h"
#include "../instance.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int max_batch;
    int max_seq_len;
} PolyGenericImportOpts;

typedef struct {
    const char *model_type;

    PolyInstance *(*from_hf_decoded)(
        const PolyHfDecoded *hf,
        const PolyGenericImportOpts *opts);

    PolyInstance *(*from_gguf_decoded)(
        const PolyGgufDecoded *gguf,
        const PolyGenericImportOpts *opts);
} PolyImportDesc;

const PolyImportDesc *poly_import_desc_find(const char *model_type);

#ifdef __cplusplus
}
#endif

#endif /* POLY_IMPORT_DESC_H */
