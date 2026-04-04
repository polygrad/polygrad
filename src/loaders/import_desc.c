/*
 * import_desc.c -- Static import descriptor table
 *
 * To add a new model:
 * 1. Write model-owned import functions in the model file
 * 2. Add extern declaration below
 * 3. Add one row to g_import_descs[]
 */

#define _POSIX_C_SOURCE 200809L
#include "import_desc.h"
#include <string.h>

/* gpt2.c */
extern PolyInstance *poly_gpt2_from_hf_decoded_generic(
    const PolyHfDecoded *hf, const PolyGenericImportOpts *opts);
extern PolyInstance *poly_gpt2_from_gguf_decoded_generic(
    const PolyGgufDecoded *gguf, const PolyGenericImportOpts *opts);

/* qwen3.c */
extern PolyInstance *poly_qwen3_from_gguf_decoded_generic(
    const PolyGgufDecoded *gguf, const PolyGenericImportOpts *opts);

static const PolyImportDesc g_import_descs[] = {
    {
        .model_type        = "gpt2",
        .from_hf_decoded   = poly_gpt2_from_hf_decoded_generic,
        .from_gguf_decoded = poly_gpt2_from_gguf_decoded_generic,
    },
    {
        .model_type        = "qwen3",
        .from_hf_decoded   = NULL,
        .from_gguf_decoded = poly_qwen3_from_gguf_decoded_generic,
    },
};

static const int g_n_import_descs =
    (int)(sizeof(g_import_descs) / sizeof(g_import_descs[0]));

const PolyImportDesc *poly_import_desc_find(const char *model_type) {
    if (!model_type) return NULL;
    for (int i = 0; i < g_n_import_descs; i++) {
        if (strcmp(g_import_descs[i].model_type, model_type) == 0)
            return &g_import_descs[i];
    }
    return NULL;
}
