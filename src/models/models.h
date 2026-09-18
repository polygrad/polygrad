#ifndef POLYGRAD_MODELS_H
#define POLYGRAD_MODELS_H

#include "mlp.h"
#include "gpt2.h"
#include "llama.h"
#include "hf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Built-in model types only. NULL type requires explicit format/type tags;
 * named factories also accept their existing untagged configuration objects.
 * NULL ctx creates an owned context; otherwise construction borrows ctx and
 * restores its defaults on every exit. */
PolyModel *poly_model_from_config(
    PolyCtx *ctx,
    const char *type,
    const char *json,
    int len,
    PolyDevice device,
    PolyModelError *err
);
enum { POLY_MODEL_CONSTRUCTIBLE = 1, POLY_MODEL_HF = 2, POLY_MODEL_GGUF = 4 };
const char *poly_model_type_name(int index); /* NULL past the last type */
/* Derived from registered callbacks; constructible means JSON construction. */
int poly_model_type_capabilities(int index);
#ifdef __cplusplus
}
#endif

#endif /* POLYGRAD_MODELS_H */
