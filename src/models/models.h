#ifndef POLYGRAD_MODELS_H
#define POLYGRAD_MODELS_H

#include "mlp.h"
#include "gpt2.h"
#include "llama.h"
#include "hf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Built-in families only. NULL family requires explicit format/type tags;
 * named factories also accept their existing untagged configuration objects.
 * NULL ctx creates an owned context; otherwise construction borrows ctx and
 * restores its defaults on every exit. */
PolyModel *poly_model_from_config(
    PolyCtx *ctx,
    const char *family,
    const char *json,
    int len,
    PolyDevice device,
    PolyModelError *err
);
const char *poly_model_family_name(int index); /* NULL past the last family */
#ifdef __cplusplus
}
#endif

#endif /* POLYGRAD_MODELS_H */
