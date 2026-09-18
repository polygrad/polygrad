#ifndef POLY_MODEL_REGISTRY_H
#define POLY_MODEL_REGISTRY_H

#include "../model.h"
#include "../loaders/hf_decode.h"
#include "../loaders/gguf_decode.h"
#include "../../vendor/cjson/cJSON.h"

typedef struct {
  PolyCtx *ctx; /* NULL requests a standalone owning Model. */
  int max_batch, max_seq_len;
  PolyDevice device;
} PolyGenericImportOpts;

/* One static descriptor for config construction and decoded imports. A missing
 * callback is an unsupported capability, not a fallback to another model. */
typedef struct {
  const char *name, *tag;
  PolyModel *(*build)(PolyCtx *, const cJSON *, PolyModelError *);
  PolyModel *(*from_hf_decoded)(const PolyHfDecoded *, const PolyGenericImportOpts *);
  PolyModel *(*from_gguf_decoded)(const PolyGgufDecoded *, const PolyGenericImportOpts *);
} PolyModelType;

const PolyModelType *model_type_find(const char *name);

/* Registry callbacks are library-internal, not independently callable ABI.
 * Private headers alone do not hide symbols in native shared-library builds. */
#if defined(__GNUC__)
#define MODEL_IMPORT_INTERNAL __attribute__((visibility("hidden")))
#else
#define MODEL_IMPORT_INTERNAL
#endif
MODEL_IMPORT_INTERNAL PolyModel *
model_gpt2_from_hf_decoded(const PolyHfDecoded *, const PolyGenericImportOpts *);
MODEL_IMPORT_INTERNAL PolyModel *
model_gpt2_from_gguf_decoded(const PolyGgufDecoded *, const PolyGenericImportOpts *);
MODEL_IMPORT_INTERNAL PolyModel *
model_llama_from_hf_decoded(const PolyHfDecoded *, const PolyGenericImportOpts *);
MODEL_IMPORT_INTERNAL PolyModel *
model_qwen3_from_gguf_decoded(const PolyGgufDecoded *, const PolyGenericImportOpts *);
MODEL_IMPORT_INTERNAL PolyModel *
model_clip_from_hf_decoded(const PolyHfDecoded *, const PolyGenericImportOpts *);
MODEL_IMPORT_INTERNAL PolyModel *
model_vit_from_hf_decoded(const PolyHfDecoded *, const PolyGenericImportOpts *);
MODEL_IMPORT_INTERNAL PolyModel *
model_dinov2_from_hf_decoded(const PolyHfDecoded *, const PolyGenericImportOpts *);
MODEL_IMPORT_INTERNAL PolyModel *
model_dinov3_from_hf_decoded(const PolyHfDecoded *, const PolyGenericImportOpts *);
#undef MODEL_IMPORT_INTERNAL

#endif
