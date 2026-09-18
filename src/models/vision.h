#ifndef POLY_MODEL_VISION_H
#define POLY_MODEL_VISION_H
#include "factory.h"
#include "layers.h"
#include "registry.h"

/* Construction helpers, not a second model/execution representation. Dimensions
 * are fixed by the declared signature; image preprocessing belongs to callers. */
typedef struct {
  int dim, hidden, heads, layers, batch, image, patch, channels, tokens;
  double eps;
  const char *activation;
} ModelVisionConfig;
bool model_vision_int(const cJSON *, const char *, int, int, int *, PolyModelError *);
bool model_vision_float(const cJSON *, const char *, double, double *, PolyModelError *);
bool model_vision_bool(const cJSON *, const char *, bool, bool *, PolyModelError *);
bool model_vision_config(const cJSON *, ModelVisionConfig *, PolyModelError *);
PolyTensor *model_vision_patch(
    PolyModel *,
    const ModelVisionConfig *,
    PolyTensor *,
    const char *,
    bool
);
PolyTensor *model_vision_slice(PolyCtx *, PolyTensor *, int, int64_t, int64_t);
PolyTensor *
model_vision_attention(PolyModel *, const ModelVisionConfig *, PolyTensor *, const char *const[4], const bool[4], bool, PolyTensor *, PolyTensor *);
PolyTensor *model_vision_activation(PolyCtx *, PolyTensor *, const char *);
PolyTensor *model_vision_scale(PolyModel *, PolyTensor *, const char *, int);
PolyModel *model_vision_finish(PolyModel *, PolyModelError *);
typedef struct {
  const char *name;
  const int64_t *shape; /* NULL accepts any shape; -1 matches one extent. */
  int ndim;
} ModelVisionSkip;
PolyModel *
model_vision_import(const PolyHfDecoded *, const PolyGenericImportOpts *, PolyModel *(*)(PolyCtx *, const cJSON *, PolyModelError *), const ModelVisionSkip *);
#endif
