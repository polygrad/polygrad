#ifndef POLY_MODEL_GEMMA_H
#define POLY_MODEL_GEMMA_H
#include "layers.h"
#include "factory.h"

/* Named checkpoint layers and scalar graph construction shared by the towers.
 * Returned tensors belong to the enclosing model-factory construction scope. */
bool model_gemma_integer(const cJSON *, const char *, int, int, int, int *, PolyModelError *);
PolyTensor *model_gemma_scale(PolyCtx *, PolyTensor *, double);
PolyTensor *model_gemma_norm(PolyModel *, const char *, const char *, PolyTensor *, int, double);
PolyTensor *model_gemma_linear(PolyModel *, const char *, const char *, PolyTensor *, int, int);

/* Model-owned normalized Q/K/V attention shared by the Gemma text and vision
 * encoders. Axial RoPE rotates independent head slices, not the entire head. */
typedef struct {
  int batch, length, dim, heads, kv_heads, head_dim, rope_axes;
  double eps;
  bool nested_linear;
} ModelGemmaAttention;
PolyTensor *
model_gemma_attention(PolyModel *, const char *, const ModelGemmaAttention *, PolyTensor *, PolyTensor *, PolyTensor *, PolyTensor *);
PolyTensor *
model_gemma4_vision(PolyModel *, const cJSON *, int, int, PolyTensor *, PolyTensor *, PolyTensor **, PolyTensor **, PolyModelError *);
PolyTensor *model_gemma_clippable_linear(
    PolyModel *,
    const char *,
    const char *,
    PolyTensor *,
    int,
    int,
    bool
);
PolyTensor *
model_gemma4_audio(PolyModel *, const cJSON *, int, int, PolyTensor *, PolyTensor *, PolyTensor **, PolyTensor **, PolyModelError *);
#endif
