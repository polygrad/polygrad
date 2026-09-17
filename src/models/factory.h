#ifndef POLY_MODEL_FACTORY_H
#define POLY_MODEL_FACTORY_H

#include "../ctx.h"
#include "../model.h"
#include "../device.h"
#include "../tensor.h"
#include "../../vendor/cjson/cJSON.h"

bool model_factory_error(PolyModelError *err, const char *path, const char *fmt, ...);
cJSON *model_factory_parse(const char *json, int len, PolyModelError *err);
bool model_config_integer(
    const cJSON *root,
    const char *key,
    int64_t lo,
    int64_t hi,
    bool required,
    PolyModelError *err
);
bool model_config_sizes(
    const cJSON *root,
    const char *key,
    int min,
    int max,
    bool required,
    PolyModelError *err
);
bool model_config_choice(
    const cJSON *root,
    const char *key,
    const char *choices,
    PolyModelError *err
);
bool model_config_training(const cJSON *root, PolyModelError *err);
/* fan_in > 0 selects the existing seed/name-keyed Kaiming initializer;
 * fan_in == 0 fills a constant. Publish through Model writes on any placement. */
int model_init_param(PolyModel *model, const char *name, uint64_t seed, int64_t fan_in, float fill);
PolyModel *poly_model_from_config(
    PolyCtx *ctx,
    const char *family,
    const char *json,
    int len,
    PolyDevice device,
    PolyModelError *err
);

/* Private family callbacks share a parsed, validated JSON object and a scoped
 * context. They publish one ordinary Model, not an execution wrapper. */
PolyModel *model_mlp_build(PolyCtx *, const cJSON *, PolyModelError *);
PolyModel *model_tabm_build(PolyCtx *, const cJSON *, PolyModelError *);
PolyModel *model_nam_build(PolyCtx *, const cJSON *, PolyModelError *);
PolyModel *model_gpt2_build(PolyCtx *, const cJSON *, PolyModelError *);
PolyModel *model_llama_build(PolyCtx *, const cJSON *, PolyModelError *);
PolyModel *model_sequential_build(PolyCtx *, const cJSON *, PolyModelError *);
PolyModel *model_graph_build(PolyCtx *, const cJSON *, PolyModelError *);

/* C-only Model construction scope. Families borrow an idle caller context or
 * create a standalone one; only a successfully built standalone Model owns it.
 * Defaults apply to new construction, never to existing caller Tensor handles. */
typedef struct {
  PolyCtx *ctx;
  PolyDevice device;
  PolyLogicalPolicy logical;
  uint64_t first_tensor_order;
  bool owned;
} PolyModelFactoryScope;

static inline bool model_factory_begin(
    PolyModelFactoryScope *scope,
    PolyCtx *ctx,
    PolyDevice device
) {
  if (device != POLY_DEVICE_AUTO && !poly_device_can_execute(device)) return false;
  if (ctx && (ctx->execution_depth || ctx->collecting || ctx->active_jit_capture)) return false;
  *scope = (PolyModelFactoryScope){.ctx = ctx, .owned = ctx == NULL};
  if (!ctx) scope->ctx = ctx = poly_ctx_new();
  if (!ctx) return false;
  scope->device = poly_ctx_get_preferred_device(ctx);
  scope->logical = poly_ctx_get_logical_policy(ctx);
  scope->first_tensor_order = ctx->next_tensor_order;
  if (device != POLY_DEVICE_AUTO) poly_ctx_set_preferred_device(ctx, device);
  poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_ALWAYS);
  return true;
}

static inline PolyModel *model_factory_end(PolyModelFactoryScope *scope, PolyModel *model) {
  /* These config-only factories publish UOp roots, never Tensor handles.
   * Drop all local references (including identity-return retains), newest first
   * so .to() source edges cannot point at a wrapper already reclaimed here.
   * No allocation is required on failure cleanup. Usually the newest handle
   * is the registry tail; scan only when inner composites left order gaps. */
  PolyCtx *ctx = scope->ctx;
  uint64_t next_order = ctx->next_tensor_order;
  while (ctx->n_tensors > 0) {
    PolyTensor *last = NULL;
    for (int i = ctx->n_tensors - 1; i >= 0; i--) {
      PolyTensor *t = ctx->tensors[i];
      if (t->order >= scope->first_tensor_order && (!last || t->order > last->order)) last = t;
      if (last && last->order == next_order - 1) break;
    }
    if (!last) break;
    next_order = last->order;
    while (last->owner_refs > 1)
      poly_tensor_release(last);
    poly_tensor_release(last);
  }
  if (!scope->owned) poly_ctx_set_preferred_device(scope->ctx, scope->device);
  poly_ctx_set_logical_policy(scope->ctx, scope->logical);
  if (scope->owned) {
    if (model)
      poly_model_own_ctx(model);
    else
      poly_ctx_destroy(scope->ctx);
  }
  return model;
}

#endif
