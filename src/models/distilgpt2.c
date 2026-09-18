#include "factory.h"

/* DistilGPT-2 is GPT-2 with six blocks. Its HF model_type and checkpoint names
 * remain gpt2; neither the execution graph nor the import mapping is special. */
PolyModel *model_distilgpt2_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  GPT2Config cfg = poly_gpt2_config_default();
  cfg.n_layer = 6;
  return model_gpt2_configure(ctx, root, cfg, err);
}
