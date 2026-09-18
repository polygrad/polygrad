/*
 * poly_model_mlp.h -- MLP family builder for PolyModel
 *
 * Deterministic weight initialization via SplitMix64 PRNG.
 */

#ifndef POLY_MODEL_MLP_H
#define POLY_MODEL_MLP_H

#include "../model.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define POLY_MLP_MAX_LAYERS 32

typedef struct {
  int layers[POLY_MLP_MAX_LAYERS]; /* layer sizes: [n_in, h1, ..., n_out] */
  int n_layers; /* number of entries in layers[] */
  const char *activation; /* "relu"|"gelu"|"silu"|"tanh"|"sigmoid"|"none" */
  int use_bias; /* 1 = bias, 0 = no bias */
  const char *loss; /* "mse"|"cross_entropy"|"none" */
  int batch_size;
  uint64_t seed;
} MLPConfig;

MLPConfig poly_mlp_config_default(void);
/* Borrows ctx and restores its defaults on every exit. Explicit NULL ctx
 * requests a standalone Model that owns its context. */
PolyModel *poly_mlp_into(PolyCtx *ctx, const MLPConfig *cfg, PolyDevice device);

#ifdef __cplusplus
}
#endif

#endif /* POLY_MODEL_MLP_H */
