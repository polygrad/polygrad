/*
 * poly_model_mlp.h -- MLP family builder for PolyInstance
 *
 * Deterministic weight initialization via SplitMix64 PRNG.
 */

#ifndef POLY_MODEL_MLP_H
#define POLY_MODEL_MLP_H

#include "../instance.h"
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
PolyInstance *poly_mlp(const MLPConfig *cfg, PolyDevice device);
PolyInstance *poly_mlp_from_json(const char *json, int len, PolyDevice device);

/* Deterministic parameter initialization.
 * Uses SplitMix64 PRNG seeded by (seed, FNV1a(name)).
 * Kaiming uniform: U(-bound, +bound) where bound = sqrt(6/fan_in). */
void poly_init_param_kaiming(
    uint64_t seed,
    const char *name,
    float *data,
    int64_t numel,
    int64_t fan_in
);

#ifdef __cplusplus
}
#endif

#endif /* POLY_MODEL_MLP_H */
