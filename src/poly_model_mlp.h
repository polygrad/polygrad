/*
 * poly_model_mlp.h -- MLP family builder for PolyInstance
 *
 * Creates PolyInstance from a JSON spec describing an MLP architecture.
 * Deterministic weight initialization via SplitMix64 PRNG.
 */

#ifndef POLY_MODEL_MLP_H
#define POLY_MODEL_MLP_H

#include "poly_instance.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Create an MLP PolyInstance from JSON spec.
 *
 * Spec format:
 *   {
 *     "layers": [n_in, h1, ..., n_out],
 *     "activation": "relu"|"gelu"|"silu"|"tanh"|"sigmoid"|"none",
 *     "bias": true|false,
 *     "loss": "mse"|"none",
 *     "batch_size": N,
 *     "seed": 42
 *   }
 *
 * Returns NULL on error. */
PolyInstance *poly_mlp_instance(const char *spec_json, int spec_len);

/* Deterministic parameter initialization.
 * Uses SplitMix64 PRNG seeded by (seed, FNV1a(name)).
 * Kaiming uniform: U(-bound, +bound) where bound = sqrt(6/fan_in). */
void poly_init_param_kaiming(uint64_t seed, const char *name,
                              float *data, int64_t numel, int64_t fan_in);

#ifdef __cplusplus
}
#endif

#endif /* POLY_MODEL_MLP_H */
