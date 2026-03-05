/*
 * model_tabm.h -- TabM (BatchEnsemble MLP) builder for PolyInstance
 *
 * TabM: MLP with per-member rank-1 weight perturbations.
 * For each linear layer with shared weight W, ensemble member i computes:
 *   l_i(x) = s_i * (W @ (r_i * x)) + b_i
 * Final prediction: mean over k ensemble members.
 *
 * Reference: Gorishniy et al. (2024), arXiv:2410.24210 (ICLR 2025)
 */

#ifndef POLY_MODEL_TABM_H
#define POLY_MODEL_TABM_H

#include "instance.h"

/*
 * Create a TabM instance from a JSON spec.
 *
 * Spec format:
 * {
 *   "layers": [n_in, h1, ..., n_out],
 *   "n_ensemble": 32,
 *   "activation": "relu",
 *   "loss": "cross_entropy" | "mse",
 *   "batch_size": 1,
 *   "seed": 42
 * }
 *
 * Weight naming:
 *   layers.{l}.weight  (out_dim, in_dim) -- shared, Kaiming init
 *   layers.{l}.r       (k, in_dim)       -- input scaling, init ones
 *   layers.{l}.s       (k, out_dim)      -- output scaling, init ones
 *   layers.{l}.b       (k, out_dim)      -- per-member bias, init zeros
 *
 * Returns NULL on error.
 */
PolyInstance *poly_tabm_instance(const char *spec_json, int spec_len);

#endif /* POLY_MODEL_TABM_H */
