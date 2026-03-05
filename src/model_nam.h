/*
 * model_nam.h -- NAM (Neural Additive Model) builder for PolyInstance
 *
 * NAM: g(E[y]) = beta + f1(x1) + f2(x2) + ... + fK(xK)
 * Each fk is a small MLP on scalar feature xk.
 *
 * Supports ReLU and ExU (Exponential Unit) activations.
 * ExU: ReLU(exp(w) * (x - b)) where w, b are learnable per-unit.
 *
 * Reference: Agarwal et al. (2021), arXiv:2004.13912 (NeurIPS 2021)
 */

#ifndef POLY_MODEL_NAM_H
#define POLY_MODEL_NAM_H

#include "instance.h"

/*
 * Create a NAM instance from a JSON spec.
 *
 * Spec format:
 * {
 *   "n_features": K,
 *   "hidden_sizes": [64, 64],
 *   "activation": "relu" | "exu",
 *   "n_outputs": 1,
 *   "loss": "mse" | "cross_entropy",
 *   "batch_size": 1,
 *   "seed": 42
 * }
 *
 * Weight naming:
 *   intercept                            (n_outputs,)   -- zero init
 *   features.{k}.layers.{l}.weight       (out, in)      -- Kaiming init
 *   features.{k}.layers.{l}.bias         (out,)         -- zero init
 *
 * When activation == "exu":
 *   features.{k}.exu.{l}.weight          (hidden,)      -- zero init (exp(0)=1)
 *   features.{k}.exu.{l}.bias            (hidden,)      -- zero init
 *
 * Returns NULL on error.
 */
PolyInstance *poly_nam_instance(const char *spec_json, int spec_len);

#endif /* POLY_MODEL_NAM_H */
