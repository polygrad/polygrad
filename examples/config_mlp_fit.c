/*
 * Config-based MLP instance, similar to loading a predefined model family.
 *
 * Build:
 *   cc -Isrc examples/config_mlp_fit.c -Lbuild -lpolygrad -lm -ldl -o temp/config_mlp_fit
 *   LD_LIBRARY_PATH=build ./temp/config_mlp_fit
 */

#include "models/mlp.h"
#include <stdio.h>

int main(void) {
  MLPConfig cfg = poly_mlp_config_default();
  cfg.layers[0] = 2;
  cfg.layers[1] = 4;
  cfg.layers[2] = 1;
  cfg.n_layers = 3;
  cfg.activation = "relu";
  cfg.use_bias = 1;
  cfg.loss = "mse";
  cfg.batch_size = 1;
  cfg.seed = 42;

  PolyInstance *inst = poly_mlp(&cfg);
  if (!inst) return 1;

  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.03f, 0.0f, 0.0f, 0.0f, 0.0f);
  float x[] = {1.0f, 2.0f};
  float y[] = {4.0f};
  PolyIOBinding io[] = {{"x", x}, {"y", y}};

  float first = 0.0f, last = 0.0f;
  for (int step = 0; step < 12; step++) {
    poly_instance_train_step(inst, io, 2, &last);
    if (step == 0) first = last;
  }
  printf("mlp loss %.6f -> %.6f\n", first, last);

  poly_instance_free(inst);
  return 0;
}
