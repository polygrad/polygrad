/* Explicit placement of retained portable logical graphs. */
#ifndef POLY_PLACER_H
#define POLY_PLACER_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reject caller-visible explicit DEVICE identities which the current runtime
 * cannot address.  An absent DEVICE and internal DEVICE(None) remain valid;
 * CALL/FUNCTION bodies are opaque, matching pinned realization traversal. */
bool poly_uop_explicit_devices_supported(PolyCtx *ctx, PolyUOp *root);

/* Compile aggregate portable roots into complete physical roots using exact
 * logical storage-binding rows. This is a pure placement kernel: caller output
 * slots change only when every candidate validates, and no prior/captured
 * physical graph or Tensor/Model/cache state is consumed or mutated. */
int poly_place_roots(
    PolyCtx *ctx,
    PolyUOp **logical_roots,
    int n_roots,
    PolyUOp **logical_bindings,
    PolyUOp **target_bindings,
    int n_bindings,
    PolyUOp **out_roots
);

/* One explicit module region for the non-uniform scalar-device placement
 * policy.  The descriptor is pass input, not graph or Model state: output
 * is the exact portable value root produced by the module, and inputs are the
 * exact portable roots at which its backward slice must stop. */
typedef struct {
  const char *name;
  PolyUOp *output;
  PolyUOp **inputs;
  int n_inputs;
  PolyUOp *device;
} PolyPlaceModule;

/* Compile pure portable roots under an ordered, explicit module/device map.
 * Each module is rebuilt from declared boundary inputs on its target device;
 * exact cross-module identity changes become ordinary COPY UOps.  The policy
 * derives physical binding homes from exact module regions and STORE values.
 * The operation is aggregate and atomic with respect to both output arrays. */
int poly_place_module_map(
    PolyCtx *ctx,
    PolyUOp **logical_roots,
    int n_roots,
    PolyUOp **logical_bindings,
    int n_bindings,
    const PolyPlaceModule *modules,
    int n_modules,
    PolyUOp **out_bindings,
    PolyUOp **out_roots
);

#ifdef __cplusplus
}
#endif
#endif
