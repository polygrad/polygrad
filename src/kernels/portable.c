#include "kernels/kernels.h"
#include <string.h>

static const char *wasm_supported(PolyCtx *ctx, const PolyGemmDesc *d) {
  const char *reason = poly_kernel_gemm_supported(ctx, d, 4, 8);
  if (reason) return reason;
  /* Packing and the extra epilogue dispatch lose on small contractions in
   * paired Node/Chrome measurements. The validated shape caps bound this product. */
  if (d->M * d->N * d->K < 262144) return "below Wasm packing crossover";
  return NULL;
}

static PolyUOp *wasm_lower(PolyCtx *ctx, const PolyGemmDesc *d) {
  /* Standard SIMD128, not relaxed SIMD: distinct multiply/add operations. */
  const PolyGemmTile tile = {.name = "wasm_gemm", .rows = 4, .lanes = 4, .vectors = 2};
  return poly_kernel_gemm_lower(ctx, d, &tile, 1, 0);
}

const PolyKernelImpl *poly_portable_kernel_impls(const char *device, int *count) {
  static const PolyKernelImpl wasm[] = {{"gemm_simd128_4x8", wasm_supported, wasm_lower}};
  static const PolyKernelImpl webgpu[] = {
      {"gemm_workgroup_32x64", poly_kernel_webgpu_supported, poly_kernel_webgpu_lower}};
  *count = 1;
  if (!strcmp(device, "WASM")) return wasm;
  if (!strcmp(device, "WEBGPU")) return webgpu;
  *count = 0;
  return NULL;
}
