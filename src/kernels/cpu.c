#include "kernels/kernels.h"
#include "renderer/cstyle.h"
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) && !defined(__EMSCRIPTEN__)
static const char *cpu_fp32(const PolyGemmDesc *d) {
  if (strcmp(d->device, "CPU") && strncmp(d->device, "CPU:", 4)) return "not a CPU graph";
  if (!poly_dtype_eq(d->root->dtype, POLY_FLOAT32) || !poly_dtype_eq(d->a->dtype, POLY_FLOAT32) ||
      !poly_dtype_eq(d->b->dtype, POLY_FLOAT32))
    return "requires float32";
  return NULL;
}

static const char *probabilities_supported(PolyCtx *ctx, const PolyGemmDesc *d) {
  (void)ctx;
  const char *reason = cpu_fp32(d);
  if (reason) return reason;
  if (!d->probabilities || d->N <= 1) return "no reused last-axis softmax";
  /* Bound added residency; this policy is not a flash-attention kernel. */
  return d->probability_elements <= 8 * 1024 * 1024 ? NULL : "probability table exceeds 32 MiB";
}

static const char *gemm_supported(PolyCtx *ctx, const PolyGemmDesc *d) {
  const char *reason = cpu_fp32(d);
  if (reason) return reason;
  return poly_kernel_gemm_supported(ctx, d, 4, 24);
}

static PolyUOp *gemm_lower(PolyCtx *ctx, const PolyGemmDesc *d) {
  static const PolyGemmTile tile = {
      .name = "avx2_gemm",
      .rows = 4,
      .lanes = 8,
      .vectors = 3,
      .fma = "__builtin_ia32_vfmaddps256({0}, {1}, {2})"};
  PolyRendererCaps caps = poly_c_renderer_caps();
  int budget = caps.has_threads ? caps.global_max[0] : 1;
  /* Pinned heuristic.py: use ~128K products/worker and an even output-axis
   * split. The target chooses distribution independently of register geometry. */
  int useful = (int)(d->M * d->N * d->K / (128 << 10));
  const int candidates[] = {32, 16, 12, 8, 6, 5, 4, 3, 2};
  int64_t tiles[] = {d->M / tile.rows, d->N / (tile.lanes * tile.vectors)};
  for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
    int workers = candidates[i];
    if (workers > budget || workers > useful) continue;
    for (int axis = 0; axis < 2; axis++)
      if (tiles[axis] % workers == 0) return poly_kernel_gemm_lower(ctx, d, &tile, workers, axis);
  }
  return poly_kernel_gemm_lower(ctx, d, &tile, 1, 0);
}
#endif

const PolyKernelImpl *poly_cpu_kernel_impls(int *count) {
  *count = 0;
#if defined(__x86_64__) && !defined(__EMSCRIPTEN__)
  const char *arch = getenv("POLY_CPU_ARCH");
  if ((arch && arch[0] && strcmp(arch, "native")) || !__builtin_cpu_supports("avx2") ||
      !__builtin_cpu_supports("fma"))
    return NULL;
  /* Materialization first; the fixed-point rewrite then considers the GEMM
   * with its new input. Each rule removes its own matching opportunity. */
  static const PolyKernelImpl impls[] = {
      {"softmax_av_materialize", probabilities_supported, poly_kernel_probabilities_lower},
      {"gemm_avx2_4x24", gemm_supported, gemm_lower}};
  *count = sizeof(impls) / sizeof(impls[0]);
  return impls;
#else
  return NULL;
#endif
}
