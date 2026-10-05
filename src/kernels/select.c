#include "kernels/kernels.h"
#include "utils.h"
#include "ctx.h"
#include <stdio.h>
#include <string.h>

typedef struct {
  bool failed, debug;
} KernelSelection;

static PolyUOp *select_match(PolyCtx *ctx, PolyUOp *u, const PolyBindings *bindings) {
  (void)bindings;
  KernelSelection *s = poly_graph_rewrite_userctx();
  if (s->failed) return NULL;
  PolySegmentDesc segment;
  if (poly_kernel_match_segment(ctx, u, &segment)) {
    const char *dev = segment.device;
    bool supported = !strcmp(dev, "CPU") || !strncmp(dev, "CPU:", 4) || !strcmp(dev, "X86") ||
                     !strcmp(dev, "INTERP") || !strcmp(dev, "WASM") || !strcmp(dev, "CUDA") ||
                     !strcmp(dev, "WEBGPU");
    if (s->debug)
      fprintf(
          stderr, "polygrad: kernel %s N=%lld S=%lld W=%lld segment_sum: %s\n", dev,
          (long long)segment.rows, (long long)segment.segments, (long long)segment.width,
          supported ? "selected" : "unsupported target"
      );
    if (supported) {
      PolyUOp *result = poly_kernel_segment_lower(ctx, &segment);
      if (!result) s->failed = true;
      return result;
    }
  }
  PolyGemmDesc d;
  if (!poly_kernel_match_gemm(ctx, u, &d)) return NULL;
  int count = 0;
  const PolyKernelImpl *impls = poly_portable_kernel_impls(d.device, &count);
  if (!count) impls = poly_cpu_kernel_impls(&count);
  for (int i = 0; i < count; i++) {
    const PolyKernelImpl *impl = &impls[i];
    const char *reason = impl->supports(ctx, &d);
    if (s->debug)
      fprintf(
          stderr, "polygrad: kernel %s M=%lld N=%lld K=%lld %s: %s\n", d.device, (long long)d.M,
          (long long)d.N, (long long)d.K, impl->name, reason ? reason : "selected"
      );
    if (reason) continue;
    PolyUOp *result = impl->lower(ctx, &d);
    if (!result) s->failed = true;
    return result;
  }
  return NULL;
}

PolyUOp *poly_kernel_select(PolyCtx *ctx, PolyUOp *sink) {
  ctx->kernel_policy_locked = true;
  if (!ctx->kernel_policy) return sink;
  KernelSelection s = {.debug = poly_getenv_int("POLY_DEBUG_KERNELS", 0) != 0};
  static _Thread_local PolyPatternMatcher *pm;
  if (!pm) {
    PolyRule rules[] = {{poly_upat_op(POLY_OP_REDUCE, NULL, 0, "x"), select_match}};
    pm = poly_pm_thread_cache(poly_pm_new(rules, 1));
  }
  /* Authored/precompiled CALL bodies are opaque. Ordinary FUNCTION bodies
   * are selected after argument substitution by the existing resolver. */
  PolyUOp *result = poly_graph_rewrite_ctx_ex2(ctx, sink, pm, &s, false, false);
  return s.failed ? NULL : result;
}
