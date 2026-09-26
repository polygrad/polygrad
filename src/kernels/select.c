#include "kernels/kernels.h"
#include "utils.h"
#include "ctx.h"
#include <stdio.h>

typedef struct {
  bool portable;
  bool failed, debug;
} KernelSelection;

static PolyUOp *select_match(PolyCtx *ctx, PolyUOp *u, const PolyBindings *bindings) {
  (void)bindings;
  KernelSelection *s = poly_graph_rewrite_userctx();
  PolyGemmDesc d;
  if (s->failed || !poly_kernel_match_gemm(ctx, u, &d)) return NULL;
  int count = 0;
  const PolyKernelImpl *impls = s->portable ? poly_portable_kernel_impls(d.device, &count) : NULL;
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
  bool portable = ctx->kernel_policy == 1;
  if (!portable && !(ctx->kernel_policy == -1 && poly_getenv_int("POLY_CPU_GEMM", 0))) return sink;
  KernelSelection s = {
      .debug = poly_getenv_int("POLY_DEBUG_KERNELS", 0) != 0, .portable = portable};
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
