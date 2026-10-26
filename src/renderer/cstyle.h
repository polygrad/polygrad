#ifndef POLY_RENDERER_CSTYLE_H
#define POLY_RENDERER_CSTYLE_H

#include "polygrad.h"
#include "renderer/renderer.h"
#include "uop/upat.h"

/* Current Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py:ClangRenderer. */
PolyRendererCaps poly_c_renderer_caps(void);
PolyPatternMatcher *poly_clang_renderer_extra_matcher(void);
PolyPatternMatcher *poly_cuda_renderer_extra_matcher(void);
PolyPatternMatcher *poly_hip_renderer_extra_matcher(void);
PolyPatternMatcher *poly_pm_manual_bf16_cast(void);
PolyPatternMatcher *poly_pm_narrow_uint(void);
bool poly_wmma_name(const PolyUOp *uop, char *out, size_t out_size);
bool poly_render_param_name(PolyCtx *ctx, PolyUOp *uop, char *out, size_t out_size);
/* Remove one balanced outer pair; caller owns the returned expression. */
char *poly_render_strip_parens(const char *expression);
/* Direct source-use counts; caller owns the returned array. */
int *poly_render_child_counts(PolyUOp **uops, int n);
/* Positional CUSTOM formatting; caller owns the returned expression. */
char *poly_render_custom(const PolyUOp *uop, const char *const *sources);
char *poly_render_c(PolyCtx *ctx, PolyUOp **uops, int n, const char *fn_name);

#endif
