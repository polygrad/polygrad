#ifndef POLY_CODEGEN_LATE_COALESCE_H
#define POLY_CODEGEN_LATE_COALESCE_H

#include "uop/upat.h"
#include "renderer/renderer.h"

/* Tinygrad 2026-08-22/a9069c177a9d codegen/late/coalesce.py. */
PolyPatternMatcher *poly_indexing_simplify(void);
PolyPatternMatcher *poly_pm_simplify_add_image(void);
PolyUOp *poly_memory_coalescing(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps);

/* C storage for Tinygrad coalesce.py's `ctx=({}, ren)` image rewrite tuple. */
typedef struct {
  PolyRendererCaps caps;
  int64_t *slots;
  int64_t *heights;
  int64_t *widths;
  int count;
  int capacity;
} PolyImageRewriteCtx;

void poly_image_rewrite_ctx_destroy(PolyImageRewriteCtx *ctx);

#endif
