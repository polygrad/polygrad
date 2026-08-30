#ifndef POLY_CODEGEN_DECOMP_DTYPE_H
#define POLY_CODEGEN_DECOMP_DTYPE_H

#include "renderer/renderer.h"
#include "uop/upat.h"

typedef struct {
  PolyDType from;
  PolyDType to;
} PolyFloatDecompContext;

typedef struct {
  PolyRendererCaps caps;
  bool seen_long;
  uint8_t seen_fp8;
  bool seen_float16;
  bool seen_bfloat16;
} PolyDTypeDecompsContext;

/* Tinygrad 2026-08-22/a9069c177a9d codegen/decomp/dtype.py. */
PolyPatternMatcher *poly_pm_float_decomp(void);
PolyPatternMatcher *poly_pm_long_decomp(void);
PolyPatternMatcher *poly_pm_dtype_decomps(void);
PolyUOp *poly_reindex(PolyCtx *ctx, PolyUOp *idx, int off, int mul);

#endif
