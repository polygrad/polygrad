#ifndef POLY_RENDERER_H
#define POLY_RENDERER_H

#include "codegen/opt/tc.h"
#include <stdbool.h>

/* Current Tinygrad 2026-08-22/a9069c177a9d renderer/__init__.py:Renderer
 * capabilities represented as immutable C call data. */
typedef struct {
  const char *device;
  const char *arch;
  bool has_mulacc;
  bool has_max;
  bool has_threefry;
  bool has_exp2;
  bool has_log2;
  bool has_sin;
  bool has_fdiv;
  bool supports_float16;
  bool supports_bfloat16;
  bool supports_fp8e4m3;
  bool supports_fp8e5m2;
  bool supports_fp8e4m3fnuz;
  bool supports_fp8e5m2fnuz;
  bool has_int64;
  bool has_local;
  bool has_threads;
  bool has_simd_int;
  bool has_simd_float;
  int max_vec_width;
  int max_threads;
  int global_max[3];
  int local_max[3];
  const PolyTensorCore *tensor_cores;
  int n_tensor_cores;
} PolyRendererCaps;

/* Tinygrad 2026-08-22/a9069c177a9d Renderer.supported_dtypes(). */
static inline bool poly_renderer_supports_dtype(PolyRendererCaps caps, PolyDType dtype) {
  if (poly_dtype_eq(dtype, POLY_FLOAT16)) return caps.supports_float16;
  if (poly_dtype_eq(dtype, POLY_BFLOAT16)) return caps.supports_bfloat16;
  if (poly_dtype_eq(dtype, POLY_FP8E4M3)) return caps.supports_fp8e4m3;
  if (poly_dtype_eq(dtype, POLY_FP8E5M2)) return caps.supports_fp8e5m2;
  if (poly_dtype_eq(dtype, POLY_FP8E4M3FNUZ)) return caps.supports_fp8e4m3fnuz;
  if (poly_dtype_eq(dtype, POLY_FP8E5M2FNUZ)) return caps.supports_fp8e5m2fnuz;
  if (poly_dtype_eq(dtype, POLY_INT64) || poly_dtype_eq(dtype, POLY_UINT64))
    return caps.has_int64;
  return true;
}

#endif
