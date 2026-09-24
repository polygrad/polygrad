#ifndef POLY_ONNX_LOADER_H
#define POLY_ONNX_LOADER_H

#include "../model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* All inputs are borrowed only during import; the resulting Model owns copies.
 * dimensions_json maps ONNX symbolic dimension names to positive integers.
 * External locations resolve by exact name, never through filesystem/network I/O.
 * Only fixed/specialized inference graphs are supported. Unsupported constructs
 * fail with poly_import_last_error_*; no partially built Model is published. */
typedef struct {
  const char *dimensions_json;
  const char *const *external_names;
  const uint8_t *const *external_data;
  const int64_t *external_lengths;
  int n_external;
} PolyOnnxOptions;

PolyModel *poly_onnx_load(
    const uint8_t *data,
    int64_t len,
    const PolyOnnxOptions *options,
    PolyDevice device
);
PolyModel *poly_onnx_load_into(
    PolyCtx *ctx,
    const uint8_t *data,
    int64_t len,
    const PolyOnnxOptions *options,
    PolyDevice device
);

#ifdef __cplusplus
}
#endif
#endif
