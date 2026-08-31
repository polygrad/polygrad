/*
 * import_error.h -- Thread-local error state for model import functions
 *
 * Set by any import function on failure. Cleared at the start of each
 * top-level import call (poly_hf_load, poly_gguf_load, poly_onnx_load).
 */

#ifndef POLY_IMPORT_ERROR_H
#define POLY_IMPORT_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  POLY_IMPORT_OK = 0,
  POLY_IMPORT_ERR_PARSE,
  POLY_IMPORT_ERR_UNSUPPORTED_MODEL,
  POLY_IMPORT_ERR_UNSUPPORTED_OP,
  POLY_IMPORT_ERR_WEIGHT_MISMATCH,
  POLY_IMPORT_ERR_SHAPE_MISMATCH,
  POLY_IMPORT_ERR_DTYPE_UNSUPPORTED,
  POLY_IMPORT_ERR_INTERNAL
} PolyImportError;

PolyImportError poly_import_last_error_code(void);
const char *poly_import_last_error_message(void);

/* Internal: called by import functions */
void poly_import_error_clear(void);
void poly_import_error_set(PolyImportError code, const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* POLY_IMPORT_ERROR_H */
