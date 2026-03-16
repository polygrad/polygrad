/*
 * poly_bundle.h -- Portable model bundle format (poly.bundle@1)
 *
 * A single-file container for a complete PolyInstance:
 *   - IR section: tensor-level UOp graph (poly.ir.uops@1)
 *   - WEIGHTS section: parameter data (safetensors format)
 *   - METADATA section: JSON (model family, entrypoint info, etc.)
 *
 * Binary layout:
 *   [8 bytes]  magic "POLYBNDL"
 *   [4 bytes]  bundle version (LE uint32, currently 1)
 *   [4 bytes]  flags (LE uint32, reserved, currently 0)
 *   [4 bytes]  n_sections (LE uint32)
 *   For each section:
 *     [4 bytes]  section_type (LE uint32)
 *     [4 bytes]  section_length (LE uint32, byte count of section_data)
 *     [N bytes]  section_data
 *
 * Section types:
 *   0x01  POLY_BUNDLE_IR        poly.ir.uops@1 payload
 *   0x02  POLY_BUNDLE_WEIGHTS   safetensors payload
 *   0x03  POLY_BUNDLE_METADATA  UTF-8 JSON
 *
 * Invariants:
 *   - No compiled kernels, no device handles, no runtime state
 *   - IR is tensor-level (pre-scheduling)
 *   - Weights are optional (may be zero-length)
 *   - Deterministic: same inputs produce same bytes (within a language)
 */

#ifndef POLY_BUNDLE_H
#define POLY_BUNDLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Section type constants */
#define POLY_BUNDLE_IR       0x01
#define POLY_BUNDLE_WEIGHTS  0x02
#define POLY_BUNDLE_METADATA 0x03

#define POLY_BUNDLE_VERSION  1
#define POLY_BUNDLE_MAGIC    "POLYBNDL"

/* ── Encode ──────────────────────────────────────────────────────────── */

/* Encode a bundle from IR bytes + optional weights bytes + optional metadata JSON.
 * ir_data/ir_len: required (poly.ir.uops@1 bytes from poly_ir_export)
 * weights_data/weights_len: optional (safetensors bytes, pass NULL/0 to omit)
 * metadata_json: optional (UTF-8 JSON string, pass NULL to omit)
 * Returns malloc'd bytes. Caller frees. Sets *out_len.
 * Returns NULL on error. */
uint8_t *poly_bundle_encode(const uint8_t *ir_data, int ir_len,
                            const uint8_t *weights_data, int weights_len,
                            const char *metadata_json,
                            int *out_len);

/* ── Decode ──────────────────────────────────────────────────────────── */

/* Decoded bundle sections (zero-copy pointers into input data). */
typedef struct {
  const uint8_t *ir_data;       /* pointer into input, or NULL */
  int ir_len;
  const uint8_t *weights_data;  /* pointer into input, or NULL */
  int weights_len;
  const char *metadata_json;    /* pointer into input, or NULL (not NUL-terminated) */
  int metadata_len;
  uint32_t version;
  uint32_t flags;
} PolyBundleSections;

/* Decode a bundle, returning zero-copy section pointers.
 * Does NOT copy data -- pointers are valid only while input data is alive.
 * Returns 0 on success, -1 on error (bad magic, truncated, etc). */
int poly_bundle_decode(const uint8_t *data, int len, PolyBundleSections *out);

/* ── Convenience: Instance round-trip ────────────────────────────────── */

/* Forward declaration */
typedef struct PolyInstance PolyInstance;

/* Save a PolyInstance as a bundle.
 * Exports IR + weights + metadata into a single byte array.
 * Caller frees returned bytes. Returns NULL on error. */
uint8_t *poly_instance_save_bundle(PolyInstance *inst, int *out_len);

/* Load a PolyInstance from a bundle.
 * Decodes bundle, imports IR, imports weights.
 * Returns NULL on error. */
PolyInstance *poly_instance_from_bundle(const uint8_t *data, int len);

#ifdef __cplusplus
}
#endif

#endif /* POLY_BUNDLE_H */
