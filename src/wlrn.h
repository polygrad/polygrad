/*
 * poly_wlrn.h -- WLRN bundle view/slice reader (read-only)
 *
 * WLRN binary layout:
 *   [4 bytes]  magic "WLRN"
 *   [4 bytes LE] version (1)
 *   [4 bytes LE] manifest JSON length
 *   [4 bytes LE] TOC JSON length
 *   [manifest_len bytes] manifest JSON
 *   [toc_len bytes] TOC JSON
 *   [remaining] blob region
 */

#ifndef POLY_WLRN_H
#define POLY_WLRN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define POLY_WLRN_MAGIC     0x4E524C57  /* "WLRN" LE */
#define POLY_WLRN_HEADER_SIZE 16

typedef struct {
  const uint8_t *manifest;    /* ptr into input data */
  uint32_t manifest_len;
  const uint8_t *toc;         /* ptr into input data */
  uint32_t toc_len;
  const uint8_t *blobs;       /* ptr into input data */
  uint32_t blobs_len;
} PolyWlrnView;

/* Parse a WLRN bundle into a zero-copy view.
 * Returns 0 on success, -1 on error. */
int poly_wlrn_view(const uint8_t *data, uint32_t len, PolyWlrnView *out);

#ifdef __cplusplus
}
#endif

#endif /* POLY_WLRN_H */
