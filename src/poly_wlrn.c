/*
 * poly_wlrn.c -- WLRN bundle view/slice reader
 */

#include "poly_wlrn.h"
#include <stdio.h>
#include <string.h>

static uint32_t read_le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int poly_wlrn_view(const uint8_t *data, uint32_t len, PolyWlrnView *out) {
  if (!data || !out) return -1;

  if (len < POLY_WLRN_HEADER_SIZE) {
    fprintf(stderr, "poly_wlrn_view: data too short (%u < %d)\n",
            len, POLY_WLRN_HEADER_SIZE);
    return -1;
  }

  /* Check magic */
  uint32_t magic = read_le32(data);
  if (magic != POLY_WLRN_MAGIC) {
    fprintf(stderr, "poly_wlrn_view: bad magic 0x%08x (expected 0x%08x)\n",
            magic, POLY_WLRN_MAGIC);
    return -1;
  }

  /* Check version */
  uint32_t version = read_le32(data + 4);
  if (version != 1) {
    fprintf(stderr, "poly_wlrn_view: unsupported version %u\n", version);
    return -1;
  }

  uint32_t manifest_len = read_le32(data + 8);
  uint32_t toc_len = read_le32(data + 12);

  /* Bounds check */
  uint64_t total_header = (uint64_t)POLY_WLRN_HEADER_SIZE + manifest_len + toc_len;
  if (total_header > len) {
    fprintf(stderr, "poly_wlrn_view: header region exceeds data length\n");
    return -1;
  }

  out->manifest = data + POLY_WLRN_HEADER_SIZE;
  out->manifest_len = manifest_len;
  out->toc = data + POLY_WLRN_HEADER_SIZE + manifest_len;
  out->toc_len = toc_len;
  out->blobs = data + total_header;
  out->blobs_len = len - (uint32_t)total_header;

  return 0;
}
