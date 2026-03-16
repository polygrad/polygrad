/*
 * bundle.c -- poly.bundle@1 container format encode/decode
 */

#include "bundle.h"
#include "ir.h"
#include "safetensors.h"
#include "instance.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Little-endian helpers ────────────────────────────────────────────── */

static void write_le32(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v);
  dst[1] = (uint8_t)(v >> 8);
  dst[2] = (uint8_t)(v >> 16);
  dst[3] = (uint8_t)(v >> 24);
}

static uint32_t read_le32(const uint8_t *src) {
  return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
         ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

/* ── Encode ──────────────────────────────────────────────────────────── */

uint8_t *poly_bundle_encode(const uint8_t *ir_data, int ir_len,
                            const uint8_t *weights_data, int weights_len,
                            const char *metadata_json,
                            int *out_len) {
  if (!ir_data || ir_len <= 0) {
    fprintf(stderr, "poly_bundle_encode: IR section is required\n");
    if (out_len) *out_len = 0;
    return NULL;
  }

  /* Count sections */
  int n_sections = 1; /* IR always present */
  if (weights_data && weights_len > 0) n_sections++;
  if (metadata_json && metadata_json[0]) n_sections++;

  int meta_len = metadata_json ? (int)strlen(metadata_json) : 0;

  /* Calculate total size:
   *   header: 8 (magic) + 4 (version) + 4 (flags) + 4 (n_sections) = 20
   *   per section: 4 (type) + 4 (length) + data */
  int total = 20;
  total += 8 + ir_len;  /* IR section header + data */
  if (weights_data && weights_len > 0)
    total += 8 + weights_len;
  if (meta_len > 0)
    total += 8 + meta_len;

  uint8_t *buf = malloc((size_t)total);
  if (!buf) { if (out_len) *out_len = 0; return NULL; }

  int pos = 0;

  /* Header */
  memcpy(buf + pos, POLY_BUNDLE_MAGIC, 8); pos += 8;
  write_le32(buf + pos, POLY_BUNDLE_VERSION); pos += 4;
  write_le32(buf + pos, 0); pos += 4; /* flags */
  write_le32(buf + pos, (uint32_t)n_sections); pos += 4;

  /* IR section */
  write_le32(buf + pos, POLY_BUNDLE_IR); pos += 4;
  write_le32(buf + pos, (uint32_t)ir_len); pos += 4;
  memcpy(buf + pos, ir_data, (size_t)ir_len); pos += ir_len;

  /* Weights section (optional) */
  if (weights_data && weights_len > 0) {
    write_le32(buf + pos, POLY_BUNDLE_WEIGHTS); pos += 4;
    write_le32(buf + pos, (uint32_t)weights_len); pos += 4;
    memcpy(buf + pos, weights_data, (size_t)weights_len); pos += weights_len;
  }

  /* Metadata section (optional) */
  if (meta_len > 0) {
    write_le32(buf + pos, POLY_BUNDLE_METADATA); pos += 4;
    write_le32(buf + pos, (uint32_t)meta_len); pos += 4;
    memcpy(buf + pos, metadata_json, (size_t)meta_len); pos += meta_len;
  }

  if (out_len) *out_len = pos;
  return buf;
}

/* ── Decode ──────────────────────────────────────────────────────────── */

int poly_bundle_decode(const uint8_t *data, int len, PolyBundleSections *out) {
  if (!data || !out) return -1;
  memset(out, 0, sizeof(*out));

  /* Minimum size: 20-byte header */
  if (len < 20) {
    fprintf(stderr, "poly_bundle_decode: too short (%d bytes)\n", len);
    return -1;
  }

  /* Check magic */
  if (memcmp(data, POLY_BUNDLE_MAGIC, 8) != 0) {
    fprintf(stderr, "poly_bundle_decode: bad magic\n");
    return -1;
  }

  out->version = read_le32(data + 8);
  out->flags = read_le32(data + 12);
  uint32_t n_sections = read_le32(data + 16);

  if (out->version != POLY_BUNDLE_VERSION) {
    fprintf(stderr, "poly_bundle_decode: unsupported version %u\n", out->version);
    return -1;
  }

  /* Parse sections */
  int pos = 20;
  for (uint32_t i = 0; i < n_sections; i++) {
    if (pos + 8 > len) {
      fprintf(stderr, "poly_bundle_decode: truncated section header at offset %d\n", pos);
      return -1;
    }
    uint32_t stype = read_le32(data + pos); pos += 4;
    uint32_t slen = read_le32(data + pos); pos += 4;

    if (pos + (int)slen > len) {
      fprintf(stderr, "poly_bundle_decode: section %u truncated (need %u, have %d)\n",
              stype, slen, len - pos);
      return -1;
    }

    switch (stype) {
      case POLY_BUNDLE_IR:
        out->ir_data = data + pos;
        out->ir_len = (int)slen;
        break;
      case POLY_BUNDLE_WEIGHTS:
        out->weights_data = data + pos;
        out->weights_len = (int)slen;
        break;
      case POLY_BUNDLE_METADATA:
        out->metadata_json = (const char *)(data + pos);
        out->metadata_len = (int)slen;
        break;
      default:
        /* Unknown section: skip (forward compatibility) */
        break;
    }
    pos += (int)slen;
  }

  if (!out->ir_data) {
    fprintf(stderr, "poly_bundle_decode: missing IR section\n");
    return -1;
  }

  return 0;
}

/* ── Instance convenience ─────────────────────────────────────────────── */

uint8_t *poly_instance_save_bundle(PolyInstance *inst, int *out_len) {
  if (!inst) { if (out_len) *out_len = 0; return NULL; }

  /* Export IR */
  int ir_len = 0;
  uint8_t *ir_data = poly_instance_export_ir(inst, &ir_len);
  if (!ir_data) { if (out_len) *out_len = 0; return NULL; }

  /* Export weights (safetensors) */
  int weights_len = 0;
  uint8_t *weights_data = poly_instance_export_weights(inst, &weights_len);
  /* weights_data may be NULL if no params -- that's ok */

  /* Encode bundle */
  uint8_t *bundle = poly_bundle_encode(ir_data, ir_len,
                                        weights_data, weights_len,
                                        NULL, /* no metadata yet */
                                        out_len);
  free(ir_data);
  free(weights_data);
  return bundle;
}

PolyInstance *poly_instance_from_bundle(const uint8_t *data, int len) {
  PolyBundleSections sections;
  if (poly_bundle_decode(data, len, &sections) != 0)
    return NULL;

  return poly_instance_from_ir(sections.ir_data, sections.ir_len,
                               sections.weights_data, sections.weights_len);
}
