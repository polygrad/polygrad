/*
 * bundle.c -- poly.bundle@1 container format encode/decode
 */

#include "bundle.h"
#include "ir.h"
#include "safetensors.h"
#include "instance.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Little-endian helpers */

static void write_le32(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v);
  dst[1] = (uint8_t)(v >> 8);
  dst[2] = (uint8_t)(v >> 16);
  dst[3] = (uint8_t)(v >> 24);
}

static uint32_t read_le32(const uint8_t *src) {
  return (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16) |
         ((uint32_t)src[3] << 24);
}

typedef struct {
  char *data;
  int len;
  int cap;
  bool failed;
} JsonBuf;

static void jb_reserve(JsonBuf *b, int extra) {
  if (!b || b->failed || extra < 0) return;
  if (b->len + extra + 1 <= b->cap) return;
  int new_cap = b->cap ? b->cap : 256;
  while (new_cap < b->len + extra + 1) {
    if (new_cap > INT32_MAX / 2) {
      b->failed = true;
      return;
    }
    new_cap *= 2;
  }
  char *new_data = realloc(b->data, (size_t)new_cap);
  if (!new_data) {
    b->failed = true;
    return;
  }
  b->data = new_data;
  b->cap = new_cap;
}

static void jb_add_n(JsonBuf *b, const char *s, int n) {
  if (!s || n <= 0) return;
  jb_reserve(b, n);
  if (!b || b->failed) return;
  memcpy(b->data + b->len, s, (size_t)n);
  b->len += n;
  b->data[b->len] = '\0';
}

static void jb_add(JsonBuf *b, const char *s) {
  if (s) jb_add_n(b, s, (int)strlen(s));
}

static void jb_add_u32(JsonBuf *b, uint32_t v) {
  char tmp[16];
  snprintf(tmp, sizeof(tmp), "%u", v);
  jb_add(b, tmp);
}

static void jb_add_json_string(JsonBuf *b, const char *s) {
  jb_add(b, "\"");
  for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
    switch (*p) {
    case '\"':
      jb_add(b, "\\\"");
      break;
    case '\\':
      jb_add(b, "\\\\");
      break;
    case '\b':
      jb_add(b, "\\b");
      break;
    case '\f':
      jb_add(b, "\\f");
      break;
    case '\n':
      jb_add(b, "\\n");
      break;
    case '\r':
      jb_add(b, "\\r");
      break;
    case '\t':
      jb_add(b, "\\t");
      break;
    default:
      if (*p < 0x20) {
        char tmp[8];
        snprintf(tmp, sizeof(tmp), "\\u%04x", *p);
        jb_add(b, tmp);
      } else {
        jb_add_n(b, (const char *)p, 1);
      }
      break;
    }
  }
  jb_add(b, "\"");
}

static void jb_add_string_array(JsonBuf *b, const char *key, const char **items, int n_items) {
  jb_add_json_string(b, key);
  jb_add(b, ":[");
  for (int i = 0; i < n_items; i++) {
    if (i) jb_add(b, ",");
    jb_add_json_string(b, items[i]);
  }
  jb_add(b, "]");
}

static char *bundle_metadata_from_ir(const uint8_t *ir_data, int ir_len) {
  PolyIrSpec spec;
  if (poly_ir_import(ir_data, ir_len, &spec) != 0) return NULL;

  JsonBuf b = {0};
  jb_add(&b, "{\"format\":\"poly.bundle@1\",\"ir_format\":\"poly.ir.uops@2\",\"entrypoints\":[");
  for (int i = 0; i < spec.n_entrypoints; i++) {
    PolyIrEntrypoint *ep = &spec.entrypoints[i];
    if (i) jb_add(&b, ",");
    jb_add(&b, "{\"name\":");
    jb_add_json_string(&b, ep->name);
    jb_add(&b, ",");
    jb_add_string_array(&b, "inputs", ep->inputs, ep->n_inputs);
    jb_add(&b, ",");
    jb_add_string_array(&b, "outputs", ep->outputs, ep->n_outputs);
    jb_add(&b, ",\"objective\":");
    if (ep->objective)
      jb_add_json_string(&b, ep->objective);
    else
      jb_add(&b, "null");
    jb_add(&b, ",\"flags\":");
    jb_add_u32(&b, ep->flags);
    jb_add(&b, "}");
  }
  jb_add(&b, "]}");

  poly_ir_spec_free(&spec);
  poly_ctx_destroy(spec.ctx);

  if (b.failed) {
    free(b.data);
    return NULL;
  }
  return b.data;
}

/* Encode */

uint8_t *poly_bundle_encode(
    const uint8_t *ir_data,
    int ir_len,
    const uint8_t *weights_data,
    int weights_len,
    const char *metadata_json,
    int *out_len
) {
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
  total += 8 + ir_len; /* IR section header + data */
  if (weights_data && weights_len > 0) total += 8 + weights_len;
  if (meta_len > 0) total += 8 + meta_len;

  uint8_t *buf = malloc((size_t)total);
  if (!buf) {
    if (out_len) *out_len = 0;
    return NULL;
  }

  int pos = 0;

  /* Header */
  memcpy(buf + pos, POLY_BUNDLE_MAGIC, 8);
  pos += 8;
  write_le32(buf + pos, POLY_BUNDLE_VERSION);
  pos += 4;
  write_le32(buf + pos, 0);
  pos += 4; /* flags */
  write_le32(buf + pos, (uint32_t)n_sections);
  pos += 4;

  /* IR section */
  write_le32(buf + pos, POLY_BUNDLE_IR);
  pos += 4;
  write_le32(buf + pos, (uint32_t)ir_len);
  pos += 4;
  memcpy(buf + pos, ir_data, (size_t)ir_len);
  pos += ir_len;

  /* Weights section (optional) */
  if (weights_data && weights_len > 0) {
    write_le32(buf + pos, POLY_BUNDLE_WEIGHTS);
    pos += 4;
    write_le32(buf + pos, (uint32_t)weights_len);
    pos += 4;
    memcpy(buf + pos, weights_data, (size_t)weights_len);
    pos += weights_len;
  }

  /* Metadata section (optional) */
  if (meta_len > 0) {
    write_le32(buf + pos, POLY_BUNDLE_METADATA);
    pos += 4;
    write_le32(buf + pos, (uint32_t)meta_len);
    pos += 4;
    memcpy(buf + pos, metadata_json, (size_t)meta_len);
    pos += meta_len;
  }

  if (out_len) *out_len = pos;
  return buf;
}

/* Decode */

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
    uint32_t stype = read_le32(data + pos);
    pos += 4;
    uint32_t slen = read_le32(data + pos);
    pos += 4;

    if (pos + (int)slen > len) {
      fprintf(
          stderr, "poly_bundle_decode: section %u truncated (need %u, have %d)\n", stype, slen,
          len - pos
      );
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

/* Instance convenience */

uint8_t *poly_instance_save_bundle_ex(PolyInstance *inst, int *out_len, uint32_t weight_flags) {
  if (!inst) {
    if (out_len) *out_len = 0;
    return NULL;
  }

  /* Export IR */
  int ir_len = 0;
  uint8_t *ir_data = poly_instance_export_ir(inst, &ir_len);
  if (!ir_data) {
    if (out_len) *out_len = 0;
    return NULL;
  }

  /* Export weights (safetensors) */
  int weights_len = 0;
  uint8_t *weights_data = poly_instance_export_weights_ex(inst, &weights_len, weight_flags);
  /* weights_data may be NULL if no params -- that's ok */

  char *metadata_json = bundle_metadata_from_ir(ir_data, ir_len);
  if (!metadata_json) {
    free(ir_data);
    free(weights_data);
    if (out_len) *out_len = 0;
    return NULL;
  }

  /* Encode bundle */
  uint8_t *bundle =
      poly_bundle_encode(ir_data, ir_len, weights_data, weights_len, metadata_json, out_len);
  free(ir_data);
  free(weights_data);
  free(metadata_json);
  return bundle;
}

uint8_t *poly_instance_save_bundle(PolyInstance *inst, int *out_len) {
  return poly_instance_save_bundle_ex(inst, out_len, POLY_EXPORT_WEIGHTS_DEFAULT);
}

PolyInstance *poly_instance_from_bundle(const uint8_t *data, int len) {
  PolyBundleSections sections;
  if (poly_bundle_decode(data, len, &sections) != 0) return NULL;

  return poly_instance_from_ir(
      sections.ir_data, sections.ir_len, sections.weights_data, sections.weights_len
  );
}
