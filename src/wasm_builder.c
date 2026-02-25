/*
 * wasm_builder.c — WASM binary module builder
 *
 * Utilities for constructing valid WASM binary modules byte-by-byte.
 * All encoding follows the WebAssembly Binary Format spec.
 *
 * Reference: https://webassembly.github.io/spec/core/binary/
 */

#include "wasm_builder.h"
#include <stdlib.h>
#include <string.h>

/* ── Byte buffer ─────────────────────────────────────────────────────── */

void wb_init(WasmBuf *b) {
  b->cap = 256;
  b->data = malloc(b->cap);
  b->len = 0;
}

void wb_free(WasmBuf *b) {
  free(b->data);
  b->data = NULL;
  b->len = b->cap = 0;
}

static void wb_ensure(WasmBuf *b, int need) {
  while (b->len + need > b->cap) {
    b->cap *= 2;
    b->data = realloc(b->data, b->cap);
  }
}

void wb_byte(WasmBuf *b, uint8_t v) {
  wb_ensure(b, 1);
  b->data[b->len++] = v;
}

void wb_bytes(WasmBuf *b, const uint8_t *src, int n) {
  wb_ensure(b, n);
  memcpy(b->data + b->len, src, n);
  b->len += n;
}

void wb_append(WasmBuf *dst, const WasmBuf *src) {
  wb_bytes(dst, src->data, src->len);
}

/* ── LEB128 encoding ─────────────────────────────────────────────────── */

void wb_uleb128(WasmBuf *b, uint64_t v) {
  do {
    uint8_t byte = v & 0x7F;
    v >>= 7;
    if (v != 0) byte |= 0x80;
    wb_byte(b, byte);
  } while (v != 0);
}

void wb_sleb128(WasmBuf *b, int64_t v) {
  bool more = true;
  while (more) {
    uint8_t byte = v & 0x7F;
    v >>= 7;
    /* sign bit of byte is bit 6 */
    if ((v == 0 && !(byte & 0x40)) || (v == -1 && (byte & 0x40))) {
      more = false;
    } else {
      byte |= 0x80;
    }
    wb_byte(b, byte);
  }
}

/* ── Numeric encoding ────────────────────────────────────────────────── */

void wb_f32(WasmBuf *b, float v) {
  uint8_t bytes[4];
  memcpy(bytes, &v, 4); /* assumes little-endian (WASM spec requires LE) */
  wb_bytes(b, bytes, 4);
}

void wb_f64(WasmBuf *b, double v) {
  uint8_t bytes[8];
  memcpy(bytes, &v, 8);
  wb_bytes(b, bytes, 8);
}

/* ── String encoding ─────────────────────────────────────────────────── */

void wb_name(WasmBuf *b, const char *s) {
  int len = (int)strlen(s);
  wb_uleb128(b, (uint64_t)len);
  wb_bytes(b, (const uint8_t *)s, len);
}

/* ── Module-level helpers ────────────────────────────────────────────── */

void wb_module_header(WasmBuf *b) {
  /* Magic number: \0asm */
  wb_byte(b, 0x00);
  wb_byte(b, 0x61); /* 'a' */
  wb_byte(b, 0x73); /* 's' */
  wb_byte(b, 0x6D); /* 'm' */
  /* Version: 1 */
  wb_byte(b, 0x01);
  wb_byte(b, 0x00);
  wb_byte(b, 0x00);
  wb_byte(b, 0x00);
}

void wb_section(WasmBuf *out, uint8_t id, const WasmBuf *content) {
  wb_byte(out, id);
  wb_uleb128(out, (uint64_t)content->len);
  wb_append(out, content);
}
