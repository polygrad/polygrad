/*
 * wasm_builder.h — WASM binary module builder
 *
 * Utilities for constructing valid WASM binary modules byte-by-byte.
 * Used by render_wasm.c to emit fused kernels as WASM bytecodes.
 *
 * Reference: WebAssembly Binary Format Specification
 * https://webassembly.github.io/spec/core/binary/
 */

#ifndef POLY_WASM_BUILDER_H
#define POLY_WASM_BUILDER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ── Byte buffer ─────────────────────────────────────────────────────── */

typedef struct {
  uint8_t *data;
  int len;
  int cap;
} WasmBuf;

void wb_init(WasmBuf *b);
void wb_free(WasmBuf *b);
void wb_byte(WasmBuf *b, uint8_t v);
void wb_bytes(WasmBuf *b, const uint8_t *src, int n);
void wb_uleb128(WasmBuf *b, uint64_t v);   /* unsigned LEB128 */
void wb_sleb128(WasmBuf *b, int64_t v);    /* signed LEB128 */
void wb_f32(WasmBuf *b, float v);          /* IEEE 754 little-endian */
void wb_f64(WasmBuf *b, double v);
void wb_name(WasmBuf *b, const char *s);   /* LEB128 length + UTF-8 bytes */

/* Append entire content of src into dst */
void wb_append(WasmBuf *dst, const WasmBuf *src);

/* ── WASM value types ────────────────────────────────────────────────── */

#define WASM_TYPE_I32     0x7F
#define WASM_TYPE_I64     0x7E
#define WASM_TYPE_F32     0x7D
#define WASM_TYPE_F64     0x7C
#define WASM_TYPE_V128    0x7B
#define WASM_TYPE_FUNCREF 0x70
#define WASM_TYPE_FUNC    0x60  /* function type constructor */

/* ── WASM section IDs ────────────────────────────────────────────────── */

#define WASM_SEC_TYPE     1
#define WASM_SEC_IMPORT   2
#define WASM_SEC_FUNCTION 3
#define WASM_SEC_TABLE    4
#define WASM_SEC_MEMORY   5
#define WASM_SEC_GLOBAL   6
#define WASM_SEC_EXPORT   7
#define WASM_SEC_START    8
#define WASM_SEC_ELEMENT  9
#define WASM_SEC_CODE     10
#define WASM_SEC_DATA     11

/* ── WASM export kinds ───────────────────────────────────────────────── */

#define WASM_EXPORT_FUNC   0x00
#define WASM_EXPORT_TABLE  0x01
#define WASM_EXPORT_MEMORY 0x02
#define WASM_EXPORT_GLOBAL 0x03

/* ── WASM opcodes: control flow ──────────────────────────────────────── */

#define WASM_OP_UNREACHABLE   0x00
#define WASM_OP_NOP           0x01
#define WASM_OP_BLOCK         0x02
#define WASM_OP_LOOP          0x03
#define WASM_OP_IF            0x04
#define WASM_OP_ELSE          0x05
#define WASM_OP_END           0x0B
#define WASM_OP_BR            0x0C
#define WASM_OP_BR_IF         0x0D
#define WASM_OP_RETURN        0x0F
#define WASM_OP_CALL          0x10
#define WASM_OP_SELECT        0x1B

/* ── WASM opcodes: variable access ───────────────────────────────────── */

#define WASM_OP_LOCAL_GET     0x20
#define WASM_OP_LOCAL_SET     0x21
#define WASM_OP_LOCAL_TEE     0x22

/* ── WASM opcodes: memory ────────────────────────────────────────────── */

#define WASM_OP_I32_LOAD      0x28
#define WASM_OP_F32_LOAD      0x2A
#define WASM_OP_F64_LOAD      0x2B
#define WASM_OP_I32_STORE     0x36
#define WASM_OP_F32_STORE     0x38
#define WASM_OP_F64_STORE     0x39

/* ── WASM opcodes: i32 ───────────────────────────────────────────────── */

#define WASM_OP_I32_CONST     0x41
#define WASM_OP_I32_EQZ       0x45
#define WASM_OP_I32_EQ        0x46
#define WASM_OP_I32_NE        0x47
#define WASM_OP_I32_LT_S      0x48
#define WASM_OP_I32_LT_U      0x49
#define WASM_OP_I32_GT_S      0x4A
#define WASM_OP_I32_GT_U      0x4B
#define WASM_OP_I32_LE_S      0x4C
#define WASM_OP_I32_LE_U      0x4D
#define WASM_OP_I32_GE_S      0x4E
#define WASM_OP_I32_GE_U      0x4F
#define WASM_OP_I32_ADD       0x6A
#define WASM_OP_I32_SUB       0x6B
#define WASM_OP_I32_MUL       0x6C
#define WASM_OP_I32_DIV_S     0x6D
#define WASM_OP_I32_REM_S     0x6F
#define WASM_OP_I32_AND       0x71
#define WASM_OP_I32_OR        0x72
#define WASM_OP_I32_XOR       0x73
#define WASM_OP_I32_SHL       0x74
#define WASM_OP_I32_SHR_S     0x75
#define WASM_OP_I32_SHR_U     0x76

/* ── WASM opcodes: f32 ───────────────────────────────────────────────── */

#define WASM_OP_F32_CONST     0x43
#define WASM_OP_F32_EQ        0x5B
#define WASM_OP_F32_NE        0x5C
#define WASM_OP_F32_LT        0x5D
#define WASM_OP_F32_GT        0x5E
#define WASM_OP_F32_ABS       0x8B
#define WASM_OP_F32_NEG       0x8C
#define WASM_OP_F32_CEIL      0x8D
#define WASM_OP_F32_FLOOR     0x8E
#define WASM_OP_F32_TRUNC     0x8F
#define WASM_OP_F32_SQRT      0x91
#define WASM_OP_F32_ADD       0x92
#define WASM_OP_F32_SUB       0x93
#define WASM_OP_F32_MUL       0x94
#define WASM_OP_F32_DIV       0x95
#define WASM_OP_F32_MIN       0x96
#define WASM_OP_F32_MAX       0x97
#define WASM_OP_F32_COPYSIGN  0x98

/* ── WASM opcodes: conversions ───────────────────────────────────────── */

#define WASM_OP_I32_TRUNC_F32_S     0xA8
#define WASM_OP_I32_TRUNC_F32_U     0xA9
#define WASM_OP_F32_CONVERT_I32_S   0xB2
#define WASM_OP_F32_CONVERT_I32_U   0xB3
#define WASM_OP_I32_REINTERPRET_F32 0xBC
#define WASM_OP_F32_REINTERPRET_I32 0xBE

/* ── WASM SIMD prefix + opcodes ──────────────────────────────────────── */

#define WASM_SIMD_PREFIX      0xFD

/* SIMD opcodes (LEB128-encoded after prefix) */
#define WASM_SIMD_V128_LOAD       0x00
#define WASM_SIMD_V128_STORE      0x0B
#define WASM_SIMD_V128_CONST      0x0C
#define WASM_SIMD_F32X4_SPLAT     0x13
#define WASM_SIMD_F32X4_EXTRACT   0x1B
#define WASM_SIMD_F32X4_ABS       0xE0
#define WASM_SIMD_F32X4_NEG       0xE1
#define WASM_SIMD_F32X4_SQRT      0xE3
#define WASM_SIMD_F32X4_ADD       0xE4
#define WASM_SIMD_F32X4_SUB       0xE5
#define WASM_SIMD_F32X4_MUL       0xE6
#define WASM_SIMD_F32X4_DIV       0xE7
#define WASM_SIMD_F32X4_MIN       0xE8
#define WASM_SIMD_F32X4_MAX       0xE9
#define WASM_SIMD_V128_BITSELECT  0x52

/* ── Block types ─────────────────────────────────────────────────────── */

#define WASM_BLOCKTYPE_VOID  0x40
#define WASM_BLOCKTYPE_I32   WASM_TYPE_I32
#define WASM_BLOCKTYPE_F32   WASM_TYPE_F32

/* ── Section-level helpers ───────────────────────────────────────────── */

/* Write WASM module header (\0asm + version 1) */
void wb_module_header(WasmBuf *b);

/* Write a section: id byte + LEB128(content.len) + content bytes */
void wb_section(WasmBuf *out, uint8_t id, const WasmBuf *content);

#endif /* POLY_WASM_BUILDER_H */
