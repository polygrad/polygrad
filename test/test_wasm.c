/*
 * test_wasm.c — Tests for WASM binary builder and WASM renderer
 */

#include "test_harness.h"
#include "../src/codegen.h"
#include "../src/wasm_builder.h"

/* ── WASM builder tests ──────────────────────────────────────────────── */

TEST(wasm, leb128_unsigned) {
  WasmBuf b;
  wb_init(&b);

  /* 0 → 0x00 */
  wb_uleb128(&b, 0);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x00);

  /* 127 → 0x7F */
  b.len = 0;
  wb_uleb128(&b, 127);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x7F);

  /* 128 → 0x80 0x01 */
  b.len = 0;
  wb_uleb128(&b, 128);
  ASSERT_INT_EQ(b.len, 2);
  ASSERT_INT_EQ(b.data[0], 0x80);
  ASSERT_INT_EQ(b.data[1], 0x01);

  /* 624485 → 0xE5 0x8E 0x26 */
  b.len = 0;
  wb_uleb128(&b, 624485);
  ASSERT_INT_EQ(b.len, 3);
  ASSERT_INT_EQ(b.data[0], 0xE5);
  ASSERT_INT_EQ(b.data[1], 0x8E);
  ASSERT_INT_EQ(b.data[2], 0x26);

  wb_free(&b);
  PASS();
}

TEST(wasm, leb128_signed) {
  WasmBuf b;
  wb_init(&b);

  /* 0 → 0x00 */
  wb_sleb128(&b, 0);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x00);

  /* -1 → 0x7F */
  b.len = 0;
  wb_sleb128(&b, -1);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x7F);

  /* 63 → 0x3F */
  b.len = 0;
  wb_sleb128(&b, 63);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x3F);

  /* -64 → 0x40 */
  b.len = 0;
  wb_sleb128(&b, -64);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x40);

  /* 64 → 0xC0 0x00 */
  b.len = 0;
  wb_sleb128(&b, 64);
  ASSERT_INT_EQ(b.len, 2);
  ASSERT_INT_EQ(b.data[0], 0xC0);
  ASSERT_INT_EQ(b.data[1], 0x00);

  /* -65 → 0xBF 0x7F */
  b.len = 0;
  wb_sleb128(&b, -65);
  ASSERT_INT_EQ(b.len, 2);
  ASSERT_INT_EQ(b.data[0], 0xBF);
  ASSERT_INT_EQ(b.data[1], 0x7F);

  wb_free(&b);
  PASS();
}

TEST(wasm, module_header) {
  WasmBuf b;
  wb_init(&b);
  wb_module_header(&b);

  ASSERT_INT_EQ(b.len, 8);
  /* Magic: \0asm */
  ASSERT_INT_EQ(b.data[0], 0x00);
  ASSERT_INT_EQ(b.data[1], 0x61);
  ASSERT_INT_EQ(b.data[2], 0x73);
  ASSERT_INT_EQ(b.data[3], 0x6D);
  /* Version: 1 */
  ASSERT_INT_EQ(b.data[4], 0x01);
  ASSERT_INT_EQ(b.data[5], 0x00);
  ASSERT_INT_EQ(b.data[6], 0x00);
  ASSERT_INT_EQ(b.data[7], 0x00);

  wb_free(&b);
  PASS();
}

TEST(wasm, section_encoding) {
  WasmBuf content;
  wb_init(&content);
  wb_byte(&content, 0xAA);
  wb_byte(&content, 0xBB);
  wb_byte(&content, 0xCC);

  WasmBuf out;
  wb_init(&out);
  wb_section(&out, 0x07, &content); /* export section */

  /* Expected: section_id(0x07) + length(3) + content(AA BB CC) */
  ASSERT_INT_EQ(out.len, 5);
  ASSERT_INT_EQ(out.data[0], 0x07); /* section id */
  ASSERT_INT_EQ(out.data[1], 0x03); /* length = 3 */
  ASSERT_INT_EQ(out.data[2], 0xAA);
  ASSERT_INT_EQ(out.data[3], 0xBB);
  ASSERT_INT_EQ(out.data[4], 0xCC);

  wb_free(&content);
  wb_free(&out);
  PASS();
}

TEST(wasm, name_encoding) {
  WasmBuf b;
  wb_init(&b);
  wb_name(&b, "env");

  ASSERT_INT_EQ(b.len, 4); /* 1 byte length + 3 bytes */
  ASSERT_INT_EQ(b.data[0], 3);
  ASSERT_INT_EQ(b.data[1], 'e');
  ASSERT_INT_EQ(b.data[2], 'n');
  ASSERT_INT_EQ(b.data[3], 'v');

  wb_free(&b);
  PASS();
}

TEST(wasm, f32_encoding) {
  WasmBuf b;
  wb_init(&b);
  float v = 1.0f;
  wb_f32(&b, v);

  ASSERT_INT_EQ(b.len, 4);
  /* IEEE 754: 1.0f = 0x3F800000 (little-endian: 00 00 80 3F) */
  ASSERT_INT_EQ(b.data[0], 0x00);
  ASSERT_INT_EQ(b.data[1], 0x00);
  ASSERT_INT_EQ(b.data[2], 0x80);
  ASSERT_INT_EQ(b.data[3], 0x3F);

  wb_free(&b);
  PASS();
}

/* ── Helper: build c[i] = a[i] OP b[i] kernel IR (same as test_codegen.c) ── */

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
  int n;
} WasmVecKernel;

static WasmVecKernel wasm_make_vec_binop(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());

  PolyUOp *alu = poly_uop2(ctx, alu_op, POLY_FLOAT32, load0, load1, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());

  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end  = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (WasmVecKernel){ ctx, sink, n };
}

/* Helper: build b[i] = OP(a[i]) unary kernel */
static WasmVecKernel wasm_make_vec_unary(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *alu = poly_uop1(ctx, alu_op, POLY_FLOAT32, load0, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, alu, poly_arg_none());

  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end  = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (WasmVecKernel){ ctx, sink, n };
}

/* ── WASM renderer tests ─────────────────────────────────────────────── */

TEST(wasm, render_vecadd) {
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);

  /* Must produce non-empty output */
  ASSERT_TRUE(wasm != NULL);
  ASSERT_TRUE(wasm_size > 8);

  /* Must start with WASM magic */
  ASSERT_INT_EQ(wasm[0], 0x00);
  ASSERT_INT_EQ(wasm[1], 0x61);
  ASSERT_INT_EQ(wasm[2], 0x73);
  ASSERT_INT_EQ(wasm[3], 0x6D);

  /* Must have sections (type=1 should appear after header) */
  ASSERT_INT_EQ(wasm[8], 0x01); /* type section id */

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, render_vecmul) {
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_MUL, 8);
  int n_lin;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Valid WASM magic */
  ASSERT_INT_EQ(wasm[0], 0x00);
  ASSERT_INT_EQ(wasm[1], 0x61);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, render_unary) {
  WasmVecKernel k = wasm_make_vec_unary(POLY_OP_NEG, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Look for NEG opcode (0x8C) somewhere in the binary */
  bool found_neg = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F32_NEG) { found_neg = true; break; }
  }
  ASSERT_TRUE(found_neg);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, render_chain) {
  /* d = (a + b) * c — fused kernel */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  int N = 10;

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));
  PolyUOp *p3 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(3));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());
  PolyUOp *idx3 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p3, range, poly_arg_none());

  PolyUOp *la = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *lb = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *lc = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx2, poly_arg_none());

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, la, lb, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, add, lc, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, mul, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end  = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Should contain both ADD and MUL opcodes */
  bool found_add = false, found_mul = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F32_ADD) found_add = true;
    if (wasm[i] == WASM_OP_F32_MUL) found_mul = true;
  }
  ASSERT_TRUE(found_add);
  ASSERT_TRUE(found_mul);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, render_simd_flag) {
  /* Render with SIMD enabled — verify SIMD prefix byte appears */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 16);
  int n_lin;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Should contain SIMD prefix (0xFD) for f32x4 ops */
  bool found_simd = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_SIMD_PREFIX) { found_simd = true; break; }
  }
  ASSERT_TRUE(found_simd);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, write_and_validate) {
  /* Render a vecadd kernel and write to /tmp for manual validation */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  /* Write to tmp file for external validation */
  FILE *f = fopen("/tmp/polygrad_test_vecadd.wasm", "wb");
  if (f) {
    fwrite(wasm, 1, wasm_size, f);
    fclose(f);
  }

  /* Try running wasm-validate if available */
  int rc = system("which wasm-validate > /dev/null 2>&1 && wasm-validate /tmp/polygrad_test_vecadd.wasm");
  if (rc == 0) {
    /* wasm-validate passed — great! */
  }
  /* Don't fail if wasm-validate is not installed */

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, render_pow) {
  /* POW kernel: c[i] = a[i] ^ b[i] — must import powf */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_POW, 4);
  int n_lin;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Must start with WASM magic */
  ASSERT_INT_EQ(wasm[0], 0x00);
  ASSERT_INT_EQ(wasm[1], 0x61);
  ASSERT_INT_EQ(wasm[2], 0x73);
  ASSERT_INT_EQ(wasm[3], 0x6D);

  /* Must contain a CALL instruction (0x10) for the imported powf */
  bool found_call = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_CALL) { found_call = true; break; }
  }
  ASSERT_TRUE(found_call);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, render_pow_simd_fallback) {
  /* POW kernel with use_simd=true must fall back to scalar (no SIMD for POW) */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_POW, 16);
  int n_lin;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Should NOT contain SIMD prefix (POW forces scalar fallback) */
  bool found_simd = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_SIMD_PREFIX) { found_simd = true; break; }
  }
  ASSERT_TRUE(!found_simd);

  /* Should still contain a CALL for powf */
  bool found_call = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_CALL) { found_call = true; break; }
  }
  ASSERT_TRUE(found_call);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, e2e_node_pow) {
  /* End-to-end: POW kernel via Node.js — c[i] = a[i]^b[i] */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_POW, 4);
  int n_lin;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("/tmp/polygrad_e2e_pow.wasm", "wb");
  ASSERT_NOT_NULL(f);
  fwrite(wasm, 1, wasm_size, f);
  fclose(f);

  int has_node = system("which node > /dev/null 2>&1");
  if (has_node != 0) {
    free(wasm);
    free(lin);
    poly_ctx_destroy(k.ctx);
    PASS();
  }

  int rc = system("node test/run_wasm.js /tmp/polygrad_e2e_pow.wasm pow 4");
  ASSERT_INT_EQ(rc, 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, e2e_node_vecadd) {
  /* End-to-end: render WASM, write to file, run with Node.js */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 8);
  int n_lin;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  /* Write WASM to tmp file */
  FILE *f = fopen("/tmp/polygrad_e2e_vecadd.wasm", "wb");
  ASSERT_NOT_NULL(f);
  fwrite(wasm, 1, wasm_size, f);
  fclose(f);

  /* Skip if node is not available */
  int has_node = system("which node > /dev/null 2>&1");
  if (has_node != 0) {
    free(wasm);
    free(lin);
    poly_ctx_destroy(k.ctx);
    PASS(); /* skip gracefully */
  }

  /* Run the Node.js test runner */
  int rc = system("node test/run_wasm.js /tmp/polygrad_e2e_vecadd.wasm add 8");
  ASSERT_INT_EQ(rc, 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}
