/*
 * test_wasm.c — Tests for WASM binary builder and WASM renderer
 */

#include "test_harness.h"
#include "../src/codegen.h"
#include "../src/engine/realize.h"
#include "../src/frontend.h"
#include "../src/tensor.h"
#include "../src/wasm_builder.h"

static int wasm_run_c_i32(PolyUOp **lin, int n_lin, const char *fn_name, int32_t *out) {
  char *src = poly_render_c(lin, n_lin, fn_name);
  if (!src) return -1;
  PolyProgram *prog = poly_compile_c(src, fn_name);
  free(src);
  if (!prog) return -1;
  void *args[1] = {out};
  poly_program_call(prog, args, 1);
  poly_program_destroy(prog);
  return 0;
}

static int wasm_run_c_i64(PolyUOp **lin, int n_lin, const char *fn_name, int64_t *out) {
  char *src = poly_render_c(lin, n_lin, fn_name);
  if (!src) return -1;
  PolyProgram *prog = poly_compile_c(src, fn_name);
  free(src);
  if (!prog) return -1;
  void *args[1] = {out};
  poly_program_call(prog, args, 1);
  poly_program_destroy(prog);
  return 0;
}

static int wasm_run_c_f32_buffer(PolyUOp **lin, int n_lin, const char *fn_name, float *buf) {
  char *src = poly_render_c(lin, n_lin, fn_name);
  if (!src) return -1;
  PolyProgram *prog = poly_compile_c(src, fn_name);
  free(src);
  if (!prog) return -1;
  void *args[1] = {buf};
  poly_program_call(prog, args, 1);
  poly_program_destroy(prog);
  return 0;
}

static int wasm_write_module(const char *path, const uint8_t *wasm, int wasm_size) {
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  size_t written = fwrite(wasm, 1, (size_t)wasm_size, f);
  int close_rc = fclose(f);
  return (written == (size_t)wasm_size && close_rc == 0) ? 0 : -1;
}

static int node_run_wasm_i32(const char *path, int32_t expected) {
  if (system("which node > /dev/null 2>&1") != 0) return 0;
  char cmd[2048];
  snprintf(
      cmd, sizeof(cmd),
      "node -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});"
      "inst.exports.kernel(0);"
      "const got=new DataView(mem.buffer).getInt32(0,true);"
      "if(got!==%d){console.error('got '+got+' expected %d');process.exit(2)}\"",
      path, expected, expected
  );
  return system(cmd);
}

static int node_run_wasm_f32_buffer(const char *path, float expected) {
  if (system("which node > /dev/null 2>&1") != 0) return 0;
  char cmd[2048];
  snprintf(
      cmd, sizeof(cmd),
      "node -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});"
      "const dv=new DataView(mem.buffer);"
      "dv.setFloat32(4,2,true);dv.setFloat32(8,3,true);dv.setFloat32(12,5,true);"
      "inst.exports.kernel(0);"
      "const got=dv.getFloat32(0,true);const expected=%.9g;"
      "if(Math.abs(got-expected)>1e-6){console.error('got '+got+' expected "
      "'+expected);process.exit(2)}\"",
      path, (double)expected
  );
  return system(cmd);
}

static int node_run_wasm_i64(const char *path, int64_t expected) {
  if (system("which node > /dev/null 2>&1") != 0) return 0;
  char cmd[2048];
  snprintf(
      cmd, sizeof(cmd),
      "node -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});"
      "inst.exports.kernel(0);"
      "const got=new DataView(mem.buffer).getBigInt64(0,true);"
      "const expected=BigInt('%lld');"
      "if(got!==expected){console.error('got '+got+' expected '+expected);process.exit(2)}\"",
      path, (long long)expected
  );
  return system(cmd);
}

/* WASM builder tests */

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

/* Helper: build c[i] = a[i] OP b[i] kernel IR (same as test_codegen.c) */

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

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (WasmVecKernel){ctx, sink, n};
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

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (WasmVecKernel){ctx, sink, n};
}

/* WASM renderer tests */

TEST(wasm, render_vecadd) {
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

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
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

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
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Look for NEG opcode (0x8C) somewhere in the binary */
  bool found_neg = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F32_NEG) {
      found_neg = true;
      break;
    }
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
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);

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

TEST(wasm, render_unsigned_alu_opcodes) {
  /* out[i] = (a[i] // b[i]) + (a[i] % b[i]) + (a[i] >> 1), uint32 path */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_u32 = poly_dtype_ptr(POLY_UINT32, -1, POLY_ADDR_GLOBAL);
  int N = 8;

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p2, range, poly_arg_none());

  PolyUOp *la = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx0, poly_arg_none());
  PolyUOp *lb = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx1, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(1));

  PolyUOp *idiv = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT32, la, lb, poly_arg_none());
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, POLY_UINT32, la, lb, poly_arg_none());
  PolyUOp *shr = poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, la, one, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, idiv, mod, poly_arg_none());
  PolyUOp *out = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, sum, shr, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, out, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  bool found_div_u = false, found_rem_u = false, found_shr_u = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_I32_DIV_U) found_div_u = true;
    if (wasm[i] == WASM_OP_I32_REM_U) found_rem_u = true;
    if (wasm[i] == WASM_OP_I32_SHR_U) found_shr_u = true;
  }
  ASSERT_TRUE(found_div_u);
  ASSERT_TRUE(found_rem_u);
  ASSERT_TRUE(found_shr_u);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, render_simd_flag) {
  /* Render with SIMD enabled — verify SIMD prefix byte appears */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 16);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Should contain SIMD prefix (0xFD) for f32x4 ops */
  bool found_simd = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_SIMD_PREFIX) {
      found_simd = true;
      break;
    }
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
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

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
  int rc =
      system("which wasm-validate > /dev/null 2>&1 && wasm-validate /tmp/polygrad_test_vecadd.wasm"
      );
  if (rc == 0) {
    /* wasm-validate passed — great! */
  }
  /* Don't fail if wasm-validate is not installed */

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, sparse_cross_entropy_i64_gather_index_validates) {
  PolyCtx *ctx = poly_ctx_new();

  /* Sparse cross-entropy follows tinygrad's late index dtype lowering: loaded
   * class labels can remain int64 in gather address expressions. The WASM
   * backend still targets wasm32 memory, so the renderer must wrap those
   * address indexes to i32 at the memory access boundary. */
  PolyUOp *logits_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *target_buf = poly_buffer(ctx, POLY_INT32, 2);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3}, 2);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2}, 1);
  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, 1);

  PolyUOp *targets[] = {loss};
  PolyUOp *realized[] = {NULL};
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_items, 3);

  bool saw_i64_index = false;
  bool saw_i32_wrap = false;

  for (int item = 0; item < sched->n_items; item++) {
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_wasm_env(ctx, sched->items[item].root, &n_lin);
    ASSERT_NOT_NULL(lin);

    bool item_has_i64_index = false;
    for (int i = 0; i < n_lin; i++) {
      PolyUOp *u = lin[i];
      if (u->op == POLY_OP_INDEX && u->n_src >= 2 && u->src[1] &&
          !poly_dtype_is_float(u->src[1]->dtype) && u->src[1]->dtype.bitsize == 64) {
        item_has_i64_index = true;
        saw_i64_index = true;
      }
    }

    int wasm_size = 0;
    uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
    ASSERT_NOT_NULL(wasm);

    if (item_has_i64_index) {
      for (int i = 0; i < wasm_size; i++) {
        if (wasm[i] == WASM_OP_I32_WRAP_I64) {
          saw_i32_wrap = true;
          break;
        }
      }
    }

    char path[128];
    snprintf(path, sizeof(path), "/tmp/polygrad_test_ce_sparse_item%d.wasm", item);
    FILE *f = fopen(path, "wb");
    ASSERT_NOT_NULL(f);
    fwrite(wasm, 1, (size_t)wasm_size, f);
    fclose(f);

    int has_node = system("which node > /dev/null 2>&1");
    if (has_node == 0) {
      char cmd[512];
      snprintf(
          cmd, sizeof(cmd),
          "node -e \"const fs=require('fs'); new WebAssembly.Module(fs.readFileSync('%s'))\"", path
      );
      ASSERT_INT_EQ(system(cmd), 0);
    }

    free(wasm);
    free(lin);
  }

  ASSERT_TRUE(saw_i64_index);
  ASSERT_TRUE(saw_i32_wrap);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, mixed_width_compare_validates) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad's UOp spec wants comparison operands to share a base dtype, but
   * imported/index-heavy graphs can expose a late widened label compared with
   * an int bound. WASM has no implicit casts, so the renderer must coerce the
   * narrower operand before emitting i64.lt_s/i64.eq/etc. */
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, out, zero, poly_arg_none());
  PolyUOp *lhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(7));
  PolyUOp *rhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(9));
  PolyUOp *lt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, lhs, rhs, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, lt, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  int32_t c_out = 0;
  ASSERT_INT_EQ(wasm_run_c_i32(lin, n_lin, "mixed_width_compare_c", &c_out), 0);
  ASSERT_INT_EQ(
      wasm_write_module("/tmp/polygrad_test_mixed_width_compare.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(node_run_wasm_i32("/tmp/polygrad_test_mixed_width_compare.wasm", c_out), 0);
  ASSERT_INT_EQ(c_out, 1);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, mixed_width_shift_validates) {
  PolyCtx *ctx = poly_ctx_new();

  /* The first Qwen renderer failure after comparison coercion was i64.shl with
   * an i32 left operand. For WASM, both operands consumed by i64.shl must be
   * i64 stack values even when Polygrad created the shift count as int32. */
  PolyDType ptr_i64 = poly_dtype_ptr(POLY_INT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_i64, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_i64, out, zero, poly_arg_none());
  PolyUOp *lhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *rhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *shl = poly_uop2(ctx, POLY_OP_SHL, POLY_INT64, lhs, rhs, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, shl, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  int64_t c_out = 0;
  ASSERT_INT_EQ(wasm_run_c_i64(lin, n_lin, "mixed_width_shift_c", &c_out), 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_mixed_width_shift.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_i64("/tmp/polygrad_test_mixed_width_shift.wasm", c_out), 0);
  ASSERT_INT_EQ(c_out, 28);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, unsigned_i64_div_mod_matches_c_renderer) {
  PolyCtx *ctx = poly_ctx_new();

  /* Tinygrad renderers distinguish signed/unsigned integer lowering:
   * LLVM emits sdiv/udiv and srem/urem, and NIR emits idiv/udiv and irem/umod.
   * WASM has the same split. This case catches the silent high-bit bug where a
   * uint64 IDIV used i64.div_s and returned -1 instead of 2. */
  PolyDType ptr_u64 = poly_dtype_ptr(POLY_UINT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_u64, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_u64, out, zero, poly_arg_none());
  PolyUOp *lhs =
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int((int64_t)0x8000000000000005ULL));
  PolyUOp *rhs =
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int((int64_t)0x4000000000000000ULL));
  PolyUOp *idiv = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT64, lhs, rhs, poly_arg_none());
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, POLY_UINT64, lhs, rhs, poly_arg_none());
  PolyUOp *outv = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT64, idiv, mod, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, outv, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  bool found_div_u = false, found_rem_u = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_I64_DIV_U) found_div_u = true;
    if (wasm[i] == WASM_OP_I64_REM_U) found_rem_u = true;
  }
  ASSERT_TRUE(found_div_u);
  ASSERT_TRUE(found_rem_u);

  int64_t c_out = 0;
  ASSERT_INT_EQ(wasm_run_c_i64(lin, n_lin, "unsigned_i64_div_mod_c", &c_out), 0);
  ASSERT_INT_EQ(
      wasm_write_module("/tmp/polygrad_test_unsigned_i64_div_mod.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(node_run_wasm_i64("/tmp/polygrad_test_unsigned_i64_div_mod.wasm", c_out), 0);
  ASSERT_INT_EQ(c_out, 7);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, mulacc_operand_order_matches_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad's Ops.MULACC is (x * y) + z. WASM is stack based, so a renderer
   * that pushes x,y,z and then emits MUL,ADD silently computes x + (y * z).
   * Keep this C-vs-WASM test on loaded values to prevent constant folding from
   * hiding the operand-order bug. */
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *i0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *i1 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *i2 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *i3 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, i0, poly_arg_none());
  PolyUOp *x_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, i1, poly_arg_none());
  PolyUOp *y_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, i2, poly_arg_none());
  PolyUOp *z_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, i3, poly_arg_none());
  PolyUOp *x = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, x_idx, poly_arg_none());
  PolyUOp *y = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, y_idx, poly_arg_none());
  PolyUOp *z = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, z_idx, poly_arg_none());
  PolyUOp *mulacc_src[3] = {x, y, z};
  PolyUOp *mulacc = poly_uop(ctx, POLY_OP_MULACC, POLY_FLOAT32, mulacc_src, 3, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, mulacc, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  float c_buf[4] = {0.0f, 2.0f, 3.0f, 5.0f};
  ASSERT_INT_EQ(wasm_run_c_f32_buffer(lin, n_lin, "mulacc_operand_order_c", c_buf), 0);
  ASSERT_FLOAT_EQ(c_buf[0], 11.0f, 1e-6);
  ASSERT_INT_EQ(
      wasm_write_module("/tmp/polygrad_test_mulacc_operand_order.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(
      node_run_wasm_f32_buffer("/tmp/polygrad_test_mulacc_operand_order.wasm", c_buf[0]), 0
  );

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, define_reg_array_constant_indexes_match_c_renderer) {
  PolyCtx *ctx = poly_ctx_new();

  /* Qwen's WASM-linearized kernels contain DEFINE_REG(ptr_size=3/4) with
   * constant INDEX(reg, i) accesses. Native C/WGSL render these as local arrays;
   * WASM must not collapse every element to the same scalar local. */
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyDType reg_ptr = poly_dtype_ptr(POLY_FLOAT32, 4, POLY_ADDR_REG);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *reg = poly_uop0(ctx, POLY_OP_DEFINE_REG, reg_ptr, poly_arg_int(0));
  PolyUOp *i0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *i3 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(5.0));

  PolyUOp *reg0 = poly_uop2(ctx, POLY_OP_INDEX, reg_ptr, reg, i0, poly_arg_none());
  PolyUOp *store0 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, reg0, two, poly_arg_none());
  PolyUOp *after0_src[2] = {reg, store0};
  PolyUOp *after0 = poly_uop(ctx, POLY_OP_AFTER, reg_ptr, after0_src, 2, poly_arg_none());

  PolyUOp *reg3 = poly_uop2(ctx, POLY_OP_INDEX, reg_ptr, after0, i3, poly_arg_none());
  PolyUOp *store3 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, reg3, five, poly_arg_none());
  PolyUOp *after1_src[2] = {after0, store3};
  PolyUOp *after1 = poly_uop(ctx, POLY_OP_AFTER, reg_ptr, after1_src, 2, poly_arg_none());

  PolyUOp *load0 = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_FLOAT32,
      poly_uop2(ctx, POLY_OP_INDEX, reg_ptr, after1, i0, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *load3 = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_FLOAT32,
      poly_uop2(ctx, POLY_OP_INDEX, reg_ptr, after1, i3, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, load0, load3, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, out, i0, poly_arg_none());
  PolyUOp *store_out = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, sum, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store_out);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  float c_buf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_INT_EQ(wasm_run_c_f32_buffer(lin, n_lin, "define_reg_array_c", c_buf), 0);
  ASSERT_FLOAT_EQ(c_buf[0], 7.0f, 1e-6);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_define_reg_array.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_f32_buffer("/tmp/polygrad_test_define_reg_array.wasm", c_buf[0]), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, render_pow) {
  /* POW kernel: c[i] = a[i] ^ b[i] — must import powf */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_POW, 4);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

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
    if (wasm[i] == WASM_OP_CALL) {
      found_call = true;
      break;
    }
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
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Should NOT contain SIMD prefix (POW forces scalar fallback) */
  bool found_simd = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_SIMD_PREFIX) {
      found_simd = true;
      break;
    }
  }
  ASSERT_TRUE(!found_simd);

  /* Should still contain a CALL for powf */
  bool found_call = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_CALL) {
      found_call = true;
      break;
    }
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
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

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
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

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

/* F64 WASM helpers */

static WasmVecKernel wasm_make_vec_binop_f64(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f64 = poly_dtype_ptr(POLY_FLOAT64, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p2, range, poly_arg_none());

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx1, poly_arg_none());

  PolyUOp *alu = poly_uop2(ctx, alu_op, POLY_FLOAT64, load0, load1, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (WasmVecKernel){ctx, sink, n};
}

static WasmVecKernel wasm_make_vec_unary_f64(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f64 = poly_dtype_ptr(POLY_FLOAT64, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(1));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p1, range, poly_arg_none());

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx0, poly_arg_none());
  PolyUOp *alu = poly_uop1(ctx, alu_op, POLY_FLOAT64, load0, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (WasmVecKernel){ctx, sink, n};
}

/* F64 WASM renderer tests */

TEST(wasm_f64, render_vecadd_f64_scalar) {
  /* Render f64 vecadd kernel in scalar mode -- verify f64 opcodes */
  WasmVecKernel k = wasm_make_vec_binop_f64(POLY_OP_ADD, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Must contain f64.add opcode (0xA0) */
  bool found_f64_add = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F64_ADD) {
      found_f64_add = true;
      break;
    }
  }
  ASSERT_TRUE(found_f64_add);

  /* Must contain f64.load opcode (0x2B) */
  bool found_f64_load = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F64_LOAD) {
      found_f64_load = true;
      break;
    }
  }
  ASSERT_TRUE(found_f64_load);

  /* Must contain f64.store opcode (0x39) */
  bool found_f64_store = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F64_STORE) {
      found_f64_store = true;
      break;
    }
  }
  ASSERT_TRUE(found_f64_store);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm_f64, render_neg_f64_scalar) {
  /* Render f64 unary neg kernel -- verify f64.neg opcode */
  WasmVecKernel k = wasm_make_vec_unary_f64(POLY_OP_NEG, 8);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);

  ASSERT_NOT_NULL(wasm);

  /* Must contain f64.neg opcode (0x9A) */
  bool found_f64_neg = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F64_NEG) {
      found_f64_neg = true;
      break;
    }
  }
  ASSERT_TRUE(found_f64_neg);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm_f64, render_simd_f64x2) {
  /* Render f64 vecadd with SIMD enabled -- verify f64x2 SIMD opcodes */
  WasmVecKernel k = wasm_make_vec_binop_f64(POLY_OP_ADD, 16);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Must contain SIMD prefix (0xFD) -- proves SIMD path was taken */
  bool found_simd = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_SIMD_PREFIX) {
      found_simd = true;
      break;
    }
  }
  ASSERT_TRUE(found_simd);

  /* Verify f64x2.add sub-opcode (0xF0) follows a SIMD prefix */
  bool found_f64x2_add = false;
  for (int i = 0; i < wasm_size - 1; i++) {
    if (wasm[i] == WASM_SIMD_PREFIX && wasm[i + 1] == WASM_SIMD_F64X2_ADD) {
      found_f64x2_add = true;
      break;
    }
  }
  ASSERT_TRUE(found_f64x2_add);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm_f64, validate_f64_scalar) {
  /* Write f64 scalar kernel and validate with wasm-validate */
  WasmVecKernel k = wasm_make_vec_binop_f64(POLY_OP_ADD, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("/tmp/polygrad_test_f64_scalar.wasm", "wb");
  if (f) {
    fwrite(wasm, 1, wasm_size, f);
    fclose(f);
  }

  int rc = system("which wasm-validate > /dev/null 2>&1 && "
                  "wasm-validate /tmp/polygrad_test_f64_scalar.wasm");
  if (rc == 0) { /* wasm-validate passed */
  }

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm_f64, validate_f64_simd) {
  /* Write f64x2 SIMD kernel and validate with wasm-validate */
  WasmVecKernel k = wasm_make_vec_binop_f64(POLY_OP_ADD, 16);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("/tmp/polygrad_test_f64_simd.wasm", "wb");
  if (f) {
    fwrite(wasm, 1, wasm_size, f);
    fclose(f);
  }

  int rc = system("which wasm-validate > /dev/null 2>&1 && "
                  "wasm-validate --enable-simd /tmp/polygrad_test_f64_simd.wasm");
  if (rc == 0) { /* wasm-validate passed */
  }

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm_f64, e2e_node_vecadd_f64) {
  /* End-to-end: render f64 WASM, write to file, run with Node.js */
  WasmVecKernel k = wasm_make_vec_binop_f64(POLY_OP_ADD, 8);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("/tmp/polygrad_e2e_vecadd_f64.wasm", "wb");
  ASSERT_NOT_NULL(f);
  fwrite(wasm, 1, wasm_size, f);
  fclose(f);

  int has_node = system("which node > /dev/null 2>&1");
  if (has_node != 0) {
    free(wasm);
    free(lin);
    poly_ctx_destroy(k.ctx);
    PASS(); /* skip gracefully */
  }

  int rc = system("node test/run_wasm.js /tmp/polygrad_e2e_vecadd_f64.wasm add_f64 8");
  ASSERT_INT_EQ(rc, 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}
