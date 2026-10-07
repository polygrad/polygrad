/*
 * test_x86.c -- current Tinygrad X86 isel and full-program tests.
 *
 * Direct isel coverage follows tinygrad@2026-08-22/a9069c177a9d
 * test/backend/test_isel.py. Remaining tests enter through current SINK,
 * LINEAR, PROGRAM, and runtime boundaries.
 */

#define _DEFAULT_SOURCE
#ifdef POLY_HAS_X86

#include "test_harness.h"
#include "../src/codegen/codegen.h"
#include "../src/ctx.h"
#include "../src/engine/schedule.h"
#include "../src/frontend.h"
#include "../src/nn/nn.h"
#include "../src/renderer/isa/x86.h"
#include "../src/tensor.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int poly_test_x86_program_call_entry(void *entry, void **args, int n_args);

TEST_BACKEND(x86, constrained_loop_live_ins_survive_nested_ranges) {
  /* Unsigned division requires a zero high dividend in RDX on every outer
   * iteration, even when an inner range reuses RDX for its counter. */
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  uint32_t indices[200];
  float values[80], actual[400];
  for (int i = 0; i < 200; i++)
    indices[i] = (uint32_t)i * 747796405u + 388445122u;
  for (int i = 0; i < 80; i++)
    values[i] = (float)i;
  PolyTensor *idx =
      poly_tensor_from_host(ctx, indices, sizeof(indices), POLY_UINT32, (int64_t[]){200}, 1);
  PolyTensor *data =
      poly_tensor_from_host(ctx, values, sizeof(values), POLY_FLOAT32, (int64_t[]){40, 2}, 2);
  idx = poly_tensor_to_device(ctx, idx, POLY_DEVICE_X86);
  data = poly_tensor_to_device(ctx, data, POLY_DEVICE_X86);
  PolyTensor *divisor = poly_tensor_const_like_int(ctx, idx, 40);
  PolyTensor *mod = poly_tensor_alu2(ctx, POLY_OP_CMOD, idx, divisor);
  PolyTensor *index = poly_tensor_cast(ctx, mod, POLY_INT32);
  PolyTensor *out = poly_tensor_index_select(ctx, data, 0, index), *realized = NULL;
  bool ok = out && poly_realize_tensors(ctx, &out, 1, &realized) == 0 && realized;
  const PolyUOp *buf = ok ? poly_uop_get_buffer_identity(poly_tensor_uop(realized)) : NULL;
  ok = buf && poly_buffer_read(ctx, (PolyUOp *)buf, actual, sizeof(actual)) == 0;
  for (int i = 0; ok && i < 400; i++)
    ok = actual[i] == values[2 * (indices[i / 2] % 40) + i % 2];
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST_BACKEND(x86, variable_shifts_preserve_each_lane_count) {
  /* Pinned x86.shift/isel allocate distinct virtual values constrained to RCX.
   * Unrolled lanes must not share one live value merely because all use CL. */
  PolyDType dtypes[] = {POLY_UINT32, POLY_INT32, POLY_UINT64, POLY_INT64};
  bool ok = true;
  for (int d = 0; d < 4; d++) {
    int bits = d < 2 ? 32 : 64, bytes = bits / 8;
    uint64_t mask = bits == 32 ? UINT32_MAX : UINT64_MAX;
    uint64_t sign = UINT64_C(1) << (bits - 1);
    uint64_t values[] = {1, 7, 12345, 123456789, sign, mask, sign + 3, 31};
    uint64_t counts[] = {1, 7, 17, 0, 1, (uint64_t)bits - 1, 3, 2};
    uint64_t input[8] = {0}, shifts[8] = {0};
    for (int i = 0; i < 8; i++) {
      if (bits == 32) {
        uint32_t v = (uint32_t)values[i], c = (uint32_t)counts[i];
        memcpy((char *)input + i * bytes, &v, bytes);
        memcpy((char *)shifts + i * bytes, &c, bytes);
      } else {
        input[i] = values[i];
        shifts[i] = counts[i];
      }
    }
    for (int right = 0; right < 2; right++) {
      PolyCtx *ctx = poly_ctx_new();
      poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
      PolyTensor *x = poly_tensor_from_host(ctx, input, 8 * bytes, dtypes[d], (int64_t[]){8}, 1);
      PolyTensor *y = poly_tensor_from_host(ctx, shifts, 8 * bytes, dtypes[d], (int64_t[]){8}, 1);
      x = poly_tensor_to_device(ctx, x, POLY_DEVICE_X86);
      y = poly_tensor_to_device(ctx, y, POLY_DEVICE_X86);
      PolyTensor *out = poly_tensor_alu2(ctx, right ? POLY_OP_SHR : POLY_OP_SHL, x, y);
      PolyTensor *realized = NULL;
      uint64_t actual[8] = {0};
      bool ran = out && poly_realize_tensors(ctx, &out, 1, &realized) == 0 && realized;
      const PolyUOp *buffer = ran ? poly_uop_get_buffer_identity(poly_tensor_uop(realized)) : NULL;
      ran = buffer && poly_buffer_read(ctx, (PolyUOp *)buffer, actual, 8 * bytes) == 0;
      ok &= ran;
      for (int i = 0; ran && i < 8; i++) {
        uint64_t expected = right ? values[i] >> counts[i] : (values[i] << counts[i]) & mask;
        if (right && (d & 1) && (values[i] & sign) && counts[i])
          expected |= mask ^ (mask >> counts[i]);
        uint64_t got = actual[i];
        if (bits == 32) {
          uint32_t lane;
          memcpy(&lane, (char *)actual + i * bytes, bytes);
          got = lane;
        }
        ok &= got == expected;
      }
      poly_ctx_destroy(ctx);
    }
  }
  ASSERT_TRUE(ok);
  PASS();
}

TEST_BACKEND(x86, program_call_without_compiler_type_prefix) {
  long page = sysconf(_SC_PAGESIZE);
  ASSERT_TRUE(page > 0);
  uint8_t *mapping = mmap(NULL, (size_t)page * 2, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_TRUE(mapping != MAP_FAILED);
  uint8_t *entry = mapping + page;
  /* mov dword ptr [rdi], 42; ret. Raw JIT code has no Clang type metadata.
   * The guard makes reads before the entry fail independently of ASLR. */
  const uint8_t code[] = {0xc7, 0x07, 0x2a, 0, 0, 0, 0xc3};
  bool ok = mprotect(entry, (size_t)page, PROT_READ | PROT_WRITE) == 0;
  if (ok) {
    memcpy(entry, code, sizeof(code));
    ok = mprotect(entry, (size_t)page, PROT_READ | PROT_EXEC) == 0;
  }
  int output = 0;
  void *args[63] = {&output};
  if (ok) {
    ok &= poly_test_x86_program_call_entry(entry + 6, args, 0) == 0;
    for (int n = 1; n <= 63; n++) {
      output = 0;
      ok &= poly_test_x86_program_call_entry(entry, args, n) == 0 && output == 42;
    }
  }
  munmap(mapping, (size_t)page * 2);
  ASSERT_TRUE(ok);
  PASS();
}

TEST_BACKEND(x86, program_call_stack_arguments) {
  int counts[] = {6, 7, 16, 17, 21, 31, 63, 65};
  for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
    int n = counts[c];
    uint8_t code[1024] = {0x31, 0xc0}; /* xor eax,eax */
    int pos = 2;
    const uint8_t add_register[][3] = {
        {0x48, 0x01, 0xf0},
        {0x48, 0x01, 0xd0},
        {0x48, 0x01, 0xc8},
        {0x4c, 0x01, 0xc0},
        {0x4c, 0x01, 0xc8}};
    for (int i = 1; i < n; i++) {
      if (i < 6) {
        memcpy(code + pos, add_register[i - 1], 3);
        pos += 3;
      } else {
        /* add rax,[rsp+disp32]: return address precedes the seventh arg. */
        const uint8_t op[] = {0x48, 0x03, 0x84, 0x24};
        int32_t offset = 8 * (i - 5);
        memcpy(code + pos, op, 4);
        memcpy(code + pos + 4, &offset, 4);
        pos += 8;
      }
    }
    /* Include entry RSP modulo 16: SysV requires eight after the return PC. */
    const uint8_t alignment[] = {0x48, 0x89, 0xe2, 0x83, 0xe2, 0x0f, 0x48, 0x01, 0xd0};
    memcpy(code + pos, alignment, sizeof(alignment));
    pos += sizeof(alignment);
    /* The last argument is core_id. Each worker writes a distinct result. */
    if (n == 6) {
      const uint8_t op[] = {0x4c, 0x89, 0xc9}; /* mov rcx,r9 */
      memcpy(code + pos, op, 3);
      pos += 3;
    } else {
      const uint8_t op[] = {0x48, 0x8b, 0x8c, 0x24};
      int32_t offset = 8 * (n - 6);
      memcpy(code + pos, op, 4);
      memcpy(code + pos + 4, &offset, 4);
      pos += 8;
    }
    const uint8_t store[] = {0x48, 0x89, 0x04, 0xcf, 0xc3}; /* [rdi+rcx*8]=rax; ret */
    memcpy(code + pos, store, sizeof(store));
    pos += sizeof(store);
    PolyX86Program *prog = poly_compile_x86(code, pos);
    ASSERT_TRUE(prog != NULL);
    uint64_t output[8] = {0};
    void *args[65] = {output};
    for (int i = 1; i < n - 1; i++)
      args[i] = (void *)(uintptr_t)i;
    uint64_t expected = (uint64_t)(n - 1) * (n - 2) / 2 + 8;
    bool ok = poly_x86_program_call(prog, args, n) == 0 && output[0] == expected;
    ok &= poly_x86_program_call_core(prog, args, n, n - 1, 7) == 0 && output[7] == expected + 7 &&
          args[n - 1] == NULL;
    ok &= poly_x86_program_call_threaded(prog, args, n, n - 1, 5) == 0;
    for (int i = 0; i < 5; i++)
      ok &= output[i] == expected + (uint64_t)i;
    poly_x86_program_destroy(prog);
    ASSERT_TRUE(ok);
  }
  PASS();
}

static bool x86_call_preserves_registers(PolyX86Program *target, void **args, int n) {
  /* A generated caller seeds all SysV callee-saved GPRs, calls the real C
   * launcher, checks them, then restores its caller's original values. */
  const int saved[] = {3, 5, 12, 13, 14, 15};
  uint8_t code[256];
  int pos = 0;
  for (int i = 0; i < 6; i++) {
    if (saved[i] >= 8) code[pos++] = 0x41;
    code[pos++] = (uint8_t)(0x50 + (saved[i] & 7));
  }
  code[pos++] = 0x57; /* push result pointer; also aligns the outgoing call */
  for (int i = 0; i < 6; i++) {
    code[pos++] = saved[i] >= 8 ? 0x49 : 0x48;
    code[pos++] = (uint8_t)(0xb8 + (saved[i] & 7));
    uint64_t sentinel = UINT64_C(0x123456789abcdef0) + (uint64_t)i;
    memcpy(code + pos, &sentinel, 8);
    pos += 8;
  }
  /* r11=callback; (rdi,rsi,rdx)=(target,args,count); call r11. */
  const uint8_t call[] = {0x49, 0x89, 0xf3, 0x48, 0x89, 0xd7, 0x48, 0x89, 0xce,
                          0x4c, 0x89, 0xc2, 0x41, 0xff, 0xd3, 0x89, 0xc2};
  memcpy(code + pos, call, sizeof(call));
  pos += sizeof(call);
  for (int i = 0; i < 6; i++) {
    code[pos++] = 0x48;
    code[pos++] = 0xb8;
    uint64_t sentinel = UINT64_C(0x123456789abcdef0) + (uint64_t)i;
    memcpy(code + pos, &sentinel, 8);
    pos += 8;
    code[pos++] = saved[i] >= 8 ? 0x4c : 0x48;
    code[pos++] = 0x31;
    code[pos++] = (uint8_t)(0xc0 | ((saved[i] & 7) << 3));
    code[pos++] = 0x48;
    code[pos++] = 0x09;
    code[pos++] = 0xc2; /* or rdx,rax */
  }
  const uint8_t store[] = {0x5f, 0x48, 0x89, 0x17}; /* pop rdi; [rdi]=rdx */
  memcpy(code + pos, store, sizeof(store));
  pos += sizeof(store);
  for (int i = 5; i >= 0; i--) {
    if (saved[i] >= 8) code[pos++] = 0x41;
    code[pos++] = (uint8_t)(0x58 + (saved[i] & 7));
  }
  code[pos++] = 0xc3;
  PolyX86Program *probe = poly_compile_x86(code, pos);
  if (!probe) return false;
  int (*call_fn)(PolyX86Program *, void **, int) = poly_x86_program_call;
  void *callback;
  memcpy(&callback, &call_fn, sizeof(callback));
  uint64_t status = UINT64_MAX;
  void *probe_args[] = {&status, callback, target, args, (void *)(uintptr_t)n};
  bool ok = poly_x86_program_call(probe, probe_args, 5) == 0 && status == 0;
  poly_x86_program_destroy(probe);
  return ok;
}

TEST_BACKEND(x86, generated_kernel_many_arguments) {
  /* Exercise real argument lowering, spills and the kernel's own prologue,
   * not only a hand-written callee. The last stack slot is a scalar control. */
  int counts[] = {17, 21, 31, 63, 65};
  for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
    int n = counts[c];
    PolyCtx *ctx = poly_ctx_new();
    PolyParamArg param = {
        .slot = n - 1, .dtype = POLY_INT32, .addrspace = POLY_ADDR_ALU, .name = "core_id"};
    PolyUOp *scalar_shape = poly_uop0(ctx, POLY_OP_STACK, POLY_VOID, poly_arg_none());
    PolyUOp *index =
        poly_uop1(ctx, POLY_OP_PARAM, POLY_INT32, scalar_shape, poly_arg_param(&param));
    PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 8, 0), *sum = NULL;
    float input[63][8], output[8] = {0}, expected[8] = {0};
    void *args[65] = {output};
    for (int i = 1; i < n - 1; i++) {
      PolyUOp *p = poly_test_program_param(ctx, POLY_FLOAT32, 8, i);
      PolyUOp *value = poly_uop1(
          ctx, POLY_OP_LOAD, POLY_FLOAT32,
          poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p, index, poly_arg_none()), poly_arg_none()
      );
      PolyUOp *weight = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(i));
      value = poly_uop_alu2(ctx, POLY_OP_MUL, value, weight);
      sum = sum ? poly_uop_alu2(ctx, POLY_OP_ADD, sum, value) : value;
      args[i] = input[i - 1];
      for (int j = 0; j < 8; j++) {
        input[i - 1][j] = (float)(i * 8 + j);
        expected[j] += input[i - 1][j] * i;
      }
    }
    PolyUOp *store = poly_uop2(
        ctx, POLY_OP_STORE, POLY_VOID,
        poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, index, poly_arg_none()), sum,
        poly_arg_none()
    );
    int n_linear = 0, n_code = 0;
    PolyUOp **linear =
        poly_linearize_x86(ctx, poly_test_kernel_sink(ctx, &store, 1, "many_args"), &n_linear);
    ASSERT_NOT_NULL(linear);
    uint8_t *code = poly_render_x86(linear, n_linear, &n_code);
    ASSERT_NOT_NULL(code);
    PolyX86Program *prog = poly_compile_x86(code, n_code);
    ASSERT_NOT_NULL(prog);
    bool ok = x86_call_preserves_registers(prog, args, n);
    for (int i = 0; i < 8; i++)
      ok &= poly_x86_program_call_core(prog, args, n, n - 1, i) == 0 && output[i] == expected[i];
    memset(output, 0, sizeof(output));
    ok &= poly_x86_program_call_threaded(prog, args, n, n - 1, 8) == 0;
    for (int i = 0; i < 8; i++)
      ok &= output[i] == expected[i];
    if (!ok)
      fprintf(
          stderr, "generated args=%d: got %g,%g expected %g,%g\n", n, output[0], output[7],
          expected[0], expected[7]
      );
    poly_x86_program_destroy(prog);
    free(code);
    free(linear);
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(ok);
  }
  PASS();
}

TEST_BACKEND(x86, compiler_hex_source_error_contract) {
  /* X86Compiler.compile delegates to bytes.fromhex: whitespace separates bytes,
   * malformed input raises ValueError rather than a recoverable RuntimeError. */
  PolyCtx *ctx = poly_ctx_new();
  const PolyBackendDesc *backend = poly_backend_get(POLY_DEVICE_X86);
  PolyUOp *sink = poly_test_kernel_sink(ctx, NULL, 0, "test");
  PolyUOp *linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, NULL, 0, poly_arg_none());
  const char *sources[] = {"c3", " \tc3\n\r\v\f", "c 3", "c", "zz", ""};
  int status[6], sizes[6];
  for (int i = 0; i < 6; i++) {
    PolyUOp *source = poly_uop0(ctx, POLY_OP_SOURCE, POLY_VOID, poly_arg_str(sources[i]));
    PolyUOp *parts[] = {sink, linear, source};
    PolyUOp *program = poly_uop(ctx, POLY_OP_PROGRAM, POLY_VOID, parts, 3, poly_arg_none());
    PolyRunner runner = {0};
    status[i] = backend->lower_item(ctx, program, "test", &runner);
    sizes[i] = runner.handle_size;
    if (!status[i]) backend->free_runner(&runner);
  }
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(status[0], 0);
  ASSERT_INT_EQ(status[1], 0);
  ASSERT_INT_EQ(sizes[0], 1);
  ASSERT_INT_EQ(sizes[1], 1);
  for (int i = 2; i < 5; i++)
    ASSERT_INT_EQ(status[i], -2);
  /* Empty bytes parse successfully but cannot back an executable mapping. */
  ASSERT_INT_EQ(status[5], -1);
  PASS();
}

static int x86_hex_nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static PolyUOp *x86_lane(PolyCtx *ctx, PolyUOp *value, int lane) {
  PolyUOp *idx = poly_uop1(
      ctx, POLY_OP_CAST, POLY_INT32,
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(lane)), poly_arg_dtype(POLY_INT32)
  );
  return poly_uop2(ctx, POLY_OP_INDEX, value->dtype, value, idx, poly_arg_none());
}

TEST_BACKEND(x86, byte_demoted_register_encoding_matches_tinygrad) {
  /* Tinygrad v0.14 x86.encode: a bool destination demotes an int32 source
   * access too. Registers 4..7 then mean SPL/BPL/SIL/DIL, not AH/CH/DH/BH. */
  PolyCtx *ctx = poly_ctx_new();
  const PolyX86Op ops[] = {POLY_X86_MOV, POLY_X86_AND, POLY_X86_OR};
  const uint8_t opcodes[] = {0x8b, 0x23, 0x0b};
  bool matches = true;
  for (int op = 0; op < 3; op++) {
    for (int reg = 0; reg < 16; reg++) {
      for (int byte = 0; byte <= 1; byte++) {
        /* Private X86 tag encoding: REAL | GPR class | physical index. */
        PolyUOp *src = poly_uop_tagged(
            ctx, POLY_OP_NOOP, POLY_INT32, NULL, 0, poly_arg_none(), 0x41000000 | reg
        );
        PolyUOp *dst = poly_uop_tagged(
            ctx, POLY_OP_INS, byte ? POLY_BOOL : POLY_INT32, &src, 1, poly_arg_int(ops[op]),
            0x41000000
        );
        int size = 0;
        uint8_t *code = poly_render_x86(&dst, 1, &size);
        bool rex = reg >= (byte ? 4 : 8);
        uint8_t expected[] = {
            (uint8_t)(0x40 | (reg >> 3)), (uint8_t)(opcodes[op] - byte),
            (uint8_t)(0xc0 | (reg & 7))};
        matches &= code && size == 2 + rex && memcmp(code, expected + !rex, (size_t)(2 + rex)) == 0;
        free(code);
      }
    }
  }
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(matches);
  PASS();
}

TEST_BACKEND(x86, isclose_nonfinite_under_register_pressure) {
  /* Exercise the mixed bool/int32 instruction operands produced after spills,
   * not just the individually correct isinf/isnan/eq components. */
  bool matches = true;
  const int widths[] = {1, 2, 4, 5, 8, 16};
  for (int k = 0; k < 6; k++) {
    for (int nan = 0; nan <= 1; nan++) {
      PolyCtx *ctx = poly_ctx_new();
      poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
      int n = widths[k];
      float data[16];
      for (int i = 0; i < n; i++)
        data[i] = nan ? NAN : INFINITY;
      PolyUOp *a = poly_uop_buffer_f32(ctx, n), *b = poly_uop_buffer_f32(ctx, n);
      poly_buffer_set(ctx, a, data, (size_t)n * sizeof(float), POLY_DEVICE_CPU);
      poly_buffer_set(ctx, b, data, (size_t)n * sizeof(float), POLY_DEVICE_CPU);
      for (int equal_nan = 0; equal_nan <= 1; equal_nan++) {
        PolyUOp *close = poly_uop_isclose(
            ctx, a, b, poly_uop_const_float(ctx, 1e-5), poly_uop_const_float(ctx, 1e-8), equal_nan
        );
        PolyUOp *realized = NULL;
        int rc = poly_realize_uops(ctx, &close, 1, &realized);
        uint8_t out[16] = {0};
        PolyUOp *buffer = (PolyUOp *)poly_uop_get_buffer_identity(realized);
        matches &= rc == 0 && buffer && poly_buffer_read(ctx, buffer, out, (size_t)n) == 0;
        for (int i = 0; i < n; i++)
          matches &= out[i] == (!nan || equal_nan);
      }
      poly_ctx_destroy(ctx);
    }
  }
  ASSERT_TRUE(matches);
  PASS();
}

TEST_BACKEND(x86, pre_isel_eliminates_current_gated_load) {
  /* tinygrad@2026-08-22/a9069c177a9d renderer/isa/x86.py:164-191:
   * gated LOAD selects the real or scratch address before instruction selection. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *src = poly_test_program_param(ctx, POLY_FLOAT32, 1, 1);
  PolyUOp *zero = poly_uop1(
      ctx, POLY_OP_CAST, POLY_INT32, poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0)),
      poly_arg_dtype(POLY_INT32)
  );
  PolyUOp *addr = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, src, zero, poly_arg_none());
  PolyUOp *one = poly_uop1(
      ctx, POLY_OP_CAST, POLY_INT32, poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1)),
      poly_arg_dtype(POLY_INT32)
  );
  PolyUOp *two = poly_uop1(
      ctx, POLY_OP_CAST, POLY_INT32, poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2)),
      poly_arg_dtype(POLY_INT32)
  );
  PolyUOp *gate_a = poly_uop2(ctx, POLY_OP_CMPEQ, POLY_BOOL, one, one, poly_arg_none());
  PolyUOp *gate_b = poly_uop2(ctx, POLY_OP_CMPEQ, POLY_BOOL, two, two, poly_arg_none());
  PolyUOp *gate = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, gate_a, gate_b, poly_arg_none());
  PolyUOp *alt = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT32,
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(7.5)),
      poly_arg_dtype(POLY_FLOAT32)
  );
  PolyUOp *load_srcs[3] = {
      addr,
      alt,
      gate,
  };
  PolyUOp *load = poly_uop(ctx, POLY_OP_LOAD, POLY_FLOAT32, load_srcs, 3, poly_arg_none());
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID,
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, zero, poly_arg_none()), load, poly_arg_none()
  );
  PolyUOp *sink = poly_uop_sink1(ctx, store);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_x86_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  for (int i = 0; i < n_lin; i++)
    ASSERT_TRUE(lin[i]->op != POLY_OP_LOAD && lin[i]->op != POLY_OP_STORE);

  int n_binary = 0;
  uint8_t *binary = poly_render_x86(lin, n_lin, &n_binary);
  ASSERT_NOT_NULL(binary);
  ASSERT_TRUE(n_binary > 0);

  free(binary);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, gated_vector_load_selects_one_scalar_address) {
  /* Tinygrad v0.14 x86.pre_isel_matcher normalizes every boolean gate.
   * A shaped uint64 address still selects one GPR with CMOVNE, not VPBLENDVB. */
  /* This direct post-decomposition fixture must fit the pinned 128-bit XMM. */
  for (int lanes = 4; lanes >= 2; lanes /= 2) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, lanes, 0);
    PolyUOp *src = poly_test_program_param(ctx, POLY_FLOAT32, lanes, 1);
    PolyUOp *flag = poly_test_program_param(ctx, POLY_INT32, 1, 2);
    PolyUOp *zero = poly_uop1(
        ctx, POLY_OP_CAST, POLY_INT32, poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0)),
        poly_arg_dtype(POLY_INT32)
    );
    PolyUOp *end = poly_uop1(
        ctx, POLY_OP_CAST, POLY_INT32,
        poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(lanes)), poly_arg_dtype(POLY_INT32)
    );
    PolyUOp *two = poly_uop1(
        ctx, POLY_OP_CAST, POLY_INT32, poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2)),
        poly_arg_dtype(POLY_INT32)
    );
    PolyUOp *value = poly_uop1(
        ctx, POLY_OP_LOAD, POLY_INT32,
        poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, flag, zero, poly_arg_none()), poly_arg_none()
    );
    PolyUOp *gate = poly_uop2(
        ctx, POLY_OP_AND, POLY_BOOL,
        poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, value, poly_arg_none()),
        poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, value, two, poly_arg_none()), poly_arg_none()
    );
    PolyUOp *alt_lane = poly_uop1(
        ctx, POLY_OP_CAST, POLY_FLOAT32,
        poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(7.5)),
        poly_arg_dtype(POLY_FLOAT32)
    );
    PolyUOp *alt_srcs[8];
    for (int i = 0; i < lanes; i++)
      alt_srcs[i] = alt_lane;
    PolyUOp *alt = poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, alt_srcs, lanes, poly_arg_none());
    PolyUOp *addr = poly_uop3(ctx, POLY_OP_SHRINK, POLY_FLOAT32, src, zero, end, poly_arg_none());
    PolyUOp *load = poly_uop3(ctx, POLY_OP_LOAD, POLY_FLOAT32, addr, alt, gate, poly_arg_none());
    PolyUOp *store = poly_uop2(
        ctx, POLY_OP_STORE, POLY_VOID,
        poly_uop3(ctx, POLY_OP_SHRINK, POLY_FLOAT32, out, zero, end, poly_arg_none()), load,
        poly_arg_none()
    );
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_x86_rewritten(ctx, poly_uop_sink1(ctx, store), &n_lin);
    ASSERT_NOT_NULL(lin);
    int cmovs = 0;
    for (int i = 0; i < n_lin; i++) {
      ASSERT_TRUE(lin[i]->op != POLY_OP_WHERE);
      if (lin[i]->op == POLY_OP_INS && lin[i]->arg.i == POLY_X86_CMOVNE) cmovs++;
    }
    ASSERT_INT_EQ(cmovs, 1);
    int n_code = 0;
    uint8_t *code = poly_render_x86(lin, n_lin, &n_code);
    ASSERT_NOT_NULL(code);
    PolyX86Program *prog = poly_compile_x86(code, n_code);
    ASSERT_NOT_NULL(prog);
    float input[8], output[8];
    for (int i = 0; i < lanes; i++)
      input[i] = (float)i + 1.0f;
    for (int32_t selected = 0; selected <= 2; selected++) {
      /* A false gate must read scratch even when the caller's source is NULL. */
      void *args[] = {output, selected == 1 ? input : NULL, &selected};
      ASSERT_INT_EQ(poly_x86_program_call(prog, args, 3), 0);
      for (int i = 0; i < lanes; i++)
        ASSERT_FLOAT_EQ(output[i], selected == 1 ? input[i] : 7.5f, 0.0f);
    }
    poly_x86_program_destroy(prog);
    free(code);
    free(lin);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST_BACKEND(x86, f64_bool_mask_uses_current_int32_immediate) {
  /* tinygrad@2026-08-22/a9069c177a9d renderer/isa/x86.py:227-232,383-385:
   * comparison masks use to_imm, so an int64 literal 1 encodes as int32. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x =
      poly_uop_variable(ctx, "x", poly_arg_int(0), poly_arg_int(0), POLY_FLOAT64, 1, false);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT64, poly_arg_float(0.0));
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, zero, poly_arg_none());
  PolyUOp *isel = poly_x86_isel(ctx, cmp);

  ASSERT_NOT_NULL(isel);
  ASSERT_INT_EQ(isel->op, POLY_OP_NOOP);
  ASSERT_TRUE(isel->n_src == 1 && isel->src[0]->op == POLY_OP_INS);
  PolyUOp *mask = isel->src[0];
  ASSERT_INT_EQ(mask->arg.i, POLY_X86_ANDi);
  ASSERT_TRUE(mask->n_src == 2);
  ASSERT_INT_EQ(mask->src[1]->dtype.bitsize, 32);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, isel_shares_compare_between_current_cmovs) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop_variable(ctx, "a", poly_arg_int(0), poly_arg_int(0), POLY_INT32, 1, false);
  PolyUOp *b = poly_uop_variable(ctx, "b", poly_arg_int(0), poly_arg_int(0), POLY_INT32, 1, false);
  PolyUOp *lt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *ne = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *c = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, lt, a, b, poly_arg_none());
  PolyUOp *d = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, ne, a, b, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, c, d, poly_arg_none());
  PolyUOp *isel = poly_x86_isel(ctx, root);

  ASSERT_NOT_NULL(isel);
  ASSERT_INT_EQ(isel->op, POLY_OP_INS);
  ASSERT_TRUE(isel->n_src >= 2);
  ASSERT_INT_EQ(isel->src[0]->arg.i, POLY_X86_CMOVL);
  ASSERT_INT_EQ(isel->src[1]->arg.i, POLY_X86_CMOVNE);
  ASSERT_TRUE(isel->src[0]->n_src >= 3 && isel->src[1]->n_src >= 3);
  ASSERT_PTR_EQ(isel->src[0]->src[2], isel->src[1]->src[2]);
  ASSERT_INT_EQ(isel->src[0]->src[2]->arg.i, POLY_X86_CMP);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, isel_stack_lanes_use_current_vinsertps) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a =
      poly_uop_variable(ctx, "a", poly_arg_int(0), poly_arg_int(0), POLY_FLOAT32, 1, false);
  PolyUOp *b =
      poly_uop_variable(ctx, "b", poly_arg_int(0), poly_arg_int(0), POLY_FLOAT32, 1, false);
  PolyUOp *c =
      poly_uop_variable(ctx, "c", poly_arg_int(0), poly_arg_int(0), POLY_FLOAT32, 1, false);
  PolyUOp *d =
      poly_uop_variable(ctx, "d", poly_arg_int(0), poly_arg_int(0), POLY_FLOAT32, 1, false);
  PolyUOp *src0[4] = {
      x86_lane(ctx, a, 0),
      x86_lane(ctx, b, 1),
      x86_lane(ctx, a, 2),
      x86_lane(ctx, b, 3),
  };
  PolyUOp *src1[4] = {
      x86_lane(ctx, a, 3),
      x86_lane(ctx, b, 2),
      x86_lane(ctx, c, 1),
      d,
  };
  PolyUOp *stack0 = poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, src0, 4, poly_arg_none());
  PolyUOp *stack1 = poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, src1, 4, poly_arg_none());
  PolyUOp *isel0 = poly_x86_isel(ctx, stack0);
  PolyUOp *isel1 = poly_x86_isel(ctx, stack1);

  ASSERT_NOT_NULL(isel0);
  ASSERT_NOT_NULL(isel1);
  ASSERT_INT_EQ(isel0->op, POLY_OP_INS);
  ASSERT_INT_EQ(isel1->op, POLY_OP_INS);
  ASSERT_INT_EQ(isel0->arg.i, POLY_X86_VINSERTPS);
  ASSERT_INT_EQ(isel1->arg.i, POLY_X86_VINSERTPS);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, isel_complex_address_scales_current_index) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop_variable(ctx, "a", poly_arg_int(0), poly_arg_int(0), POLY_INT32, 1, false);
  PolyUOp *param = poly_test_program_param(ctx, POLY_INT32, 16, 0);
  PolyUOp *one = poly_uop1(
      ctx, POLY_OP_CAST, POLY_INT32, poly_uop_const_int(ctx, 1), poly_arg_dtype(POLY_INT32)
  );
  PolyUOp *index = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, one, poly_arg_none());
  PolyUOp *addr = poly_uop_index(ctx, param, &index, 1);
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, addr, poly_arg_none());
  PolyUOp *isel = poly_x86_isel(ctx, load);

  ASSERT_NOT_NULL(isel);
  ASSERT_INT_EQ(isel->op, POLY_OP_INS);
  ASSERT_TRUE(isel->n_src >= 3);
  PolyUOp *disp = isel->src[2];
  ASSERT_INT_EQ(disp->dtype.priority, POLY_INT8.priority);
  ASSERT_TRUE(disp->n_src == 1 && disp->src[0]->op == POLY_OP_CONST);
  ASSERT_INT_EQ(disp->src[0]->arg.i, 4);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, to_program_attaches_linear_source_hex_and_binary_children) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  poly_program_source_render_count_reset();

  PolyUOp *a = poly_uop_buffer_f32(ctx, 8);
  PolyUOp *b = poly_uop_buffer_f32(ctx, 8);
  PolyUOp *out = poly_uop_buffer_f32(ctx, 8);
  PolyUOp *sum = poly_uop_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *sink = poly_uop_sink1(ctx, poly_uop_store_val(ctx, out, sum));

  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_INT_EQ(linear_schedule->n_src, 1);

  PolyUOp *compiled = poly_compile_linear(ctx, linear_schedule, -1);
  ASSERT_NOT_NULL(compiled);
  PolyUOp *program = poly_test_linear_call_body(compiled, 0);
  ASSERT_NOT_NULL(program);
  ASSERT_INT_EQ(program->op, POLY_OP_PROGRAM);
  ASSERT_INT_EQ(program->arg.kind, POLY_ARG_PROGRAM_INFO);
  ASSERT_INT_EQ(program->n_src, 4);
  ASSERT_NOT_NULL(poly_program_linear(program));
  ASSERT_INT_EQ(program->src[2]->op, POLY_OP_SOURCE);
  ASSERT_INT_EQ(program->src[2]->arg.kind, POLY_ARG_STRING);
  ASSERT_NOT_NULL(program->src[2]->arg.str);
  ASSERT_INT_EQ(program->src[3]->op, POLY_OP_BINARY);
  ASSERT_INT_EQ(program->src[3]->arg.kind, POLY_ARG_BYTES);
  ASSERT_NOT_NULL(program->src[3]->arg.bytes.data);
  ASSERT_TRUE(program->src[3]->arg.bytes.n > 0);

  const char *hex = program->src[2]->arg.str;
  int n_hex = (int)strlen(hex);
  ASSERT_INT_EQ(n_hex, program->src[3]->arg.bytes.n * 2);
  for (int i = 0; i < program->src[3]->arg.bytes.n; i++) {
    int hi = x86_hex_nibble(hex[2 * i]);
    int lo = x86_hex_nibble(hex[2 * i + 1]);
    ASSERT_TRUE(hi >= 0 && lo >= 0);
    ASSERT_INT_EQ((hi << 4) | lo, program->src[3]->arg.bytes.data[i]);
  }
  ASSERT_INT_EQ(poly_program_source_render_count(), 1);

  PolyUOp *again = poly_test_linear_call_body(poly_compile_linear(ctx, linear_schedule, -1), 0);
  ASSERT_PTR_EQ(again, program);
  ASSERT_INT_EQ((int)poly_to_program_cache_len(ctx), 1);
  ASSERT_INT_EQ(poly_program_source_render_count(), 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, to_program_allows_f16_after_x86_extra_legalization) {
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT16, 4, POLY_DEVICE_X86);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT16, 4, POLY_DEVICE_X86);
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, a, a, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT16, mul, a, poly_arg_none());
  PolyUOp *sink = poly_uop_sink1(ctx, poly_uop_store_val(ctx, out, sum));

  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_INT_EQ(linear_schedule->n_src, 1);
  PolyUOp *program = poly_test_linear_call_body(poly_compile_linear(ctx, linear_schedule, -1), 0);
  ASSERT_NOT_NULL(program);
  ASSERT_INT_EQ(program->n_src, 4);
  ASSERT_INT_EQ(program->src[2]->op, POLY_OP_SOURCE);
  ASSERT_INT_EQ(program->src[3]->op, POLY_OP_BINARY);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_vecadd_uses_x86_device) {
  enum { N = 8 };
  float a_data[N], b_data[N], out[N];
  for (int i = 0; i < N; i++) {
    a_data[i] = (float)i - 3.0f;
    b_data[i] = 0.25f * (float)i;
  }
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  PolyUOp *a = poly_uop_buffer_f32(ctx, N);
  PolyUOp *b = poly_uop_buffer_f32(ctx, N);
  poly_buffer_set(ctx, a, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, b_data, sizeof(b_data), POLY_DEVICE_CPU);
  PolyUOp *sum = poly_uop_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &sum, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *buf = (PolyUOp *)poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf, out, sizeof(out)), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], a_data[i] + b_data[i], 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_reduce_sum_axis1_matches_tinygrad_probe_class) {
  enum { N = 16, OUT = 4 };
  float a_data[N], out[OUT];
  for (int i = 0; i < N; i++)
    a_data[i] = (float)i;
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  PolyUOp *a = poly_uop_buffer_f32(ctx, N);
  poly_buffer_set(ctx, a, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  int64_t shape[] = {4, 4};
  PolyUOp *a2d = poly_uop_reshape(ctx, a, shape, 2);
  int64_t axes[] = {1};
  PolyUOp *sum = poly_uop_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1);
  PolyUOp *realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &sum, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *buf = (PolyUOp *)poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf, out, sizeof(out)), 0);
  const float expected[OUT] = {6.0f, 22.0f, 38.0f, 54.0f};
  for (int i = 0; i < OUT; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_dot_matches_tinygrad_probe_class) {
  enum { XN = 8, WN = 8, OUT = 4 };
  float x_data[XN] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float w_data[WN] = {0.5f, 1.0f, 1.5f, 2.0f, 2.0f, 1.5f, 1.0f, 0.5f};
  float out[OUT] = {0};
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  PolyUOp *xb = poly_uop_buffer_f32(ctx, XN);
  PolyUOp *wb = poly_uop_buffer_f32(ctx, WN);
  poly_buffer_set(ctx, xb, x_data, sizeof(x_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wb, w_data, sizeof(w_data), POLY_DEVICE_CPU);
  PolyUOp *x = poly_uop_reshape(ctx, xb, (int64_t[]){2, 4}, 2);
  PolyUOp *w = poly_uop_reshape(ctx, wb, (int64_t[]){2, 4}, 2);
  PolyUOp *wt = poly_uop_permute(ctx, w, (int64_t[]){1, 0}, 2);
  PolyUOp *dot = poly_uop_dot(ctx, x, wt);
  ASSERT_NOT_NULL(dot);
  PolyUOp *realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &dot, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *buf = (PolyUOp *)poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf, out, sizeof(out)), 0);
  const float expected[OUT] = {15.0f, 10.0f, 35.0f, 30.0f};
  for (int i = 0; i < OUT; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, threaded_vecadd_program_core_id_shards_match_tinygrad_cpu_x86) {
  enum { N = 262144 };
  float *a = malloc((size_t)N * sizeof(float));
  float *b = malloc((size_t)N * sizeof(float));
  float *out = calloc((size_t)N, sizeof(float));
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  ASSERT_NOT_NULL(out);
  for (int i = 0; i < N; i++) {
    a[i] = (float)i;
    b[i] = (float)i * 0.25f;
  }

  setenv("NUM_CPU_THREADS", "2", 1);
  setenv("THREADS", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  PolyUOp *abuf = poly_uop_buffer_f32(ctx, N);
  PolyUOp *bbuf = poly_uop_buffer_f32(ctx, N);
  PolyUOp *obuf = poly_uop_buffer_f32(ctx, N);
  PolyUOp *sum = poly_uop_alu2(ctx, POLY_OP_ADD, abuf, bbuf);
  PolyUOp *sink = poly_uop_sink1(ctx, poly_uop_store_val(ctx, obuf, sum));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_INT_EQ(linear_schedule->n_src, 1);

  PolyUOp *program = poly_test_linear_call_body(poly_compile_linear(ctx, linear_schedule, -1), 0);
  ASSERT_NOT_NULL(program);
  const PolyProgramInfo *info = poly_program_info(ctx, program);
  ASSERT_NOT_NULL(info);
  ASSERT_INT_EQ(info->global_size[0], 2);
  ASSERT_TRUE(info->n_vars >= 1);
  int core_id_slot = -1;
  for (int i = 0; i < info->n_vars; i++) {
    PolyUOp *var = info->vars[i];
    if (var && var->arg.kind == POLY_ARG_PARAM && var->arg.param &&
        var->arg.param->addrspace == POLY_ADDR_ALU && var->arg.param->name &&
        strcmp(var->arg.param->name, "core_id") == 0)
      core_id_slot = (int)var->arg.param->slot;
  }
  ASSERT_INT_EQ(core_id_slot, 3);

  ASSERT_TRUE(program->n_src >= 2);
  PolyUOp *linear = program->src[1];
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(program->n_src, 4);
  PolyUOp *binary = program->src[3];
  ASSERT_NOT_NULL(binary);
  ASSERT_INT_EQ(binary->op, POLY_OP_BINARY);
  ASSERT_INT_EQ(binary->arg.kind, POLY_ARG_BYTES);
  PolyX86Program *prog = poly_compile_x86(binary->arg.bytes.data, binary->arg.bytes.n);
  ASSERT_NOT_NULL(prog);

  void *args[4] = {out, a, b, NULL};
  ASSERT_INT_EQ(poly_x86_program_call_core(prog, args, 4, core_id_slot, 0), 0);
  ASSERT_INT_EQ(poly_x86_program_call_core(prog, args, 4, core_id_slot, 1), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], a[i] + b[i], 1e-6f);

  poly_x86_program_destroy(prog);
  poly_ctx_destroy(ctx);
  free(a);
  free(b);
  free(out);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_cross_entropy_dense_axis1_keeps_fifth_arg_live) {
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);

  PolyUOp *logits_buf = poly_uop_buffer_f32(ctx, 12);
  PolyUOp *target_buf = poly_uop_buffer_f32(ctx, 12);
  PolyUOp *out_buf = poly_uop_buffer_f32(ctx, 1);
  PolyUOp *logits = poly_uop_reshape(ctx, logits_buf, (int64_t[]){2, 3, 2}, 3);
  PolyUOp *target = poly_uop_reshape(ctx, target_buf, (int64_t[]){2, 3, 2}, 3);
  PolyUOp *loss = poly_uop_cross_entropy(ctx, logits, target, 1);
  ASSERT_NOT_NULL(loss);
  PolyUOp *sink = poly_uop_sink1(ctx, poly_uop_store_val(ctx, out_buf, loss));

  float logits_data[12] = {0};
  float target_data[12] = {
      1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
  };
  float out_data[1] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(logits_buf, logits_data),
      POLY_TEST_HOST_VIEW(target_buf, target_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };

  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, bindings, 3), 0);
  ASSERT_FLOAT_EQ(out_data[0], logf(3.0f), 1e-5f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, canonical_reg_buffer_extent_survives_isel) {
  /* Pinned tinygrad renderer/isa/x86.py:371-379 retains BUFFER size sources
   * through isel, and codegen/late/regalloc.py:87-89 allocates
   * max_numel*itemsize bytes. Four float lanes therefore require 16 bytes. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *size = poly_uop1(
      ctx, POLY_OP_CAST, POLY_INT32, poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4)),
      poly_arg_dtype(POLY_INT32)
  );
  PolyParamArg param = {.slot = 0, .addrspace = POLY_ADDR_REG};
  PolyUOp *buf = poly_uop1(ctx, POLY_OP_BUFFER, POLY_FLOAT32, size, poly_arg_param(&param));
  PolyUOp *stores[4];
  for (int i = 0; i < 4; i++) {
    PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
    PolyUOp *addr = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, buf, idx, poly_arg_none());
    stores[i] = poly_uop2(
        ctx, POLY_OP_STORE, POLY_VOID, addr,
        poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float((double)i)), poly_arg_none()
    );
  }
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 4, poly_arg_none());
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_x86_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  int64_t stack_bytes = -1;
  for (int i = 0; i < n_lin; i++) {
    PolyUOp *u = lin[i];
    if (!u || u->op != POLY_OP_INS || u->arg.kind != POLY_ARG_INT || u->arg.i != POLY_X86_SUBi ||
        u->n_src != 1 || !u->src[0] || u->src[0]->op != POLY_OP_CAST || u->src[0]->n_src != 1 ||
        u->src[0]->tag_arg.kind != POLY_ARG_BOOL || !u->src[0]->tag_arg.b || !u->src[0]->src[0] ||
        u->src[0]->src[0]->op != POLY_OP_CONST || u->src[0]->src[0]->arg.kind != POLY_ARG_INT)
      continue;
    stack_bytes = u->src[0]->src[0]->arg.i;
    break;
  }
  ASSERT_INT_EQ(stack_bytes, 16);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_computed_log2_keeps_loop_live_ins_like_tinygrad) {
  enum { N = 16 };
  float out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);

  PolyUOp *obuf = poly_uop_buffer_f32(ctx, N);
  PolyUOp *rf = poly_uop_arange(ctx, 0.0, (double)N, 1.0);
  ASSERT_NOT_NULL(rf);
  PolyUOp *x = poly_uop_alu2(
      ctx, POLY_OP_FDIV, poly_uop_alu2(ctx, POLY_OP_ADD, rf, poly_uop_const_float(ctx, 1.0)),
      poly_uop_const_float(ctx, 17.0)
  );
  PolyUOp *y = poly_uop_alu1(ctx, POLY_OP_LOG2, x);
  PolyUOp *sink = poly_uop_sink1(ctx, poly_uop_store_val(ctx, obuf, y));
  PolyTestBufferView view = POLY_TEST_HOST_VIEW(obuf, out);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &view, 1), 0);

  for (int i = 0; i < N; i++) {
    float expected = log2f(((float)i + 1.0f) / 17.0f);
    ASSERT_TRUE(isfinite(out[i]));
    ASSERT_FLOAT_EQ(out[i], expected, 1e-4f);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_pow_const_exponents_match_tinygrad) {
  enum { N = 4 };
  float in[N] = {1.0f, 2.0f, 3.0f, 4.0f};
  float out_i[N] = {0}, out_h[N] = {0}, out_n[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);

  PolyUOp *a = poly_uop_buffer_f32(ctx, N);
  PolyUOp *oi = poly_uop_buffer_f32(ctx, N);
  PolyUOp *oh = poly_uop_buffer_f32(ctx, N);
  PolyUOp *on = poly_uop_buffer_f32(ctx, N);
  poly_buffer_set(ctx, a, in, sizeof(in), POLY_DEVICE_CPU);

  PolyUOp *pow_i = poly_uop_alu2(ctx, POLY_OP_POW, a, poly_uop_const_float(ctx, 2.0));
  PolyUOp *pow_h = poly_uop_alu2(ctx, POLY_OP_POW, a, poly_uop_const_float(ctx, 1.5));
  PolyUOp *pow_n = poly_uop_alu2(ctx, POLY_OP_POW, a, poly_uop_const_float(ctx, -1.0));
  PolyUOp *stores[3] = {
      poly_uop_store_val(ctx, oi, pow_i),
      poly_uop_store_val(ctx, oh, pow_h),
      poly_uop_store_val(ctx, on, pow_n),
  };
  PolyUOp *sink = poly_uop_sink_n(ctx, stores, 3);

  PolyTestBufferView views[] = {
      POLY_TEST_HOST_VIEW(oi, out_i),
      POLY_TEST_HOST_VIEW(oh, out_h),
      POLY_TEST_HOST_VIEW(on, out_n),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, views, 3), 0);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(out_i[i], in[i] * in[i], 1e-5f);
    ASSERT_FLOAT_EQ(out_h[i], in[i] * sqrtf(in[i]), 1e-4f);
    ASSERT_FLOAT_EQ(out_n[i], 1.0f / in[i], 1e-5f);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_pow_dynamic_exponent_uses_xpow_like_tinygrad) {
  enum { N = 4 };
  float base[N] = {2.0f, 3.0f, 4.0f, 5.0f};
  float expv[N] = {3.0f, 2.0f, 0.5f, 1.0f};
  float out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);

  PolyUOp *b = poly_uop_buffer_f32(ctx, N);
  PolyUOp *e = poly_uop_buffer_f32(ctx, N);
  PolyUOp *o = poly_uop_buffer_f32(ctx, N);
  poly_buffer_set(ctx, b, base, sizeof(base), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, e, expv, sizeof(expv), POLY_DEVICE_CPU);

  PolyUOp *pow = poly_uop_alu2(ctx, POLY_OP_POW, b, e);
  PolyUOp *sink = poly_uop_sink1(ctx, poly_uop_store_val(ctx, o, pow));
  PolyTestBufferView view = POLY_TEST_HOST_VIEW(o, out);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &view, 1), 0);

  const float expected[N] = {8.0f, 9.0f, 2.0f, 5.0f};
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 2e-3f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_integer_pow_matches_current_tinygrad_bits) {
  enum { N = 10 };
  int32_t base[N] = {2, 3, -2, -1, 0, 1, 11, 0, -1, 2};
  int32_t expv[N] = {3, 2, 3, -3, -1, -2, 7, 0, INT32_MIN, INT32_MIN};
  int32_t out[N] = {0};
  /* Tinygrad 2026-08-22/a9069c177a9d lowers integer POW through floating
   * xpow; CPU:X86 exposes the resulting float bits through the int buffer. */
  const int32_t expected[N] = {
      1090519040, 1091567614, -1056964608, -1082130432, 2139095040,
      1065353216, 1268034793, 1065353216,  1065353216,  0,
  };
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);

  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_INT32, N, POLY_DEVICE_X86);
  PolyUOp *e = poly_test_buffer_on_device(ctx, POLY_INT32, N, POLY_DEVICE_X86);
  PolyUOp *o = poly_test_buffer_on_device(ctx, POLY_INT32, N, POLY_DEVICE_X86);
  poly_buffer_set(ctx, b, base, sizeof(base), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, e, expv, sizeof(expv), POLY_DEVICE_CPU);

  PolyUOp *pow = poly_uop_alu2(ctx, POLY_OP_POW, b, e);
  PolyUOp *sink = poly_uop_sink1(ctx, poly_uop_store_val(ctx, o, pow));
  PolyTestBufferView view = POLY_TEST_HOST_VIEW(o, out);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &view, 1), 0);

  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], expected[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

static void x86_make_data(float *data, int n, uint32_t seed, float scale) {
  uint32_t x = seed;
  for (int i = 0; i < n; i++) {
    x = x * 1664525u + 1013904223u;
    data[i] = (float)((int)((x >> 8) % 1009u) - 504) * scale / 504.0f;
  }
}

TEST_BACKEND(x86, schedule_runtime_qwen_ffn_fused_large_matches_tinygrad_probe_class) {
  enum { D = 256, H = 1536 };
  setenv("NUM_CPU_THREADS", "1", 1);
  setenv("THREADS", "0", 1);

  float *x = malloc((size_t)D * sizeof(float));
  float *wg = malloc((size_t)H * D * sizeof(float));
  float *wu = malloc((size_t)H * D * sizeof(float));
  float *wd = malloc((size_t)D * H * sizeof(float));
  float *ref_prod = malloc((size_t)H * sizeof(float));
  float *ref = calloc((size_t)D, sizeof(float));
  float *out = calloc((size_t)D, sizeof(float));
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(wg);
  ASSERT_NOT_NULL(wu);
  ASSERT_NOT_NULL(wd);
  ASSERT_NOT_NULL(ref_prod);
  ASSERT_NOT_NULL(ref);
  ASSERT_NOT_NULL(out);

  x86_make_data(x, D, 1, 0.02f);
  x86_make_data(wg, H * D, 6, 0.02f);
  x86_make_data(wu, H * D, 7, 0.02f);
  x86_make_data(wd, D * H, 8, 0.02f);

  for (int h = 0; h < H; h++) {
    float gate = 0.0f, up = 0.0f;
    for (int d = 0; d < D; d++) {
      gate += x[d] * wg[h * D + d];
      up += x[d] * wu[h * D + d];
    }
    gate = gate / (1.0f + expf(-gate));
    ref_prod[h] = gate * up;
  }
  for (int d = 0; d < D; d++)
    for (int h = 0; h < H; h++)
      ref[d] += ref_prod[h] * wd[d * H + h];

  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  PolyUOp *xb = poly_uop_buffer_f32(ctx, D);
  PolyUOp *wgb = poly_uop_buffer_f32(ctx, (int64_t)H * D);
  PolyUOp *wub = poly_uop_buffer_f32(ctx, (int64_t)H * D);
  PolyUOp *wdb = poly_uop_buffer_f32(ctx, (int64_t)D * H);
  poly_buffer_set(ctx, xb, x, (size_t)D * sizeof(float), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wgb, wg, (size_t)H * D * sizeof(float), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wub, wu, (size_t)H * D * sizeof(float), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wdb, wd, (size_t)D * H * sizeof(float), POLY_DEVICE_CPU);

  PolyUOp *x2 = poly_uop_reshape(ctx, xb, (int64_t[]){1, D}, 2);
  PolyUOp *wg2 = poly_uop_reshape(ctx, wgb, (int64_t[]){H, D}, 2);
  PolyUOp *wu2 = poly_uop_reshape(ctx, wub, (int64_t[]){H, D}, 2);
  PolyUOp *wd2 = poly_uop_reshape(ctx, wdb, (int64_t[]){D, H}, 2);
  PolyUOp *gate =
      poly_uop_silu(ctx, poly_uop_dot(ctx, x2, poly_uop_permute(ctx, wg2, (int64_t[]){1, 0}, 2)));
  PolyUOp *up = poly_uop_dot(ctx, x2, poly_uop_permute(ctx, wu2, (int64_t[]){1, 0}, 2));
  PolyUOp *prod = poly_uop_alu2(ctx, POLY_OP_MUL, gate, up);
  PolyUOp *res = poly_uop_dot(ctx, prod, poly_uop_permute(ctx, wd2, (int64_t[]){1, 0}, 2));
  PolyUOp *realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &res, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *buf = (PolyUOp *)poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf, out, (size_t)D * sizeof(float)), 0);

  for (int i = 0; i < D; i++)
    ASSERT_FLOAT_ABS(out[i], ref[i], 2e-6f);

  poly_ctx_destroy(ctx);
  free(x);
  free(wg);
  free(wu);
  free(wd);
  free(ref_prod);
  free(ref);
  free(out);
  PASS();
}

#endif /* POLY_HAS_X86 */
