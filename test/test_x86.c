/*
 * test_x86.c -- current Tinygrad X86 isel and full-program tests.
 *
 * Direct isel coverage follows tinygrad@2026-08-22/a9069c177a9d
 * test/backend/test_isel.py. Remaining tests enter through current SINK,
 * LINEAR, PROGRAM, and runtime boundaries.
 */

#ifdef POLY_HAS_X86

#include "test_harness.h"
#include "../src/codegen/codegen.h"
#include "../src/ctx.h"
#include "../src/engine/schedule.h"
#include "../src/frontend.h"
#include "../src/nn.h"
#include "../src/renderer/isa/x86.h"
#include "../src/tensor.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
  PolyUOp *sink = poly_sink1(ctx, store);

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

TEST_BACKEND(x86, f64_bool_mask_uses_current_int32_immediate) {
  /* tinygrad@2026-08-22/a9069c177a9d renderer/isa/x86.py:227-232,383-385:
   * comparison masks use to_imm, so an int64 literal 1 encodes as int32. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop_variable(ctx, "x", 0, 0, POLY_FLOAT64, 1, false);
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
  PolyUOp *a = poly_uop_variable(ctx, "a", 0, 0, POLY_INT32, 1, false);
  PolyUOp *b = poly_uop_variable(ctx, "b", 0, 0, POLY_INT32, 1, false);
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
  PolyUOp *a = poly_uop_variable(ctx, "a", 0, 0, POLY_FLOAT32, 1, false);
  PolyUOp *b = poly_uop_variable(ctx, "b", 0, 0, POLY_FLOAT32, 1, false);
  PolyUOp *c = poly_uop_variable(ctx, "c", 0, 0, POLY_FLOAT32, 1, false);
  PolyUOp *d = poly_uop_variable(ctx, "d", 0, 0, POLY_FLOAT32, 1, false);
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
  PolyUOp *a = poly_uop_variable(ctx, "a", 0, 0, POLY_INT32, 1, false);
  PolyUOp *param = poly_test_program_param(ctx, POLY_INT32, 16, 0);
  PolyUOp *one =
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, poly_const_int(ctx, 1), poly_arg_dtype(POLY_INT32));
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

  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *b = poly_buffer_f32(ctx, 8);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

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
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

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
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  poly_buffer_set(ctx, a, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, b_data, sizeof(b_data), POLY_DEVICE_CPU);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
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
  PolyUOp *a = poly_buffer_f32(ctx, N);
  poly_buffer_set(ctx, a, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  int64_t shape[] = {4, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, shape, 2);
  int64_t axes[] = {1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1);
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
  PolyUOp *xb = poly_buffer_f32(ctx, XN);
  PolyUOp *wb = poly_buffer_f32(ctx, WN);
  poly_buffer_set(ctx, xb, x_data, sizeof(x_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wb, w_data, sizeof(w_data), POLY_DEVICE_CPU);
  PolyUOp *x = poly_reshape(ctx, xb, (int64_t[]){2, 4}, 2);
  PolyUOp *w = poly_reshape(ctx, wb, (int64_t[]){2, 4}, 2);
  PolyUOp *wt = poly_permute(ctx, w, (int64_t[]){1, 0}, 2);
  PolyUOp *dot = poly_dot(ctx, x, wt);
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
  PolyUOp *abuf = poly_buffer_f32(ctx, N);
  PolyUOp *bbuf = poly_buffer_f32(ctx, N);
  PolyUOp *obuf = poly_buffer_f32(ctx, N);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, abuf, bbuf);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, obuf, sum));
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

  PolyUOp *logits_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *target_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3, 2}, 3);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2, 3, 2}, 3);
  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, 1);
  ASSERT_NOT_NULL(loss);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out_buf, loss));

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

  PolyUOp *obuf = poly_buffer_f32(ctx, N);
  PolyUOp *rf = poly_arange(ctx, 0.0, (double)N, 1.0);
  ASSERT_NOT_NULL(rf);
  PolyUOp *x = poly_alu2(
      ctx, POLY_OP_FDIV, poly_alu2(ctx, POLY_OP_ADD, rf, poly_const_float(ctx, 1.0)),
      poly_const_float(ctx, 17.0)
  );
  PolyUOp *y = poly_alu1(ctx, POLY_OP_LOG2, x);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, obuf, y));
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

  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *oi = poly_buffer_f32(ctx, N);
  PolyUOp *oh = poly_buffer_f32(ctx, N);
  PolyUOp *on = poly_buffer_f32(ctx, N);
  poly_buffer_set(ctx, a, in, sizeof(in), POLY_DEVICE_CPU);

  PolyUOp *pow_i = poly_alu2(ctx, POLY_OP_POW, a, poly_const_float(ctx, 2.0));
  PolyUOp *pow_h = poly_alu2(ctx, POLY_OP_POW, a, poly_const_float(ctx, 1.5));
  PolyUOp *pow_n = poly_alu2(ctx, POLY_OP_POW, a, poly_const_float(ctx, -1.0));
  PolyUOp *stores[3] = {
      poly_store_val(ctx, oi, pow_i),
      poly_store_val(ctx, oh, pow_h),
      poly_store_val(ctx, on, pow_n),
  };
  PolyUOp *sink = poly_sink_n(ctx, stores, 3);

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

  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *e = poly_buffer_f32(ctx, N);
  PolyUOp *o = poly_buffer_f32(ctx, N);
  poly_buffer_set(ctx, b, base, sizeof(base), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, e, expv, sizeof(expv), POLY_DEVICE_CPU);

  PolyUOp *pow = poly_alu2(ctx, POLY_OP_POW, b, e);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, o, pow));
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

  PolyUOp *pow = poly_alu2(ctx, POLY_OP_POW, b, e);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, o, pow));
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
  PolyUOp *xb = poly_buffer_f32(ctx, D);
  PolyUOp *wgb = poly_buffer_f32(ctx, (int64_t)H * D);
  PolyUOp *wub = poly_buffer_f32(ctx, (int64_t)H * D);
  PolyUOp *wdb = poly_buffer_f32(ctx, (int64_t)D * H);
  poly_buffer_set(ctx, xb, x, (size_t)D * sizeof(float), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wgb, wg, (size_t)H * D * sizeof(float), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wub, wu, (size_t)H * D * sizeof(float), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wdb, wd, (size_t)D * H * sizeof(float), POLY_DEVICE_CPU);

  PolyUOp *x2 = poly_reshape(ctx, xb, (int64_t[]){1, D}, 2);
  PolyUOp *wg2 = poly_reshape(ctx, wgb, (int64_t[]){H, D}, 2);
  PolyUOp *wu2 = poly_reshape(ctx, wub, (int64_t[]){H, D}, 2);
  PolyUOp *wd2 = poly_reshape(ctx, wdb, (int64_t[]){D, H}, 2);
  PolyUOp *gate = poly_silu(ctx, poly_dot(ctx, x2, poly_permute(ctx, wg2, (int64_t[]){1, 0}, 2)));
  PolyUOp *up = poly_dot(ctx, x2, poly_permute(ctx, wu2, (int64_t[]){1, 0}, 2));
  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, gate, up);
  PolyUOp *res = poly_dot(ctx, prod, poly_permute(ctx, wd2, (int64_t[]){1, 0}, 2));
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
