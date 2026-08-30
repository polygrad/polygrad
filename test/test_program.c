/* Bound compiled-program artifact tests. */

#include "test_harness.h"
#include "../src/frontend.h"
#include "../src/frontend_internal.h"
#include "../src/instance.h"
#include "../src/ir.h"

#include <string.h>

static uint8_t *program_test_ir(bool with_param, int *out_len) {
  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return NULL;
  PolyUOp *x = poly_uop_new_logical_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *rhs = poly_uop_new_logical_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *out = poly_uop_new_logical_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_MUL, x, rhs)));
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyIrBufEntry bufs[] = {
      {.name = "x", .role = POLY_IR_ROLE_INPUT, .buffer = x, .shape = {2}, .ndim = 1},
      {.name = with_param ? "weight" : "rhs",
       .role = with_param ? POLY_IR_ROLE_PARAM : POLY_IR_ROLE_INPUT,
       .buffer = rhs,
       .shape = {2},
       .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out, .shape = {2}, .ndim = 1},
  };
  const char *two_inputs[] = {"x", "rhs"};
  PolyIrEntrypoint ep = {
      .name = "forward",
      .sink = sink,
      .inputs = with_param ? inputs : two_inputs,
      .n_inputs = with_param ? 1 : 2,
      .outputs = outputs,
      .n_outputs = 1,
  };
  PolyIrSpec spec = {
      .ctx = ctx,
      .bufs = bufs,
      .n_bufs = 3,
      .entrypoints = &ep,
      .n_entrypoints = 1,
  };
  uint8_t *ir = poly_ir_export(&spec, out_len);
  poly_ctx_destroy(ctx);
  return ir;
}

static int program_call_mul(PolyInstance *inst, const float x[2], const float *rhs, float out[2]) {
  PolyIOBinding io[2] = {
      {.name = "x",
       .data = x,
       .nbytes = 2 * sizeof(float),
       .dtype_id = poly_dtype_id_by_name("float32")},
      {.name = "rhs",
       .data = rhs,
       .nbytes = 2 * sizeof(float),
       .dtype_id = poly_dtype_id_by_name("float32")},
  };
  int n_io = rhs ? 2 : 1;
  if (poly_instance_call(inst, "forward", io, n_io) != 0) return -1;
  return poly_instance_read_buf_named(inst, "output", out, 2 * sizeof(float));
}

TEST(program, compiled_roundtrip_exact_bytes_and_values) {
  int ir_len = 0;
  uint8_t *ir = program_test_ir(false, &ir_len);
  ASSERT_NOT_NULL(ir);
  PolyInstance *source = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(source);

  int program_len = 0;
  uint8_t *program = poly_instance_export_program(source, &program_len);
  ASSERT_NOT_NULL(program);
  ASSERT_TRUE(program_len > 32);
  ASSERT_TRUE(memcmp(program, "PGPM", 4) == 0);
  ASSERT_INT_EQ(program[4], POLY_PROGRAM_VERSION);
  ASSERT_INT_EQ(program[8], POLYGRAD_ABI_VERSION);

  poly_program_source_render_count_reset();
  PolyInstance *loaded = poly_instance_from_program(program, program_len, NULL, 0);
  ASSERT_NOT_NULL(loaded);
  ASSERT_INT_EQ(poly_program_source_render_count(), 0);
  PolyUOp *linear = poly_instance_get_sink(loaded, "forward");
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(linear->n_src, 1);
  ASSERT_INT_EQ(linear->src[0]->op, POLY_OP_CALL);
  PolyUOp *body = linear->src[0]->src[0];
  ASSERT_INT_EQ(body->op, POLY_OP_PROGRAM);
  ASSERT_INT_EQ(body->src[0]->op, POLY_OP_SINK);
  if (poly_uop_device(body) == POLY_DEVICE_INTERP) {
    /* PG-DIV-004: the C interpreter retains Tinygrad's compiled LINEAR but
     * has no PythonRenderer SOURCE/BINARY payload. */
    ASSERT_INT_EQ(body->n_src, 2);
    ASSERT_INT_EQ(body->src[1]->op, POLY_OP_LINEAR);
  } else {
    ASSERT_TRUE(body->n_src == 3 || body->n_src == 4);
    ASSERT_INT_EQ(body->src[1]->op, POLY_OP_LINEAR);
    ASSERT_INT_EQ(body->src[2]->op, POLY_OP_SOURCE);
    if (body->n_src == 4) ASSERT_INT_EQ(body->src[3]->op, POLY_OP_BINARY);
  }

  bool saw_value_param = false;
  int n_tag_bool = 0, n_tag_int_tuple = 0, n_tag_string = 0;
  int n_body = 0;
  PolyUOp **body_topo = poly_toposort_alloc(poly_instance_ctx(loaded), body, &n_body);
  ASSERT_NOT_NULL(body_topo);
  for (int i = 0; i < n_body; i++) {
    saw_value_param |= body_topo[i]->op == POLY_OP_PARAM && body_topo[i]->n_src == 1 &&
                       body_topo[i]->arg.kind == POLY_ARG_PARAM;
    n_tag_bool += body_topo[i]->tag_arg.kind == POLY_ARG_BOOL;
    n_tag_int_tuple += body_topo[i]->tag_arg.kind == POLY_ARG_INT_TUPLE;
    n_tag_string += body_topo[i]->tag_arg.kind == POLY_ARG_STRING;
  }
  poly_toposort_free(body_topo);
  ASSERT_TRUE(saw_value_param);
  if (poly_uop_device(body) == POLY_DEVICE_X86) {
    /* X86 stores immediate markers, register lists, and labels in tag_arg.
     * Dropping these fields changes UOp identity and collapses the imported
     * graph through CSE even when a retained BINARY still happens to run. */
    ASSERT_TRUE(n_tag_bool > 0);
    ASSERT_TRUE(n_tag_int_tuple > 0);
    ASSERT_TRUE(n_tag_string > 0);
  }

  const float x[2] = {2, 3}, rhs[2] = {4, 5};
  float got[2] = {0};
  ASSERT_INT_EQ(program_call_mul(loaded, x, rhs, got), 0);
  ASSERT_FLOAT_EQ(got[0], 8.0f, 0.0f);
  ASSERT_FLOAT_EQ(got[1], 15.0f, 0.0f);

  int program2_len = 0;
  uint8_t *program2 = poly_instance_export_program(loaded, &program2_len);
  ASSERT_NOT_NULL(program2);
  ASSERT_INT_EQ(program2_len, program_len);
  ASSERT_TRUE(memcmp(program2, program, (size_t)program_len) == 0);

  int portable_len = 17;
  ASSERT_TRUE(poly_instance_export_ir(loaded, &portable_len) == NULL);
  ASSERT_INT_EQ(portable_len, 0);
  ASSERT_INT_EQ(poly_instance_set_device(loaded, POLY_DEVICE_INTERP), -1);
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(loaded, POLY_OPTIM_SGD, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f), -1
  );

  free(program2);
  poly_instance_free(loaded);
  free(program);
  poly_instance_free(source);
  free(ir);
  PASS();
}

TEST(program, weights_are_separate_and_required) {
  int ir_len = 0;
  uint8_t *ir = program_test_ir(true, &ir_len);
  ASSERT_NOT_NULL(ir);
  PolyInstance *source = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(source);
  float weight[2] = {4, 5};
  ASSERT_INT_EQ(poly_instance_write_buf_named(source, "weight", weight, sizeof(weight)), 0);

  int program_len = 0, weights_len = 0;
  uint8_t *program = poly_instance_export_program(source, &program_len);
  uint8_t *weights = poly_instance_export_weights(source, &weights_len);
  ASSERT_NOT_NULL(program);
  ASSERT_NOT_NULL(weights);
  ASSERT_TRUE(poly_instance_from_program(program, program_len, NULL, 0) == NULL);

  PolyInstance *loaded = poly_instance_from_program(program, program_len, weights, weights_len);
  ASSERT_NOT_NULL(loaded);
  const float x[2] = {2, 3};
  float got[2] = {0};
  ASSERT_INT_EQ(program_call_mul(loaded, x, NULL, got), 0);
  ASSERT_FLOAT_EQ(got[0], 8.0f, 0.0f);
  ASSERT_FLOAT_EQ(got[1], 15.0f, 0.0f);

  poly_instance_free(loaded);
  free(weights);
  free(program);
  poly_instance_free(source);
  free(ir);
  PASS();
}

TEST(program, rejects_wrong_format_abi_and_truncation) {
  int ir_len = 0;
  uint8_t *ir = program_test_ir(false, &ir_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_TRUE(poly_instance_from_program(ir, ir_len, NULL, 0) == NULL);

  PolyInstance *source = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(source);
  int program_len = 0;
  uint8_t *program = poly_instance_export_program(source, &program_len);
  ASSERT_NOT_NULL(program);

  PolyIrSpec wrong_decoder = {0};
  ASSERT_INT_EQ(poly_ir_import(program, program_len, &wrong_decoder), -1);
  ASSERT_TRUE(poly_instance_from_program(program, program_len - 1, NULL, 0) == NULL);

  uint8_t *bad_abi = malloc((size_t)program_len);
  ASSERT_NOT_NULL(bad_abi);
  memcpy(bad_abi, program, (size_t)program_len);
  bad_abi[8] ^= 1;
  ASSERT_TRUE(poly_instance_from_program(bad_abi, program_len, NULL, 0) == NULL);

  uint8_t *trailing = malloc((size_t)program_len + 1);
  ASSERT_NOT_NULL(trailing);
  memcpy(trailing, program, (size_t)program_len);
  trailing[program_len] = 0;
  ASSERT_TRUE(poly_instance_from_program(trailing, program_len + 1, NULL, 0) == NULL);

  free(trailing);
  free(bad_abi);
  free(program);
  poly_instance_free(source);
  free(ir);
  PASS();
}
