/*
 * test_ir.c -- Tests for poly_ir binary export/import
 */

#include "test_harness.h"
#include "../src/ir.h"
#include "../src/bigint.h"
#include "../src/ctx.h"
#include "../src/frontend.h"
#include "../src/frontend_internal.h"
#include "../src/device.h"
#include "../src/engine/schedule.h"
#include <string.h>
#include <stdlib.h>
#include <limits.h>

/* Round-trip: simple add graph */

TEST(ir, round_trip_add) {
  PolyCtx *ctx = poly_ctx_new();

  /* Build: out = a + b */
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *store = poly_store_val(ctx, out_buf, sum);
  PolyUOp *sink = poly_sink1(ctx, store);

  int64_t shape4[] = {4};
  PolyIrBufEntry bufs[] = {
      {.name = "a", .role = POLY_IR_ROLE_INPUT, .buffer = a, .shape = {4}, .ndim = 1},
      {.name = "b", .role = POLY_IR_ROLE_INPUT, .buffer = b, .shape = {4}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out_buf, .shape = {4}, .ndim = 1},
  };
  (void)shape4;

  PolyIrEntrypoint eps[] = {
      {.name = "forward", .sink = sink},
  };

  PolyIrSpec spec = {ctx, bufs, 3, eps, 1, NULL, 0};

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);
  ASSERT_TRUE(out_len > 32);

  /* Import */
  PolyIrSpec imported;
  int ret = poly_ir_import(bytes, out_len, &imported);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_NOT_NULL(imported.ctx);
  ASSERT_INT_EQ(imported.n_bufs, 3);
  ASSERT_INT_EQ(imported.n_entrypoints, 1);

  /* Check buffer names */
  ASSERT_STR_EQ(imported.bufs[0].name, "a");
  ASSERT_INT_EQ(imported.bufs[0].role, POLY_IR_ROLE_INPUT);
  ASSERT_INT_EQ(imported.bufs[0].ndim, 1);
  ASSERT_INT_EQ(imported.bufs[0].shape[0], 4);

  ASSERT_STR_EQ(imported.bufs[1].name, "b");
  ASSERT_STR_EQ(imported.bufs[2].name, "output");

  ASSERT_STR_EQ(imported.entrypoints[0].name, "forward");
  ASSERT_NOT_NULL(imported.entrypoints[0].sink);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

TEST(ir, export_rewinds_scratch_root_toposorts) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out_buf, sum));

  PolyIrBufEntry bufs[] = {
      {.name = "a", .role = POLY_IR_ROLE_INPUT, .buffer = a, .shape = {4}, .ndim = 1},
      {.name = "b", .role = POLY_IR_ROLE_INPUT, .buffer = b, .shape = {4}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out_buf, .shape = {4}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, bufs, 3, eps, 1, NULL, 0};

  size_t scratch_before = poly_arena_used(ctx->scratch);
  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);
  ASSERT_TRUE(out_len > 0);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  free(bytes);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Round-trip: graph with CONST args */

TEST(ir, round_trip_const) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *two = poly_const_float(ctx, 2.0);
  PolyUOp *scaled = poly_alu2(ctx, POLY_OP_MUL, a, two);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *store = poly_store_val(ctx, out, scaled);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIrBufEntry bufs[] = {
      {.name = "input", .role = POLY_IR_ROLE_INPUT, .buffer = a, .shape = {4}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out, .shape = {4}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, bufs, 2, eps, 1, NULL, 0};

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  int ret = poly_ir_import(bytes, out_len, &imported);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_INT_EQ(imported.n_bufs, 2);
  ASSERT_STR_EQ(imported.bufs[0].name, "input");
  ASSERT_STR_EQ(imported.bufs[1].name, "output");

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

TEST(ir, round_trip_exact_bigint_arg_current_format) {
  /* Pinned tinygrad UOp args retain positive uint64 values above INT64_MAX.
   * The current Polygrad IR format must preserve the same exact identity, not
   * a signed surrogate. */
  PolyCtx *ctx = poly_ctx_new();
  PolyInt value = {0};
  ASSERT_TRUE(poly_int_from_decimal(&value, "18446744073709550593"));
  PolyUOp *constant =
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_int_as_arg(&value));
  poly_int_free(&value);
  ASSERT_NOT_NULL(constant);

  PolyUOp *out = poly_buffer(ctx, POLY_UINT64, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, constant));
  PolyIrBufEntry bufs[] = {
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out, .shape = {1}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, bufs, 1, eps, 1, NULL, 0};

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);
  ASSERT_TRUE(out_len > 32);
  ASSERT_INT_EQ(bytes[4], 10);
  ASSERT_INT_EQ(bytes[5], 0);

  PolyIrSpec imported;
  ASSERT_INT_EQ(poly_ir_import(bytes, out_len, &imported), 0);
  int n_topo = 0;
  PolyUOp **topo =
      poly_toposort_alloc(imported.ctx, imported.entrypoints[0].sink, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *imported_big = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_CONST && topo[i]->arg.kind == POLY_ARG_BIGINT)
      imported_big = topo[i];
  ASSERT_NOT_NULL(imported_big);
  char *decimal = poly_arg_integer_to_decimal(imported_big->arg);
  ASSERT_STR_EQ(decimal, "18446744073709550593");
  free(decimal);
  poly_toposort_free(topo);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

TEST(ir, round_trip_bufferize_opts_arg) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *device = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *copy = poly_uop2(ctx, POLY_OP_COPY, POLY_FLOAT32, a, device, poly_arg_none());
  PolyUOp *bound = poly_const_int(ctx, 8);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *bsrc[] = {copy, range};
  PolyUOp *bufferize = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, bsrc, 2,
      poly_arg_bufferize_opts("CPU:1", POLY_ADDR_GLOBAL, false)
  );
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *store = poly_store_val(ctx, out, bufferize);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIrBufEntry bufs[] = {
      {.name = "input", .role = POLY_IR_ROLE_INPUT, .buffer = a, .shape = {8}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out, .shape = {8}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, bufs, 2, eps, 1, NULL, 0};

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  ASSERT_INT_EQ(poly_ir_import(bytes, out_len, &imported), 0);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(imported.ctx, imported.entrypoints[0].sink, &n_topo);
  ASSERT_NOT_NULL(topo);
  bool found = false, found_device = false;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_DEVICE) {
      ASSERT_EQ(topo[i]->arg.kind, POLY_ARG_STRING);
      ASSERT_STR_EQ(topo[i]->arg.str, "CPU:1");
      found_device = true;
    }
    if (topo[i]->op != POLY_OP_STAGE) continue;
    ASSERT_INT_EQ(topo[i]->arg.kind, POLY_ARG_BUFFERIZE_OPTS);
    ASSERT_STR_EQ(poly_bufferize_arg_device(topo[i]->arg), "CPU:1");
    ASSERT_INT_EQ(poly_bufferize_arg_addrspace(topo[i]->arg), POLY_ADDR_GLOBAL);
    ASSERT_FALSE(poly_bufferize_arg_removable(topo[i]->arg));
    found = true;
  }
  ASSERT_TRUE(found);
  ASSERT_TRUE(found_device);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

TEST(ir, round_trip_bufferize_opts_tuple_device) {
  /* Pinned BufferizeOpts.device preserves ordered device tuples
   * (tinygrad/schedule/indexing.py:38-43). PGIR must not collapse this to
   * unknown/AUTO metadata. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *devices[] = {"CPU", "CPU:1"};
  PolyUOp *input = poly_buffer_f32(ctx, 8);
  PolyUOp *bound = poly_const_int(ctx, 8);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *stage_src[] = {input, range};
  PolyUOp *stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_src, 2,
      poly_arg_bufferize_opts_tuple(devices, 2, POLY_ADDR_GLOBAL, false)
  );
  PolyUOp *output = poly_buffer_f32(ctx, 8);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, output, stage));
  PolyIrBufEntry bufs[] = {
      {.name = "input", .role = POLY_IR_ROLE_INPUT, .buffer = input, .shape = {8}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = output, .shape = {8}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, bufs, 2, eps, 1, NULL, 0};

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);
  ASSERT_INT_EQ(bytes[4], 10);
  PolyIrSpec imported;
  ASSERT_INT_EQ(poly_ir_import(bytes, out_len, &imported), 0);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(imported.ctx, imported.entrypoints[0].sink, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *imported_stage = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_STAGE) imported_stage = topo[i];
  ASSERT_NOT_NULL(imported_stage);
  ASSERT_TRUE(poly_bufferize_arg_device_is_tuple(imported_stage->arg));
  ASSERT_INT_EQ(poly_bufferize_arg_n_devices(imported_stage->arg), 2);
  ASSERT_STR_EQ(poly_bufferize_arg_devices(imported_stage->arg)[0], "CPU");
  ASSERT_STR_EQ(poly_bufferize_arg_devices(imported_stage->arg)[1], "CPU:1");
  PolyUOp *imported_device = poly_uop_device_uop_cached(imported.ctx, imported_stage, NULL);
  ASSERT_NOT_NULL(imported_device);
  ASSERT_INT_EQ(imported_device->arg.kind, POLY_ARG_STRING_TUPLE);
  ASSERT_INT_EQ(imported_device->arg.string_tuple.n, 2);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

TEST(ir, round_trip_tuple_device_arg) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  const char *devices[] = {"CPU:0", "cpu:1"};
  PolyUOp *tuple_device = poly_device_uop_from_names(ctx, devices, 2);
  PolyUOp *input = poly_buffer_f32(ctx, 4);
  PolyUOp *copy = poly_uop2(ctx, POLY_OP_COPY, POLY_FLOAT32, input, tuple_device, poly_arg_none());
  PolyUOp *output = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, output, copy));
  PolyIrBufEntry bufs[] = {
      {.name = "input", .role = POLY_IR_ROLE_INPUT, .buffer = input, .shape = {4}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = output, .shape = {4}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, bufs, 2, eps, 1, NULL, 0};

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);
  ASSERT_INT_EQ(bytes[4], 10);

  PolyIrSpec imported;
  ASSERT_INT_EQ(poly_ir_import(bytes, out_len, &imported), 0);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(imported.ctx, imported.entrypoints[0].sink, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *imported_device = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_DEVICE && topo[i]->arg.kind == POLY_ARG_STRING_TUPLE) {
      imported_device = topo[i];
      break;
    }
  }
  ASSERT_NOT_NULL(imported_device);
  ASSERT_INT_EQ(imported_device->arg.string_tuple.n, 2);
  ASSERT_STR_EQ(imported_device->arg.string_tuple.vals[0], "CPU");
  ASSERT_STR_EQ(imported_device->arg.string_tuple.vals[1], "CPU:1");

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

TEST(ir, round_trip_paramarg_exact_device_identity) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *shape = poly_const_int(ctx, 4);
  PolyParamArg param_arg = {
      .slot = 3,
      .name = NULL,
      .addrspace = POLY_ADDR_GLOBAL,
      .device = "CPU:1",
  };
  PolyUOp *param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&param_arg));
  PolyUOp *sink = poly_sink1(ctx, param);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(sink);

  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, NULL, 0, eps, 1, NULL, 0};
  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  ASSERT_INT_EQ(poly_ir_import(bytes, out_len, &imported), 0);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(imported.ctx, imported.entrypoints[0].sink, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *imported_param = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_PARAM && topo[i]->arg.kind == POLY_ARG_PARAM)
      imported_param = topo[i];
  ASSERT_NOT_NULL(imported_param);
  ASSERT_NOT_NULL(imported_param->arg.param);
  ASSERT_INT_EQ(imported_param->arg.param->slot, 3);
  ASSERT_STR_EQ(imported_param->arg.param->device, "CPU:1");
  ASSERT_STR_EQ(poly_uop_device_name(imported.ctx, imported_param), "CPU:1");

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

TEST(ir, round_trip_paramarg_ordered_device_tuple) {
  /* Pinned ParamArg.device preserves str | tuple[str, ...] | None
   * (tinygrad/uop/ops.py:1071-1076). PGIR v9 must retain tuple order and keep
   * it CSE-distinct from a scalar identity. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *shape = poly_const_int(ctx, 4);
  const char *devices[] = {"CPU", "CPU:1"};
  const char *reversed[] = {"CPU:1", "CPU"};
  PolyParamArg tuple_arg = {
      .slot = 4,
      .addrspace = POLY_ADDR_GLOBAL,
      .devices = devices,
      .n_devices = 2,
      .device_is_tuple = true,
  };
  PolyParamArg reversed_arg = {
      .slot = 4,
      .addrspace = POLY_ADDR_GLOBAL,
      .devices = reversed,
      .n_devices = 2,
      .device_is_tuple = true,
  };
  PolyParamArg scalar_arg = {
      .slot = 4,
      .addrspace = POLY_ADDR_GLOBAL,
      .device = "CPU",
  };
  PolyUOp *param =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&tuple_arg));
  PolyUOp *same =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&tuple_arg));
  PolyUOp *reverse =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&reversed_arg));
  PolyUOp *scalar =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&scalar_arg));
  ASSERT_NOT_NULL(param);
  ASSERT_PTR_EQ(param, same);
  ASSERT_TRUE(param != reverse);
  ASSERT_TRUE(param != scalar);
  PolyUOp *multi = poly_uop1(ctx, POLY_OP_MULTI, POLY_FLOAT32, param, poly_arg_int(0));
  PolyShape multi_shape = poly_uop_max_shape(ctx, multi);
  ASSERT_INT_EQ(multi_shape.ndim, 1);
  ASSERT_INT_EQ(multi_shape.dims[0], 8);
  free(multi_shape.dims);

  PolyUOp *sink = poly_sink1(ctx, param);
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, NULL, 0, eps, 1, NULL, 0};
  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  ASSERT_INT_EQ(poly_ir_import(bytes, out_len, &imported), 0);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(imported.ctx, imported.entrypoints[0].sink, &n_topo);
  PolyUOp *imported_param = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_PARAM && topo[i]->arg.kind == POLY_ARG_PARAM)
      imported_param = topo[i];
  ASSERT_NOT_NULL(imported_param);
  ASSERT_TRUE(imported_param->arg.param->device_is_tuple);
  ASSERT_TRUE(imported_param->arg.param->device == NULL);
  ASSERT_INT_EQ(imported_param->arg.param->n_devices, 2);
  ASSERT_STR_EQ(imported_param->arg.param->devices[0], "CPU");
  ASSERT_STR_EQ(imported_param->arg.param->devices[1], "CPU:1");
  PolyUOp *imported_multi =
      poly_uop1(imported.ctx, POLY_OP_MULTI, POLY_FLOAT32, imported_param, poly_arg_int(0));
  PolyShape imported_shape = poly_uop_max_shape(imported.ctx, imported_multi);
  ASSERT_INT_EQ(imported_shape.ndim, 1);
  ASSERT_INT_EQ(imported_shape.dims[0], 8);
  free(imported_shape.dims);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

TEST(ir, round_trip_default_call_info) {
  /* PGIR v9 carries the serializable default CallInfo subset used by pinned
   * value-producing FUNCTIONs (tinygrad/uop/ops.py:1083-1092,1158-1170). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *value = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.0));
  PolyUOp *body = poly_uop1(ctx, POLY_OP_TUPLE, POLY_VOID, value, poly_arg_none());
  PolyCallInfo info = {.name = "round_trip"};
  PolyUOp *function = poly_uop1(
      ctx, POLY_OP_FUNCTION, POLY_VOID, body, poly_arg_call_info(&info));
  PolyUOp *selected = poly_uop1(
      ctx, POLY_OP_GETTUPLE, POLY_FLOAT32, function, poly_arg_int(0));
  PolyUOp *sink = poly_sink1(ctx, selected);
  ASSERT_NOT_NULL(sink);

  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, NULL, 0, eps, 1, NULL, 0};
  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);
  PolyIrSpec imported;
  ASSERT_INT_EQ(poly_ir_import(bytes, out_len, &imported), 0);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(
      imported.ctx, imported.entrypoints[0].sink, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *imported_function = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_FUNCTION) imported_function = topo[i];
  ASSERT_NOT_NULL(imported_function);
  ASSERT_INT_EQ(imported_function->arg.kind, POLY_ARG_CALL_INFO);
  ASSERT_NOT_NULL(imported_function->arg.call_info);
  ASSERT_STR_EQ(imported_function->arg.call_info->name, "round_trip");
  ASSERT_FALSE(imported_function->arg.call_info->precompile);
  ASSERT_FALSE(imported_function->arg.call_info->precompile_backward);
  ASSERT_FALSE(imported_function->arg.call_info->has_grad_fxn);
  ASSERT_FALSE(imported_function->arg.call_info->has_metadata);
  ASSERT_FALSE(imported_function->arg.call_info->has_aux);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

/* Round-trip: multiple entrypoints */

TEST(ir, round_trip_multi_entry) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = poly_buffer_f32(ctx, 4);
  PolyUOp *fwd_out = poly_buffer_f32(ctx, 4);
  PolyUOp *fwd_store = poly_store_val(ctx, fwd_out, x);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);

  PolyUOp *loss_out = poly_buffer_f32(ctx, 1);
  PolyUOp *loss_val = poly_const_float(ctx, 0.0);
  PolyUOp *loss_store = poly_store_val(ctx, loss_out, loss_val);
  PolyUOp *loss_sink = poly_sink1(ctx, loss_store);

  PolyIrBufEntry bufs[] = {
      {.name = "x", .role = POLY_IR_ROLE_INPUT, .buffer = x, .shape = {4}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = fwd_out, .shape = {4}, .ndim = 1},
      {.name = "loss", .role = POLY_IR_ROLE_OUTPUT, .buffer = loss_out, .shape = {1}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {
      {.name = "forward", .sink = fwd_sink},
      {.name = "loss", .sink = loss_sink},
  };
  PolyIrSpec spec = {ctx, bufs, 3, eps, 2, NULL, 0};

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  int ret = poly_ir_import(bytes, out_len, &imported);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_INT_EQ(imported.n_entrypoints, 2);
  ASSERT_STR_EQ(imported.entrypoints[0].name, "forward");
  ASSERT_STR_EQ(imported.entrypoints[1].name, "loss");

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

TEST(ir, import_reserves_unique_ids_for_future_buffers) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = poly_buffer_f32(ctx, 4);
  PolyUOp *loss_out = poly_buffer_f32(ctx, 1);
  PolyUOp *loss_store = poly_store_val(ctx, loss_out, poly_const_float(ctx, 0.0));
  PolyUOp *sink = poly_sink1(ctx, loss_store);

  PolyIrBufEntry bufs[] = {
      {.name = "x", .role = POLY_IR_ROLE_INPUT, .buffer = x, .shape = {4}, .ndim = 1},
      {.name = "loss", .role = POLY_IR_ROLE_OUTPUT, .buffer = loss_out, .shape = {1}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "loss", .sink = sink}};
  PolyIrSpec spec = {ctx, bufs, 2, eps, 1, NULL, 0};

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  ASSERT_INT_EQ(poly_ir_import(bytes, out_len, &imported), 0);
  PolyUOp *fresh = poly_buffer(imported.ctx, POLY_FLOAT32, 1);
  ASSERT_NOT_NULL(fresh);
  for (int i = 0; i < imported.n_bufs; i++)
    ASSERT_TRUE(fresh != imported.bufs[i].buffer);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

TEST(ir, round_trip_entrypoint_metadata) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = poly_buffer_f32(ctx, 4);
  PolyUOp *y = poly_buffer_f32(ctx, 1);
  PolyUOp *fwd_out = poly_buffer_f32(ctx, 4);
  PolyUOp *fwd_store = poly_store_val(ctx, fwd_out, x);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);

  PolyUOp *loss_out = poly_buffer_f32(ctx, 1);
  PolyUOp *loss_val = poly_const_float(ctx, 0.0);
  PolyUOp *loss_store = poly_store_val(ctx, loss_out, loss_val);
  PolyUOp *loss_sink = poly_sink1(ctx, loss_store);

  PolyIrBufEntry bufs[] = {
      {.name = "x", .role = POLY_IR_ROLE_INPUT, .buffer = x, .shape = {4}, .ndim = 1},
      {.name = "y", .role = POLY_IR_ROLE_TARGET, .buffer = y, .shape = {1}, .ndim = 1},
      {.name = "logits", .role = POLY_IR_ROLE_OUTPUT, .buffer = fwd_out, .shape = {4}, .ndim = 1},
      {.name = "loss", .role = POLY_IR_ROLE_OUTPUT, .buffer = loss_out, .shape = {1}, .ndim = 1},
  };
  const char *forward_inputs[] = {"x"};
  const char *forward_outputs[] = {"logits"};
  const char *loss_inputs[] = {"x", "y"};
  const char *loss_outputs[] = {"loss"};
  PolyIrEntrypoint eps[] = {
      {
          .name = "forward",
          .sink = fwd_sink,
          .inputs = forward_inputs,
          .n_inputs = 1,
          .outputs = forward_outputs,
          .n_outputs = 1,
      },
      {
          .name = "loss",
          .sink = loss_sink,
          .inputs = loss_inputs,
          .n_inputs = 2,
          .outputs = loss_outputs,
          .n_outputs = 1,
          .objective = "loss",
          .flags = 7,
      },
  };
  PolyIrSpec spec = {ctx, bufs, 4, eps, 2, NULL, 0};

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  ASSERT_INT_EQ(poly_ir_import(bytes, out_len, &imported), 0);
  ASSERT_INT_EQ(imported.n_entrypoints, 2);
  ASSERT_STR_EQ(imported.entrypoints[0].name, "forward");
  ASSERT_INT_EQ(imported.entrypoints[0].n_inputs, 1);
  ASSERT_STR_EQ(imported.entrypoints[0].inputs[0], "x");
  ASSERT_INT_EQ(imported.entrypoints[0].n_outputs, 1);
  ASSERT_STR_EQ(imported.entrypoints[0].outputs[0], "logits");
  ASSERT_TRUE(imported.entrypoints[0].objective == NULL);

  ASSERT_STR_EQ(imported.entrypoints[1].name, "loss");
  ASSERT_INT_EQ(imported.entrypoints[1].n_inputs, 2);
  ASSERT_STR_EQ(imported.entrypoints[1].inputs[0], "x");
  ASSERT_STR_EQ(imported.entrypoints[1].inputs[1], "y");
  ASSERT_INT_EQ(imported.entrypoints[1].n_outputs, 1);
  ASSERT_STR_EQ(imported.entrypoints[1].outputs[0], "loss");
  ASSERT_STR_EQ(imported.entrypoints[1].objective, "loss");
  ASSERT_INT_EQ(imported.entrypoints[1].flags, 7);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

/* Round-trip: param roles */

TEST(ir, round_trip_roles) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *w = poly_buffer_f32(ctx, 6);
  PolyUOp *x = poly_buffer_f32(ctx, 3);
  PolyUOp *out = poly_buffer_f32(ctx, 2);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, w, x);
  PolyUOp *store = poly_store_val(ctx, out, sum);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIrBufEntry bufs[] = {
      {.name = "layers.0.weight",
       .role = POLY_IR_ROLE_PARAM,
       .buffer = w,
       .shape = {2, 3},
       .ndim = 2},
      {.name = "x", .role = POLY_IR_ROLE_INPUT, .buffer = x, .shape = {3}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out, .shape = {2}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, bufs, 3, eps, 1, NULL, 0};

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  int ret = poly_ir_import(bytes, out_len, &imported);
  ASSERT_INT_EQ(ret, 0);

  /* Check roles preserved */
  ASSERT_STR_EQ(imported.bufs[0].name, "layers.0.weight");
  ASSERT_INT_EQ(imported.bufs[0].role, POLY_IR_ROLE_PARAM);
  ASSERT_INT_EQ(imported.bufs[0].ndim, 2);
  ASSERT_INT_EQ(imported.bufs[0].shape[0], 2);
  ASSERT_INT_EQ(imported.bufs[0].shape[1], 3);

  ASSERT_INT_EQ(imported.bufs[1].role, POLY_IR_ROLE_INPUT);
  ASSERT_INT_EQ(imported.bufs[2].role, POLY_IR_ROLE_OUTPUT);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

/* Genuine scalar-dtype poly.ir.uops@1/@2 payloads. These fixtures use the
 * historical 11-byte node header; they are not v3 bytes with a changed version.
 * Graph: one void SINK named "main", with no interface buffers. */
static const uint8_t IR_V1_SCALAR_FIXTURE[] = {
    0x50, 0x47, 0x49, 0x52, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x6d, 0x61, 0x69, 0x6e,
    0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

static const uint8_t IR_V2_SCALAR_FIXTURE[] = {
    0x50, 0x47, 0x49, 0x52, 0x02, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x6d, 0x61, 0x69, 0x6e,
    0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xff, 0xff, 0xff, 0xff,
};

TEST(ir, import_genuine_v1_scalar_fixture) {
  PolyIrSpec imported;
  ASSERT_INT_EQ(
      poly_ir_import(
          IR_V1_SCALAR_FIXTURE, (int)sizeof(IR_V1_SCALAR_FIXTURE), &imported),
      0);
  ASSERT_NOT_NULL(imported.ctx);
  ASSERT_INT_EQ(imported.n_bufs, 0);
  ASSERT_INT_EQ(imported.n_entrypoints, 1);
  ASSERT_STR_EQ(imported.entrypoints[0].name, "main");
  ASSERT_INT_EQ(imported.entrypoints[0].flags, 0);
  ASSERT_NOT_NULL(imported.entrypoints[0].sink);
  ASSERT_INT_EQ(imported.entrypoints[0].sink->op, POLY_OP_SINK);
  ASSERT_TRUE(poly_dtype_eq(imported.entrypoints[0].sink->dtype, POLY_VOID));
  ASSERT_INT_EQ(imported.entrypoints[0].sink->dtype.count, 1);
  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  PASS();
}

TEST(ir, import_genuine_v2_scalar_fixture) {
  PolyIrSpec imported;
  ASSERT_INT_EQ(
      poly_ir_import(
          IR_V2_SCALAR_FIXTURE, (int)sizeof(IR_V2_SCALAR_FIXTURE), &imported),
      0);
  ASSERT_NOT_NULL(imported.ctx);
  ASSERT_INT_EQ(imported.n_bufs, 0);
  ASSERT_INT_EQ(imported.n_entrypoints, 1);
  ASSERT_STR_EQ(imported.entrypoints[0].name, "main");
  ASSERT_INT_EQ(imported.entrypoints[0].flags, 7);
  ASSERT_NOT_NULL(imported.entrypoints[0].sink);
  ASSERT_INT_EQ(imported.entrypoints[0].sink->op, POLY_OP_SINK);
  ASSERT_TRUE(poly_dtype_eq(imported.entrypoints[0].sink->dtype, POLY_VOID));
  ASSERT_INT_EQ(imported.entrypoints[0].sink->dtype.count, 1);
  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  PASS();
}

/* Invalid data */

TEST(ir, import_bad_magic) {
  uint8_t data[32] = {0};
  PolyIrSpec spec;
  int ret = poly_ir_import(data, 32, &spec);
  ASSERT_INT_EQ(ret, -1);
  PASS();
}

TEST(ir, import_truncated) {
  uint8_t data[16] = {'P', 'G', 'I', 'R'};
  PolyIrSpec spec;
  int ret = poly_ir_import(data, 16, &spec);
  ASSERT_INT_EQ(ret, -1);
  PASS();
}

TEST(ir, interface_shape_capacity_is_enforced_on_export_and_import) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *value = poly_buffer_f32(ctx, 1);
  PolyUOp *sink = poly_sink1(ctx, value);
  int64_t rank_eight[] = {1, 1, 1, 1, 1, 1, 1, 1};
  PolyIrBufEntry row = {
      .name = "rank_boundary", .role = POLY_IR_ROLE_PARAM, .buffer = value,
      .shape = {1, 1, 1, 1, 1, 1, 1, 1}, .ndim = POLY_IR_MAX_DIMS,
  };
  PolyIrEntrypoint ep = {.name = "forward", .sink = sink};
  PolyIrSpec spec = {.ctx = ctx, .bufs = &row, .n_bufs = 1,
                     .entrypoints = &ep, .n_entrypoints = 1};

  int ir_len = 0;
  uint8_t *ir = poly_ir_export(&spec, &ir_len);
  ASSERT_NOT_NULL(ir);
  PolyIrSpec imported = {0};
  ASSERT_INT_EQ(poly_ir_import(ir, ir_len, &imported), 0);
  ASSERT_INT_EQ(imported.bufs[0].ndim, POLY_IR_MAX_DIMS);
  ASSERT_TRUE(memcmp(imported.bufs[0].shape, rank_eight, sizeof(rank_eight)) == 0);
  PolyCtx *imported_ctx = imported.ctx;
  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported_ctx);

  /* The rank field is the four bytes immediately before the exact eight
   * serialized dimensions. Locate that unique interface payload and turn a
   * valid rank-8 artifact into a truncated rank-9 artifact. Import must reject
   * it before reading beyond the fixed interface row. */
  int rank_offset = -1;
  for (int i = 4; i + (int)sizeof(rank_eight) <= ir_len; i++) {
    if (ir[i - 4] == POLY_IR_MAX_DIMS && ir[i - 3] == 0 && ir[i - 2] == 0 &&
        ir[i - 1] == 0 && memcmp(ir + i, rank_eight, sizeof(rank_eight)) == 0) {
      ASSERT_INT_EQ(rank_offset, -1);
      rank_offset = i - 4;
    }
  }
  ASSERT_TRUE(rank_offset >= 0);
  ir[rank_offset] = POLY_IR_MAX_DIMS + 1;
  ASSERT_INT_EQ(poly_ir_import(ir, ir_len, &imported), -1);
  free(ir);

  row.ndim = POLY_IR_MAX_DIMS + 1;
  ir_len = 7;
  ASSERT_EQ(poly_ir_export(&spec, &ir_len), NULL);
  ASSERT_INT_EQ(ir_len, 0);

  row.ndim = 1;
  row.shape[0] = -1;
  ASSERT_EQ(poly_ir_export(&spec, &ir_len), NULL);
  row.ndim = 2;
  row.shape[0] = INT64_MAX;
  row.shape[1] = 2;
  ASSERT_EQ(poly_ir_export(&spec, &ir_len), NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Round-trip: graph with int tuple args */

TEST(ir, round_trip_reshape) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 6);
  int64_t new_shape[] = {2, 3};
  PolyUOp *reshaped = poly_reshape(ctx, a, new_shape, 2);
  PolyUOp *out = poly_buffer_f32(ctx, 6);
  PolyUOp *store = poly_store_val(ctx, out, reshaped);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIrBufEntry bufs[] = {
      {.name = "input", .role = POLY_IR_ROLE_INPUT, .buffer = a, .shape = {6}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out, .shape = {2, 3}, .ndim = 2},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyUOp *module_inputs[] = {a};
  PolyIrModule modules[] = {
      {.name = "reshape", .inputs = module_inputs, .n_inputs = 1, .output = reshaped},
  };
  PolyIrSpec spec = {ctx, bufs, 2, eps, 1, modules, 1};

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  int ret = poly_ir_import(bytes, out_len, &imported);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_INT_EQ(imported.n_bufs, 2);
  ASSERT_INT_EQ(imported.n_modules, 1);
  ASSERT_STR_EQ(imported.modules[0].name, "reshape");
  ASSERT_INT_EQ(imported.modules[0].n_inputs, 1);
  ASSERT_INT_EQ(imported.modules[0].inputs[0]->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(imported.modules[0].output->op, POLY_OP_RESHAPE);
  PolyUOp *imported_sink = imported.entrypoints[0].sink;
  ASSERT_NOT_NULL(imported_sink);
  ASSERT_INT_EQ(imported_sink->op, POLY_OP_SINK);
  ASSERT_INT_EQ(imported_sink->n_src, 1);
  PolyUOp *imported_store = imported_sink->src[0];
  ASSERT_INT_EQ(imported_store->op, POLY_OP_STORE);
  PolyUOp *imported_reshape = imported_store->src[1];
  ASSERT_INT_EQ(imported_reshape->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(imported_reshape->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(imported_reshape->n_src, 2);
  ASSERT_INT_EQ(imported_reshape->src[1]->op, POLY_OP_STACK);
  ASSERT_TRUE(
      poly_dtype_eq(imported_reshape->src[1]->dtype, poly_dtype_vec(POLY_INDEX, 2))
  );
  ASSERT_INT_EQ(imported_reshape->src[1]->n_src, 2);
  ASSERT_INT_EQ(imported_reshape->src[1]->src[0]->arg.i, 2);
  ASSERT_INT_EQ(imported_reshape->src[1]->src[1]->arg.i, 3);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}
