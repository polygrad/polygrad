/* Current Tinygrad LINEAR compilation and execution gates. */

#define _POSIX_C_SOURCE 200809L
#include "test_harness.h"

#include "../src/ctx.h"
#include "../src/device.h"
#include "../src/engine/jit.h"
#include "../src/engine/realize.h"
#include "../src/engine/schedule.h"
#include "../src/frontend.h"
#include "../src/codegen/codegen.h"
#include "../src/schedule/memory.h"
#include "../src/schedule/rangeify.h"
#include "../src/schedule/schedule.h"
#include "../src/uop/spec.h"
#include "../src/uop/ops.h"
#include "../src/tensor.h"

TEST(schedule_runtime, owner_call_arguments_exclude_only_bound_variables) {
  /* get_call_arg_uops filters bound variables, not every ALU BUFFER/PARAM. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *var =
      poly_uop_variable(ctx, "offset", poly_arg_int(0), poly_arg_int(4), POLY_WEAKINT, 1, false);
  PolyUOp *bound = poly_uop_bind(ctx, var, 2);
  PolyUOp *a = poly_test_program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *b = poly_test_program_param(ctx, POLY_FLOAT32, 1, 1);
  PolyUOp *sources[] = {
      poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none()), bound, a, bound, b, var};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, sources, 6, poly_arg_none());
  bool correct = poly_call_n_buffer_args(call) == 3 && poly_call_buffer_arg(call, 0) == a &&
                 poly_call_buffer_arg(call, 1) == b && poly_call_buffer_arg(call, 2) == var &&
                 poly_call_buffer_arg(call, 3) == NULL && poly_call_buffer_arg(call, -1) == NULL;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

#ifdef POLY_TESTING
TEST(schedule_runtime, closeout_resolve_preserves_invocation_graph_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_test_program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *b = poly_test_program_param(ctx, POLY_FLOAT32, 4, 1);
  PolyUOp *inputs[] = {
      poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU),
      poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU)};
  bool correct = poly_test_resolve_linear_param(ctx, a, inputs, 2) == inputs[0] &&
                 poly_test_resolve_linear_param(ctx, inputs[0], inputs, 2) == inputs[0];
  PolyUOp *select = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, a, poly_arg_int(0));
  PolyUOp *slice = poly_shrink(ctx, b, (int64_t[][2]){{1, 3}}, 1);
  PolyUOp *views[] = {select, slice};
  for (int i = 0; i < 2; i++) {
    PolyUOp **src = malloc((size_t)views[i]->n_src * sizeof(*src));
    memcpy(src, views[i]->src, (size_t)views[i]->n_src * sizeof(*src));
    src[0] = inputs[i];
    PolyUOp *expected = poly_uop_tagged_arg(
        ctx, views[i]->op, views[i]->dtype, src, views[i]->n_src, views[i]->arg, views[i]->tag,
        views[i]->tag_arg
    );
    correct &= poly_test_resolve_linear_param(ctx, views[i], inputs, 2) == expected;
    free(src);
  }
  PolyUOp *stack = poly_uop2(ctx, POLY_OP_MSTACK, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *expected =
      poly_uop2(ctx, POLY_OP_MSTACK, POLY_FLOAT32, inputs[0], inputs[1], poly_arg_none());
  correct &= poly_test_resolve_linear_param(ctx, stack, inputs, 2) == expected;
  correct &= !poly_test_resolve_linear_param(ctx, b, inputs, 1);
  correct &= !poly_test_resolve_linear_param(ctx, stack, inputs, 1);
  correct &= !poly_test_resolve_linear_param(ctx, a, NULL, 2);
  /* No resolution publishes its substitutions into the reusable template. */
  correct &= select->src[0] == a && slice->src[0] == b && stack->src[0] == a;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}
#endif

static int count_root_ops(PolyCtx *ctx, PolyUOp *root, PolyOps op) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++)
    count += topo[i]->op == op;
  poly_toposort_free(topo);
  return count;
}

static bool contains_uop(PolyUOp **items, int count, PolyUOp *item) {
  for (int i = 0; i < count; i++)
    if (items[i] == item) return true;
  return false;
}

/* resolve_linear_call replaces p0 in each CALL source, not positional data
 * inside another CALL's body. NOOP avoids unrelated kernel compilation here. */
static PolyUOp *binding_publication_call(PolyCtx *ctx, int64_t value, bool used) {
  PolyUOp *var = poly_uop_variable(
      ctx, "value", poly_arg_int(-(INT64_C(1) << 40)), poly_arg_int(INT64_C(1) << 40), POLY_INT64,
      1, false
  );
  PolyUOp *param = poly_uop_variable(
      ctx, "p0", poly_arg_int(-(INT64_C(1) << 40)), poly_arg_int(INT64_C(1) << 40), POLY_INT64, 1,
      true
  );
  PolyUOp *body =
      poly_uop(ctx, POLY_OP_NOOP, POLY_VOID, used ? &param : NULL, used ? 1 : 0, poly_arg_none());
  PolyUOp *sources[] = {body, param};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, sources, used ? 2 : 1, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  return poly_uop2(
      ctx, POLY_OP_CALL, POLY_VOID, linear, poly_uop_bind(ctx, var, value), poly_arg_none()
  );
}

TEST(schedule_runtime, binding_publication_rejects_narrowing) {
  /* Tinygrad retains Python ints. C's existing PolyVarBinding is int32_t:
   * an unsupported value must fail, never become a different binding. */
  const int64_t values[] = {
      INT32_MIN, INT32_MAX, (int64_t)INT32_MIN - 1, (int64_t)INT32_MAX + 1, INT64_C(4294967303)};
  bool correct = true;
  for (int used = 0; used < 2; used++) {
    for (size_t i = 0; i < sizeof(values) / sizeof(*values); i++) {
      PolyCtx *ctx = poly_ctx_new();
      PolyUOp *call = binding_publication_call(ctx, values[i], used);
      PolyVarBinding *bindings = NULL;
      int count = 0;
      PolyUOp *linear = poly_create_linear_with_vars(ctx, call, &bindings, &count);
      if (!used)
        correct &= linear && count == 0;
      else if (values[i] < INT32_MIN || values[i] > INT32_MAX) {
        if (linear && count)
          fprintf(stderr, "bound %lld became %d\n", (long long)values[i], bindings[0].value);
        correct &= !linear && !bindings && count == 0;
      } else {
        correct &= linear && count == 1 && bindings[0].value == values[i];
        if (linear) {
          PolyUOp *resolved = linear->src[0]->src[0]->src[0];
          correct &= resolved->op == POLY_OP_PARAM && strcmp(poly_uop_expr(resolved), "value") == 0;
        }
      }
      free(bindings);
      poly_ctx_destroy(ctx);
    }
  }
  ASSERT_TRUE(correct);
  PASS();
}

TEST(schedule_runtime, binding_publication_memory_size_limits) {
  /* C size_t admission is not Python's arbitrary-precision size domain.
   * These are graph-only probes; never attempt the huge physical allocation. */
  const int64_t sizes[] = {1, 128, (int64_t)(SIZE_MAX / 2), INT64_MAX};
  bool correct = true;
  for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *buffer = poly_test_buffer_on_device(ctx, POLY_UINT16, sizes[i], POLY_DEVICE_CPU);
    PolyUOp *body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
    PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, body, buffer, poly_arg_none());
    PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
    PolyUOp *planned = poly_memory_plan_rewrite(ctx, linear, NULL, 0);
    correct &= i < 2 ? planned && planned->src[0]->src[1]->op == POLY_OP_BITCAST : !planned;
    correct &= call->src[1] == buffer && !poly_buffer_get(ctx, buffer);
    poly_ctx_destroy(ctx);
  }
  ASSERT_TRUE(correct);
  PASS();
}

#ifdef POLY_TESTING
TEST(schedule_runtime, binding_publication_substitution_failure) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *call = binding_publication_call(ctx, 7, true);
  PolyUOp *original_param = call->src[0]->src[0]->src[0]->src[0];
  PolyVarBinding *bindings = NULL;
  int count = 0;
  bool correct = true;
  /* Also fail after one source was rewritten: no partially bound CALL may
   * escape, and the original must remain reusable. */
  for (int failure = 0; failure < 2; failure++) {
    poly_test_substitute_fail_after(failure);
    PolyUOp *failed = poly_create_linear_with_vars(ctx, call, &bindings, &count);
    poly_test_substitute_fail_after(-1);
    correct &= !failed && !bindings && count == 0;
    free(bindings);
  }
  PolyUOp *retry = poly_create_linear_with_vars(ctx, call, &bindings, &count);
  correct &= retry && count == 1 && bindings[0].value == 7;
  correct &= call->src[0]->src[0]->src[0]->src[0] == original_param;
  if (retry) {
    correct &= strcmp(poly_uop_expr(retry->src[0]->src[0]->src[0]), "value") == 0;
    correct &= retry->src[0]->src[1] == retry->src[0]->src[0]->src[0];
  }
  free(bindings);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(schedule_runtime, binding_publication_memory_failure) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 16, POLY_DEVICE_CPU);
  PolyUOp *body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, body, buffer, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  poly_test_substitute_fail_after(0);
  PolyUOp *failed = poly_memory_plan_rewrite(ctx, linear, NULL, 0);
  poly_test_substitute_fail_after(-1);
  PolyUOp *retry = poly_memory_plan_rewrite(ctx, linear, NULL, 0);
  bool correct = !failed && retry && retry != linear && retry->src[0]->src[0] == body &&
                 retry->src[0]->src[1]->op == POLY_OP_BITCAST && call->src[1] == buffer;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(schedule_runtime, binding_publication_jit_failure) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *input = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_INTERP);
  PolyUOp *output = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_INTERP);
  PolyUOp *copy = poly_uop1(ctx, POLY_OP_COPY, POLY_FLOAT32, input, poly_arg_str("INTERP"));
  PolyUOp *src[] = {copy, output, input};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, 3, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  PolyUOp *held[] = {input, output};
  poly_test_substitute_fail_after(0);
  PolyUOp *failed = poly_jit_lower(ctx, linear, held, 2, &input, 1);
  poly_test_substitute_fail_after(-1);
  PolyUOp *retry = poly_jit_lower(ctx, linear, held, 2, &input, 1);
  bool correct = retry && call->src[2] == input;
  if (retry) {
    correct &= retry->src[0]->src[2]->op == POLY_OP_PARAM;
    correct &= retry->src[0]->src[2]->arg.param->slot == 0;
    PolyUOp *replacement = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_INTERP);
    float value = 13.0f, got = 0;
    correct &= poly_buffer_write(ctx, replacement, &value, sizeof(value)) == 0;
    correct &= poly_run_linear(ctx, retry, NULL, 0, &replacement, 1, true, true, false) == 0;
    correct &= poly_buffer_read(ctx, output, &got, sizeof(got)) == 0 && got == value;
  }
  if (failed) fprintf(stderr, "failed JIT input substitution returned an executable\n");
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(failed == NULL);
  ASSERT_TRUE(correct);
  PASS();
}
#endif

#ifdef POLY_TESTING
TEST(schedule_runtime, jit_input_scalar_substitution_failure_is_not_capture) {
  bool correct = true;
  for (int failure = 0; failure < 2; failure++) {
    PolyCtx *ctx = poly_ctx_new();
    int64_t shape[] = {6};
    PolyTensor *input = poly_tensor_empty_by_id(
        ctx, poly_dtype_id_by_name("float32"), shape, 1, POLY_DEVICE_INTERP
    );
    PolyUOp *var =
        poly_uop_variable(ctx, "offset", poly_arg_int(0), poly_arg_int(4), POLY_WEAKINT, 1, false);
    PolyUOp *start = poly_uop_bind(ctx, var, 2), *size = poly_const_int(ctx, 2);
    PolyTensor *view = poly_tensor_shrink_uop(ctx, input, &start, &size, 1);
    PolyUOp *original = poly_tensor_uop_physical(view);
    PolyJit *jit = poly_jit_new(ctx);
    poly_test_substitute_fail_after(failure);
    int rc = poly_jit_begin_capture(jit, &view, 1);
    poly_test_substitute_fail_after(-1);
    correct &= rc == -1 && !poly_jit_is_capturing(jit) && ctx->active_jit_capture == NULL;
    if (!rc) poly_jit_cancel_capture(jit);
    correct &= poly_jit_begin_capture(jit, &view, 1) == 0;
    correct &= poly_tensor_uop_physical(view) == original;
    poly_jit_cancel_capture(jit);
    poly_jit_free(jit);
    poly_ctx_destroy(ctx);
  }
  ASSERT_TRUE(correct);
  PASS();
}
#endif

TEST(schedule_runtime, interp_scalar_params_use_numeric_bindings) {
  /* PythonProgram consumes numeric vals. Polygrad's runner ABI supplies
   * int32_t pointers regardless of the PARAM's computation dtype. */
  PolyDType types[] = {POLY_INT32, POLY_INT64, POLY_FLOAT32, POLY_FLOAT64};
  bool correct = true;
  for (size_t t = 0; t < sizeof(types) / sizeof(*types); t++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *buffer = poly_test_program_param(ctx, POLY_FLOAT64, 1, 0);
    PolyParamArg arg = {
        .slot = 1,
        .addrspace = POLY_ADDR_ALU,
        .name = "scalar",
        .has_minmax = true,
        .min_val = poly_arg_int(INT32_MIN),
        .max_val = poly_arg_int(INT32_MAX)};
    PolyUOp *scalar = poly_uop0(ctx, POLY_OP_PARAM, types[t], poly_arg_param(&arg));
    PolyUOp *zero = poly_const_int(ctx, 0);
    PolyUOp *ptr = poly_uop_index(ctx, buffer, &zero, 1);
    PolyUOp *store = poly_uop2(
        ctx, POLY_OP_STORE, POLY_VOID, ptr, poly_cast(ctx, scalar, POLY_FLOAT64), poly_arg_none()
    );
    PolyKernelInfo info = {.name = "scalar_numeric_binding"};
    PolyUOp *sink =
        poly_uop_tagged(ctx, POLY_OP_SINK, POLY_VOID, &store, 1, poly_arg_kernel_info(&info), 1);
    PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, sink, buffer, poly_arg_none());
    PolyRunner runner = {0};
    int rc = poly_time_call_prepare(ctx, call, POLY_DEVICE_INTERP, &runner);
    correct &= rc == 0;
    if (!rc) {
      /* Exactly four bytes: sanitizers must catch a future dtype-sized read. */
      int32_t *value = malloc(sizeof(*value));
      if (!value) {
        poly_time_call_finish(ctx, &runner, POLY_DEVICE_INTERP);
        poly_ctx_destroy(ctx);
        FAIL("scalar test allocation failed");
      }
      const int32_t values[] = {-3, 7, 16777217};
      for (size_t i = 0; i < sizeof(values) / sizeof(*values); i++) {
        *value = values[i];
        double got = NAN;
        void *args[] = {&got, value};
        PolyVarBinding binding = {.var = scalar, .value = *value};
        double elapsed =
            poly_time_call(&runner, POLY_DEVICE_INTERP, args, 2, &binding, 1, 1, INFINITY, 0);
        /* PythonProgram preserves numeric pvals until a consuming operation;
         * a float32 PARAM alone does not round the int32 binding. */
        double expected = *value;
        correct &= isfinite(elapsed) && got == expected;
      }
      free(value);
      poly_time_call_finish(ctx, &runner, POLY_DEVICE_INTERP);
    }
    poly_ctx_destroy(ctx);
  }
  ASSERT_TRUE(correct);
  PASS();
}

TEST(schedule_runtime, beam_time_call_reads_scalar_values) {
#ifdef __EMSCRIPTEN__
  PolyDevice devices[] = {POLY_DEVICE_WASM, POLY_DEVICE_INTERP};
#else
  PolyDevice devices[] = {POLY_DEVICE_CPU, POLY_DEVICE_INTERP};
#endif
  for (size_t d = 0; d < sizeof(devices) / sizeof(devices[0]); d++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *buffer = poly_test_program_param(ctx, POLY_FLOAT32, 1, 0);
    PolyParamArg arg = {
        .slot = 1,
        .addrspace = POLY_ADDR_ALU,
        .name = "scalar",
        .has_minmax = true,
        .min_val = poly_arg_int(-8),
        .max_val = poly_arg_int(8)};
    PolyUOp *scalar = poly_uop0(ctx, POLY_OP_PARAM, POLY_INT32, poly_arg_param(&arg));
    PolyUOp *zero = poly_const_int(ctx, 0);
    PolyUOp *ptr = poly_uop_index(ctx, buffer, &zero, 1);
    PolyUOp *store = poly_uop2(
        ctx, POLY_OP_STORE, POLY_VOID, ptr, poly_cast(ctx, scalar, POLY_FLOAT32), poly_arg_none()
    );
    PolyKernelInfo info = {.name = "test"};
    PolyUOp *sink =
        poly_uop_tagged(ctx, POLY_OP_SINK, POLY_VOID, &store, 1, poly_arg_kernel_info(&info), 1);
    PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, sink, buffer, poly_arg_none());
    PolyRunner runner = {0};
    int rc = poly_time_call_prepare(ctx, call, devices[d], &runner);
    bool correct = rc == 0 && runner.n_params == 1 && runner.n_vars == 1;
    if (rc == 0) {
      int values[] = {7, -3};
      for (int i = 0; i < 2; i++) {
        float output = NAN;
        void *args[] = {&output, &values[i]};
        PolyVarBinding binding = {.var = scalar, .value = values[i]};
        double elapsed = poly_time_call(&runner, devices[d], args, 2, &binding, 1, 1, INFINITY, 0);
        correct &= isfinite(elapsed) && output == values[i];
        if (output != values[i])
          fprintf(stderr, "scalar backend%d: expected%d, got%g\n", devices[d], values[i], output);
      }
      poly_time_call_finish(ctx, &runner, devices[d]);
    }
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(correct);
  }
  PASS();
}

TEST(schedule_runtime, allocation_free_linear_does_not_collect_residency) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  float value = 1.0f;
  poly_buffer_set(ctx, buffer, &value, sizeof(value), POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(poly_buffer_get(ctx, buffer));
  ctx->collection_dirty = true;

  PolyUOp *linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, NULL, 0, poly_arg_none());
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(poly_run_linear(ctx, linear, NULL, 0, NULL, 0, true, true, false), 0);
  ASSERT_NOT_NULL(poly_buffer_get(ctx, buffer));
  ASSERT_TRUE(ctx->collection_dirty);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, create_schedule_returns_ordered_linear_calls) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));

  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(linear->n_src, 1);
  ASSERT_EQ(poly_test_linear_call(linear, 0)->op, POLY_OP_CALL);
  ASSERT_EQ(poly_test_linear_call_body(linear, 0)->op, POLY_OP_SINK);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(linear, 0), 3);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(linear, 0, 0), out);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(linear, 0, 1), a);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(linear, 0, 2), b);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, effect_sink_parameterizes_concrete_buffers) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));
  PolyVarBinding *vars = NULL;
  int n_vars = 0;

  PolyUOp *linear = poly_linear_effect_sink(ctx, sink, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(n_vars, 0);
  ASSERT_INT_EQ(linear->n_src, 1);
  PolyUOp *body = poly_test_linear_call_body(linear, 0);
  ASSERT_NOT_NULL(body);
  ASSERT_INT_EQ(count_root_ops(ctx, body, POLY_OP_BUFFER), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, body, POLY_OP_PARAM), 3);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(linear, 0), 3);

  free(vars);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, call_rejects_unused_out_of_range_program_globals) {
  /* exec_kernel indexes resolved arguments for every global, including unused
   * parameters absent from outs/ins. C rejects before indexing runtime arrays. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  int slot = 1;
  PolyProgramInfo info = {.name = "invalid_global", .globals = &slot, .n_globals = 1};
  PolyUOp *program = poly_uop0(ctx, POLY_OP_PROGRAM, POLY_VOID, poly_arg_program_info(&info));
  PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, program, buffer, poly_arg_none());
  bool outs[1], ins[1];
  int result = poly_call_get_outs_ins(ctx, call, outs, ins, 1);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(result, -1);
  PASS();
}

TEST(schedule_runtime, compiler_sink_uses_kernel_info) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *sink = poly_uop_sink_ex(ctx, NULL, 0, "named", 1);
  ASSERT_NOT_NULL(sink);

  ASSERT_EQ(sink->arg.kind, POLY_ARG_KERNEL_INFO);
  ASSERT_NOT_NULL(sink->arg.kernel_info);
  ASSERT_STR_EQ(sink->arg.kernel_info->name, "named");
  ASSERT_INT_EQ(sink->arg.kernel_info->beam, 0);
  ASSERT_EQ(sink->tag_arg.kind, POLY_ARG_NONE);

  poly_ctx_destroy(ctx);
  PASS();
}

static PolyUOp *runtime_test_graph(PolyCtx *ctx, PolyUOp *linear) {
  PolyUOp *function =
      poly_uop1(ctx, POLY_OP_CUSTOM_FUNCTION, POLY_VOID, linear, poly_arg_str("graph"));
  PolyUOp *call = poly_uop1(ctx, POLY_OP_CALL, POLY_VOID, function, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
}

#ifdef POLY_HAS_CUDA
static bool runtime_scalar_integer_args(bool graph) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_INT64, 1, POLY_DEVICE_CUDA);
  PolyUOp *ptr = poly_test_program_param(ctx, POLY_INT64, 1, 0);
  PolyParamArg arg = {
      .slot = 1,
      .addrspace = POLY_ADDR_ALU,
      .name = "scalar",
      .has_minmax = true,
      .min_val = poly_arg_int(INT32_MIN),
      .max_val = poly_arg_int(INT32_MAX)};
  PolyUOp *var = poly_uop0(ctx, POLY_OP_PARAM, POLY_INT64, poly_arg_param(&arg));
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, ptr, &zero, 1), var, poly_arg_none()
  );
  PolyKernelInfo info = {.name = "scalar_integer_args"};
  PolyUOp *sink =
      poly_uop_tagged(ctx, POLY_OP_SINK, POLY_VOID, &store, 1, poly_arg_kernel_info(&info), 1);
  PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, sink, out, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  linear = poly_compile_linear(ctx, linear, 0);
  bool correct = linear != NULL;
  if (linear && graph) linear = runtime_test_graph(ctx, linear);
  const int32_t values[] = {-3, 7, INT32_MIN, INT32_MAX};
  for (size_t i = 0; linear && i < sizeof(values) / sizeof(*values); i++) {
    PolyVarBinding binding = {.var = var, .value = values[i]};
    int64_t got = 0;
    int rc = poly_run_linear(ctx, linear, &binding, 1, NULL, 0, true, false, true);
    correct &= rc == 0 && poly_buffer_read(ctx, out, &got, sizeof(got)) == 0;
    if (got != values[i])
      fprintf(stderr, "CUDA graph=%d scalar=%d got=%lld\n", graph, values[i], (long long)got);
    correct &= got == values[i];
  }
  poly_ctx_destroy(ctx);
  return correct;
}

TEST_BACKEND(cuda, runtime_scalar_integer_args) {
  ASSERT_TRUE(runtime_scalar_integer_args(false));
  PASS();
}

TEST_BACKEND(cuda, graph_scalar_integer_args) {
  ASSERT_TRUE(runtime_scalar_integer_args(true));
  PASS();
}
#endif

static bool runtime_allocates_only_program_globals(PolyDevice device) {
  /* exec_kernel resolves CALL arguments, but only ProgramInfo.globals own
   * runtime allocations. An eliminated argument remains in the CALL. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, device);
  PolyUOp *unused = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, device);
  PolyUOp *param = poly_test_program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, param, &zero, 1),
      poly_const_float(ctx, 7.0), poly_arg_none()
  );
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "unused_arg");
  PolyUOp *sources[] = {sink, out, unused};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, sources, 3, poly_arg_none());
  PolyUOp *linear = poly_compile_linear(
      ctx, poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none()), -1
  );
  if (!linear) {
    poly_ctx_destroy(ctx);
    return false;
  }
  const PolyProgramInfo *info = poly_program_info(ctx, linear->src[0]->src[0]);
  bool ok = info && info->n_globals == 1 && info->globals[0] == 0 &&
            poly_call_n_buffer_args(linear->src[0]) == 2;
  for (int replay = 0; replay < 2; replay++) {
    float got = 0;
    ok &= poly_run_linear(ctx, linear, NULL, 0, NULL, 0, true, true, true) == 0;
    ok &= poly_buffer_read(ctx, out, &got, sizeof(got)) == 0 && got == 7.0f;
    PolyBuffer *handle = poly_buffer_get(ctx, unused);
    ok &= !handle || !handle->ptr;
  }
  /* Unused does not mean an invalid lexical PARAM slot may be ignored. */
  PolyUOp *bad = poly_test_program_param(ctx, POLY_FLOAT32, 1, 7);
  PolyUOp *bad_sources[] = {linear->src[0]->src[0], out, bad};
  PolyUOp *bad_call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, bad_sources, 3, poly_arg_none());
  PolyUOp *bad_linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, bad_call, poly_arg_none());
  ok &= poly_run_linear(ctx, bad_linear, NULL, 0, NULL, 0, false, true, true) == -1;
  poly_ctx_destroy(ctx);
  return ok;
}

TEST(schedule_runtime, runtime_allocates_only_program_globals) {
  ASSERT_TRUE(runtime_allocates_only_program_globals(POLY_DEVICE_AUTO));
  PASS();
}

static bool runtime_copy_accepts_empty_storage(PolyDevice device, bool graph) {
  /* COPY preserves unspecified empty-storage bytes, not an initialization
   * promise. Do not read them as values or require a particular bit pattern. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, device);
  PolyUOp *in = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, device);
  PolyUOp *copy = poly_uop1(
      ctx, POLY_OP_COPY, POLY_FLOAT32, in,
      poly_arg_str(poly_uop_device_uop_cached(ctx, out, NULL)->arg.str)
  );
  if (!copy) {
    poly_ctx_destroy(ctx);
    return false;
  }
  PolyUOp *sources[] = {copy, out, in};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, sources, 3, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  if (graph) linear = runtime_test_graph(ctx, linear);
  int rc = poly_run_linear(ctx, linear, NULL, 0, NULL, 0, true, true, true);
  PolyBuffer *dst = poly_buffer_get(ctx, out);
  PolyBuffer *src = poly_buffer_get(ctx, in);
  bool allocated = dst && dst->ptr && src && src->ptr;
  if (!rc) rc = poly_run_linear(ctx, linear, NULL, 0, NULL, 0, true, true, true);
  poly_ctx_destroy(ctx);
  if (rc) fprintf(stderr, "empty COPY device=%d graph=%d returned %d\n", device, graph, rc);
  return rc == 0 && allocated;
}

TEST(schedule_runtime, runtime_copy_accepts_empty_storage) {
  ASSERT_TRUE(runtime_copy_accepts_empty_storage(POLY_DEVICE_AUTO, false));
  PASS();
}

#ifdef POLY_HAS_CUDA
TEST_BACKEND(cuda, runtime_graph_copy_accepts_empty_storage) {
  ASSERT_TRUE(runtime_copy_accepts_empty_storage(POLY_DEVICE_CUDA, true));
  PASS();
}
#endif

TEST(schedule_runtime, compile_linear_replaces_compute_body_with_program) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  PolyUOp *raw_body = poly_test_linear_call_body(linear, 0);
  ASSERT_EQ(raw_body->op, POLY_OP_SINK);

  PolyUOp *compiled = poly_compile_linear(ctx, linear, -1);
  ASSERT_NOT_NULL(compiled);
  ASSERT_EQ(compiled->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(compiled->n_src, 1);
  PolyUOp *program = poly_test_linear_call_body(compiled, 0);
  ASSERT_EQ(program->op, POLY_OP_PROGRAM);
  ASSERT_TRUE(program->n_src == 3 || program->n_src == 4);
  ASSERT_EQ(program->src[0]->op, POLY_OP_SINK);
  ASSERT_EQ(program->src[1]->op, POLY_OP_LINEAR);
  ASSERT_EQ(program->src[2]->op, POLY_OP_SOURCE);
  if (program->n_src == 4) {
    ASSERT_EQ(program->src[3]->op, POLY_OP_BINARY);
    ASSERT_TRUE(poly_dtype_eq(program->src[3]->dtype, POLY_UINT8));
  }
  ASSERT_EQ(poly_program_linear(program)->op, POLY_OP_LINEAR);
  ASSERT_PTR_NEQ(program->src[0], raw_body);
  ASSERT_NOT_NULL(poly_program_info(ctx, program));

  poly_ctx_destroy(ctx);
  PASS();
}

static bool boundary_program_resume(PolyDevice device, int missing_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, device);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, device);
  float input[] = {1, -2, 3.5f, 0}, got[4];
  bool written = poly_buffer_write(ctx, a, input, sizeof(input)) == 0;
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, a)));
  PolyUOp *compiled = poly_compile_linear(ctx, poly_test_create_linear(ctx, sink), -1);
  PolyUOp *full = compiled ? poly_test_linear_call_body(compiled, 0) : NULL;
  bool ok = written && full && full->op == POLY_OP_PROGRAM && full->n_src >= 2;
  for (int n = 1; ok && n <= full->n_src; n++) {
    if (missing_metadata == 2 && n != 2) continue;
    PolyUOp *prefix[4];
    memcpy(prefix, full->src, (size_t)n * sizeof(*prefix));
    if (missing_metadata == 2) {
      PolyKernelInfo info = *prefix[0]->arg.kernel_info;
      info.estimates = NULL;
      prefix[0] = poly_uop_tagged_arg(
          ctx, POLY_OP_SINK, prefix[0]->dtype, prefix[0]->src, prefix[0]->n_src,
          poly_arg_kernel_info(&info), prefix[0]->tag, prefix[0]->tag_arg
      );
    }
    PolyUOp *partial = poly_uop(
        ctx, full->op, full->dtype, prefix, n, missing_metadata == 1 ? poly_arg_none() : full->arg
    );
    PolyUOp *call = compiled->src[0];
    PolyUOp **src = malloc((size_t)call->n_src * sizeof(*src));
    if (!src) {
      ok = false;
      break;
    }
    memcpy(src, call->src, (size_t)call->n_src * sizeof(*src));
    src[0] = partial;
    PolyUOp *partial_call = poly_uop_replace_src(ctx, call, src);
    free(src);
    PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, partial_call, poly_arg_none());
    PolyUOp *resumed = poly_compile_linear(ctx, linear, -1);
    PolyUOp *program = resumed ? poly_test_linear_call_body(resumed, 0) : NULL;
    /* X86 also loads its hex SOURCE directly when BINARY is absent. */
    int expected_sources = device == POLY_DEVICE_X86 && n == 3 ? 3 : full->n_src;
    ok = program && program->n_src == expected_sources && poly_program_info(ctx, program) &&
         program->src[0]->arg.kernel_info->estimates;
    for (int i = 0; ok && i < n; i++)
      ok = program->src[i] == full->src[i];
    if (ok)
      ok = poly_run_linear(ctx, resumed, NULL, 0, NULL, 0, true, true, false) == 0 &&
           poly_buffer_read(ctx, out, got, sizeof(got)) == 0;
    for (int i = 0; ok && i < 4; i++)
      ok = got[i] == 2 * input[i];
    if (!ok) fprintf(stderr, "PROGRAM resume device=%d prefix=%d failed\n", device, n);
  }
  poly_ctx_destroy(ctx);
  return ok;
}

TEST(schedule_runtime, boundary_program_resume_cpu) {
  ASSERT_TRUE(boundary_program_resume(POLY_DEVICE_CPU, 0));
  PASS();
}

TEST(schedule_runtime, boundary_program_resume_interp) {
  ASSERT_TRUE(boundary_program_resume(POLY_DEVICE_INTERP, 0));
  PASS();
}

#ifdef POLY_HAS_X86
TEST(schedule_runtime, boundary_program_resume_x86) {
  ASSERT_TRUE(boundary_program_resume(POLY_DEVICE_X86, 0));
  ASSERT_TRUE(boundary_program_resume(POLY_DEVICE_X86, 1));
  ASSERT_TRUE(boundary_program_resume(POLY_DEVICE_X86, 2));
  PASS();
}
#endif

TEST(schedule_runtime, boundary_program_resume_missing_info) {
  ASSERT_TRUE(boundary_program_resume(POLY_DEVICE_CPU, 1));
  PASS();
}

TEST(schedule_runtime, boundary_program_resume_missing_estimates) {
  ASSERT_TRUE(boundary_program_resume(POLY_DEVICE_CPU, 2));
  PASS();
}

TEST(schedule_runtime, compile_linear_rewrites_interp_before_program_verification) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_INTERP);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_INTERP);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_INTERP);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  PolyUOp *raw = poly_test_linear_call_body(linear, 0);
  ASSERT_NOT_NULL(raw);

  int n_raw = 0;
  PolyUOp **raw_topo = poly_toposort_alloc(ctx, raw, &n_raw);
  ASSERT_NOT_NULL(raw_topo);
  int weak_ranges = 0;
  for (int i = 0; i < n_raw; i++)
    weak_ranges += raw_topo[i]->op == POLY_OP_RANGE && poly_dtype_is_weak(raw_topo[i]->dtype);
  poly_toposort_free(raw_topo);
  ASSERT_TRUE(weak_ranges > 0);

  PolyUOp *compiled = poly_compile_linear(ctx, linear, -1);
  ASSERT_NOT_NULL(compiled);
  PolyUOp *program = poly_test_linear_call_body(compiled, 0);
  ASSERT_NOT_NULL(program);
  ASSERT_EQ(program->op, POLY_OP_PROGRAM);
  ASSERT_TRUE(program->n_src >= 2);
  ASSERT_EQ(program->src[0]->op, POLY_OP_SINK);
  ASSERT_EQ(program->src[1]->op, POLY_OP_LINEAR);
  ASSERT_TRUE(poly_type_verify_program(ctx, program->src[0]));

  int n_program = 0;
  PolyUOp **program_topo = poly_toposort_alloc(ctx, program->src[0], &n_program);
  ASSERT_NOT_NULL(program_topo);
  for (int i = 0; i < n_program; i++)
    ASSERT_FALSE(
        program_topo[i]->op == POLY_OP_RANGE && poly_dtype_is_weak(program_topo[i]->dtype)
    );
  poly_toposort_free(program_topo);

  float a_data[2] = {1.0f, 2.0f};
  float b_data[2] = {3.0f, 4.0f};
  float got[2] = {0};
  ASSERT_INT_EQ(poly_buffer_write(ctx, a, a_data, sizeof(a_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, b, b_data, sizeof(b_data)), 0);
  ASSERT_INT_EQ(poly_run_linear(ctx, compiled, NULL, 0, NULL, 0, true, true, false), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 4.0f, 1e-6f);
  ASSERT_FLOAT_EQ(got[1], 6.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, compile_linear_sets_kernel_info_beam) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  PolyUOp *call = poly_test_linear_call(linear, 0);
  PolyUOp *body = poly_test_linear_call_body(linear, 0);
  ASSERT_NOT_NULL(call);
  ASSERT_NOT_NULL(body);
  ASSERT_EQ(body->arg.kind, POLY_ARG_KERNEL_INFO);

  /* Tinygrad compile_linear's pm_beam changes only KernelInfo.beam. Tagging
   * the SINK skips optimization so this regression tests topology, not search. */
  PolyUOp *tagged = poly_uop_tagged_arg(
      ctx, POLY_OP_SINK, body->dtype, body->src, body->n_src, body->arg, 1, body->tag_arg
  );
  PolyUOp **call_src = malloc((size_t)call->n_src * sizeof(*call_src));
  ASSERT_NOT_NULL(call_src);
  memcpy(call_src, call->src, (size_t)call->n_src * sizeof(*call_src));
  call_src[0] = tagged;
  PolyUOp *tagged_call = poly_uop(ctx, POLY_OP_CALL, call->dtype, call_src, call->n_src, call->arg);
  free(call_src);
  ASSERT_NOT_NULL(tagged_call);
  PolyUOp *tagged_linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, tagged_call, poly_arg_none());
  PolyUOp *compiled = poly_compile_linear(ctx, tagged_linear, 4);
  ASSERT_NOT_NULL(compiled);
  PolyUOp *program = poly_test_linear_call_body(compiled, 0);
  ASSERT_EQ(program->op, POLY_OP_PROGRAM);
  ASSERT_EQ(program->src[0]->arg.kind, POLY_ARG_KERNEL_INFO);
  ASSERT_STR_EQ(program->src[0]->arg.kernel_info->name, body->arg.kernel_info->name);
  ASSERT_INT_EQ(program->src[0]->arg.kernel_info->beam, 4);
  ASSERT_INT_EQ(program->src[0]->tag, 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, run_linear_executes_current_call_graph) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  float a_data[4] = {1, 2, 3, 4};
  float b_data[4] = {10, 20, 30, 40};
  float got[4] = {0};
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  ASSERT_INT_EQ(poly_buffer_write(ctx, a, a_data, sizeof(a_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, b, b_data, sizeof(b_data)), 0);
  PolyUOp *value = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *realized = NULL;
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, &value, 1, &realized, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_NOT_NULL(realized);

  ASSERT_INT_EQ(poly_run_linear(ctx, linear, vars, n_vars, NULL, 0, true, false, false), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, realized, got, sizeof(got)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got[i], a_data[i] + b_data[i], 1e-6f);

  free(vars);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, noopt_separates_program_cache) {
  int old = poly_get_noopt();
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 128, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 128, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, a)));
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  poly_set_noopt(0);
  PolyUOp *optimized = poly_compile_linear(ctx, linear, 0);
  if (optimized) poly_uop_retain(ctx, optimized);
  size_t first = poly_to_program_cache_len(ctx);
  poly_set_noopt(1);
  PolyUOp *unoptimized = poly_compile_linear(ctx, linear, 0);
  if (unoptimized) poly_uop_retain(ctx, unoptimized);
  size_t second = poly_to_program_cache_len(ctx);
  poly_set_noopt(0);
  PolyUOp *restored = poly_compile_linear(ctx, linear, 0);
  bool valid = optimized && unoptimized && optimized != unoptimized && restored == optimized &&
               second > first;
  float values[128], got[128];
  for (int i = 0; i < 128; i++)
    values[i] = (float)i;
  valid &= poly_buffer_write(ctx, a, values, sizeof(values)) == 0;
  PolyUOp *programs[] = {optimized, unoptimized};
  for (int p = 0; valid && p < 2; p++) {
    valid &= poly_run_linear(ctx, programs[p], NULL, 0, NULL, 0, true, true, false) == 0;
    valid &= poly_buffer_read(ctx, out, got, sizeof(got)) == 0;
    for (int i = 0; valid && i < 128; i++)
      valid &= got[i] == 2 * values[i];
  }
  poly_set_noopt(old);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(valid);
  PASS();
}

TEST(schedule_runtime, default_dtypes_separate_program_cache) {
  int old_f = poly_get_default_float(), old_i = poly_get_default_int();
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *linear =
      poly_test_create_linear(ctx, poly_sink1(ctx, poly_store_val(ctx, out, poly_add(ctx, a, a))));
  poly_set_default_float(12);
  poly_set_default_int(6);
  PolyUOp *first = poly_compile_linear(ctx, linear, 0);
  if (first) poly_uop_retain(ctx, first);
  size_t n_first = poly_to_program_cache_len(ctx);
  poly_set_default_float(13);
  PolyUOp *second = poly_compile_linear(ctx, linear, 0);
  if (second) poly_uop_retain(ctx, second);
  size_t n_second = poly_to_program_cache_len(ctx);
  poly_set_default_int(8);
  PolyUOp *third = poly_compile_linear(ctx, linear, 0);
  if (third) poly_uop_retain(ctx, third);
  size_t n_third = poly_to_program_cache_len(ctx);
  poly_set_default_float(12);
  poly_set_default_int(6);
  PolyUOp *restored = poly_compile_linear(ctx, linear, 0);
  /* Identical emitted instructions may CSE to the same PROGRAM; distinct
   * policy cache entries, not pointer inequality, prove the configuration key. */
  bool ok =
      first && second && third && restored == first && n_second > n_first && n_third > n_second;
  poly_set_default_float(old_f);
  poly_set_default_int(old_i);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(schedule_runtime, owner_compiler_options_separate_program_cache) {
  /* to_program_key includes compiler policy even when this particular graph
   * emits identical code. Restoring policy must reuse its original entry. */
  const char *names[] = {"NUM_CPU_THREADS", "NOLOCALS", "TC_SELECT", "TC_OPT"};
  const char *first_values[] = {"1", "0", "-1", "0"};
  const char *second_values[] = {"2", "1", "0", "256"};
  bool ok = true;
  for (int i = 0; i < 4; i++) {
    const char *prior = getenv(names[i]);
    char *saved = prior ? strdup(prior) : NULL;
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
    PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
    PolyUOp *linear = poly_test_create_linear(
        ctx, poly_sink1(ctx, poly_store_val(ctx, out, poly_add(ctx, a, a)))
    );
    setenv(names[i], first_values[i], 1);
    PolyUOp *first = poly_compile_linear(ctx, linear, 0);
    if (first) poly_uop_retain(ctx, first);
    size_t n_first = poly_to_program_cache_len(ctx);
    setenv(names[i], second_values[i], 1);
    PolyUOp *second = poly_compile_linear(ctx, linear, 0);
    size_t n_second = poly_to_program_cache_len(ctx);
    setenv(names[i], first_values[i], 1);
    PolyUOp *restored = poly_compile_linear(ctx, linear, 0);
    bool matched = first && second && restored == first && n_second > n_first &&
                   poly_to_program_cache_len(ctx) == n_second;
    if (!matched)
      fprintf(stderr, "cache policy %s: entries %zu -> %zu\n", names[i], n_first, n_second);
    ok &= matched;
    if (saved)
      setenv(names[i], saved, 1);
    else
      unsetenv(names[i]);
    free(saved);
    poly_ctx_destroy(ctx);
  }
  ASSERT_TRUE(ok);
  PASS();
}

TEST(schedule_runtime, closeout_runtime_cache_preserves_call_device_identity) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *param = poly_test_program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, param, &zero, 1),
      poly_const_float(ctx, 7.0), poly_arg_none()
  );
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "call_device_identity");
  PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, sink, out, poly_arg_none());
  PolyUOp *linear = poly_compile_linear(
      ctx, poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none()), -1
  );
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(poly_uop_retain(ctx, linear), 0);
  poly_runtime_cache_clear(ctx);
  bool correct = poly_run_linear(ctx, linear, NULL, 0, NULL, 0, false, true, false) == 0;
  size_t first = poly_runtime_cache_len(ctx);
  PolyUOp *other = poly_uop_new_buffer(
      ctx, poly_device_uop_from_name(ctx, "CPU:1"), 1, POLY_FLOAT32, poly_ctx_next_unique_id(ctx)
  );
  PolyUOp *other_call =
      poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, linear->src[0]->src[0], other, poly_arg_none());
  PolyUOp *other_linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, other_call, poly_arg_none());
  correct &= poly_run_linear(ctx, other_linear, NULL, 0, NULL, 0, false, true, false) == 0;
  float value = 0;
  correct &= poly_buffer_read(ctx, other, &value, sizeof(value)) == 0 && value == 7.0f;
  if (poly_runtime_cache_len(ctx) != first + 1)
    fprintf(
        stderr, "CPU:1 reused CPU runtime identity: %zu -> %zu\n", first,
        poly_runtime_cache_len(ctx)
    );
  correct &= first == 1 && poly_runtime_cache_len(ctx) == first + 1;
  poly_uop_release(ctx, linear);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(schedule_runtime, program_and_runtime_caches_reuse_current_keys) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  poly_to_program_cache_clear(ctx);
  poly_runtime_cache_clear(ctx);

  PolyUOp *compiled0 = poly_compile_linear(ctx, linear, -1);
  ASSERT_NOT_NULL(compiled0);
  /* The C local is the owner corresponding to Tinygrad's live Python LINEAR. */
  ASSERT_INT_EQ(poly_uop_retain(ctx, compiled0), 0);
  size_t program_entries = poly_to_program_cache_len(ctx);
  ASSERT_TRUE(program_entries > 0);
  PolyUOp *compiled1 = poly_compile_linear(ctx, linear, -1);
  ASSERT_PTR_EQ(compiled1, compiled0);
  ASSERT_INT_EQ((int)poly_to_program_cache_len(ctx), (int)program_entries);

  float a_data[4] = {1, 2, 3, 4};
  float b_data[4] = {5, 6, 7, 8};
  ASSERT_INT_EQ(poly_buffer_write(ctx, a, a_data, sizeof(a_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, b, b_data, sizeof(b_data)), 0);
  ASSERT_INT_EQ(poly_run_linear(ctx, compiled0, NULL, 0, NULL, 0, true, true, false), 0);
  size_t runtime_entries = poly_runtime_cache_len(ctx);
  ASSERT_TRUE(runtime_entries > 0);
  ASSERT_INT_EQ(poly_run_linear(ctx, compiled1, NULL, 0, NULL, 0, true, true, false), 0);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), (int)runtime_entries);

  PolyUOp *cached_program = compiled0->src[0]->src[0];
  ASSERT_INT_EQ(cached_program->op, POLY_OP_PROGRAM);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  PolyUOp *rebuilt = cached_program->tag || cached_program->tag_arg.kind != POLY_ARG_NONE
                         ? poly_uop_tagged_arg(
                               ctx, cached_program->op, cached_program->dtype, cached_program->src,
                               cached_program->n_src, cached_program->arg, cached_program->tag,
                               cached_program->tag_arg
                           )
                         : poly_uop(
                               ctx, cached_program->op, cached_program->dtype, cached_program->src,
                               cached_program->n_src, cached_program->arg
                           );
  ASSERT_PTR_EQ(rebuilt, cached_program);
  ASSERT_INT_EQ(poly_run_linear(ctx, compiled0, NULL, 0, NULL, 0, true, true, false), 0);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), (int)runtime_entries);

  poly_uop_release(ctx, compiled0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, symbolic_template_replays_without_model_owner) {
  /* symcpg-style candidate x row loss evaluation stays a direct LINEAR client. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *inputs[8], *expanded[8];
  int64_t shape[] = {2, 3}, rows[] = {1, 3}, candidates[] = {2, 1}, axis[] = {1};
  for (int i = 0; i < 8; i++) {
    inputs[i] = poly_test_buffer_on_device(ctx, POLY_FLOAT32, i < 4 ? 3 : 2, POLY_DEVICE_CPU);
    expanded[i] =
        poly_expand(ctx, poly_reshape(ctx, inputs[i], i < 4 ? rows : candidates, 2), shape, 2);
  }
  PolyUOp *pred = poly_add(
      ctx,
      poly_add(
          ctx, poly_mul(ctx, expanded[0], expanded[4]), poly_mul(ctx, expanded[1], expanded[5])
      ),
      poly_add(ctx, poly_mul(ctx, expanded[2], expanded[6]), expanded[7])
  );
  PolyUOp *diff = poly_sub(ctx, pred, expanded[3]);
  PolyUOp *loss = poly_mul(
      ctx, poly_reduce_axis(ctx, POLY_OP_ADD, poly_mul(ctx, diff, diff), axis, 1),
      poly_const_float(ctx, 1.0 / 3.0)
  );
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  PolyUOp *linear = poly_compile_linear(
      ctx, poly_test_create_linear(ctx, poly_sink1(ctx, poly_store_val(ctx, out, loss))), -1
  );
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(poly_uop_retain(ctx, linear), 0);
  size_t programs = poly_to_program_cache_len(ctx), runtimes = 0;
  uint64_t resident = 0;
  float data[8][3] = {{1, 2, 3}, {2, -1, 1}, {2, -2, 3}, {0, 1, 2},
                      {1, 2},    {0, 1},     {1, -1},    {0, 2}};
  for (int repeat = 0; repeat < 12; repeat++) {
    data[0][0] = (float)repeat;
    data[4][1] = (float)repeat / 4;
    for (int i = 0; i < 8; i++)
      ASSERT_INT_EQ(poly_buffer_write(ctx, inputs[i], data[i], (i < 4 ? 3 : 2) * sizeof(float)), 0);
    ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
    ASSERT_INT_EQ(poly_run_linear(ctx, linear, NULL, 0, NULL, 0, true, true, false), 0);
    float actual[2];
    ASSERT_INT_EQ(poly_buffer_read(ctx, out, actual, sizeof(actual)), 0);
    for (int k = 0; k < 2; k++) {
      float expected = 0;
      for (int r = 0; r < 3; r++) {
        float residual = data[4][k] * data[0][r] + data[5][k] * data[1][r] +
                         data[6][k] * data[2][r] + data[7][k] - data[3][r];
        expected += residual * residual / 3;
      }
      ASSERT_TRUE(isfinite(actual[k]));
      ASSERT_FLOAT_EQ(actual[k], expected, 1e-4f);
    }
    if (repeat == 0) {
      runtimes = poly_runtime_cache_len(ctx);
      resident = ctx->mem_used;
    }
    ASSERT_TRUE(poly_to_program_cache_len(ctx) == programs);
    ASSERT_TRUE(poly_runtime_cache_len(ctx) == runtimes);
    ASSERT_TRUE(ctx->mem_used == resident);
  }
  poly_uop_release(ctx, linear);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_TRUE(ctx->mem_used == 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, memory_plan_honors_no_memory_planner) {
  const char *value = getenv("NO_MEMORY_PLANNER");
  char *saved = value ? strdup(value) : NULL;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 16, POLY_DEVICE_CPU);
  PolyUOp *body = poly_uop_sink_ex(ctx, NULL, 0, "test", 1);
  PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, body, buffer, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  setenv("NO_MEMORY_PLANNER", "1", 1);
  bool disabled = poly_memory_plan_rewrite(ctx, linear, NULL, 0) == linear;
  setenv("NO_MEMORY_PLANNER", "0", 1);
  PolyUOp *planned = poly_memory_plan_rewrite(ctx, linear, NULL, 0);
  bool enabled = planned && planned != linear && planned->src[0]->src[1]->op == POLY_OP_BITCAST;
  if (saved)
    setenv("NO_MEMORY_PLANNER", saved, 1);
  else
    unsetenv("NO_MEMORY_PLANNER");
  free(saved);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(disabled);
  ASSERT_TRUE(enabled);
  PASS();
}

TEST(schedule_runtime, memory_plan_rewrite_uses_one_arena_for_disjoint_temporaries) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *input = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 256, POLY_DEVICE_CPU);
  PolyUOp *tmp0 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 256, POLY_DEVICE_CPU);
  PolyUOp *tmp1 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 256, POLY_DEVICE_CPU);
  PolyUOp *output = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 256, POLY_DEVICE_CPU);
  PolyUOp *body = poly_uop_sink_ex(ctx, NULL, 0, "test", 1);
  PolyUOp *call0_src[] = {body, tmp0, input};
  PolyUOp *call1_src[] = {body, tmp1, tmp0};
  PolyUOp *call2_src[] = {body, output, tmp1};
  PolyUOp *calls[] = {
      poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call0_src, 3, poly_arg_none()),
      poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call1_src, 3, poly_arg_none()),
      poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call2_src, 3, poly_arg_none()),
  };
  PolyUOp *linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, calls, 3, poly_arg_none());
  PolyUOp *held[] = {input, output};
  PolyUOp *planned = poly_memory_plan_rewrite(ctx, linear, held, 2);
  ASSERT_NOT_NULL(planned);
  PolyUOp *planned_tmp0 = planned->src[0]->src[1];
  PolyUOp *planned_tmp1 = planned->src[1]->src[1];
  ASSERT_EQ(planned_tmp0->op, POLY_OP_BITCAST);
  ASSERT_EQ(planned_tmp1->op, POLY_OP_BITCAST);
  ASSERT_EQ(planned_tmp0->src[0]->op, POLY_OP_SHRINK);
  ASSERT_EQ(planned_tmp1->src[0]->op, POLY_OP_SHRINK);
  ASSERT_PTR_EQ(planned_tmp0->src[0]->src[0], planned_tmp1->src[0]->src[0]);
  ASSERT_PTR_EQ(planned->src[0]->src[2], input);
  ASSERT_PTR_EQ(planned->src[2]->src[1], output);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, memory_plan_tuple_view_resolves_before_arena_allocation) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *devices[] = {"CPU", "CPU:1"};
  PolyUOp *device = poly_device_uop_from_names(ctx, devices, 2);
  PolyUOp *arena = poly_uop_new_buffer(ctx, device, 256, POLY_INT8, poly_ctx_next_unique_id(ctx));
  ASSERT_NOT_NULL(arena);
  PolyUOp *slice = poly_shrink(ctx, arena, (int64_t[][2]){{0, 16}}, 1);
  PolyUOp *view = poly_uop1(ctx, POLY_OP_BITCAST, POLY_FLOAT32, slice, poly_arg_none());
  ASSERT_NOT_NULL(view);

  /* Tinygrad 2026-08-22 a9069c17 UOp.buffer recursively creates base Buffer
   * metadata before constructing the memory-planner view (uop/ops.py:922-946). */
  ASSERT_FALSE(poly_buffer_is_allocated(ctx, arena));
  PolyBuffer *resolved = poly_uop_buffer_handle(ctx, view);
  ASSERT_NOT_NULL(resolved);
  ASSERT_TRUE(poly_buffer_is_multi(resolved));
  ASSERT_INT_EQ(resolved->n_bufs, 2);
  PolyBuffer *resolved_children[2];
  for (int lane = 0; lane < resolved->n_bufs; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(resolved, lane);
    ASSERT_NOT_NULL(child);
    resolved_children[lane] = child;
    ASSERT_NOT_NULL(child->base);
    ASSERT_INT_EQ(child->nbytes, 16);
    ASSERT_INT_EQ(child->base->nbytes, 256);
    ASSERT_PTR_EQ(child->base->ptr, NULL);
  }
  ASSERT_PTR_EQ(poly_uop_buffer_handle(ctx, view), resolved);
  for (int lane = 0; lane < resolved->n_bufs; lane++)
    ASSERT_PTR_EQ(poly_buffer_multi_child(resolved, lane), resolved_children[lane]);
  PolyBuffer *lane0 = poly_buffer_multi_child(resolved, 0);
  ASSERT_INT_EQ(poly_buffer_handle_ensure_allocated(ctx, lane0), 0);
  ASSERT_NOT_NULL(lane0->ptr);
  ASSERT_NOT_NULL(lane0->base->ptr);
  ASSERT_PTR_EQ(poly_buffer_multi_child(resolved, 1)->base->ptr, NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, memory_plan_rewrite_plans_tuple_device_buffer) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *names[] = {"CPU", "CPU:1"};
  PolyUOp *device = poly_device_uop_from_names(ctx, names, 2);
  ASSERT_NOT_NULL(device);
  PolyUOp *buffer = poly_uop_new_buffer(ctx, device, 4, POLY_FLOAT32, poly_ctx_next_unique_id(ctx));
  ASSERT_NOT_NULL(buffer);
  PolyUOp *body = poly_uop0(ctx, POLY_OP_CUSTOM_FUNCTION, POLY_VOID, poly_arg_str("probe"));
  PolyUOp *call_src[] = {body, buffer};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 2, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  PolyUOp *planned = poly_memory_plan_rewrite(ctx, linear, NULL, 0);
  ASSERT_NOT_NULL(planned);
  PolyUOp *arg = planned->src[0]->src[1];
  ASSERT_EQ(arg->op, POLY_OP_BITCAST);
  ASSERT_EQ(arg->src[0]->op, POLY_OP_SHRINK);
  ASSERT_EQ(arg->src[0]->src[0]->op, POLY_OP_BUFFER);
  PolyUOp *arena_device = poly_uop_device_uop_cached(ctx, arg->src[0]->src[0], NULL);
  ASSERT_NOT_NULL(arena_device);
  ASSERT_EQ(arena_device->arg.kind, POLY_ARG_STRING_TUPLE);
  ASSERT_INT_EQ(arena_device->arg.string_tuple.n, 2);
  ASSERT_STR_EQ(arena_device->arg.string_tuple.vals[0], "CPU");
  ASSERT_STR_EQ(arena_device->arg.string_tuple.vals[1], "CPU:1");

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, jit_capture_records_linear_before_memory_plan) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, NULL, 0), 0);

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 256, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 256, POLY_DEVICE_CPU);
  PolyUOp *value = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *realized = NULL;
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *returned = poly_linear_with_vars(ctx, &value, 1, &realized, &vars, &n_vars);
  ASSERT_NOT_NULL(returned);
  ASSERT_NOT_NULL(realized);

  /* Tinygrad records the resolved LINEAR and returns an empty LINEAR. */
  ASSERT_EQ(returned->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(returned->n_src, 0);
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 1);

  free(vars);
  poly_jit_cancel_capture(jit);
  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, jit_held_buffers_match_runtime_and_live_tensors) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *materialized = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *live_buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *unheld = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  float data[4] = {0};
  ASSERT_INT_EQ(poly_buffer_write(ctx, materialized, data, sizeof(data)), 0);
  PolyTensor *live = poly_tensor_create_with_roots(
      ctx, live_buffer, live_buffer, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(live);

  PolyUOp **held = NULL;
  int n_held = 0;
  ASSERT_INT_EQ(poly_jit_collect_held_bufs(ctx, &live, 1, &held, &n_held), 0);
  ASSERT_TRUE(contains_uop(held, n_held, materialized));
  ASSERT_TRUE(contains_uop(held, n_held, live_buffer));
  ASSERT_FALSE(contains_uop(held, n_held, unheld));

  free(held);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, create_graph_call_deduplicates_params) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *b0 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CUDA);
  PolyUOp *b1 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CUDA);
  PolyUOp *p0 = poly_uop_param(ctx, 0, b0);
  PolyUOp *p1 = poly_uop_param(ctx, 1, b1);
  PolyUOp *program = poly_uop0(ctx, POLY_OP_PROGRAM, POLY_VOID, poly_arg_none());
  PolyUOp *call0_src[] = {program, p0, p1};
  PolyUOp *call1_src[] = {program, p1, p0};
  PolyUOp *calls[] = {
      poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call0_src, 3, poly_arg_none()),
      poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call1_src, 3, poly_arg_none()),
  };
  PolyUOp *graph = poly_create_graph_call(ctx, calls, 2);
  ASSERT_NOT_NULL(graph);
  ASSERT_EQ(graph->op, POLY_OP_CALL);
  ASSERT_INT_EQ(graph->n_src, 3);
  ASSERT_EQ(graph->src[0]->op, POLY_OP_CUSTOM_FUNCTION);
  ASSERT_STR_EQ(graph->src[0]->arg.str, "graph");
  ASSERT_EQ(graph->src[0]->src[0]->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(graph->src[0]->src[0]->n_src, 2);
  ASSERT_PTR_EQ(graph->src[1], p0);
  ASSERT_PTR_EQ(graph->src[2], p1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, multi_kernel_independent_values_execute_linear) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  float a_data[4] = {1, 2, 3, 4};
  float b_data[4] = {10, 20, 30, 40};
  float got_add[4] = {0}, got_mul[4] = {0};
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  ASSERT_INT_EQ(poly_buffer_write(ctx, a, a_data, sizeof(a_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, b, b_data, sizeof(b_data)), 0);
  PolyUOp *values[] = {
      poly_alu2(ctx, POLY_OP_ADD, a, b),
      poly_alu2(ctx, POLY_OP_MUL, a, b),
  };
  PolyUOp *realized[2] = {0};
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, values, 2, realized, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->n_src, 2);
  ASSERT_INT_EQ(poly_run_linear(ctx, linear, vars, n_vars, NULL, 0, true, false, false), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, realized[0], got_add, sizeof(got_add)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, realized[1], got_mul, sizeof(got_mul)), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(got_add[i], a_data[i] + b_data[i], 1e-6f);
    ASSERT_FLOAT_EQ(got_mul[i], a_data[i] * b_data[i], 1e-6f);
  }

  free(vars);
  poly_ctx_destroy(ctx);
  PASS();
}
