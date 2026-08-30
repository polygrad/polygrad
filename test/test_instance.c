/*
 * test_instance.c -- Tests for PolyInstance runtime
 */

#include "test_harness.h"
#include "../src/instance.h"
#include "../src/codegen/codegen.h"
#include "../src/ctx.h"
#include "../src/engine/jit.h"
#include "../src/engine/realize.h"
#include "../src/engine/schedule.h"
#include "../src/ir.h"
#include "../src/frontend.h"
#include "../src/frontend_internal.h"
#include "../src/engine/schedule.h"
#include "../src/optim.h"
#include "../src/tensor.h"
#include "../src/device.h"
#include "../src/safetensors.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

static int test_find_instance_buf(PolyInstance *inst, const char *name);

static PolyTensor *test_cpu_f32_tensor(
    PolyCtx *ctx,
    int64_t numel,
    float *data,
    PolyUOp **out_buffer
) {
  int64_t shape[] = {numel};
  PolyTensor *tensor = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyUOp *buffer = tensor
                        ? (PolyUOp *)poly_uop_get_buffer_identity(
                              poly_tensor_uop_physical(tensor)
                          )
                        : NULL;
  if (!buffer) return NULL;
  poly_buffer_set(ctx, buffer, data, (size_t)numel * sizeof(float), POLY_DEVICE_CPU);
  if (out_buffer) *out_buffer = buffer;
  return tensor;
}

static PolyDevice test_ctx_execution_device(PolyCtx *ctx) {
  PolyDevice device = poly_ctx_get_preferred_device(ctx);
  return poly_device_can_execute(device) ? device : poly_device_default();
}

static int instance_count_root_op(PolyCtx *ctx, PolyUOp *root, PolyOps op) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_scratch(ctx, root, &n_topo);
  if (!topo) return -1;
  int count = 0;
  for (int i = 0; i < n_topo; i++)
    if (topo[i] && topo[i]->op == op) count++;
  return count;
}

static PolyUOp *instance_logical_f32(PolyCtx *ctx, int64_t size) {
  return poly_uop_new_logical_buffer(ctx, POLY_FLOAT32, size);
}

/* Helper: build IR bytes for a simple add graph */
/* out = a + b, forward entrypoint */
static uint8_t *make_add_ir(int *out_len) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = instance_logical_f32(ctx, 4);
  PolyUOp *b = instance_logical_f32(ctx, 4);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out_buf = instance_logical_f32(ctx, 4);
  PolyUOp *store = poly_store_val(ctx, out_buf, sum);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIrBufEntry bufs[] = {
      {.name = "a", .role = POLY_IR_ROLE_INPUT, .buffer = a, .shape = {4}, .ndim = 1},
      {.name = "b", .role = POLY_IR_ROLE_INPUT, .buffer = b, .shape = {4}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out_buf, .shape = {4}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, bufs, 3, eps, 1, NULL, 0};

  uint8_t *bytes = poly_ir_export(&spec, out_len);
  poly_ctx_destroy(ctx);
  return bytes;
}

TEST(optim, build_step_sgd_uses_current_lr_tensor_with_after_store) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *p_buf = NULL;
  PolyUOp *g_buf = NULL;
  PolyUOp *lr_buf = NULL;
  float p_data[] = {1.0f};
  float g_data[] = {2.0f};
  float lr_data[] = {0.1f};
  PolyTensor *param = test_cpu_f32_tensor(ctx, 1, p_data, &p_buf);
  PolyTensor *grad = test_cpu_f32_tensor(ctx, 1, g_data, &g_buf);
  PolyTensor *lr = test_cpu_f32_tensor(ctx, 1, lr_data, &lr_buf);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(grad);
  ASSERT_NOT_NULL(lr);

  PolyOptimConfig cfg = {
      .kind = POLY_OPTIM_SGD,
      .beta1 = 0.0f,
      .beta2 = 0.0f,
      .eps = 0.0f,
      .weight_decay = 0.0f,
      .momentum = 0.0f,
      .nesterov = false,
      .classic = false,
  };
  int need =
      poly_optim_build_step(ctx, &cfg, lr, &param, &grad, 1, NULL, NULL, NULL, NULL, NULL, 0);
  ASSERT_INT_EQ(need, 1);
  PolyTensor *outs[1] = {NULL};
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr, &param, &grad, 1, NULL, NULL, NULL, NULL, outs, 1), 1
  );
  ASSERT_PTR_EQ(outs[0], param);
  ASSERT_INT_EQ(poly_tensor_uop(param)->op, POLY_OP_AFTER);
  ASSERT_TRUE(poly_uop_reachable(ctx, poly_tensor_uop(param), lr_buf));

  /* The update graph was built while lr contained 0.1. Changing the existing
   * LR buffer before realization must make the update use 0.2. */
  lr_data[0] = 0.2f;
  ASSERT_INT_EQ(poly_buffer_write(ctx, lr_buf, lr_data, sizeof(lr_data)), 0);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, outs, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, param);
  float p_after = 0.0f;
  ASSERT_INT_EQ(poly_buffer_read(ctx, p_buf, &p_after, sizeof(p_after)), 0);
  ASSERT_FLOAT_EQ(p_after, 0.6f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(optim, build_step_sgd_realizes_lazy_state_and_gradient_effects) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *p_buf = NULL;
  PolyUOp *m_buf = NULL;
  PolyUOp *grad_buf = NULL;
  PolyUOp *lr_buf = NULL;
  PolyUOp *g_value_buf = NULL;
  float p_data[] = {1.0f};
  float m_data[] = {0.0f};
  float grad_data[] = {0.0f};
  float lr_data[] = {0.02f};
  float g_data[] = {1.0f};

  PolyTensor *param = test_cpu_f32_tensor(ctx, 1, p_data, &p_buf);
  PolyTensor *momentum = test_cpu_f32_tensor(ctx, 1, m_data, &m_buf);
  PolyTensor *grad = test_cpu_f32_tensor(ctx, 1, grad_data, &grad_buf);
  PolyTensor *lr = test_cpu_f32_tensor(ctx, 1, lr_data, &lr_buf);
  PolyTensor *g_value = test_cpu_f32_tensor(ctx, 1, g_data, &g_value_buf);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(momentum);
  ASSERT_NOT_NULL(grad);
  ASSERT_NOT_NULL(lr);
  ASSERT_NOT_NULL(g_value);

  PolyUOp *zero = poly_full(ctx, (int64_t[]){1}, 1, 0.0f);
  PolyUOp *m_logical = poly_tensor_uop_logical(momentum);
  PolyUOp *m_physical = poly_tensor_uop_physical(momentum);
  PolyUOp *m_init_store_logical = poly_store_val(ctx, m_logical, zero);
  PolyUOp *m_init_store_physical = poly_store_val(ctx, m_physical, zero);
  PolyUOp *m_init_logical_src[2] = {m_logical, m_init_store_logical};
  PolyUOp *m_init_physical_src[2] = {m_physical, m_init_store_physical};
  PolyUOp *m_init_logical = poly_uop(
      ctx, POLY_OP_AFTER, POLY_FLOAT32, m_init_logical_src, 2, poly_arg_none()
  );
  PolyUOp *m_init_physical = poly_uop(
      ctx, POLY_OP_AFTER, POLY_FLOAT32, m_init_physical_src, 2, poly_arg_none()
  );
  PolyUOp *grad_logical = poly_tensor_uop_logical(grad);
  PolyUOp *grad_physical = poly_tensor_uop_physical(grad);
  PolyUOp *grad_store_logical =
      poly_store_val(ctx, grad_logical, poly_tensor_uop_logical(g_value));
  PolyUOp *grad_store_physical =
      poly_store_val(ctx, grad_physical, poly_tensor_uop_physical(g_value));
  PolyUOp *grad_logical_src[2] = {grad_logical, grad_store_logical};
  PolyUOp *grad_physical_src[2] = {grad_physical, grad_store_physical};
  PolyUOp *grad_effect_logical = poly_uop(
      ctx, POLY_OP_AFTER, POLY_FLOAT32, grad_logical_src, 2, poly_arg_none()
  );
  PolyUOp *grad_effect_physical = poly_uop(
      ctx, POLY_OP_AFTER, POLY_FLOAT32, grad_physical_src, 2, poly_arg_none()
  );
  ASSERT_NOT_NULL(zero);
  ASSERT_NOT_NULL(m_init_store_logical);
  ASSERT_NOT_NULL(m_init_store_physical);
  ASSERT_NOT_NULL(m_init_logical);
  ASSERT_NOT_NULL(m_init_physical);
  ASSERT_NOT_NULL(grad_store_logical);
  ASSERT_NOT_NULL(grad_store_physical);
  ASSERT_NOT_NULL(grad_effect_logical);
  ASSERT_NOT_NULL(grad_effect_physical);
  ASSERT_INT_EQ(
      poly_tensor_replace_roots(
          ctx, momentum, m_init_logical, m_init_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
      ),
      0
  );
  ASSERT_INT_EQ(
      poly_tensor_replace_roots(
          ctx, grad, grad_effect_logical, grad_effect_physical, POLY_TENSOR_VALUE,
          POLY_DEVICE_CPU
      ),
      0
  );

  PolyOptimConfig cfg = {
      .kind = POLY_OPTIM_SGD,
      .weight_decay = 0.0f,
      .momentum = 0.85f,
      .nesterov = true,
      .classic = false,
  };
  PolyTensor *m_state[] = {momentum};
  ASSERT_INT_EQ(
      poly_optim_build_step(
          ctx, &cfg, lr, &param, &grad, 1, m_state, NULL, NULL, NULL, NULL, 0
      ),
      2
  );
  PolyTensor *targets[2] = {NULL, NULL};
  ASSERT_INT_EQ(
      poly_optim_build_step(
          ctx, &cfg, lr, &param, &grad, 1, m_state, NULL, NULL, NULL, targets, 2
      ),
      2
  );
  ASSERT_PTR_EQ(targets[0], momentum);
  ASSERT_PTR_EQ(targets[1], param);
  PolyUOp *momentum_logical = poly_tensor_uop_logical(momentum);
  PolyUOp *param_logical = poly_tensor_uop_logical(param);
  ASSERT_NOT_NULL(momentum_logical);
  ASSERT_NOT_NULL(param_logical);
  ASSERT_EQ(momentum_logical->op, POLY_OP_AFTER);
  ASSERT_EQ(param_logical->op, POLY_OP_AFTER);
  ASSERT_TRUE(poly_uop_reachable(ctx, param_logical, momentum_logical));

  /* Current Tensor.linear_with_vars publishes callify's becomes-map before
   * create_linear_with_vars. Inspect that exact boundary before execution. */
  PolyUOp *physical_targets[2] = {poly_tensor_uop_physical(targets[0]),
                                  poly_tensor_uop_physical(targets[1])};
  PolyUOp *callified_targets[2] = {NULL, NULL};
  PolyUOp **map_orig = NULL;
  PolyUOp **map_repl = NULL;
  int map_n = 0;
  PolyUOp *big_call = poly_transform_to_call_with_map(
      ctx, physical_targets, 2, callified_targets, &map_orig, &map_repl, &map_n
  );
  ASSERT_NOT_NULL(big_call);
  ASSERT_INT_EQ(
      poly_tensor_apply_realize_map(ctx, map_orig, map_repl, map_n, POLY_DEVICE_AUTO), 0
  );
  PolyVarBinding *bindings = NULL;
  int n_bindings = 0;
  PolyUOp *linear = poly_create_linear_with_vars(
      ctx, big_call, &bindings, &n_bindings
  );
  ASSERT_NOT_NULL(linear);
  int max_call_args = 0;
  for (int i = 0; i < linear->n_src; i++) {
    PolyUOp *call = linear->src[i];
    ASSERT_EQ(call->op, POLY_OP_CALL);
    if (call->n_src - 1 > max_call_args) max_call_args = call->n_src - 1;
    int n_body_topo = 0;
    PolyUOp **body_topo = poly_toposort(ctx, call->src[0], &n_body_topo);
    ASSERT_NOT_NULL(body_topo);
    bool seen_slots[16] = {false};
    int n_storage_params = 0;
    for (int j = 0; j < n_body_topo; j++) {
      PolyUOp *u = body_topo[j];
      if (u->op != POLY_OP_PARAM || u->arg.kind != POLY_ARG_PARAM || !u->arg.param ||
          u->arg.param->slot < 0)
        continue;
      int slot = (int)u->arg.param->slot;
      ASSERT_TRUE(slot < (int)(sizeof(seen_slots) / sizeof(seen_slots[0])));
      ASSERT_FALSE(seen_slots[slot]);
      seen_slots[slot] = true;
      n_storage_params++;
    }
    ASSERT_INT_EQ(n_storage_params, call->n_src - 1);
    for (int slot = 0; slot < n_storage_params; slot++) ASSERT_TRUE(seen_slots[slot]);
  }
  ASSERT_INT_EQ(max_call_args, 5);
  ASSERT_INT_EQ(
      poly_run_linear(ctx, linear, bindings, n_bindings, NULL, 0, true, false, false), 0
  );
  free(bindings);
  free(map_repl);
  free(map_orig);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(momentum), momentum_logical);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(param), param_logical);
  ASSERT_PTR_EQ(poly_tensor_uop(momentum), m_buf);
  ASSERT_PTR_EQ(poly_tensor_uop(param), p_buf);

  float momentum_after = 0.0f;
  float param_after = 0.0f;
  ASSERT_INT_EQ(poly_buffer_read(ctx, m_buf, &momentum_after, sizeof(momentum_after)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, p_buf, &param_after, sizeof(param_after)), 0);
  ASSERT_FLOAT_EQ(momentum_after, 1.0f, 1e-6f);
  ASSERT_FLOAT_EQ(param_after, 0.963f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(optim, build_step_sgd_broadcasts_shape_one_lr_across_vector_param) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *p_buf = NULL;
  PolyUOp *g_buf = NULL;
  PolyUOp *lr_buf = NULL;
  float p_data[] = {1.0f, 2.0f};
  float g_data[] = {1.0f, 2.0f};
  float lr_data[] = {0.1f};
  PolyTensor *param = test_cpu_f32_tensor(ctx, 2, p_data, &p_buf);
  PolyTensor *grad = test_cpu_f32_tensor(ctx, 2, g_data, &g_buf);
  PolyTensor *lr = test_cpu_f32_tensor(ctx, 1, lr_data, &lr_buf);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(grad);
  ASSERT_NOT_NULL(lr);

  PolyOptimConfig cfg = {.kind = POLY_OPTIM_SGD};
  PolyTensor *outs[1] = {NULL};
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr, &param, &grad, 1, NULL, NULL, NULL, NULL, outs, 1), 1
  );
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, outs, 1, &realized), 0);

  float p_after[2] = {0.0f, 0.0f};
  ASSERT_INT_EQ(poly_buffer_read(ctx, p_buf, p_after, sizeof(p_after)), 0);
  ASSERT_FLOAT_EQ(p_after[0], 0.9f, 1e-5f);
  ASSERT_FLOAT_EQ(p_after[1], 1.8f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(optim, build_step_sgd_realizes_momentum_before_dependent_param) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *p_buf = NULL;
  PolyUOp *g_buf = NULL;
  PolyUOp *m_buf = NULL;
  PolyUOp *lr_buf = NULL;
  float p_data[] = {1.0f, 2.0f};
  float g_data[] = {0.25f, -0.5f};
  float m_data[] = {0.0f, 0.0f};
  float lr_data[] = {0.1f};
  PolyTensor *param = test_cpu_f32_tensor(ctx, 2, p_data, &p_buf);
  PolyTensor *grad = test_cpu_f32_tensor(ctx, 2, g_data, &g_buf);
  PolyTensor *momentum = test_cpu_f32_tensor(ctx, 2, m_data, &m_buf);
  PolyTensor *lr = test_cpu_f32_tensor(ctx, 1, lr_data, &lr_buf);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(grad);
  ASSERT_NOT_NULL(momentum);
  ASSERT_NOT_NULL(lr);

  PolyOptimConfig cfg = {
      .kind = POLY_OPTIM_SGD,
      .weight_decay = 0.1f,
      .momentum = 0.9f,
      .nesterov = true,
  };
  PolyTensor *m_arr[] = {momentum};
  PolyTensor *outs[2] = {NULL};
  ASSERT_INT_EQ(
      poly_optim_build_step(
          ctx, &cfg, lr, &param, &grad, 1, m_arr, NULL, NULL, NULL, outs, 2
      ),
      2
  );
  ASSERT_PTR_EQ(outs[0], momentum);
  ASSERT_PTR_EQ(outs[1], param);
  ASSERT_TRUE(poly_uop_reachable(ctx, poly_tensor_uop(param), poly_tensor_uop(momentum)));

  PolyTensor *realized[2] = {NULL};
  ASSERT_INT_EQ(poly_realize_tensors(ctx, outs, 2, realized), 0);
  float p_after[2] = {0.0f, 0.0f};
  float m_after[2] = {0.0f, 0.0f};
  ASSERT_INT_EQ(poly_buffer_read(ctx, p_buf, p_after, sizeof(p_after)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, m_buf, m_after, sizeof(m_after)), 0);
  ASSERT_FLOAT_EQ(m_after[0], 0.35f, 1e-6f);
  ASSERT_FLOAT_EQ(m_after[1], -0.3f, 1e-6f);
  ASSERT_FLOAT_EQ(p_after[0], 0.9335f, 1e-6f);
  ASSERT_FLOAT_EQ(p_after[1], 2.057f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(optim, build_step_sgd_lazy_parameter_executes_shared_momentum_effect_once) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t param_shape[2] = {2, 3};
  float g_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float lr_data[] = {0.1f};
  int f32_id = poly_dtype_id_by_name("float32");
  int64_t grad_shape[1] = {6};
  int64_t lr_shape[1] = {1};
  PolyTensor *indices =
      poly_tensor_arange_float_by_id(ctx, 0.0, 6.0, 1.0, f32_id, POLY_DEVICE_CPU);
  PolyTensor *half =
      poly_tensor_const_float_by_id(ctx, 0.5, f32_id, POLY_DEVICE_CPU);
  PolyTensor *one =
      poly_tensor_const_float_by_id(ctx, 1.0, f32_id, POLY_DEVICE_CPU);
  PolyTensor *param_values =
      poly_tensor_alu2(ctx, POLY_OP_ADD, poly_tensor_alu2(ctx, POLY_OP_MUL, indices, half), one);
  PolyTensor *param = poly_tensor_reshape(ctx, param_values, param_shape, 2);
  PolyTensor *grad_host =
      poly_tensor_from_host_by_id(ctx, g_data, sizeof(g_data), f32_id, grad_shape, 1);
  PolyTensor *grad = poly_tensor_reshape(
      ctx, poly_tensor_to_device(ctx, grad_host, POLY_DEVICE_CPU), param_shape, 2
  );
  PolyTensor *momentum_source =
      poly_tensor_full_float_by_id(
          ctx, param_shape, 2, 0.0, f32_id, POLY_DEVICE_CPU, true, false
      );
  PolyTensor *momentum =
      poly_tensor_empty(ctx, POLY_FLOAT32, param_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *lr_host =
      poly_tensor_from_host_by_id(ctx, lr_data, sizeof(lr_data), f32_id, lr_shape, 1);
  PolyTensor *lr = poly_tensor_to_device(ctx, lr_host, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(indices);
  ASSERT_NOT_NULL(half);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(param_values);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(grad_host);
  ASSERT_NOT_NULL(grad);
  ASSERT_NOT_NULL(momentum_source);
  ASSERT_NOT_NULL(momentum);
  ASSERT_NOT_NULL(lr_host);
  ASSERT_NOT_NULL(lr);
  ASSERT_FALSE(poly_uop_has_buffer_identity(poly_tensor_uop_physical(param)));

  /* Pinned _new_optim_param constructs a lazy zeros Tensor shaped like the
   * parameter (nn/optim.py:24-27). Keep the same lazy initialization effect on
   * the already-complete physical root; default realization must not place it. */
  ASSERT_PTR_EQ(poly_tensor_clone_into(ctx, momentum, momentum_source), momentum);
  PolyShape momentum_shape =
      poly_uop_max_shape_cached(ctx, poly_tensor_uop_physical(momentum));
  ASSERT_INT_EQ(momentum_shape.ndim, 2);
  ASSERT_INT_EQ(momentum_shape.dims[0], 2);
  ASSERT_INT_EQ(momentum_shape.dims[1], 3);
  ASSERT_EQ(poly_tensor_uop_physical(momentum)->op, POLY_OP_AFTER);

  PolyOptimConfig cfg = {.kind = POLY_OPTIM_SGD, .momentum = 0.9f};
  PolyTensor *m_arr[] = {momentum};
  PolyTensor *outs[2] = {NULL, NULL};
  ASSERT_INT_EQ(
      poly_optim_build_step(
          ctx, &cfg, lr, &param, &grad, 1, m_arr, NULL, NULL, NULL, outs, 2
      ),
      2
  );
  ASSERT_PTR_EQ(outs[0], momentum);
  ASSERT_PTR_EQ(outs[1], param);
  ASSERT_EQ(poly_tensor_uop_logical(momentum)->op, POLY_OP_AFTER);
  ASSERT_EQ(poly_tensor_uop_physical(momentum)->op, POLY_OP_AFTER);
  ASSERT_EQ(poly_tensor_uop_logical(param)->op, POLY_OP_AFTER);
  ASSERT_EQ(poly_tensor_uop_physical(param)->op, POLY_OP_AFTER);
  ASSERT_PTR_NEQ(poly_tensor_uop_logical(momentum), poly_tensor_uop_physical(momentum));
  ASSERT_PTR_NEQ(poly_tensor_uop_logical(param), poly_tensor_uop_physical(param));
  ASSERT_TRUE(poly_uop_reachable(
      ctx, poly_tensor_uop_logical(param), poly_tensor_uop_logical(momentum)
  ));
  ASSERT_TRUE(poly_uop_reachable(
      ctx, poly_tensor_uop_physical(param), poly_tensor_uop_physical(momentum)
  ));

  /* Pinned Optimizer.step realizes the returned update roots directly
   * (nn/optim.py:35-57). These roots are already the complete eager physical
   * graph, so scheduling them must not invoke a placement projection. */
  PolyUOp *targets[2] = {
      poly_tensor_uop_physical(outs[0]), poly_tensor_uop_physical(outs[1])
  };
  PolyUOp *resolved[2] = {NULL, NULL};
  ASSERT_NOT_NULL(targets[0]);
  ASSERT_NOT_NULL(targets[1]);
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, targets, 2, resolved, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_NOT_NULL(resolved[0]);
  ASSERT_NOT_NULL(resolved[1]);
  ASSERT_INT_EQ(linear->n_src, 5);

  const PolyUOp *m_buf = poly_uop_get_buffer_identity(resolved[0]);
  ASSERT_NOT_NULL(m_buf);
  int momentum_writes = 0;
  int momentum_read_writes = 0;
  for (int k = 0; k < linear->n_src; k++) {
    PolyUOp *call = poly_test_linear_call(linear, k);
    int n_args = poly_call_n_buffer_args(call);
    bool *call_outs = calloc((size_t)n_args, sizeof(*call_outs));
    bool *call_ins = calloc((size_t)n_args, sizeof(*call_ins));
    ASSERT_TRUE(n_args == 0 || (call_outs && call_ins));
    ASSERT_INT_EQ(poly_call_get_outs_ins(ctx, call, call_outs, call_ins, n_args), 0);
    for (int i = 0; i < n_args; i++) {
      if (poly_call_buffer_arg(call, i) == m_buf && call_outs[i]) {
        momentum_writes++;
        if (call_ins[i]) momentum_read_writes++;
      }
    }
    free(call_outs);
    free(call_ins);
  }
  /* Pinned tinygrad's exact first lazy step has one output-only state
   * initialization followed by one input/output momentum update. */
  ASSERT_INT_EQ(momentum_writes, 2);
  ASSERT_INT_EQ(momentum_read_writes, 1);

  ASSERT_INT_EQ(poly_run_linear(ctx, linear, vars, n_vars, NULL, 0, true, false, false), 0);
  free(vars);
  float m_after[6] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)m_buf, m_after, sizeof(m_after)), 0);
  const PolyUOp *param_identity = poly_uop_get_buffer_identity(resolved[1]);
  ASSERT_NOT_NULL(param_identity);
  float p_after[6] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(ctx, (PolyUOp *)param_identity, p_after, sizeof(p_after)), 0
  );
  for (int i = 0; i < 6; i++) {
    ASSERT_FLOAT_EQ(m_after[i], (float)(i + 1), 1e-6f);
    ASSERT_FLOAT_EQ(p_after[i], 1.0f + 0.5f * (float)i - 0.1f * (float)(i + 1), 1e-6f);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(optim, build_step_adam_updates_beta_power_state_in_graph) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *p_buf = NULL;
  PolyUOp *g_buf = NULL;
  PolyUOp *m_buf = NULL;
  PolyUOp *v_buf = NULL;
  PolyUOp *bc1_buf = NULL;
  PolyUOp *bc2_buf = NULL;
  PolyUOp *lr_buf = NULL;
  float p_data[] = {1.0f}, g_data[] = {1.0f}, m_data[] = {0.0f}, v_data[] = {0.0f};
  float bc1_data[] = {1.0f}, bc2_data[] = {1.0f}, lr_data[] = {0.1f};
  PolyTensor *param = test_cpu_f32_tensor(ctx, 1, p_data, &p_buf);
  PolyTensor *grad = test_cpu_f32_tensor(ctx, 1, g_data, &g_buf);
  PolyTensor *m = test_cpu_f32_tensor(ctx, 1, m_data, &m_buf);
  PolyTensor *v = test_cpu_f32_tensor(ctx, 1, v_data, &v_buf);
  PolyTensor *bc1 = test_cpu_f32_tensor(ctx, 1, bc1_data, &bc1_buf);
  PolyTensor *bc2 = test_cpu_f32_tensor(ctx, 1, bc2_data, &bc2_buf);
  PolyTensor *lr = test_cpu_f32_tensor(ctx, 1, lr_data, &lr_buf);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(grad);
  ASSERT_NOT_NULL(m);
  ASSERT_NOT_NULL(v);
  ASSERT_NOT_NULL(bc1);
  ASSERT_NOT_NULL(bc2);
  ASSERT_NOT_NULL(lr);

  PolyOptimConfig cfg = {
      .kind = POLY_OPTIM_ADAM,
      .beta1 = 0.9f,
      .beta2 = 0.999f,
      .eps = 1e-8f,
      .weight_decay = 0.0f,
      .momentum = 0.0f,
      .nesterov = false,
      .classic = false,
  };
  PolyTensor *m_arr[] = {m};
  PolyTensor *v_arr[] = {v};
  int need =
      poly_optim_build_step(ctx, &cfg, lr, &param, &grad, 1, m_arr, v_arr, bc1, bc2, NULL, 0);
  ASSERT_INT_EQ(need, 5);
  PolyTensor *outs[5] = {0};
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr, &param, &grad, 1, m_arr, v_arr, bc1, bc2, outs, 5), 5
  );
  ASSERT_PTR_EQ(outs[0], bc1);
  ASSERT_PTR_EQ(outs[1], bc2);
  ASSERT_PTR_EQ(outs[2], m);
  ASSERT_PTR_EQ(outs[3], v);
  ASSERT_PTR_EQ(outs[4], param);
  ASSERT_TRUE(poly_uop_reachable(ctx, poly_tensor_uop(param), lr_buf));
  ASSERT_TRUE(poly_uop_reachable(ctx, poly_tensor_uop(param), poly_tensor_uop(m)));
  ASSERT_TRUE(poly_uop_reachable(ctx, poly_tensor_uop(param), poly_tensor_uop(v)));
  ASSERT_TRUE(poly_uop_reachable(ctx, poly_tensor_uop(param), poly_tensor_uop(bc1)));
  ASSERT_TRUE(poly_uop_reachable(ctx, poly_tensor_uop(param), poly_tensor_uop(bc2)));
  PolyTensor *realized[5] = {0};
  ASSERT_INT_EQ(poly_realize_tensors(ctx, outs, 5, realized), 0);

  float p_after = 0.0f, m_after = 0.0f, v_after = 0.0f, bc1_after = 0.0f, bc2_after = 0.0f;
  ASSERT_INT_EQ(poly_buffer_read(ctx, p_buf, &p_after, sizeof(p_after)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, m_buf, &m_after, sizeof(m_after)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, v_buf, &v_after, sizeof(v_after)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, bc1_buf, &bc1_after, sizeof(bc1_after)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, bc2_buf, &bc2_after, sizeof(bc2_after)), 0);
  ASSERT_FLOAT_EQ(p_after, 0.9f, 1e-4f);
  ASSERT_FLOAT_EQ(m_after, 0.1f, 1e-5f);
  ASSERT_FLOAT_EQ(v_after, 0.001f, 1e-6f);
  ASSERT_FLOAT_EQ(bc1_after, 0.9f, 1e-6f);
  ASSERT_FLOAT_EQ(bc2_after, 0.999f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(optim, build_update_adamw_depends_on_lr_uop) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *lr = poly_buffer_f32(ctx, 1);
  PolyUOp *param = poly_buffer_f32(ctx, 1);
  PolyUOp *grad = poly_buffer_f32(ctx, 1);
  PolyUOp *m = poly_buffer_f32(ctx, 1);
  PolyUOp *v = poly_buffer_f32(ctx, 1);
  PolyUOp *bc1 = poly_buffer_f32(ctx, 1);
  PolyUOp *bc2 = poly_buffer_f32(ctx, 1);
  ASSERT_NOT_NULL(lr);

  PolyOptimConfig cfg = {
      .kind = POLY_OPTIM_ADAMW,
      .beta1 = 0.9f,
      .beta2 = 0.999f,
      .eps = 1e-8f,
      .weight_decay = 0.01f,
  };
  PolyOptimUpdate upd;
  ASSERT_INT_EQ(poly_optim_build_update(ctx, &cfg, lr, param, grad, m, v, bc1, bc2, 1, &upd), 0);
  ASSERT_NOT_NULL(upd.param_new);
  ASSERT_TRUE(poly_uop_reachable(ctx, upd.param_new, lr));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(optim, build_step_late_failure_leaves_all_params_and_state_unchanged) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0_buf = NULL;
  PolyUOp *p1_buf = NULL;
  PolyUOp *g0_buf = NULL;
  PolyUOp *g1_buf = NULL;
  PolyUOp *m0_buf = NULL;
  PolyUOp *m1_buf = NULL;
  PolyUOp *lr_buf = NULL;
  float p0_data[] = {1.0f}, p1_data[] = {2.0f};
  float g0_data[] = {1.0f}, g1_data[] = {1.0f};
  float m0_data[] = {0.3f}, m1_data[] = {0.4f}, lr_data[] = {0.1f};
  PolyTensor *p0 = test_cpu_f32_tensor(ctx, 1, p0_data, &p0_buf);
  PolyTensor *p1 = test_cpu_f32_tensor(ctx, 1, p1_data, &p1_buf);
  PolyTensor *g0 = test_cpu_f32_tensor(ctx, 1, g0_data, &g0_buf);
  PolyTensor *g1 = test_cpu_f32_tensor(ctx, 1, g1_data, &g1_buf);
  PolyTensor *m0 = test_cpu_f32_tensor(ctx, 1, m0_data, &m0_buf);
  PolyTensor *m1 = test_cpu_f32_tensor(ctx, 1, m1_data, &m1_buf);
  PolyTensor *lr = test_cpu_f32_tensor(ctx, 1, lr_data, &lr_buf);
  ASSERT_NOT_NULL(p0);
  ASSERT_NOT_NULL(p1);
  ASSERT_NOT_NULL(g0);
  ASSERT_NOT_NULL(g1);
  ASSERT_NOT_NULL(m0);
  ASSERT_NOT_NULL(m1);
  ASSERT_NOT_NULL(lr);
  ASSERT_INT_EQ(
      poly_tensor_replace_roots(
          ctx, p1, poly_tensor_uop_logical(p1), poly_tensor_uop_physical(p1),
          POLY_TENSOR_VALUE, POLY_DEVICE_CUDA
      ),
      0
  );
  ASSERT_INT_EQ(
      poly_tensor_replace_roots(
          ctx, g1, poly_tensor_uop_logical(g1), poly_tensor_uop_physical(g1),
          POLY_TENSOR_VALUE, POLY_DEVICE_CUDA
      ),
      0
  );
  ASSERT_INT_EQ(
      poly_tensor_replace_roots(
          ctx, m1, poly_tensor_uop_logical(m1), poly_tensor_uop_physical(m1),
          POLY_TENSOR_VALUE, POLY_DEVICE_CUDA
      ),
      0
  );

  PolyUOp *p0_root = poly_tensor_uop(p0);
  PolyUOp *p1_root = poly_tensor_uop(p1);
  PolyUOp *m0_root = poly_tensor_uop(m0);
  PolyUOp *m1_root = poly_tensor_uop(m1);
  PolyTensor *params[] = {p0, p1};
  PolyTensor *grads[] = {g0, g1};
  PolyTensor *momenta[] = {m0, m1};
  PolyTensor *outs[4] = {NULL};
  PolyOptimConfig cfg = {.kind = POLY_OPTIM_SGD, .momentum = 0.9f};
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr, params, grads, 2, momenta, NULL, NULL, NULL, outs, 4), -1
  );

  ASSERT_PTR_EQ(poly_tensor_uop(p0), p0_root);
  ASSERT_PTR_EQ(poly_tensor_uop(p1), p1_root);
  ASSERT_PTR_EQ(poly_tensor_uop(m0), m0_root);
  ASSERT_PTR_EQ(poly_tensor_uop(m1), m1_root);
  for (int i = 0; i < 4; i++)
    ASSERT_PTR_EQ(outs[i], NULL);

  float p0_after = 0.0f, p1_after = 0.0f, m0_after = 0.0f, m1_after = 0.0f;
  ASSERT_INT_EQ(poly_buffer_read(ctx, p0_buf, &p0_after, sizeof(p0_after)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, p1_buf, &p1_after, sizeof(p1_after)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, m0_buf, &m0_after, sizeof(m0_after)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, m1_buf, &m1_after, sizeof(m1_after)), 0);
  ASSERT_FLOAT_EQ(p0_after, 1.0f, 1e-6f);
  ASSERT_FLOAT_EQ(p1_after, 2.0f, 1e-6f);
  ASSERT_FLOAT_EQ(m0_after, 0.3f, 1e-6f);
  ASSERT_FLOAT_EQ(m1_after, 0.4f, 1e-6f);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &p0, 1, &realized), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, p0_buf, &p0_after, sizeof(p0_after)), 0);
  ASSERT_FLOAT_EQ(p0_after, 1.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(optim, build_step_rejects_invalid_lr_tensor_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyCtx *other_ctx = poly_ctx_new();
  int64_t scalar_shape[] = {1};
  int64_t vector_shape[] = {2};
  PolyTensor *param =
      poly_tensor_empty(ctx, POLY_FLOAT32, scalar_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *grad =
      poly_tensor_empty(ctx, POLY_FLOAT32, scalar_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *lr_int =
      poly_tensor_empty(ctx, POLY_INT32, scalar_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *lr_vec =
      poly_tensor_empty(ctx, POLY_FLOAT32, vector_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *lr_f16 =
      poly_tensor_empty(ctx, POLY_FLOAT16, scalar_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *lr_bf16 =
      poly_tensor_empty(ctx, POLY_BFLOAT16, scalar_shape, 1, POLY_DEVICE_CPU);
  PolyUOp *n = poly_uop_variable(ctx, "optim_lr_n", 0, 1, POLY_WEAKINT, 1, false);
  PolyUOp *symbolic_dims[] = {n};
  PolyUOp *lr_symbolic_uop =
      poly_expand_uop(ctx, poly_buffer_f32(ctx, 1), symbolic_dims, 1);
  PolyUOp *lr_symbolic_physical = poly_expand_uop(
      ctx, poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU), symbolic_dims, 1
  );
  PolyTensor *lr_symbolic = poly_tensor_create_with_roots(
      ctx, lr_symbolic_uop, lr_symbolic_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  PolyTensor *lr_scalar = poly_tensor_const_float_by_id(
      ctx, 0.1, poly_dtype_id_by_name("float32"), POLY_DEVICE_CPU
  );
  PolyTensor *lr_other =
      poly_tensor_empty(other_ctx, POLY_FLOAT32, scalar_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *lr_cuda =
      poly_tensor_empty(ctx, POLY_FLOAT32, scalar_shape, 1, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(grad);
  ASSERT_NOT_NULL(lr_int);
  ASSERT_NOT_NULL(lr_vec);
  ASSERT_NOT_NULL(lr_f16);
  ASSERT_NOT_NULL(lr_bf16);
  ASSERT_NOT_NULL(lr_symbolic);
  ASSERT_NOT_NULL(lr_scalar);
  ASSERT_NOT_NULL(lr_other);
  ASSERT_NOT_NULL(lr_cuda);
  ASSERT_NOT_NULL(lr_symbolic_physical);
  ASSERT_NOT_NULL(poly_uop_shape_dim(ctx, lr_symbolic_uop, 0));

  PolyOptimConfig cfg = {.kind = POLY_OPTIM_SGD};
  PolyTensor *out[1] = {NULL};
  ASSERT_INT_EQ(
      poly_optim_build_step(
          ctx, &cfg, lr_scalar, &param, &grad, 1, NULL, NULL, NULL, NULL, NULL, 0
      ),
      1
  );
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr_int, &param, &grad, 1, NULL, NULL, NULL, NULL, out, 1), -1
  );
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr_vec, &param, &grad, 1, NULL, NULL, NULL, NULL, out, 1), -1
  );
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr_f16, &param, &grad, 1, NULL, NULL, NULL, NULL, out, 1), -1
  );
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr_bf16, &param, &grad, 1, NULL, NULL, NULL, NULL, out, 1),
      -1
  );
  ASSERT_INT_EQ(
      poly_optim_build_step(
          ctx, &cfg, lr_symbolic, &param, &grad, 1, NULL, NULL, NULL, NULL, out, 1
      ),
      -1
  );
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr_other, &param, &grad, 1, NULL, NULL, NULL, NULL, out, 1),
      -1
  );
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr_cuda, &param, &grad, 1, NULL, NULL, NULL, NULL, out, 1),
      -1
  );

  poly_ctx_destroy(other_ctx);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Helper: build IR bytes for a trainable model */
/* out = w * x, loss = sum((out - y)^2) / N */
static uint8_t *make_train_ir(int n, int *out_len) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *w = instance_logical_f32(ctx, n);
  PolyUOp *x = instance_logical_f32(ctx, n);
  PolyUOp *y = instance_logical_f32(ctx, n);
  PolyUOp *out_buf = instance_logical_f32(ctx, n);
  PolyUOp *loss_buf = instance_logical_f32(ctx, 1);

  /* Forward: out = w * x */
  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, w, x);
  PolyUOp *fwd_store = poly_store_val(ctx, out_buf, prod);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);

  /* Loss: sum((prod - y)^2) */
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_ADD, prod, poly_alu1(ctx, POLY_OP_NEG, y));
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
  int64_t axes[] = {0};
  PolyUOp *loss_val = poly_reduce_axis(ctx, POLY_OP_ADD, sq, axes, 1);

  /* Scale by 1/N */
  PolyUOp *scale = poly_const_float(ctx, 1.0 / (double)n);
  PolyUOp *mse = poly_alu2(ctx, POLY_OP_MUL, loss_val, scale);

  PolyUOp *loss_store = poly_store_val(ctx, loss_buf, mse);
  PolyUOp *loss_sink = poly_sink1(ctx, loss_store);

  PolyIrBufEntry bufs[] = {
      {.name = "w", .role = POLY_IR_ROLE_PARAM, .buffer = w, .shape = {n}, .ndim = 1},
      {.name = "x", .role = POLY_IR_ROLE_INPUT, .buffer = x, .shape = {n}, .ndim = 1},
      {.name = "y", .role = POLY_IR_ROLE_TARGET, .buffer = y, .shape = {n}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out_buf, .shape = {n}, .ndim = 1},
      {.name = "loss", .role = POLY_IR_ROLE_OUTPUT, .buffer = loss_buf, .shape = {1}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {
      {.name = "forward", .sink = fwd_sink},
      {.name = "loss", .sink = loss_sink},
  };
  PolyIrSpec spec = {ctx, bufs, 5, eps, 2, NULL, 0};

  uint8_t *bytes = poly_ir_export(&spec, out_len);
  poly_ctx_destroy(ctx);
  return bytes;
}

/* Tests */

TEST(instance, create_from_ir) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  ASSERT_NOT_NULL(ir);

  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  ASSERT_INT_EQ(poly_instance_buf_count(inst), 3);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 0);

  ASSERT_STR_EQ(poly_instance_buf_name(inst, 0), "a");
  ASSERT_INT_EQ(poly_instance_buf_role(inst, 0), POLY_ROLE_INPUT);
  ASSERT_STR_EQ(poly_instance_buf_name(inst, 1), "b");
  ASSERT_STR_EQ(poly_instance_buf_name(inst, 2), "output");
  ASSERT_INT_EQ(poly_instance_buf_role(inst, 2), POLY_ROLE_OUTPUT);

  int64_t shape[8];
  int ndim = poly_instance_buf_shape(inst, 2, shape, 8);
  ASSERT_INT_EQ(ndim, 1);
  ASSERT_INT_EQ((int)shape[0], 4);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, forward_add) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyIOBinding inputs[] = {
      POLY_IO_BINDING_ARRAY("a", a_data, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("b", b_data, POLY_FLOAT32),
  };

  int ret = poly_instance_forward(inst, inputs, 2);
  ASSERT_INT_EQ(ret, 0);

  /* Read output */
  int64_t numel;
  float *out = poly_instance_buf_data(inst, 2, &numel);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ((int)numel, 4);
  ASSERT_TRUE(fabsf(out[0] - 11.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[1] - 22.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[2] - 33.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[3] - 44.0f) < 1e-5f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, read_write_buf_named_forward_e2e) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
  ASSERT_INT_EQ(poly_instance_write_buf_named(inst, "a", a_data, sizeof(a_data)), 0);
  ASSERT_INT_EQ(poly_instance_write_buf_named(inst, "b", b_data, sizeof(b_data)), 0);
  ASSERT_INT_EQ(poly_instance_write_buf_named(inst, "missing", b_data, sizeof(b_data)), -1);

  ASSERT_INT_EQ(poly_instance_forward(inst, NULL, 0), 0);

  float out[4] = {0};
  ASSERT_INT_EQ(poly_instance_read_buf_named(inst, "output", out, sizeof(out)), 0);
  ASSERT_FLOAT_EQ(out[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[3], 44.0f, 1e-5f);
  ASSERT_INT_EQ(poly_instance_read_buf_named(inst, "missing", out, sizeof(out)), -1);

  float a2[] = {5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_INT_EQ(poly_instance_write_buf_named(inst, "a", a2, sizeof(a2)), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, NULL, 0), 0);

  int out_idx = -1;
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    if (strcmp(poly_instance_buf_name(inst, i), "output") == 0) out_idx = i;
  ASSERT_TRUE(out_idx >= 0);
  memset(out, 0, sizeof(out));
  ASSERT_INT_EQ(poly_instance_read_buf(inst, out_idx, out, sizeof(out)), 0);
  ASSERT_FLOAT_EQ(out[0], 15.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[1], 26.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[2], 37.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[3], 48.0f, 1e-5f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, forward_rebuilds_runtime_cache_after_clear) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  float a1[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b1[] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyIOBinding io1[] = {POLY_IO_BINDING_ARRAY("a", a1, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("b", b1, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io1, 2), 0);

  ASSERT_TRUE(poly_runtime_cache_len(poly_instance_ctx(inst)) > 0);
  poly_runtime_cache_clear(poly_instance_ctx(inst));
  ASSERT_INT_EQ((int)poly_runtime_cache_len(poly_instance_ctx(inst)), 0);

  float a2[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float b2[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyIOBinding io2[] = {POLY_IO_BINDING_ARRAY("a", a2, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("b", b2, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io2, 2), 0);
  ASSERT_TRUE(poly_runtime_cache_len(poly_instance_ctx(inst)) > 0);

  int64_t numel = 0;
  float *out = poly_instance_buf_data(inst, 2, &numel);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ((int)numel, 4);
  ASSERT_FLOAT_EQ(out[0], 6.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[1], 8.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[2], 10.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[3], 12.0f, 1e-5f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, staged_build_forward_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_BUILDING);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);

  /* Pinned Tensor ALU constructs from current Tensor.uop operands. Use the C
   * Tensor boundary so Instance retains both the portable logical graph and
   * the eager physical execution graph before realization. */
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, w);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_BUILT);
  ASSERT_INT_EQ(poly_instance_buf_count(inst), 3);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);

  int64_t numel = 0;
  float *w_data = poly_instance_param_data(inst, 0, &numel);
  ASSERT_NOT_NULL(w_data);
  ASSERT_INT_EQ((int)numel, 4);
  w_data[0] = 10.0f;
  w_data[1] = 20.0f;
  w_data[2] = 30.0f;
  w_data[3] = 40.0f;

  float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);

  float *y = poly_instance_buf_data_named(inst, "output", &numel);
  ASSERT_NOT_NULL(y);
  ASSERT_INT_EQ((int)numel, 4);
  ASSERT_FLOAT_EQ(y[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y[3], 44.0f, 1e-5f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_logical_only_output_places_and_runs) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(ctx);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);
  PolyUOp *logical = poly_alu2(
      ctx, POLY_OP_ADD, poly_tensor_uop_logical(x), poly_tensor_uop_logical(w)
  );
  PolyTensor *out = poly_tensor_create_with_roots(
      ctx, logical, NULL, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(out);
  ASSERT_EQ(poly_tensor_uop_physical(out), NULL);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );

  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_BUILT);

  int64_t n = 0;
  float *w_data = poly_instance_param_data(inst, 0, &n);
  ASSERT_NOT_NULL(w_data);
  ASSERT_INT_EQ((int)n, 4);
  for (int i = 0; i < 4; i++) w_data[i] = (float)(10 * (i + 1));
  float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  float *result = poly_instance_buf_data_named(inst, "output", &n);
  ASSERT_NOT_NULL(result);
  ASSERT_FLOAT_EQ(result[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(result[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(result[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(result[3], 44.0f, 1e-5f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, typed_io_preserves_integer_bytes_and_rejects_partial_updates) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {3};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_INT32, shape, 1);
  PolyTensor *y = poly_instance_input(inst, "y", POLY_INT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(y);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, x, y);
  ASSERT_NOT_NULL(sum);
  PolyTensor *out = poly_tensor_cast_by_id(ctx, sum, poly_dtype_id_by_name("float32"));
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x", "y"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 2, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  int32_t x_data[] = {0, 1, 2};
  int32_t y_data[] = {3, 4, 5};
  PolyIOBinding valid[] = {
      POLY_IO_BINDING_ARRAY("x", x_data, POLY_INT32),
      POLY_IO_BINDING_ARRAY("y", y_data, POLY_INT32),
  };
  ASSERT_INT_EQ(poly_instance_forward(inst, valid, 2), 0);

  int32_t stored_x[3] = {0};
  int32_t stored_y[3] = {0};
  ASSERT_INT_EQ(poly_instance_read_buf_named(inst, "x", stored_x, sizeof(stored_x)), 0);
  ASSERT_INT_EQ(poly_instance_read_buf_named(inst, "y", stored_y, sizeof(stored_y)), 0);
  ASSERT_INT_EQ(stored_x[0], 0);
  ASSERT_INT_EQ(stored_x[1], 1);
  ASSERT_INT_EQ(stored_x[2], 2);
  ASSERT_INT_EQ(stored_y[0], 3);
  ASSERT_INT_EQ(stored_y[1], 4);
  ASSERT_INT_EQ(stored_y[2], 5);

  int64_t n = 0;
  float *out_data = poly_instance_buf_data_named(inst, "output", &n);
  ASSERT_NOT_NULL(out_data);
  ASSERT_INT_EQ((int)n, 3);
  ASSERT_FLOAT_EQ(out_data[0], 3.0f, 0.0f);
  ASSERT_FLOAT_EQ(out_data[1], 5.0f, 0.0f);
  ASSERT_FLOAT_EQ(out_data[2], 7.0f, 0.0f);

  /* Pinned TinyJit rejects a replay whose complete input signature differs
   * (tinygrad/engine/jit.py:303-309). Missing and duplicate rows must fail
   * before either input buffer is changed. */
  int32_t replacement_x[] = {9, 9, 9};
  PolyIOBinding missing_y[] = {
      POLY_IO_BINDING_ARRAY("x", replacement_x, POLY_INT32),
  };
  ASSERT_TRUE(poly_instance_forward(inst, missing_y, 1) < 0);
  ASSERT_INT_EQ(poly_instance_read_buf_named(inst, "x", stored_x, sizeof(stored_x)), 0);
  ASSERT_INT_EQ(stored_x[0], 0);

  PolyIOBinding duplicate_x[] = {
      POLY_IO_BINDING_ARRAY("x", replacement_x, POLY_INT32),
      POLY_IO_BINDING_ARRAY("x", replacement_x, POLY_INT32),
      POLY_IO_BINDING_ARRAY("y", y_data, POLY_INT32),
  };
  ASSERT_TRUE(poly_instance_forward(inst, duplicate_x, 3) < 0);
  ASSERT_INT_EQ(poly_instance_read_buf_named(inst, "x", stored_x, sizeof(stored_x)), 0);
  ASSERT_INT_EQ(stored_x[0], 0);

  float wrong_y[] = {1.0f, 2.0f, 3.0f};
  PolyIOBinding wrong_dtype[] = {
      POLY_IO_BINDING_ARRAY("x", replacement_x, POLY_INT32),
      POLY_IO_BINDING_ARRAY("y", wrong_y, POLY_FLOAT32),
  };
  ASSERT_TRUE(poly_instance_forward(inst, wrong_dtype, 2) < 0);
  ASSERT_INT_EQ(poly_instance_read_buf_named(inst, "x", stored_x, sizeof(stored_x)), 0);
  ASSERT_INT_EQ(stored_x[0], 0);
  ASSERT_INT_EQ(stored_x[1], 1);
  ASSERT_INT_EQ(stored_x[2], 2);

  PolyIOBinding wrong_length =
      POLY_IO_BINDING_BYTES("x", replacement_x, 2 * sizeof(int32_t), POLY_INT32);
  ASSERT_TRUE(poly_instance_forward(inst, &wrong_length, 1) < 0);
  ASSERT_INT_EQ(poly_instance_read_buf_named(inst, "x", stored_x, sizeof(stored_x)), 0);
  ASSERT_INT_EQ(stored_x[0], 0);
  ASSERT_INT_EQ(stored_x[1], 1);
  ASSERT_INT_EQ(stored_x[2], 2);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, existing_bindings_retain_physical_entrypoint_and_portable_ir) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  int64_t shape[] = {4};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *w = poly_tensor_empty(
      ctx, POLY_FLOAT32, shape, 1, poly_ctx_get_preferred_device(ctx)
  );
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, w);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);
  ASSERT_NOT_NULL(out);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "w", .role = POLY_ROLE_PARAM, .tensor = w},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entries[] = {{
      .name = "forward",
      .inputs = inputs,
      .n_inputs = 1,
      .outputs = outputs,
      .n_outputs = 1,
  }};
  PolyInstance *inst = poly_instance_from_bindings(ctx, bindings, 3, entries, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);

  PolyUOp *physical_sink = poly_instance_get_sink(inst, "forward");
  ASSERT_NOT_NULL(physical_sink);
  ASSERT_INT_EQ(physical_sink->op, POLY_OP_SINK);
  ASSERT_INT_EQ(physical_sink->n_src, 1);
  PolyUOp *physical_store = physical_sink->src[0];
  ASSERT_INT_EQ(physical_store->op, POLY_OP_STORE);
  ASSERT_INT_EQ(physical_store->n_src, 2);
  ASSERT_INT_EQ(poly_uop_device(physical_store->src[0]), POLY_DEVICE_CPU);
  PolyUOp *physical_add = physical_store->src[1];
  ASSERT_INT_EQ(physical_add->op, POLY_OP_ADD);
  ASSERT_INT_EQ(poly_uop_device(physical_add->src[0]), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(poly_uop_device(physical_add->src[1]), POLY_DEVICE_CPU);
  ASSERT_PTR_EQ(physical_add->src[0], poly_uop_get_buffer_identity(poly_tensor_uop_physical(x)));
  ASSERT_PTR_EQ(physical_add->src[1], poly_uop_get_buffer_identity(poly_tensor_uop_physical(w)));

  int ir_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  ASSERT_NOT_NULL(ir);
  PolyIrSpec imported = {0};
  ASSERT_INT_EQ(poly_ir_import(ir, ir_len, &imported), 0);
  ASSERT_INT_EQ(imported.n_entrypoints, 1);
  PolyUOp *logical_sink = imported.entrypoints[0].sink;
  ASSERT_INT_EQ(logical_sink->op, POLY_OP_SINK);
  PolyUOp *logical_store = logical_sink->src[0];
  ASSERT_INT_EQ(logical_store->op, POLY_OP_STORE);
  ASSERT_INT_EQ(poly_uop_device(logical_store->src[0]), POLY_DEVICE_AUTO);
  PolyUOp *logical_add = logical_store->src[1];
  ASSERT_INT_EQ(logical_add->op, POLY_OP_ADD);
  ASSERT_INT_EQ(poly_uop_device(logical_add->src[0]), POLY_DEVICE_AUTO);
  ASSERT_INT_EQ(poly_uop_device(logical_add->src[1]), POLY_DEVICE_AUTO);

  int64_t numel = 0;
  float *w_data = poly_instance_param_data(inst, 0, &numel);
  ASSERT_NOT_NULL(w_data);
  ASSERT_INT_EQ((int)numel, 4);
  w_data[0] = 10.0f;
  w_data[1] = 20.0f;
  w_data[2] = 30.0f;
  w_data[3] = 40.0f;
  float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  float *result = poly_instance_buf_data_named(inst, "output", &numel);
  ASSERT_NOT_NULL(result);
  ASSERT_FLOAT_EQ(result[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(result[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(result[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(result[3], 44.0f, 1e-5f);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, owns_physical_residency_after_source_tensors_retire) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  int64_t shape[] = {4};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *w = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, w);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);
  ASSERT_NOT_NULL(out);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "w", .role = POLY_ROLE_PARAM, .tensor = w},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entries[] = {{
      .name = "forward",
      .inputs = inputs,
      .n_inputs = 1,
      .outputs = outputs,
      .n_outputs = 1,
  }};
  PolyInstance *inst = poly_instance_from_bindings(ctx, bindings, 3, entries, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);
  ASSERT_TRUE(ctx->mem_used > 0);

  poly_tensor_release(out);
  poly_tensor_release(w);
  poly_tensor_release(x);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_TRUE(ctx->mem_used > 0);

  float *w_data = poly_instance_param_data(inst, 0, NULL);
  ASSERT_NOT_NULL(w_data);
  for (int i = 0; i < 4; i++) w_data[i] = (float)(10 * (i + 1));
  float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  float *result = poly_instance_buf_data_named(inst, "output", NULL);
  ASSERT_NOT_NULL(result);
  ASSERT_FLOAT_EQ(result[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(result[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(result[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(result[3], 44.0f, 1e-5f);

  poly_instance_free(inst);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(ctx->mem_used, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_build_validation_rewinds_scratch_on_success) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);

  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, w);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );

  size_t scratch_before = poly_arena_used(ctx->scratch);
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_stage_guards) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_forward(inst, NULL, 0), -1);

  int64_t shape[] = {1};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", x), POLY_STATUS_OK);
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_TRUE(poly_instance_param(inst, "late", POLY_FLOAT32, shape, 1) == NULL);
  const PolyInstanceError *err = poly_instance_last_error(inst);
  ASSERT_NOT_NULL(err);
  ASSERT_INT_EQ(err->code, POLY_STATUS_BAD_STAGE);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_build_validation_rewinds_scratch_on_unbound_storage) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);

  PolyTensor *w =
      poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, test_ctx_execution_device(ctx));
  ASSERT_NOT_NULL(w);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, w);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );

  size_t scratch_before = poly_arena_used(ctx->scratch);
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_INVALID);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_build_rejects_unbound_storage_leaf) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);

  PolyTensor *w =
      poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, test_ctx_execution_device(ctx));
  ASSERT_NOT_NULL(w);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, w);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_INVALID);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_FAILED);
  const PolyInstanceError *err = poly_instance_last_error(inst);
  ASSERT_NOT_NULL(err);
  ASSERT_TRUE(strstr(err->message, "unbound storage") != NULL);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_bindings_set_tensor_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  PolyTensor *y = poly_instance_target(inst, "y", POLY_FLOAT32, shape, 1);
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(y);
  ASSERT_NOT_NULL(w);

  ASSERT_TRUE(!poly_tensor_requires_grad(x));
  ASSERT_TRUE(!poly_tensor_requires_grad(y));
  ASSERT_TRUE(poly_tensor_requires_grad(w));
  ASSERT_INT_EQ(poly_tensor_provenance(x), POLY_TENSOR_PROVENANCE_USER_INPUT);
  ASSERT_INT_EQ(poly_tensor_provenance(y), POLY_TENSOR_PROVENANCE_USER_INPUT);
  ASSERT_INT_EQ(poly_tensor_provenance(w), POLY_TENSOR_PROVENANCE_PARAM_INIT);

  PolyTensor *bindings[] = {x, y, w};
  PolyDevice expected_device = poly_ctx_get_preferred_device(ctx);
  if (!poly_device_can_execute(expected_device)) expected_device = poly_device_default();
  for (int i = 0; i < 3; i++) {
    PolyUOp *logical = (PolyUOp *)poly_uop_get_buffer_identity(
        poly_tensor_uop_logical(bindings[i])
    );
    PolyUOp *physical = (PolyUOp *)poly_uop_get_buffer_identity(
        poly_tensor_uop_physical(bindings[i])
    );
    ASSERT_NOT_NULL(logical);
    ASSERT_NOT_NULL(physical);
    ASSERT_PTR_NEQ(logical, physical);
    ASSERT_INT_EQ(logical->n_src, 1);
    ASSERT_INT_EQ(physical->n_src, 1);
    ASSERT_INT_EQ(physical->src[0]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(physical->arg.kind, POLY_ARG_PARAM);
    ASSERT_INT_EQ(logical->src[0]->arg.i, physical->arg.param->slot);
    ASSERT_INT_EQ(poly_uop_device(physical), expected_device);
  }

  PolyTensor *w_cuda = poly_tensor_to_device(ctx, w, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(w_cuda);
  ASSERT_TRUE(poly_tensor_requires_grad(w_cuda));
  ASSERT_INT_EQ(poly_tensor_provenance(w_cuda), POLY_TENSOR_PROVENANCE_PARAM_INIT);

  PolyTensor *state = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(state);
  ASSERT_INT_EQ(poly_instance_state(inst, "loaded", state, 0), POLY_STATUS_OK);
  ASSERT_TRUE(poly_tensor_requires_grad(state));
  ASSERT_INT_EQ(poly_tensor_provenance(state), POLY_TENSOR_PROVENANCE_STATE_LOADED);

  PolyTensor *aux = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(aux);
  ASSERT_INT_EQ(poly_instance_aux(inst, "aux", aux, 0), POLY_STATUS_OK);
  ASSERT_TRUE(!poly_tensor_requires_grad(aux));
  ASSERT_INT_EQ(poly_tensor_provenance(aux), POLY_TENSOR_PROVENANCE_STATE_LOADED);

  ASSERT_INT_EQ(poly_instance_output(inst, "echo", x), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_tensor_provenance(x), POLY_TENSOR_PROVENANCE_USER_INPUT);

  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, w);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_tensor_provenance(out), POLY_TENSOR_PROVENANCE_COMPUTED);
  ASSERT_INT_EQ(poly_instance_output(inst, "computed", out), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_tensor_provenance(out), POLY_TENSOR_PROVENANCE_COMPUTED);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_build_rejects_unbound_trainable_storage_leaf) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t x_shape[] = {2, 2};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, x_shape, 2);
  ASSERT_NOT_NULL(x);

  int64_t w_shape[] = {2, 2};
  PolyDevice device = test_ctx_execution_device(ctx);
  PolyTensor *w = poly_tensor_empty(ctx, POLY_FLOAT32, w_shape, 2, device);
  ASSERT_NOT_NULL(w);
  poly_tensor_set_requires_grad(w, true);
  poly_tensor_set_provenance(w, POLY_TENSOR_PROVENANCE_PARAM_INIT);

  PolyTensor *w_alias = poly_tensor_create_with_roots(
      ctx, poly_tensor_uop_logical(w), poly_tensor_uop_physical(w),
      POLY_TENSOR_VALUE, device
  );
  ASSERT_NOT_NULL(w_alias);

  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, w_alias);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_INVALID);
  const PolyInstanceError *err = poly_instance_last_error(inst);
  ASSERT_NOT_NULL(err);
  ASSERT_TRUE(strstr(err->message, "unbound trainable storage") != NULL);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_runtime_trainability_uses_tensor_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t x_shape[] = {1, 1};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, x_shape, 2);
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT32, x_shape, 2);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);
  ASSERT_TRUE(poly_tensor_requires_grad(w));

  poly_tensor_set_requires_grad(w, false);

  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_MUL, x, w);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_TRUE(!poly_instance_param_trainable(inst, 0));

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_objective_accepts_one_element_output) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {1};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_INT_EQ(poly_instance_output(inst, "loss", x), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"loss"};
  PolyEntrypointOptions opts = {.objective = "loss"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "loss", inputs, 1, outputs, 1, &opts), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_objective_round_trips_through_ir) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t x_shape[] = {1};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, x_shape, 1);
  ASSERT_NOT_NULL(x);
  PolyTensor *y = poly_instance_target(inst, "y", POLY_FLOAT32, x_shape, 1);
  ASSERT_NOT_NULL(y);

  ASSERT_INT_EQ(poly_instance_output(inst, "logits", x), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_output(inst, "loss", y), POLY_STATUS_OK);

  const char *forward_inputs[] = {"x"};
  const char *forward_outputs[] = {"logits"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", forward_inputs, 1, forward_outputs, 1, NULL),
      POLY_STATUS_OK
  );

  const char *loss_inputs[] = {"x", "y"};
  const char *loss_outputs[] = {"loss"};
  PolyEntrypointOptions opts = {.objective = "loss", .flags = 5};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "loss", loss_inputs, 2, loss_outputs, 1, &opts), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  int ir_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  ASSERT_NOT_NULL(ir);

  PolyIrSpec imported;
  ASSERT_INT_EQ(poly_ir_import(ir, ir_len, &imported), 0);
  ASSERT_INT_EQ(imported.n_entrypoints, 2);
  ASSERT_STR_EQ(imported.entrypoints[1].name, "loss");
  ASSERT_INT_EQ(imported.entrypoints[1].n_inputs, 2);
  ASSERT_STR_EQ(imported.entrypoints[1].inputs[0], "x");
  ASSERT_STR_EQ(imported.entrypoints[1].inputs[1], "y");
  ASSERT_INT_EQ(imported.entrypoints[1].n_outputs, 1);
  ASSERT_STR_EQ(imported.entrypoints[1].outputs[0], "loss");
  ASSERT_STR_EQ(imported.entrypoints[1].objective, "loss");
  ASSERT_INT_EQ(imported.entrypoints[1].flags, 5);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_objective_rejects_vector_output) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {2};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_INT_EQ(poly_instance_output(inst, "loss", x), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"loss"};
  PolyEntrypointOptions opts = {.objective = "loss"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "loss", inputs, 1, outputs, 1, &opts), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_INVALID);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_FAILED);
  const PolyInstanceError *err = poly_instance_last_error(inst);
  ASSERT_NOT_NULL(err);
  ASSERT_TRUE(strstr(err->message, "scalar or one element") != NULL);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, named_value_state_preserves_logical_program_and_binds_physical_storage) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  PolyDevice device = test_ctx_execution_device(ctx);

  PolyTensor *a = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  PolyTensor *b = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  float av[] = {1.0f, 2.0f};
  float bv[] = {3.0f, 4.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, poly_tensor_uop_physical(a), av, sizeof(av)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, poly_tensor_uop_physical(b), bv, sizeof(bv)), 0);

  PolyTensor *w = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  PolyTensor *y = poly_tensor_alu2(ctx, POLY_OP_MUL, x, w);
  ASSERT_NOT_NULL(w);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(y);
  poly_tensor_set_requires_grad(w, true);
  PolyUOp *source_logical = poly_tensor_uop_logical(w);
  PolyUOp *source_physical = poly_tensor_uop_physical(w);
  ASSERT_INT_EQ(source_logical->op, POLY_OP_ADD);
  ASSERT_INT_EQ(source_physical->op, POLY_OP_ADD);
  ASSERT_EQ(poly_uop_get_buffer_identity(source_logical), NULL);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "w", .role = POLY_ROLE_PARAM, .tensor = w},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = y},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {
      {.name = "forward", .inputs = inputs, .n_inputs = 1,
       .outputs = outputs, .n_outputs = 1},
  };
  PolyInstanceError error = {0};
  PolyInstance *inst = poly_instance_from_bindings(
      ctx, bindings, 3, entrypoints, 1, NULL, &error
  );
  ASSERT_NOT_NULL(inst);

  ASSERT_PTR_EQ(poly_tensor_uop_logical(w), source_logical);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(w), source_physical);
  ASSERT_INT_EQ(
      instance_count_root_op(ctx, poly_instance_get_sink(inst, "forward"), POLY_OP_ADD), 0
  );

  int64_t n = 0;
  float *state_data = poly_instance_param_data(inst, 0, &n);
  ASSERT_NOT_NULL(state_data);
  ASSERT_INT_EQ((int)n, 2);
  ASSERT_FLOAT_EQ(state_data[0], 4.0f, 1e-6f);
  ASSERT_FLOAT_EQ(state_data[1], 6.0f, 1e-6f);

  float xv[] = {2.0f, 3.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", xv, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  float *out = poly_instance_buf_data_named(inst, "output", &n);
  ASSERT_NOT_NULL(out);
  ASSERT_FLOAT_EQ(out[0], 8.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[1], 18.0f, 1e-5f);

  int ir_len = 0, weights_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_NOT_NULL(weights);

  /* Tinygrad load_state_dict replaces only the named Tensor's current UOp
   * (`nn/state.py:126-160`); it does not rewrite an already-built program.
   * Polygrad's portable divergence keeps that exact program and binds the
   * named value only while constructing the active physical graph. */
  PolyIrSpec portable = {0};
  ASSERT_INT_EQ(poly_ir_import(ir, ir_len, &portable), 0);
  PolyUOp *portable_w = NULL;
  for (int i = 0; i < portable.n_bufs; i++)
    if (strcmp(portable.bufs[i].name, "w") == 0) portable_w = portable.bufs[i].buffer;
  ASSERT_NOT_NULL(portable_w);
  ASSERT_INT_EQ(portable_w->op, POLY_OP_ADD);
  ASSERT_INT_EQ(
      instance_count_root_op(portable.ctx, portable.entrypoints[0].sink, POLY_OP_ADD), 1
  );
  ASSERT_TRUE(poly_uop_reachable(portable.ctx, portable.entrypoints[0].sink, portable_w));
  PolyCtx *portable_ctx = portable.ctx;
  poly_ir_spec_free(&portable);
  poly_ctx_destroy(portable_ctx);

  ASSERT_EQ(poly_instance_from_ir(ir, ir_len, NULL, 0), NULL);
  float unrelated_data[] = {9.0f, 9.0f};
  PolySafetensorEntry unrelated = {
      .name = "other", .data = unrelated_data, .shape = shape, .ndim = 1,
  };
  int partial_len = 0;
  uint8_t *partial = poly_safetensors_encode(&unrelated, 1, NULL, &partial_len);
  ASSERT_NOT_NULL(partial);
  ASSERT_EQ(poly_instance_from_ir(ir, ir_len, partial, partial_len), NULL);
  free(partial);
  PolyInstance *restored = poly_instance_from_ir(ir, ir_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);
  ASSERT_INT_EQ(
      instance_count_root_op(
          poly_instance_ctx(restored), poly_instance_get_sink(restored, "forward"), POLY_OP_ADD
      ),
      0
  );
  ASSERT_INT_EQ(poly_instance_forward(restored, io, 1), 0);
  out = poly_instance_buf_data_named(restored, "output", &n);
  ASSERT_NOT_NULL(out);
  ASSERT_FLOAT_EQ(out[0], 8.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[1], 18.0f, 1e-5f);

  poly_instance_free(restored);
  free(weights);
  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, fresh_import_evaluates_closed_named_initializer_once) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  PolyUOp *x = poly_uop_new_logical_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *three = poly_full(ctx, shape, 1, 3.0);
  PolyUOp *one = poly_full(ctx, shape, 1, 1.0);
  PolyUOp *w = poly_add(ctx, three, one);
  PolyUOp *y = poly_mul(ctx, x, w);
  PolyUOp *out_buffer = poly_uop_new_logical_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *store = poly_store_val(ctx, out_buffer, y);
  PolyUOp *sink = poly_sink1(ctx, store);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);
  ASSERT_NOT_NULL(y);
  ASSERT_NOT_NULL(out_buffer);
  ASSERT_NOT_NULL(sink);
  ASSERT_INT_EQ(w->op, POLY_OP_ADD);
  ASSERT_EQ(poly_uop_get_buffer_identity(w), NULL);

  PolyIrBufEntry bufs[] = {
      {.name = "x", .role = POLY_IR_ROLE_INPUT, .buffer = x, .shape = {2}, .ndim = 1},
      {.name = "w", .role = POLY_IR_ROLE_PARAM, .buffer = w, .shape = {2}, .ndim = 1,
       .trainable = true, .trainable_set = true},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out_buffer,
       .shape = {2}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {
      .ctx = ctx, .bufs = bufs, .n_bufs = 3, .entrypoints = eps, .n_entrypoints = 1,
  };
  int ir_len = 0;
  uint8_t *ir = poly_ir_export(&spec, &ir_len);
  ASSERT_NOT_NULL(ir);

  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);
  PolyCtx *imported_ctx = poly_instance_ctx(inst);
  ASSERT_INT_EQ(
      instance_count_root_op(imported_ctx, poly_instance_get_sink(inst, "forward"), POLY_OP_ADD),
      0
  );
  int exported_len = 0;
  uint8_t *exported = poly_instance_export_ir(inst, &exported_len);
  ASSERT_NOT_NULL(exported);
  PolyIrSpec portable = {0};
  ASSERT_INT_EQ(poly_ir_import(exported, exported_len, &portable), 0);
  PolyUOp *portable_w = NULL;
  for (int i = 0; i < portable.n_bufs; i++)
    if (strcmp(portable.bufs[i].name, "w") == 0) portable_w = portable.bufs[i].buffer;
  ASSERT_NOT_NULL(portable_w);
  ASSERT_INT_EQ(portable_w->op, POLY_OP_ADD);
  ASSERT_TRUE(poly_uop_reachable(portable.ctx, portable.entrypoints[0].sink, portable_w));
  PolyCtx *portable_ctx = portable.ctx;
  poly_ir_spec_free(&portable);
  poly_ctx_destroy(portable_ctx);
  free(exported);

  int64_t n = 0;
  float *weight = poly_instance_param_data(inst, 0, &n);
  ASSERT_NOT_NULL(weight);
  ASSERT_INT_EQ(n, 2);
  ASSERT_FLOAT_EQ(weight[0], 4.0f, 1e-6f);
  ASSERT_FLOAT_EQ(weight[1], 4.0f, 1e-6f);
  float xv[] = {2.0f, 3.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", xv, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  float *output = poly_instance_buf_data_named(inst, "output", &n);
  ASSERT_NOT_NULL(output);
  ASSERT_FLOAT_EQ(output[0], 8.0f, 1e-5f);
  ASSERT_FLOAT_EQ(output[1], 12.0f, 1e-5f);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  weight = poly_instance_param_data(inst, 0, &n);
  ASSERT_FLOAT_EQ(weight[0], 4.0f, 1e-6f);
  ASSERT_FLOAT_EQ(weight[1], 4.0f, 1e-6f);

  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, stochastic_named_state_requires_checkpoint_for_portable_activation) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  const int f32_id = poly_dtype_id_by_name("float32");
  PolyDevice device = test_ctx_execution_device(ctx);
  poly_tensor_manual_seed(ctx, 11);
  PolyTensor *w = poly_tensor_rand_by_id(ctx, shape, 1, f32_id, device, 1);
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_MUL, x, w);
  ASSERT_NOT_NULL(w);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_tensor_uop_logical(w)->op, POLY_OP_CONTIGUOUS);
  ASSERT_TRUE(instance_count_root_op(ctx, poly_tensor_uop_logical(w), POLY_OP_AFTER) > 0);
  ASSERT_TRUE(instance_count_root_op(ctx, poly_tensor_uop_logical(w), POLY_OP_STORE) > 0);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "w", .role = POLY_ROLE_PARAM, .tensor = w},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {{
      .name = "forward", .inputs = inputs, .n_inputs = 1,
      .outputs = outputs, .n_outputs = 1,
  }};
  PolyInstanceError error = {0};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 3, entrypoints, 1, NULL, &error);
  ASSERT_NOT_NULL(inst);

  int ir_len = 0, weights_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_NOT_NULL(weights);

  /* Pinned Tensor RNG advances storage-backed per-device counters through
   * STORE/AFTER (tensor.py:473-504).  The portable deterministic Instance
   * contract cannot replay that history, so bytes are mandatory. */
  ASSERT_EQ(poly_instance_from_ir(ir, ir_len, NULL, 0), NULL);
  PolyInstance *restored = poly_instance_from_ir(ir, ir_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);
  ASSERT_INT_EQ(
      poly_instance_param_nbytes(restored, 0), poly_instance_param_nbytes(inst, 0)
  );
  ASSERT_TRUE(
      memcmp(
          poly_instance_param_data_raw(inst, 0, NULL),
          poly_instance_param_data_raw(restored, 0, NULL),
          poly_instance_param_nbytes(inst, 0)
      ) == 0
  );

  float xv[] = {2.0f, 3.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", xv, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  ASSERT_INT_EQ(poly_instance_forward(restored, io, 1), 0);
  int64_t source_n = 0, restored_n = 0;
  float *source_out = poly_instance_buf_data_named(inst, "output", &source_n);
  float *restored_out = poly_instance_buf_data_named(restored, "output", &restored_n);
  ASSERT_NOT_NULL(source_out);
  ASSERT_NOT_NULL(restored_out);
  ASSERT_INT_EQ(source_n, restored_n);
  for (int64_t i = 0; i < source_n; i++)
    ASSERT_FLOAT_EQ(source_out[i], restored_out[i], 0.0f);

  poly_instance_free(restored);
  free(weights);
  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, duplicate_abi_storage_alias_fails_closed_like_tinyjit) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  PolyTensor *x = poly_tensor_empty(
      ctx, POLY_FLOAT32, shape, 1, test_ctx_execution_device(ctx)
  );
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, x);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(out);

  PolyBindingSpec bindings[] = {
      {.name = "a", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "b", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"a", "b"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entries[] = {{
      .name = "forward", .inputs = inputs, .n_inputs = 2,
      .outputs = outputs, .n_outputs = 1,
  }};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 3, entries, 1, NULL, NULL);
  ASSERT_EQ(inst, NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, dynamic_abi_storage_alias_with_persistent_state_fails_closed) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  PolyTensor *x = poly_tensor_empty(
      ctx, POLY_FLOAT32, shape, 1, test_ctx_execution_device(ctx)
  );
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, x);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(out);

  /* Rejected packaging must not leave transaction rows dangling when a
   * collection safe point runs inside the build. */
  PolyTensor *retired = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(retired);
  float retired_data[] = {1.0f, 2.0f};
  ASSERT_INT_EQ(
      poly_buffer_write(
          ctx, poly_tensor_uop_physical(retired), retired_data, sizeof(retired_data)
      ),
      0
  );
  poly_tensor_release(retired);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "w", .role = POLY_ROLE_PARAM, .tensor = x},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entries[] = {{
      .name = "forward", .inputs = inputs, .n_inputs = 1,
      .outputs = outputs, .n_outputs = 1,
  }};
  ASSERT_EQ(
      poly_instance_from_bindings(ctx, bindings, 3, entries, 1, NULL, NULL),
      NULL
  );

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, output_alias_of_dynamic_input_round_trips) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  PolyTensor *x = poly_tensor_empty(
      ctx, POLY_FLOAT32, shape, 1, test_ctx_execution_device(ctx)
  );
  ASSERT_NOT_NULL(x);
  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = x},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entries[] = {{
      .name = "forward", .inputs = inputs, .n_inputs = 1,
      .outputs = outputs, .n_outputs = 1,
  }};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 2, entries, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);
  float value[] = {3.0f, 4.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", value, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  int64_t n = 0;
  float *out = poly_instance_buf_data_named(inst, "output", &n);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(n, 2);
  ASSERT_FLOAT_EQ(out[0], 3.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[1], 4.0f, 0.0f);

  int ir_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  ASSERT_NOT_NULL(ir);
  PolyInstance *restored = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(restored);
  ASSERT_INT_EQ(poly_instance_forward(restored, io, 1), 0);
  out = poly_instance_buf_data_named(restored, "output", &n);
  ASSERT_NOT_NULL(out);
  ASSERT_FLOAT_EQ(out[0], 3.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[1], 4.0f, 0.0f);

  poly_instance_free(restored);
  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, named_partial_view_state_fails_closed) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {4};
  int64_t x_shape[] = {2};
  int64_t bounds[][2] = {{1, 3}};
  PolyDevice device = test_ctx_execution_device(ctx);
  PolyTensor *base = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  PolyTensor *view = poly_tensor_shrink(ctx, base, bounds, 1);
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, x_shape, 1, device);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, view);
  ASSERT_NOT_NULL(base);
  ASSERT_NOT_NULL(view);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(out);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "base", .role = POLY_ROLE_PARAM, .tensor = base},
      {.name = "view", .role = POLY_ROLE_PARAM, .tensor = view},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entries[] = {{
      .name = "forward", .inputs = inputs, .n_inputs = 1,
      .outputs = outputs, .n_outputs = 1,
  }};
  PolyInstanceError error = {0};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 4, entries, 1, NULL, &error);
  ASSERT_EQ(inst, NULL);
  ASSERT_TRUE(strstr(error.message, "unsupported named view state") != NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, input_dependent_named_state_effect_fails_closed) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {1};
  PolyDevice device = test_ctx_execution_device(ctx);
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  PolyTensor *w = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, w, x);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);
  ASSERT_NOT_NULL(sum);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, w, sum), w);
  ASSERT_TRUE(instance_count_root_op(ctx, poly_tensor_uop_logical(w), POLY_OP_AFTER) > 0);
  ASSERT_TRUE(instance_count_root_op(ctx, poly_tensor_uop_logical(w), POLY_OP_STORE) > 0);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "w", .role = POLY_ROLE_PARAM, .tensor = w},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = w},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entries[] = {{
      .name = "forward", .inputs = inputs, .n_inputs = 1,
      .outputs = outputs, .n_outputs = 1,
  }};
  PolyInstanceError error = {0};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 3, entries, 1, NULL, &error);
  ASSERT_EQ(inst, NULL);
  ASSERT_TRUE(strstr(error.message, "depends on an input or target") != NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, assigned_realized_input_history_fails_closed) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  PolyDevice device = test_ctx_execution_device(ctx);
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  PolyTensor *value = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, x, value), x);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &x, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, x);
  ASSERT_TRUE(poly_uop_has_buffer_identity(poly_tensor_uop_physical(x)));
  ASSERT_FALSE(poly_uop_has_buffer_identity(poly_tensor_uop_logical(x)));
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, x);
  ASSERT_NOT_NULL(out);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entries[] = {{
      .name = "forward", .inputs = inputs, .n_inputs = 1,
      .outputs = outputs, .n_outputs = 1,
  }};
  PolyInstanceError error = {0};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 2, entries, 1, NULL, &error);
  ASSERT_EQ(inst, NULL);
  ASSERT_TRUE(strstr(error.message, "has no buffer identity") != NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, stochastic_output_effect_requires_named_rng_state) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  PolyDevice device = test_ctx_execution_device(ctx);
  poly_tensor_manual_seed(ctx, 123);
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  PolyTensor *random = poly_tensor_rand_by_id(
      ctx, shape, 1, poly_dtype_id_by_name("float32"), device, 1
  );
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, random);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(random);
  ASSERT_NOT_NULL(out);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entries[] = {{
      .name = "forward", .inputs = inputs, .n_inputs = 1,
      .outputs = outputs, .n_outputs = 1,
  }};
  PolyInstanceError error = {0};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 2, entries, 1, NULL, &error);
  ASSERT_EQ(inst, NULL);
  ASSERT_TRUE(strstr(error.message, "unbound storage") != NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, import_rejects_duplicate_abi_storage_alias) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *input = poly_buffer_f32(ctx, 2);
  PolyUOp *output = poly_buffer_f32(ctx, 2);
  PolyUOp *sink = poly_sink1(
      ctx, poly_store_val(ctx, output, poly_alu2(ctx, POLY_OP_ADD, input, input))
  );
  ASSERT_NOT_NULL(input);
  ASSERT_NOT_NULL(output);
  ASSERT_NOT_NULL(sink);
  PolyIrBufEntry bufs[] = {
      {.name = "a", .role = POLY_IR_ROLE_INPUT, .buffer = input,
       .shape = {2}, .ndim = 1},
      {.name = "b", .role = POLY_IR_ROLE_INPUT, .buffer = input,
       .shape = {2}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = output,
       .shape = {2}, .ndim = 1},
  };
  const char *inputs[] = {"a", "b"};
  const char *outputs[] = {"output"};
  PolyIrEntrypoint entry = {
      .name = "forward", .sink = sink, .inputs = inputs, .n_inputs = 2,
      .outputs = outputs, .n_outputs = 1,
  };
  PolyIrSpec spec = {
      .ctx = ctx, .bufs = bufs, .n_bufs = 3,
      .entrypoints = &entry, .n_entrypoints = 1,
  };
  int ir_len = 0;
  uint8_t *ir = poly_ir_export(&spec, &ir_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_EQ(poly_instance_from_ir(ir, ir_len, NULL, 0), NULL);

  free(ir);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, import_rejects_dynamic_abi_alias_with_persistent_state) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shared = poly_buffer_f32(ctx, 2);
  PolyUOp *output = poly_buffer_f32(ctx, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, output, shared));
  ASSERT_NOT_NULL(shared);
  ASSERT_NOT_NULL(output);
  ASSERT_NOT_NULL(sink);
  PolyIrBufEntry bufs[] = {
      {.name = "x", .role = POLY_IR_ROLE_INPUT, .buffer = shared,
       .shape = {2}, .ndim = 1},
      {.name = "w", .role = POLY_IR_ROLE_PARAM, .buffer = shared,
       .shape = {2}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = output,
       .shape = {2}, .ndim = 1},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyIrEntrypoint entry = {
      .name = "forward", .sink = sink, .inputs = inputs, .n_inputs = 1,
      .outputs = outputs, .n_outputs = 1,
  };
  PolyIrSpec spec = {
      .ctx = ctx, .bufs = bufs, .n_bufs = 3,
      .entrypoints = &entry, .n_entrypoints = 1,
  };
  int ir_len = 0;
  uint8_t *ir = poly_ir_export(&spec, &ir_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_EQ(poly_instance_from_ir(ir, ir_len, NULL, 0), NULL);

  free(ir);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, import_rejects_named_partial_view_state) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *base = poly_buffer_f32(ctx, 4);
  int64_t bounds[][2] = {{1, 3}};
  PolyUOp *view = poly_shrink(ctx, base, bounds, 1);
  PolyUOp *output = poly_buffer_f32(ctx, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, output, view));
  ASSERT_NOT_NULL(base);
  ASSERT_NOT_NULL(view);
  ASSERT_NOT_NULL(output);
  ASSERT_NOT_NULL(sink);
  PolyIrBufEntry bufs[] = {
      {.name = "base", .role = POLY_IR_ROLE_PARAM, .buffer = base,
       .shape = {4}, .ndim = 1},
      {.name = "view", .role = POLY_IR_ROLE_PARAM, .buffer = view,
       .shape = {2}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = output,
       .shape = {2}, .ndim = 1},
  };
  const char *outputs[] = {"output"};
  PolyIrEntrypoint entry = {
      .name = "forward", .sink = sink, .outputs = outputs, .n_outputs = 1,
  };
  PolyIrSpec spec = {
      .ctx = ctx, .bufs = bufs, .n_bufs = 3,
      .entrypoints = &entry, .n_entrypoints = 1,
  };
  int ir_len = 0;
  uint8_t *ir = poly_ir_export(&spec, &ir_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_EQ(poly_instance_from_ir(ir, ir_len, NULL, 0), NULL);

  free(ir);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, checkpoint_import_rejects_input_dependent_named_state_effect) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *input = poly_buffer_f32(ctx, 1);
  PolyUOp *state = poly_buffer_f32(ctx, 1);
  PolyUOp *updated = poly_uop_after(
      ctx, state,
      poly_store_val(ctx, state, poly_alu2(ctx, POLY_OP_ADD, state, input))
  );
  PolyUOp *output = poly_buffer_f32(ctx, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, output, updated));
  ASSERT_NOT_NULL(input);
  ASSERT_NOT_NULL(state);
  ASSERT_NOT_NULL(updated);
  ASSERT_NOT_NULL(output);
  ASSERT_NOT_NULL(sink);
  PolyIrBufEntry bufs[] = {
      {.name = "x", .role = POLY_IR_ROLE_INPUT, .buffer = input,
       .shape = {1}, .ndim = 1},
      {.name = "w", .role = POLY_IR_ROLE_PARAM, .buffer = updated,
       .shape = {1}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = output,
       .shape = {1}, .ndim = 1},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyIrEntrypoint entry = {
      .name = "forward", .sink = sink, .inputs = inputs, .n_inputs = 1,
      .outputs = outputs, .n_outputs = 1,
  };
  PolyIrSpec spec = {
      .ctx = ctx, .bufs = bufs, .n_bufs = 3,
      .entrypoints = &entry, .n_entrypoints = 1,
  };
  int ir_len = 0;
  uint8_t *ir = poly_ir_export(&spec, &ir_len);
  ASSERT_NOT_NULL(ir);
  float weight[] = {7.0f};
  int64_t shape[] = {1};
  PolySafetensorEntry weight_entry = {
      .name = "w", .data = weight, .shape = shape, .ndim = 1,
      .dtype = POLY_ST_F32,
  };
  int checkpoint_len = 0;
  uint8_t *checkpoint =
      poly_safetensors_encode(&weight_entry, 1, NULL, &checkpoint_len);
  ASSERT_NOT_NULL(checkpoint);
  ASSERT_EQ(poly_instance_from_ir(ir, ir_len, checkpoint, checkpoint_len), NULL);

  free(checkpoint);
  free(ir);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, multi_entrypoint_shared_state_round_trips_across_policy_replacement) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  PolyDevice initial = test_ctx_execution_device(ctx);
  PolyDevice replacement = initial == POLY_DEVICE_INTERP ? POLY_DEVICE_CPU
                                                          : POLY_DEVICE_INTERP;
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, initial);
  PolyTensor *w = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, initial);
  PolyTensor *shared = poly_tensor_alu2(ctx, POLY_OP_MUL, x, w);
  PolyTensor *plus = poly_tensor_alu2(ctx, POLY_OP_ADD, shared, x);
  PolyTensor *minus = poly_tensor_alu2(ctx, POLY_OP_SUB, shared, x);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);
  ASSERT_NOT_NULL(shared);
  ASSERT_NOT_NULL(plus);
  ASSERT_NOT_NULL(minus);
  float weights_data[] = {2.0f, 3.0f};
  ASSERT_INT_EQ(
      poly_buffer_write(
          ctx, poly_tensor_uop_physical(w), weights_data, sizeof(weights_data)
      ),
      0
  );

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "w", .role = POLY_ROLE_PARAM, .tensor = w},
      {.name = "plus", .role = POLY_ROLE_OUTPUT, .tensor = plus},
      {.name = "minus", .role = POLY_ROLE_OUTPUT, .tensor = minus},
  };
  const char *inputs[] = {"x"};
  const char *plus_outputs[] = {"plus"};
  const char *minus_outputs[] = {"minus"};
  PolyEntrypointSpec entries[] = {
      {
          .name = "plus_ep", .inputs = inputs, .n_inputs = 1,
          .outputs = plus_outputs, .n_outputs = 1,
      },
      {
          .name = "minus_ep", .inputs = inputs, .n_inputs = 1,
          .outputs = minus_outputs, .n_outputs = 1,
      },
  };
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 4, entries, 2, NULL, NULL);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_entrypoint_count(inst), 2);
  ASSERT_STR_EQ(poly_instance_entrypoint_name(inst, 0), "plus_ep");
  ASSERT_INT_EQ(poly_instance_entrypoint_input_count(inst, "plus_ep"), 1);
  ASSERT_STR_EQ(poly_instance_entrypoint_input_name(inst, "plus_ep", 0), "x");
  ASSERT_INT_EQ(poly_instance_entrypoint_output_count(inst, "plus_ep"), 1);
  ASSERT_STR_EQ(poly_instance_entrypoint_output_name(inst, "plus_ep", 0), "plus");
  ASSERT_INT_EQ(poly_instance_entrypoint_output_count(inst, "minus_ep"), 1);
  ASSERT_STR_EQ(poly_instance_entrypoint_output_name(inst, "minus_ep", 0), "minus");
  ASSERT_EQ(poly_instance_entrypoint_output_count(inst, "missing"), -1);

  int ir_before_len = 0;
  uint8_t *ir_before = poly_instance_export_ir(inst, &ir_before_len);
  ASSERT_NOT_NULL(ir_before);
  float input_data[] = {4.0f, 5.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", input_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_call(inst, "plus_ep", io, 1), 0);
  int64_t n = 0;
  float *out = poly_instance_buf_data_named(inst, "plus", &n);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(n, 2);
  ASSERT_FLOAT_EQ(out[0], 12.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[1], 20.0f, 0.0f);
  ASSERT_INT_EQ(poly_instance_call(inst, "minus_ep", io, 1), 0);
  out = poly_instance_buf_data_named(inst, "minus", &n);
  ASSERT_NOT_NULL(out);
  ASSERT_FLOAT_EQ(out[0], 4.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[1], 10.0f, 0.0f);

  ASSERT_INT_EQ(poly_instance_set_device(inst, replacement), 0);
  ASSERT_INT_EQ(poly_instance_call(inst, "plus_ep", io, 1), 0);
  out = poly_instance_buf_data_named(inst, "plus", &n);
  ASSERT_NOT_NULL(out);
  ASSERT_FLOAT_EQ(out[0], 12.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[1], 20.0f, 0.0f);
  ASSERT_INT_EQ(poly_instance_call(inst, "minus_ep", io, 1), 0);
  out = poly_instance_buf_data_named(inst, "minus", &n);
  ASSERT_NOT_NULL(out);
  ASSERT_FLOAT_EQ(out[0], 4.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[1], 10.0f, 0.0f);

  int ir_after_len = 0;
  uint8_t *ir_after = poly_instance_export_ir(inst, &ir_after_len);
  ASSERT_NOT_NULL(ir_after);
  ASSERT_INT_EQ(ir_after_len, ir_before_len);
  ASSERT_TRUE(memcmp(ir_before, ir_after, (size_t)ir_before_len) == 0);
  int weights_len = 0;
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(weights);
  PolyInstance *restored =
      poly_instance_from_ir(ir_after, ir_after_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);
  ASSERT_INT_EQ(poly_instance_call(restored, "plus_ep", io, 1), 0);
  out = poly_instance_buf_data_named(restored, "plus", &n);
  ASSERT_NOT_NULL(out);
  ASSERT_FLOAT_EQ(out[0], 12.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[1], 20.0f, 0.0f);
  ASSERT_INT_EQ(poly_instance_call(restored, "minus_ep", io, 1), 0);
  out = poly_instance_buf_data_named(restored, "minus", &n);
  ASSERT_NOT_NULL(out);
  ASSERT_FLOAT_EQ(out[0], 4.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[1], 10.0f, 0.0f);

  poly_instance_free(restored);
  free(weights);
  free(ir_after);
  free(ir_before);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, float16_state_exports_and_imports_exact_storage_bytes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);
  int64_t shape[] = {2};
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT16, shape, 1);
  ASSERT_NOT_NULL(w);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", w), POLY_STATUS_OK);
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", NULL, 0, outputs, 1, NULL),
      POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  int64_t numel = 0;
  uint16_t *raw = poly_instance_param_data_raw(inst, 0, &numel);
  ASSERT_NOT_NULL(raw);
  ASSERT_INT_EQ(numel, 2);
  ASSERT_EQ(poly_instance_param_data(inst, 0, NULL), NULL);
  ASSERT_INT_EQ(poly_instance_param_dtype_id(inst, 0), poly_dtype_id_by_name("float16"));
  ASSERT_INT_EQ(poly_instance_param_nbytes(inst, 0), sizeof(uint16_t) * 2);
  raw[0] = 0x3e00; /* 1.5 */
  raw[1] = 0xc000; /* -2.0 */

  int weights_len = 0;
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(weights);
  int n_views = 0;
  PolySafetensorViewEx *views =
      poly_safetensors_decode_ex(weights, weights_len, &n_views, NULL);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(n_views, 1);
  ASSERT_STR_EQ(views[0].name, "w");
  ASSERT_INT_EQ(views[0].dtype, POLY_ST_F16);
  ASSERT_INT_EQ(views[0].numel, 2);
  ASSERT_TRUE(memcmp(views[0].raw_data, raw, sizeof(uint16_t) * 2) == 0);
  free(views[0].name);
  free(views);

  int ir_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  ASSERT_NOT_NULL(ir);
  PolyInstance *restored = poly_instance_from_ir(ir, ir_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);
  uint16_t *restored_raw = poly_instance_param_data_raw(restored, 0, &numel);
  ASSERT_NOT_NULL(restored_raw);
  ASSERT_INT_EQ(numel, 2);
  ASSERT_TRUE(memcmp(restored_raw, raw, sizeof(uint16_t) * 2) == 0);
  ASSERT_INT_EQ(poly_instance_param_nbytes(restored, 0), sizeof(uint16_t) * 2);

  poly_instance_free(restored);
  free(ir);
  free(weights);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, supported_scalar_state_dtypes_round_trip_exact_storage_bytes) {
  const char *dtype_names[] = {
      "float32", "float16", "bfloat16", "float64", "int64", "int32", "int16",
      "int8", "uint8", "bool", "uint16", "uint32", "uint64",
  };
  for (size_t dtype_index = 0;
       dtype_index < sizeof(dtype_names) / sizeof(dtype_names[0]); dtype_index++) {
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);
    PolyDType dtype;
    int dtype_id = poly_dtype_id_by_name(dtype_names[dtype_index]);
    ASSERT_TRUE(poly_dtype_by_id(dtype_id, &dtype));
    int itemsize = poly_dtype_itemsize(dtype);
    ASSERT_TRUE(itemsize > 0 && itemsize <= 8);

    PolyInstance *inst = poly_instance_new(ctx, NULL);
    ASSERT_NOT_NULL(inst);
    int64_t shape[] = {2};
    PolyTensor *w = poly_instance_param(inst, "w", dtype, shape, 1);
    ASSERT_NOT_NULL(w);
    ASSERT_INT_EQ(poly_instance_output(inst, "output", w), POLY_STATUS_OK);
    const char *outputs[] = {"output"};
    ASSERT_INT_EQ(
        poly_instance_entrypoint(inst, "forward", NULL, 0, outputs, 1, NULL),
        POLY_STATUS_OK
    );
    ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

    size_t nbytes = (size_t)itemsize * 2;
    uint8_t expected[16] = {0};
    for (size_t i = 0; i < nbytes; i++)
      expected[i] = (uint8_t)(17 + i * 29 + dtype_index);
    if (strcmp(dtype_names[dtype_index], "bool") == 0) {
      expected[0] = 0;
      expected[1] = 1;
    }
    void *raw = poly_instance_param_data_raw(inst, 0, NULL);
    ASSERT_NOT_NULL(raw);
    ASSERT_INT_EQ(poly_instance_param_nbytes(inst, 0), nbytes);
    memcpy(raw, expected, nbytes);

    int ir_len = 0, weights_len = 0;
    uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
    uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
    ASSERT_NOT_NULL(ir);
    ASSERT_NOT_NULL(weights);
    PolyInstance *restored = poly_instance_from_ir(ir, ir_len, weights, weights_len);
    ASSERT_NOT_NULL(restored);
    ASSERT_INT_EQ(poly_instance_param_dtype_id(restored, 0), dtype_id);
    ASSERT_INT_EQ(poly_instance_param_nbytes(restored, 0), nbytes);
    ASSERT_TRUE(
        memcmp(poly_instance_param_data_raw(restored, 0, NULL), expected, nbytes) == 0
    );

    poly_instance_free(restored);
    free(weights);
    free(ir);
    poly_instance_free(inst);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(instance, checkpoint_validation_is_shape_exact_and_transactional) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);
  int64_t shape[] = {1};
  PolyTensor *a = poly_instance_param(inst, "a", POLY_FLOAT32, shape, 1);
  PolyTensor *b = poly_instance_param(inst, "b", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
  ASSERT_NOT_NULL(sum);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", sum), POLY_STATUS_OK);
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", NULL, 0, outputs, 1, NULL),
      POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  int64_t numel = 0;
  float *a_data = poly_instance_param_data(inst, 0, &numel);
  float *b_data = poly_instance_param_data(inst, 1, &numel);
  ASSERT_NOT_NULL(a_data);
  ASSERT_NOT_NULL(b_data);
  a_data[0] = 1.0f;
  b_data[0] = 2.0f;

  float replacement_a[] = {9.0f};
  float invalid_b[] = {7.0f, 8.0f};
  int64_t bad_shape[] = {2};
  PolySafetensorEntry rows[] = {
      {.name = "a", .data = replacement_a, .shape = shape, .ndim = 1,
       .dtype = POLY_ST_F32},
      {.name = "b", .data = invalid_b, .shape = bad_shape, .ndim = 1,
       .dtype = POLY_ST_F32},
  };
  int checkpoint_len = 0;
  uint8_t *checkpoint = poly_safetensors_encode(rows, 2, NULL, &checkpoint_len);
  ASSERT_NOT_NULL(checkpoint);
  ASSERT_INT_EQ(poly_instance_import_weights(inst, checkpoint, checkpoint_len), -1);
  ASSERT_FLOAT_EQ(poly_instance_param_data(inst, 0, NULL)[0], 1.0f, 0.0f);
  ASSERT_FLOAT_EQ(poly_instance_param_data(inst, 1, NULL)[0], 2.0f, 0.0f);

  free(checkpoint);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, alias_checkpoint_payloads_follow_deterministic_name_order) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {1};
  PolyTensor *w = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(w);
  float initial[] = {1.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, poly_tensor_uop_physical(w), initial, sizeof(initial)), 0);
  PolyBindingSpec bindings[] = {
      {.name = "a", .role = POLY_ROLE_PARAM, .tensor = w},
      {.name = "b", .role = POLY_ROLE_PARAM, .tensor = w},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = w},
  };
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {
      {.name = "forward", .outputs = outputs, .n_outputs = 1},
  };
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 3, entrypoints, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 2);
  ASSERT_PTR_EQ(
      poly_instance_param_data_raw(inst, 0, NULL),
      poly_instance_param_data_raw(inst, 1, NULL)
  );

  float first[] = {3.0f};
  float second[] = {9.0f};
  PolySafetensorEntry rows[] = {
      {.name = "b", .data = second, .shape = shape, .ndim = 1, .dtype = POLY_ST_F32},
      {.name = "a", .data = first, .shape = shape, .ndim = 1, .dtype = POLY_ST_F32},
  };
  int checkpoint_len = 0;
  uint8_t *checkpoint = poly_safetensors_encode(rows, 2, NULL, &checkpoint_len);
  ASSERT_NOT_NULL(checkpoint);
  ASSERT_INT_EQ(poly_instance_import_weights(inst, checkpoint, checkpoint_len), 0);
  ASSERT_FLOAT_EQ(poly_instance_param_data(inst, 0, NULL)[0], 9.0f, 0.0f);
  ASSERT_FLOAT_EQ(poly_instance_param_data(inst, 1, NULL)[0], 9.0f, 0.0f);

  free(checkpoint);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, fresh_import_rejects_special_before_scheduling) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *bound = poly_const_int(ctx, 4);
  PolyUOp *special =
      poly_uop1(ctx, POLY_OP_SPECIAL, POLY_WEAKINT, bound, poly_arg_str("gidx0"));
  PolyUOp *value = poly_cast(ctx, special, POLY_FLOAT32);
  PolyUOp *output = poly_buffer_f32(ctx, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, output, value));
  ASSERT_NOT_NULL(value);
  ASSERT_NOT_NULL(sink);
  PolyIrBufEntry bufs[] = {
      {.name = "w", .role = POLY_IR_ROLE_PARAM, .buffer = value, .ndim = 0,
       .trainable = true, .trainable_set = true},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = output,
       .shape = {1}, .ndim = 1},
  };
  PolyIrEntrypoint ep = {.name = "forward", .sink = sink};
  PolyIrSpec spec = {.ctx = ctx, .bufs = bufs, .n_bufs = 2,
                     .entrypoints = &ep, .n_entrypoints = 1};
  int ir_len = 0;
  uint8_t *ir = poly_ir_export(&spec, &ir_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_EQ(poly_instance_from_ir(ir, ir_len, NULL, 0), NULL);

  free(ir);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, failed_from_bindings_does_not_snapshot_lazy_state) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  float init[] = {2.0f, 3.0f};
  PolyTensor *host_w =
      poly_tensor_from_host(ctx, init, sizeof(init), POLY_FLOAT32, shape, 1);
  PolyTensor *w = poly_tensor_to_device(ctx, host_w, POLY_DEVICE_CPU);
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *unbound = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(w);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(unbound);
  poly_tensor_set_requires_grad(unbound, true);
  PolyTensor *mul = poly_tensor_alu2(ctx, POLY_OP_MUL, x, w);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, mul, unbound);
  ASSERT_NOT_NULL(out);
  PolyUOp *logical_before = poly_tensor_uop_logical(w);
  PolyUOp *physical_before = poly_tensor_uop_physical(w);
  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "w", .role = POLY_ROLE_PARAM, .tensor = w},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {
      {.name = "forward", .inputs = inputs, .n_inputs = 1,
       .outputs = outputs, .n_outputs = 1},
  };
  int tensors_before = ctx->n_tensors;
  size_t writes_before = ctx->buffer_write_count;
  uint64_t memory_before = ctx->mem_used;
  bool w_requires_grad_before = w->requires_grad;
  bool w_requires_grad_set_before = w->requires_grad_set;
  PolyTensorProvenance w_provenance_before = w->provenance;
  ASSERT_EQ(
      poly_instance_from_bindings(ctx, bindings, 3, entrypoints, 1, NULL, NULL), NULL
  );
  ASSERT_INT_EQ(ctx->n_tensors, tensors_before);
  ASSERT_INT_EQ(ctx->buffer_write_count, writes_before);
  ASSERT_INT_EQ(ctx->mem_used, memory_before);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(w), logical_before);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(w), physical_before);
  ASSERT_EQ(w->requires_grad, w_requires_grad_before);
  ASSERT_EQ(w->requires_grad_set, w_requires_grad_set_before);
  ASSERT_EQ(w->provenance, w_provenance_before);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, later_build_failure_unwinds_named_value_snapshot_and_residency) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {1};
  float av[] = {2.0f};
  float bv[] = {3.0f};
  PolyTensor *a_source =
      poly_tensor_from_host(ctx, av, sizeof(av), POLY_FLOAT32, shape, 1);
  PolyTensor *b_source =
      poly_tensor_from_host(ctx, bv, sizeof(bv), POLY_FLOAT32, shape, 1);
  PolyTensor *a = poly_tensor_to_device(ctx, a_source, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_to_device(ctx, b_source, POLY_DEVICE_CPU);
  PolyTensor *good = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
  ASSERT_NOT_NULL(a_source);
  ASSERT_NOT_NULL(b_source);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  ASSERT_NOT_NULL(good);

  /* The computed parameter above is snapshotted first.  This internal-only
  * output fixture keeps a valid movement shape but assigns the root VOID
  * dtype, so packing rejects it before any backend runs. */
  PolyUOp *bad_value = poly_full(ctx, shape, 1, 1.0);
  PolyUOp *bad_reshape = poly_reshape(ctx, bad_value, shape, 1);
  ASSERT_NOT_NULL(bad_reshape);
  PolyUOp *bad_root = poly_uop(
      ctx, bad_reshape->op, POLY_VOID, bad_reshape->src, bad_reshape->n_src,
      bad_reshape->arg
  );
  PolyTensor *bad_output = poly_tensor_create_with_roots(
      ctx, bad_root, bad_root, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(bad_output);

  PolyBindingSpec bindings[] = {
      {.name = "good", .role = POLY_ROLE_PARAM, .tensor = good},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = bad_output},
  };
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {
      {.name = "forward", .outputs = outputs, .n_outputs = 1},
  };
  int tensors_before = ctx->n_tensors;
  uint64_t memory_before = ctx->mem_used;
  size_t writes_before = ctx->buffer_write_count;
  PolyCtxStats stats_before = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats_before), 0);
  ASSERT_INT_EQ(poly_tensor_uop_physical(a)->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_tensor_uop_physical(b)->op, POLY_OP_COPY);
  PolyUOp *a_storage =
      (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(a)->src[0]);
  PolyUOp *b_storage =
      (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(b)->src[0]);
  ASSERT_NOT_NULL(a_storage);
  ASSERT_NOT_NULL(b_storage);
  PolyBuffer *a_head_before = poly_buffer_get(ctx, a_storage);
  PolyBuffer *b_head_before = poly_buffer_get(ctx, b_storage);
  ASSERT_NOT_NULL(a_head_before);
  ASSERT_NOT_NULL(b_head_before);
  PolyBuffer a_residency_before = *a_head_before;
  PolyBuffer b_residency_before = *b_head_before;
  PolyInstanceError error = {0};
  PolyInstance *failed =
      poly_instance_from_bindings(ctx, bindings, 2, entrypoints, 1, NULL, &error);
  if (failed) poly_instance_free(failed);
  ASSERT_EQ(failed, NULL);
  ASSERT_TRUE(strstr(error.message, "failed to pack runtime instance") != NULL);
  ASSERT_INT_EQ(ctx->n_tensors, tensors_before);
  ASSERT_INT_EQ(ctx->mem_used, memory_before);
  ASSERT_INT_EQ(ctx->buffer_write_count, writes_before);
  PolyCtxStats stats_after = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats_after), 0);
  ASSERT_INT_EQ(stats_after.buffer_entries, stats_before.buffer_entries);
  ASSERT_PTR_EQ(poly_buffer_get(ctx, a_storage), a_head_before);
  ASSERT_PTR_EQ(poly_buffer_get(ctx, b_storage), b_head_before);
  ASSERT_PTR_EQ(a_head_before->ptr, a_residency_before.ptr);
  ASSERT_PTR_EQ(b_head_before->ptr, b_residency_before.ptr);
  ASSERT_PTR_EQ(a_head_before->src, a_residency_before.src);
  ASSERT_PTR_EQ(b_head_before->src, b_residency_before.src);
  ASSERT_INT_EQ(a_head_before->device, a_residency_before.device);
  ASSERT_INT_EQ(b_head_before->device, b_residency_before.device);
  ASSERT_EQ(a_head_before->valid, a_residency_before.valid);
  ASSERT_EQ(b_head_before->valid, b_residency_before.valid);

  /* The failure remains repeatable and does not grow the long-lived caller
   * context on each packaging attempt. */
  memset(&error, 0, sizeof(error));
  failed = poly_instance_from_bindings(ctx, bindings, 2, entrypoints, 1, NULL, &error);
  if (failed) poly_instance_free(failed);
  ASSERT_EQ(failed, NULL);
  ASSERT_INT_EQ(ctx->n_tensors, tensors_before);
  ASSERT_INT_EQ(ctx->mem_used, memory_before);
  ASSERT_INT_EQ(ctx->buffer_write_count, writes_before);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats_after), 0);
  ASSERT_INT_EQ(stats_after.buffer_entries, stats_before.buffer_entries);
  ASSERT_PTR_EQ(poly_buffer_get(ctx, a_storage), a_head_before);
  ASSERT_PTR_EQ(poly_buffer_get(ctx, b_storage), b_head_before);
  ASSERT_PTR_EQ(a_head_before->ptr, a_residency_before.ptr);
  ASSERT_PTR_EQ(b_head_before->ptr, b_residency_before.ptr);
  ASSERT_PTR_EQ(a_head_before->src, a_residency_before.src);
  ASSERT_PTR_EQ(b_head_before->src, b_residency_before.src);
  ASSERT_INT_EQ(a_head_before->device, a_residency_before.device);
  ASSERT_INT_EQ(b_head_before->device, b_residency_before.device);
  ASSERT_EQ(a_head_before->valid, a_residency_before.valid);
  ASSERT_EQ(b_head_before->valid, b_residency_before.valid);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, named_value_aliases_share_one_runtime_storage) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  PolyDevice device = test_ctx_execution_device(ctx);
  PolyTensor *a = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  PolyTensor *b = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  float av[] = {1.0f, 2.0f};
  float bv[] = {3.0f, 4.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, poly_tensor_uop_physical(a), av, sizeof(av)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, poly_tensor_uop_physical(b), bv, sizeof(bv)), 0);
  PolyTensor *state = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
  ASSERT_NOT_NULL(state);
  poly_tensor_set_requires_grad(state, true);

  PolyBindingSpec bindings[] = {
      {.name = "encoder.weight", .role = POLY_ROLE_PARAM, .tensor = state},
      {.name = "lm_head.weight", .role = POLY_ROLE_PARAM, .tensor = state},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = state},
  };
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {
      {.name = "forward", .outputs = outputs, .n_outputs = 1},
  };
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 3, entrypoints, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);
  int64_t n0 = 0, n1 = 0;
  float *p0 = poly_instance_buf_data_named(inst, "encoder.weight", &n0);
  float *p1 = poly_instance_buf_data_named(inst, "lm_head.weight", &n1);
  ASSERT_NOT_NULL(p0);
  ASSERT_NOT_NULL(p1);
  ASSERT_PTR_EQ(p0, p1);
  ASSERT_INT_EQ((int)n0, 2);
  ASSERT_INT_EQ((int)n1, 2);
  ASSERT_FLOAT_EQ(p0[0], 4.0f, 1e-6f);
  ASSERT_FLOAT_EQ(p0[1], 6.0f, 1e-6f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, imported_alias_extent_must_match_exact_named_value) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t value_shape[] = {2};
  PolyUOp *value = poly_add(
      ctx, poly_full(ctx, value_shape, 1, 3.0),
      poly_full(ctx, value_shape, 1, 1.0)
  );
  PolyUOp *output = poly_buffer_f32(ctx, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, output, value));
  ASSERT_NOT_NULL(value);
  ASSERT_NOT_NULL(output);
  ASSERT_NOT_NULL(sink);

  PolyIrBufEntry bufs[] = {
      {.name = "a", .role = POLY_IR_ROLE_PARAM, .buffer = value,
       .shape = {2}, .ndim = 1, .trainable = true, .trainable_set = true},
      {.name = "b", .role = POLY_IR_ROLE_PARAM, .buffer = value,
       .shape = {3}, .ndim = 1, .trainable = true, .trainable_set = true},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = output,
       .shape = {2}, .ndim = 1},
  };
  PolyIrEntrypoint entrypoint = {.name = "forward", .sink = sink};
  PolyIrSpec spec = {
      .ctx = ctx,
      .bufs = bufs,
      .n_bufs = 3,
      .entrypoints = &entrypoint,
      .n_entrypoints = 1,
  };
  int ir_len = 0;
  uint8_t *ir = poly_ir_export(&spec, &ir_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_TRUE(ir_len > 0);
  ASSERT_EQ(poly_instance_from_ir(ir, ir_len, NULL, 0), NULL);

  free(ir);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, ambiguous_equal_logical_state_occurrences_fail_closed) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  PolyDevice device = test_ctx_execution_device(ctx);
  PolyTensor *a = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  PolyTensor *b = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  PolyTensor *c = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, device);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  ASSERT_NOT_NULL(c);

  PolyUOp *logical = poly_alu2(
      ctx, POLY_OP_ADD, poly_tensor_uop_logical(a), poly_tensor_uop_logical(b)
  );
  PolyUOp *physical0 = poly_alu2(
      ctx, POLY_OP_ADD, poly_tensor_uop_physical(a), poly_tensor_uop_physical(b)
  );
  PolyUOp *physical1 = poly_alu2(
      ctx, POLY_OP_ADD, poly_tensor_uop_physical(a), poly_tensor_uop_physical(c)
  );
  ASSERT_NOT_NULL(logical);
  ASSERT_NOT_NULL(physical0);
  ASSERT_NOT_NULL(physical1);
  ASSERT_TRUE(physical0 != physical1);
  PolyTensor *state0 = poly_tensor_create_with_roots(
      ctx, logical, physical0, POLY_TENSOR_VALUE, device
  );
  PolyTensor *state1 = poly_tensor_create_with_roots(
      ctx, logical, physical1, POLY_TENSOR_VALUE, device
  );
  PolyTensor *output = poly_tensor_alu2(ctx, POLY_OP_ADD, state0, state1);
  ASSERT_NOT_NULL(state0);
  ASSERT_NOT_NULL(state1);
  ASSERT_NOT_NULL(output);

  PolyBindingSpec bindings[] = {
      {.name = "state0", .role = POLY_ROLE_PARAM, .tensor = state0},
      {.name = "state1", .role = POLY_ROLE_PARAM, .tensor = state1},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = output},
  };
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {
      {.name = "forward", .outputs = outputs, .n_outputs = 1},
  };
  PolyInstanceError error = {0};
  PolyInstance *inst = poly_instance_from_bindings(
      ctx, bindings, 3, entrypoints, 1, NULL, &error
  );
  ASSERT_EQ(inst, NULL);
  ASSERT_TRUE(strstr(error.message, "ambiguous state occurrence") != NULL);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(state0), logical);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(state0), physical0);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(state1), logical);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(state1), physical1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_state_aliases_share_storage) {
  PolyCtx *ctx = poly_ctx_new();
  float init[] = {3.0f, 4.0f};
  int64_t shape[] = {2};
  PolyTensor *state = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(state);
  ASSERT_INT_EQ(
      poly_buffer_write(ctx, poly_tensor_uop_physical(state), init, sizeof(init)), 0
  );

  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_state(inst, "encoder.weight", state, 0), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_state(inst, "lm_head.weight", state, 0), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", state), POLY_STATUS_OK);
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", NULL, 0, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  int64_t n0 = 0, n1 = 0;
  float *a = poly_instance_buf_data_named(inst, "encoder.weight", &n0);
  float *b = poly_instance_buf_data_named(inst, "lm_head.weight", &n1);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  ASSERT_PTR_EQ(a, b);
  ASSERT_INT_EQ((int)n0, 2);
  ASSERT_INT_EQ((int)n1, 2);
  ASSERT_FLOAT_EQ(a[0], 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(a[1], 4.0f, 1e-6f);

  /* Pinned tinygrad state traversal emits both names while preserving the
   * shared Tensor object (nn/state.py:87-110).  Polygrad's portable spelling
   * is two interface rows referencing the same serialized logical BUFFER. */
  int ir_len = 0;
  int weights_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_NOT_NULL(weights);
  PolyInstance *restored = poly_instance_from_ir(ir, ir_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);

  int64_t rn0 = 0, rn1 = 0;
  float *ra = poly_instance_buf_data_named(restored, "encoder.weight", &rn0);
  float *rb = poly_instance_buf_data_named(restored, "lm_head.weight", &rn1);
  ASSERT_NOT_NULL(ra);
  ASSERT_NOT_NULL(rb);
  ASSERT_PTR_EQ(ra, rb);
  ASSERT_INT_EQ((int)rn0, 2);
  ASSERT_INT_EQ((int)rn1, 2);
  ASSERT_FLOAT_EQ(ra[0], 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(ra[1], 4.0f, 1e-6f);

  ASSERT_INT_EQ(poly_instance_forward(restored, NULL, 0), 0);
  float *restored_output = poly_instance_buf_data_named(restored, "output", NULL);
  ASSERT_NOT_NULL(restored_output);
  ASSERT_FLOAT_EQ(restored_output[0], 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(restored_output[1], 4.0f, 1e-6f);

  poly_instance_free(restored);
  free(weights);
  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, full_reshape_alias_round_trips_as_one_resource) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t flat_shape[] = {4};
  int64_t view_shape[] = {2, 2};
  float init[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyTensor *base =
      poly_tensor_empty(ctx, POLY_FLOAT32, flat_shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(base);
  ASSERT_INT_EQ(
      poly_buffer_write(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(base)),
          init, sizeof(init)
      ),
      0
  );
  PolyTensor *view = poly_tensor_reshape(ctx, base, view_shape, 2);
  PolyTensor *view_flat = poly_tensor_reshape(ctx, view, flat_shape, 1);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, base, view_flat);
  ASSERT_NOT_NULL(view);
  ASSERT_NOT_NULL(view_flat);
  ASSERT_NOT_NULL(out);

  PolyBindingSpec bindings[] = {
      {.name = "base", .role = POLY_ROLE_PARAM, .tensor = base},
      {.name = "view", .role = POLY_ROLE_PARAM, .tensor = view},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {{
      .name = "forward",
      .outputs = outputs,
      .n_outputs = 1,
  }};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 3, entrypoints, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);

  int base_idx = test_find_instance_buf(inst, "base");
  int view_idx = test_find_instance_buf(inst, "view");
  ASSERT_TRUE(base_idx >= 0);
  ASSERT_TRUE(view_idx >= 0);
  ASSERT_PTR_EQ(
      poly_instance_buf_data_named(inst, "base", NULL),
      poly_instance_buf_data_named(inst, "view", NULL)
  );
  int64_t shape_out[8] = {0};
  ASSERT_INT_EQ(poly_instance_buf_shape(inst, base_idx, shape_out, 8), 1);
  ASSERT_INT_EQ(shape_out[0], 4);
  ASSERT_INT_EQ(poly_instance_buf_shape(inst, view_idx, shape_out, 8), 2);
  ASSERT_INT_EQ(shape_out[0], 2);
  ASSERT_INT_EQ(shape_out[1], 2);

  int ir_len = 0;
  int weights_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_NOT_NULL(weights);
  PolyInstance *restored = poly_instance_from_ir(ir, ir_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);
  ASSERT_PTR_EQ(
      poly_instance_buf_data_named(restored, "base", NULL),
      poly_instance_buf_data_named(restored, "view", NULL)
  );
  base_idx = test_find_instance_buf(restored, "base");
  view_idx = test_find_instance_buf(restored, "view");
  ASSERT_INT_EQ(poly_instance_buf_shape(restored, base_idx, shape_out, 8), 1);
  ASSERT_INT_EQ(shape_out[0], 4);
  ASSERT_INT_EQ(poly_instance_buf_shape(restored, view_idx, shape_out, 8), 2);
  ASSERT_INT_EQ(shape_out[0], 2);
  ASSERT_INT_EQ(shape_out[1], 2);
  ASSERT_INT_EQ(poly_instance_forward(restored, NULL, 0), 0);
  float *result = poly_instance_buf_data_named(restored, "output", NULL);
  ASSERT_NOT_NULL(result);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(result[i], 2.0f * init[i], 1e-6f);

  poly_instance_free(restored);
  free(weights);
  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, readonly_partial_view_round_trips_through_base_resource) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {4};
  int64_t pairs[][2] = {{1, 3}};
  float init[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyTensor *base = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(base);
  ASSERT_INT_EQ(
      poly_buffer_write(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(base)),
          init, sizeof(init)
      ),
      0
  );
  PolyTensor *partial = poly_tensor_shrink(ctx, base, pairs, 1);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, partial, partial);
  ASSERT_NOT_NULL(partial);
  ASSERT_NOT_NULL(out);

  PolyBindingSpec bindings[] = {
      {.name = "base", .role = POLY_ROLE_PARAM, .tensor = base},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {{
      .name = "forward",
      .outputs = outputs,
      .n_outputs = 1,
  }};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 2, entrypoints, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);

  int ir_len = 0;
  int weights_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_NOT_NULL(weights);
  PolyInstance *restored = poly_instance_from_ir(ir, ir_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);
  ASSERT_INT_EQ(poly_instance_forward(restored, NULL, 0), 0);
  int64_t numel = 0;
  float *result = poly_instance_buf_data_named(restored, "output", &numel);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(numel, 2);
  ASSERT_FLOAT_EQ(result[0], 4.0f, 1e-6f);
  ASSERT_FLOAT_EQ(result[1], 6.0f, 1e-6f);

  poly_instance_free(restored);
  free(weights);
  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_canonical_names_round_trip_through_ir) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {2};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_INT_EQ(poly_instance_scope_push(inst, "layers.%d", 0), POLY_STATUS_OK);
  PolyTensor *w = poly_instance_param(inst, "weight", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(w);
  ASSERT_INT_EQ(poly_instance_scope_pop(inst), POLY_STATUS_OK);

  PolyTensor *logits = poly_tensor_alu2(ctx, POLY_OP_MUL, x, w);
  ASSERT_NOT_NULL(logits);
  ASSERT_INT_EQ(poly_instance_output(inst, "logits", logits), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"logits"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "layers.0.weight");

  int ir_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_TRUE(ir_len > 0);

  PolyInstance *inst2 = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst2);
  ASSERT_INT_EQ(poly_instance_param_count(inst2), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst2, 0), "layers.0.weight");

  int found_x = 0, found_logits = 0;
  for (int i = 0; i < poly_instance_buf_count(inst2); i++) {
    const char *name = poly_instance_buf_name(inst2, i);
    if (name && strcmp(name, "x") == 0) found_x = 1;
    if (name && strcmp(name, "logits") == 0) found_logits = 1;
  }
  ASSERT_TRUE(found_x);
  ASSERT_TRUE(found_logits);

  poly_instance_free(inst2);
  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, from_binding_arrays_forward_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t shape[] = {2};
  float w_init[] = {2.0f, 3.0f};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *w = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);
  ASSERT_INT_EQ(
      poly_buffer_write(ctx, poly_tensor_uop_physical(w), w_init, sizeof(w_init)), 0
  );

  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_MUL, x, w);
  ASSERT_NOT_NULL(out);

  const char *binding_names[] = {"x", "w", "output"};
  int binding_roles[] = {POLY_ROLE_INPUT, POLY_ROLE_PARAM, POLY_ROLE_OUTPUT};
  PolyTensor *binding_tensors[] = {x, w, out};
  uint32_t binding_flags[] = {0, 0, 0};

  const char *entry_names[] = {"forward"};
  const char *entry_inputs[] = {"x"};
  int entry_input_counts[] = {1};
  const char *entry_outputs[] = {"output"};
  int entry_output_counts[] = {1};
  const char *entry_objectives[] = {NULL};
  uint32_t entry_flags[] = {0};

  PolyInstanceError err = {0};
  PolyInstance *inst = poly_instance_from_binding_arrays(
      ctx, binding_names, binding_roles, binding_tensors, binding_flags, 3, entry_names,
      entry_inputs, entry_input_counts, entry_outputs, entry_output_counts, entry_objectives,
      entry_flags, 1, NULL, &err
  );
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_ctx_named_count(ctx), 0);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "w");

  float x_data[] = {10.0f, 20.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);

  int64_t numel = 0;
  float *y = poly_instance_buf_data_named(inst, "output", &numel);
  ASSERT_NOT_NULL(y);
  ASSERT_INT_EQ((int)numel, 2);
  ASSERT_FLOAT_EQ(y[0], 20.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y[1], 60.0f, 1e-5f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, from_bindings_snapshots_lazy_host_parameter) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t shape[] = {2};
  float w_init[] = {2.0f, 3.0f};
  PolyTensor *host_w =
      poly_tensor_from_host(ctx, w_init, sizeof(w_init), POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(host_w);
  PolyTensor *w = poly_tensor_to_device(ctx, host_w, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(w);
  ASSERT_FALSE(poly_uop_has_buffer_identity(poly_tensor_uop_physical(w)));

  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_MUL, x, w);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(out);

  const char *binding_names[] = {"x", "w", "output"};
  int binding_roles[] = {POLY_ROLE_INPUT, POLY_ROLE_PARAM, POLY_ROLE_OUTPUT};
  PolyTensor *binding_tensors[] = {x, w, out};
  uint32_t binding_flags[] = {0, 0, 0};
  const char *entry_names[] = {"forward"};
  const char *entry_inputs[] = {"x"};
  int entry_input_counts[] = {1};
  const char *entry_outputs[] = {"output"};
  int entry_output_counts[] = {1};
  const char *entry_objectives[] = {NULL};
  uint32_t entry_flags[] = {0};

  PolyInstanceError err = {0};
  PolyInstance *inst = poly_instance_from_binding_arrays(
      ctx, binding_names, binding_roles, binding_tensors, binding_flags, 3, entry_names,
      entry_inputs, entry_input_counts, entry_outputs, entry_output_counts, entry_objectives,
      entry_flags, 1, NULL, &err
  );
  ASSERT_NOT_NULL(inst);

  float x_data[] = {10.0f, 20.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  int64_t numel = 0;
  float *y = poly_instance_buf_data_named(inst, "output", &numel);
  ASSERT_NOT_NULL(y);
  ASSERT_INT_EQ((int)numel, 2);
  ASSERT_FLOAT_EQ(y[0], 20.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y[1], 60.0f, 1e-5f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, from_bindings_snapshots_weak_scalar_as_strong_storage) {
  /* Tinygrad 2026-08-22/a9069c177a9d UOp.new_buffer rejects weak storage,
   * while UOp.empty_like commits an inferred weak value through strong_dtype
   * (uop/ops.py:814-827). Instance activation is the equivalent persistent
   * storage boundary for a named scalar initializer. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int weakfloat = poly_dtype_id_by_name("weakfloat");
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, NULL, 0, POLY_DEVICE_CPU);
  PolyTensor *w =
      poly_tensor_const_float_by_id(ctx, 3.0, weakfloat, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_MUL, x, w);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);
  ASSERT_NOT_NULL(out);

  /* A prior retired Tensor makes the next allocation a collection safe point.
   * The build-time named snapshot must stay owned until Instance adopts it. */
  PolyTensor *retired = poly_tensor_empty(ctx, POLY_FLOAT32, NULL, 0, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(retired);
  poly_tensor_release(retired);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "w", .role = POLY_ROLE_PARAM, .tensor = w},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {{
      .name = "forward",
      .inputs = inputs,
      .n_inputs = 1,
      .outputs = outputs,
      .n_outputs = 1,
  }};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 3, entrypoints, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_param_dtype_id(inst, 0), poly_dtype_id_by_name("float32"));
  ASSERT_FLOAT_EQ(poly_instance_param_data(inst, 0, NULL)[0], 3.0f, 0.0f);

  int ir_len = 0, weights_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_NOT_NULL(weights);
  PolyInstance *restored = poly_instance_from_ir(ir, ir_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);

  float xv = 2.0f;
  PolyIOBinding io[] = {POLY_IO_BINDING_BYTES("x", &xv, sizeof(xv), POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(restored, io, 1), 0);
  ASSERT_FLOAT_EQ(
      poly_instance_buf_data_named(restored, "output", NULL)[0], 6.0f, 0.0f
  );
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  int64_t numel = 0;
  float *y = poly_instance_buf_data_named(inst, "output", &numel);
  ASSERT_NOT_NULL(y);
  ASSERT_INT_EQ(numel, 1);
  ASSERT_FLOAT_EQ(y[0], 6.0f, 0.0f);

  poly_instance_free(restored);
  free(weights);
  free(ir);
  poly_instance_free(inst);
  poly_tensor_release(out);
  poly_tensor_release(w);
  poly_tensor_release(x);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, from_binding_arrays_train_after_set_device_auto_updates_param) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t matrix_shape[] = {1, 1};
  float w_init[] = {1.0f};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, matrix_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *y = poly_tensor_empty(ctx, POLY_FLOAT32, matrix_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *w = poly_tensor_empty(ctx, POLY_FLOAT32, matrix_shape, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(y);
  ASSERT_NOT_NULL(w);
  ASSERT_INT_EQ(
      poly_buffer_write(ctx, poly_tensor_uop_physical(w), w_init, sizeof(w_init)), 0
  );

  PolyTensor *pred = poly_tensor_dot(ctx, x, w);
  ASSERT_NOT_NULL(pred);

  PolyTensor *diff = poly_tensor_alu2(ctx, POLY_OP_SUB, pred, y);
  PolyTensor *loss = poly_tensor_alu2(ctx, POLY_OP_MUL, diff, diff);
  int64_t loss_axes[] = {0, 1};
  loss = poly_tensor_sum(ctx, loss, loss_axes, 2, false);
  PolyTensor *count = poly_tensor_const_float_by_id(
      ctx, 1.0, poly_dtype_id_by_name("float32"), POLY_DEVICE_CPU
  );
  loss = poly_tensor_div(ctx, loss, count);
  ASSERT_NOT_NULL(loss);

  const char *binding_names[] = {"fit_x", "fit_y", "fit_w", "fit_out", "loss"};
  int binding_roles[] = {
      POLY_ROLE_INPUT,
      POLY_ROLE_TARGET,
      POLY_ROLE_PARAM,
      POLY_ROLE_OUTPUT,
      POLY_ROLE_OUTPUT,
  };
  PolyTensor *binding_tensors[] = {x, y, w, pred, loss};
  uint32_t binding_flags[] = {0, 0, 0, 0, 0};

  const char *entry_names[] = {"forward", "loss"};
  const char *entry_inputs[] = {"fit_x", "fit_x", "fit_y"};
  int entry_input_counts[] = {1, 2};
  const char *entry_outputs[] = {"fit_out", "loss"};
  int entry_output_counts[] = {1, 1};
  const char *entry_objectives[] = {NULL, "loss"};
  uint32_t entry_flags[] = {0, 0};

  PolyInstanceError err = {0};
  PolyInstance *inst = poly_instance_from_binding_arrays(
      ctx, binding_names, binding_roles, binding_tensors, binding_flags, 5, entry_names,
      entry_inputs, entry_input_counts, entry_outputs, entry_output_counts, entry_objectives,
      entry_flags, 2, NULL, &err
  );
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_AUTO), 0);
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f), 0
  );

  float x_data[] = {1.0f};
  float y_data[] = {3.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("fit_x", x_data, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("fit_y", y_data, POLY_FLOAT32)};

  float first = 0.0f;
  float last = 0.0f;
  for (int step = 0; step < 4; step++) {
    float loss_out = 0.0f;
    ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss_out), 0);
    if (step == 0) first = loss_out;
    last = loss_out;
  }

  int64_t numel = 0;
  float *w_data = poly_instance_param_data(inst, 0, &numel);
  ASSERT_NOT_NULL(w_data);
  ASSERT_INT_EQ((int)numel, 1);
  ASSERT_TRUE(last < first);
  ASSERT_TRUE(w_data[0] > 1.0f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, from_sinks_wraps_selected_lazy_tensor_graph) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {4};

  PolyUOp *w = poly_buffer_f32(ctx, 4);
  float w_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
  poly_buffer_set(ctx, w, w_data, sizeof(w_data), POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(poly_register_existing_buffer(ctx, POLY_ROLE_PARAM, w, shape, 1, "w", true));

  PolyUOp *x = poly_register_buffer_by_id(
      ctx, POLY_ROLE_INPUT, poly_dtype_id_by_name("float32"), shape, 1, "x"
  );
  PolyUOp *out = poly_register_buffer_by_id(
      ctx, POLY_ROLE_OUTPUT, poly_dtype_id_by_name("float32"), shape, 1, "output"
  );
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(out);

  /* A stale registered entrypoint should not leak into this package when the
   * frontend asks for a selected sink set. This is what lets multiple lazy
   * models share a ctx but export one ABI at a time. */
  PolyUOp *stale = poly_register_buffer_by_id(
      ctx, POLY_ROLE_OUTPUT, poly_dtype_id_by_name("float32"), shape, 1, "stale"
  );
  poly_register_entrypoint(
      ctx, "stale", poly_sink1(ctx, poly_store_val(ctx, stale, poly_alu2(ctx, POLY_OP_ADD, x, w)))
  );

  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, x, w);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, prod));
  const char *names[] = {"forward"};
  PolyUOp *sinks[] = {sink};
  PolyInstance *inst = poly_instance_from_sinks(ctx, names, sinks, 1);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "w");
  ASSERT_INT_EQ(poly_instance_buf_count(inst), 3);
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    ASSERT_TRUE(strcmp(poly_instance_buf_name(inst, i), "stale") != 0);

  float x_data[] = {10.0f, 10.0f, 10.0f, 10.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  int out_idx = -1;
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    if (strcmp(poly_instance_buf_name(inst, i), "output") == 0) out_idx = i;
  ASSERT_TRUE(out_idx >= 0);
  int64_t numel = 0;
  float *out_data = poly_instance_buf_data(inst, out_idx, &numel);
  ASSERT_INT_EQ((int)numel, 4);
  ASSERT_FLOAT_EQ(out_data[0], 20.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[1], 30.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[2], 40.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[3], 50.0f, 1e-5f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_build_places_preserved_logical_after_output_realize) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};

  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);

  float x_initial[] = {1.0f, 2.0f};
  float w_data[] = {3.0f, 4.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, poly_tensor_uop_physical(x), x_initial, sizeof(x_initial)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, poly_tensor_uop_physical(w), w_data, sizeof(w_data)), 0);

  /* Pinned Tensor ALU composes the current Tensor.uop operands.  Use the C
   * Tensor boundary so this Instance captures the retained logical graph and
   * the eager physical template before realization. */
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, w);
  ASSERT_NOT_NULL(out);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(out));

  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_BUILT);

  float runtime_x[] = {5.0f, 6.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", runtime_x, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  int64_t n = 0;
  float *result = poly_instance_buf_data_named(inst, "output", &n);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ((int)n, 2);
  ASSERT_FLOAT_EQ(result[0], 8.0f, 1e-5f);
  ASSERT_FLOAT_EQ(result[1], 10.0f, 1e-5f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, param_enumeration) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "w");

  int64_t shape[8];
  int ndim = poly_instance_param_shape(inst, 0, shape, 8);
  ASSERT_INT_EQ(ndim, 1);
  ASSERT_INT_EQ((int)shape[0], 4);

  int64_t numel;
  float *data = poly_instance_param_data(inst, 0, &numel);
  ASSERT_NOT_NULL(data);
  ASSERT_INT_EQ((int)numel, 4);

  /* Params should be zero-initialized */
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(data[i] == 0.0f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, weights_round_trip) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* Set param values */
  int64_t numel;
  float *w = poly_instance_param_data(inst, 0, &numel);
  w[0] = 1.0f;
  w[1] = 2.0f;
  w[2] = 3.0f;
  w[3] = 4.0f;

  /* Export */
  int st_len = 0;
  uint8_t *st = poly_instance_export_weights(inst, &st_len);
  ASSERT_NOT_NULL(st);
  ASSERT_TRUE(st_len > 0);

  /* Create new instance and import */
  PolyInstance *inst2 = poly_instance_from_ir(ir, ir_len, st, st_len);
  ASSERT_NOT_NULL(inst2);

  float *w2 = poly_instance_param_data(inst2, 0, &numel);
  ASSERT_TRUE(fabsf(w2[0] - 1.0f) < 1e-6f);
  ASSERT_TRUE(fabsf(w2[1] - 2.0f) < 1e-6f);
  ASSERT_TRUE(fabsf(w2[2] - 3.0f) < 1e-6f);
  ASSERT_TRUE(fabsf(w2[3] - 4.0f) < 1e-6f);

  poly_instance_free(inst);
  poly_instance_free(inst2);
  free(ir);
  free(st);
  PASS();
}

TEST(instance, export_ir_round_trip) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* Re-export IR */
  int ir2_len = 0;
  uint8_t *ir2 = poly_instance_export_ir(inst, &ir2_len);
  ASSERT_NOT_NULL(ir2);
  ASSERT_TRUE(ir2_len > 0);

  /* Create new instance from re-exported IR */
  PolyInstance *inst2 = poly_instance_from_ir(ir2, ir2_len, NULL, 0);
  ASSERT_NOT_NULL(inst2);
  ASSERT_INT_EQ(poly_instance_buf_count(inst2), 3);
  ASSERT_STR_EQ(poly_instance_buf_name(inst2, 0), "a");

  poly_instance_free(inst);
  poly_instance_free(inst2);
  free(ir);
  free(ir2);
  PASS();
}

TEST(instance, train_step_sgd) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* Init weights to 1.0 */
  int64_t numel;
  float *w = poly_instance_param_data(inst, 0, &numel);
  for (int i = 0; i < 4; i++)
    w[i] = 1.0f;

  /* Configure SGD */
  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f);

  /* Training data: x=1,1,1,1; y=3,3,3,3 (target: w=3) */
  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
  };

  /* Run several train steps and check loss decreases */
  float prev_loss = 1e10f;
  for (int step = 0; step < 50; step++) {
    float loss;
    int ret = poly_instance_train_step(inst, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    if (step > 0) ASSERT_TRUE(loss <= prev_loss + 1e-6f);
    prev_loss = loss;
  }

  /* Loss should decrease substantially */
  ASSERT_TRUE(prev_loss < 2.0f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

static PolyInstance *make_tied_scalar_train_instance(PolyCtx *ctx) {
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  int64_t shape[] = {1};
  PolyTensor *w = inst ? poly_instance_param(inst, "w0", POLY_FLOAT32, shape, 1) : NULL;
  if (!w || poly_instance_state(inst, "w1", w, 0) != POLY_STATUS_OK) goto fail;
  PolyTensor *twice = poly_tensor_alu2(ctx, POLY_OP_ADD, w, w);
  PolyTensor *loss = twice ? poly_tensor_alu2(ctx, POLY_OP_MUL, twice, twice) : NULL;
  if (!loss || poly_instance_output(inst, "loss", loss) != POLY_STATUS_OK) goto fail;
  const char *outputs[] = {"loss"};
  PolyEntrypointOptions options = {.objective = "loss"};
  if (poly_instance_entrypoint(inst, "loss", NULL, 0, outputs, 1, &options) !=
          POLY_STATUS_OK ||
      poly_instance_build(inst, NULL) != POLY_STATUS_OK)
    goto fail;
  return inst;

fail:
  poly_instance_free(inst);
  return NULL;
}

TEST(instance, tied_parameter_aliases_receive_one_sgd_update) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = make_tied_scalar_train_instance(ctx);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 2);
  int64_t numel = 0;
  float *w0 = poly_instance_param_data(inst, 0, &numel);
  float *w1 = poly_instance_param_data(inst, 1, &numel);
  ASSERT_NOT_NULL(w0);
  ASSERT_PTR_EQ(w0, w1);
  w0[0] = 1.0f;
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f), 0
  );
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, NULL, 0, &loss), 0);
  /* Execution may migrate the current residency.  Instance data pointers are
   * borrowed host views and must be reacquired after any run. */
  w0 = poly_instance_param_data(inst, 0, &numel);
  ASSERT_NOT_NULL(w0);
  /* Pinned tinygrad nn/optim.py:13 deduplicates [w,w].  The loss gradient is
   * 8 at w=1, so one lr=0.1 update produces 0.2, not the pre-fix double-update
   * result 0.04. */
  ASSERT_FLOAT_EQ(w0[0], 0.2f, 1e-6f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, tied_parameter_aliases_share_one_adam_state_family) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = make_tied_scalar_train_instance(ctx);
  ASSERT_NOT_NULL(inst);
  int64_t numel = 0;
  float *w0 = poly_instance_param_data(inst, 0, &numel);
  ASSERT_NOT_NULL(w0);
  w0[0] = 1.0f;
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(inst, POLY_OPTIM_ADAM, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f), 0
  );
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, NULL, 0, &loss), 0);
  w0 = poly_instance_param_data(inst, 0, &numel);
  ASSERT_NOT_NULL(w0);
  ASSERT_FLOAT_EQ(w0[0], 0.9f, 1e-5f);
  int optimizer_state = 0;
  bool has_w0_m = false, has_w0_v = false, has_w1_state = false;
  for (int i = 0; i < poly_instance_buf_count(inst); i++) {
    const char *name = poly_instance_buf_name(inst, i);
    if (!name || !strstr(name, "optim.")) continue;
    optimizer_state++;
    if (strcmp(name, "optim.adam.m.w0") == 0) has_w0_m = true;
    if (strcmp(name, "optim.adam.v.w0") == 0) has_w0_v = true;
    if (strstr(name, ".w1") != NULL) has_w1_state = true;
  }
  ASSERT_INT_EQ(optimizer_state, 4); /* b1_t, b2_t, m.w0, v.w0 */
  ASSERT_TRUE(has_w0_m);
  ASSERT_TRUE(has_w0_v);
  ASSERT_FALSE(has_w1_state);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, train_step_rewinds_vag_shape_scan_scratch) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  PolyTensor *y = poly_instance_target(inst, "y", POLY_FLOAT32, shape, 1);
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(y);
  ASSERT_NOT_NULL(w);

  PolyTensor *prod = poly_tensor_alu2(ctx, POLY_OP_MUL, w, x);
  ASSERT_NOT_NULL(prod);
  PolyTensor *neg_y = poly_tensor_alu1(ctx, POLY_OP_NEG, y);
  PolyTensor *diff = poly_tensor_alu2(ctx, POLY_OP_ADD, prod, neg_y);
  PolyTensor *sq = poly_tensor_alu2(ctx, POLY_OP_MUL, diff, diff);
  int64_t axes[] = {0};
  PolyTensor *loss_val = poly_tensor_sum(ctx, sq, axes, 1, false);
  PolyTensor *quarter = poly_tensor_const_float_by_id(
      ctx, 0.25, poly_dtype_id_by_name("float32"), poly_ctx_get_preferred_device(ctx)
  );
  PolyTensor *loss = poly_tensor_alu2(ctx, POLY_OP_MUL, loss_val, quarter);
  ASSERT_NOT_NULL(loss);

  ASSERT_INT_EQ(poly_instance_output(inst, "output", prod), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_output(inst, "loss", loss), POLY_STATUS_OK);

  const char *forward_inputs[] = {"x"};
  const char *forward_outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", forward_inputs, 1, forward_outputs, 1, NULL),
      POLY_STATUS_OK
  );

  const char *loss_inputs[] = {"x", "y"};
  const char *loss_outputs[] = {"loss"};
  PolyEntrypointOptions opts = {.objective = "loss"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "loss", loss_inputs, 2, loss_outputs, 1, &opts),
      POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f), 0
  );

  int64_t numel = 0;
  float *w_data = poly_instance_param_data(inst, 0, &numel);
  ASSERT_NOT_NULL(w_data);
  ASSERT_INT_EQ((int)numel, 4);
  for (int i = 0; i < 4; i++)
    w_data[i] = 1.0f;

  float x_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y_data[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("y", y_data, POLY_FLOAT32)};

  size_t scratch_before = poly_arena_used(ctx->scratch);
  float loss_out = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss_out), 0);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);
  ASSERT_TRUE(loss_out > 0.0f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, train_step_adam) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* Init weights */
  int64_t numel;
  float *w = poly_instance_param_data(inst, 0, &numel);
  for (int i = 0; i < 4; i++)
    w[i] = 0.5f;

  /* Configure Adam */
  poly_instance_set_optimizer(inst, POLY_OPTIM_ADAM, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32)};

  float first_loss = -1.0f;
  float prev_loss = 1e10f;
  for (int step = 0; step < 50; step++) {
    float loss;
    int ret = poly_instance_train_step(inst, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    if (step == 0) first_loss = loss;
    prev_loss = loss;
  }

  /* Loss should decrease from initial */
  ASSERT_TRUE(prev_loss < first_loss);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

static int test_find_instance_buf(PolyInstance *inst, const char *name) {
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    if (strcmp(poly_instance_buf_name(inst, i), name) == 0) return i;
  return -1;
}

static bool test_safetensors_has_name(PolySafetensorView *views, int n, const char *name) {
  for (int i = 0; i < n; i++)
    if (strcmp(views[i].name, name) == 0) return true;
  return false;
}

static void test_safetensors_free_views(PolySafetensorView *views, int n, char *metadata) {
  if (views) {
    for (int i = 0; i < n; i++)
      free(views[i].name);
  }
  free(views);
  free(metadata);
}

TEST(instance, adam_optimizer_state_is_named_checkpoint_state) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  int64_t numel;
  float *w = poly_instance_param_data(inst, 0, &numel);
  ASSERT_INT_EQ(numel, 4);
  for (int i = 0; i < 4; i++)
    w[i] = 0.5f;

  ASSERT_INT_EQ(poly_instance_set_optimizer(inst, POLY_OPTIM_ADAM, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f), 0);

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32)};
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss), 0);

  int b1_idx = test_find_instance_buf(inst, "optim.adam.b1_t");
  int b2_idx = test_find_instance_buf(inst, "optim.adam.b2_t");
  int m_idx = test_find_instance_buf(inst, "optim.adam.m.w");
  int v_idx = test_find_instance_buf(inst, "optim.adam.v.w");
  ASSERT_TRUE(b1_idx >= 0);
  ASSERT_TRUE(b2_idx >= 0);
  ASSERT_TRUE(m_idx >= 0);
  ASSERT_TRUE(v_idx >= 0);
  ASSERT_INT_EQ(poly_instance_buf_role(inst, b1_idx), POLY_ROLE_AUX);
  ASSERT_INT_EQ(poly_instance_buf_role(inst, m_idx), POLY_ROLE_AUX);

  int64_t n_b1 = 0;
  float *b1 = poly_instance_buf_data(inst, b1_idx, &n_b1);
  ASSERT_INT_EQ(n_b1, 1);
  ASSERT_FLOAT_EQ(b1[0], 0.9f, 1e-6f);
  int64_t n_m = 0;
  float *m = poly_instance_buf_data(inst, m_idx, &n_m);
  ASSERT_INT_EQ(n_m, 4);
  ASSERT_TRUE(fabsf(m[0]) > 0.0f);

  int weights_len = 0;
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(weights);
  int n_views = 0;
  char *metadata = NULL;
  PolySafetensorView *views = poly_safetensors_decode(weights, weights_len, &n_views, &metadata);
  ASSERT_NOT_NULL(views);
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "w"));
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "optim.adam.b1_t"));
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "optim.adam.b2_t"));
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "optim.adam.m.w"));
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "optim.adam.v.w"));
  test_safetensors_free_views(views, n_views, metadata);

  int model_only_len = 0;
  uint8_t *model_only =
      poly_instance_export_weights_ex(inst, &model_only_len, POLY_EXPORT_WEIGHTS_PARAMS);
  ASSERT_NOT_NULL(model_only);
  n_views = 0;
  metadata = NULL;
  views = poly_safetensors_decode(model_only, model_only_len, &n_views, &metadata);
  ASSERT_NOT_NULL(views);
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "w"));
  ASSERT_FALSE(test_safetensors_has_name(views, n_views, "optim.adam.b1_t"));
  ASSERT_FALSE(test_safetensors_has_name(views, n_views, "optim.adam.m.w"));
  test_safetensors_free_views(views, n_views, metadata);
  free(model_only);

  int ir2_len = 0;
  uint8_t *ir2 = poly_instance_export_ir(inst, &ir2_len);
  ASSERT_NOT_NULL(ir2);
  PolyInstance *restored = poly_instance_from_ir(ir2, ir2_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);
  int rb1_idx = test_find_instance_buf(restored, "optim.adam.b1_t");
  ASSERT_TRUE(rb1_idx >= 0);
  float *rb1 = poly_instance_buf_data(restored, rb1_idx, NULL);
  ASSERT_FLOAT_EQ(rb1[0], 0.9f, 1e-6f);

  ASSERT_INT_EQ(
      poly_instance_set_optimizer(restored, POLY_OPTIM_ADAM, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f), 0
  );
  float source_loss = 0.0f, restored_loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &source_loss), 0);
  ASSERT_INT_EQ(poly_instance_train_step(restored, io, 2, &restored_loss), 0);
  ASSERT_FLOAT_EQ(source_loss, restored_loss, 0.0f);
  float *source_w = poly_instance_param_data(inst, 0, NULL);
  float *restored_w = poly_instance_param_data(restored, 0, NULL);
  ASSERT_NOT_NULL(source_w);
  ASSERT_NOT_NULL(restored_w);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(source_w[i], restored_w[i], 0.0f);
  const char *adam_state_names[] = {
      "optim.adam.b1_t", "optim.adam.b2_t", "optim.adam.m.w", "optim.adam.v.w",
  };
  for (size_t i = 0; i < sizeof(adam_state_names) / sizeof(adam_state_names[0]); i++) {
    int source_idx = test_find_instance_buf(inst, adam_state_names[i]);
    int restored_idx = test_find_instance_buf(restored, adam_state_names[i]);
    ASSERT_TRUE(source_idx >= 0);
    ASSERT_TRUE(restored_idx >= 0);
    int64_t source_n = 0, restored_n = 0;
    float *source_state = poly_instance_buf_data(inst, source_idx, &source_n);
    float *restored_state = poly_instance_buf_data(restored, restored_idx, &restored_n);
    ASSERT_NOT_NULL(source_state);
    ASSERT_NOT_NULL(restored_state);
    ASSERT_INT_EQ(source_n, restored_n);
    for (int64_t j = 0; j < source_n; j++)
      ASSERT_FLOAT_EQ(source_state[j], restored_state[j], 0.0f);
  }
  rb1_idx = test_find_instance_buf(restored, "optim.adam.b1_t");
  rb1 = poly_instance_buf_data(restored, rb1_idx, NULL);
  ASSERT_FLOAT_EQ(rb1[0], 0.81f, 1e-5f);

  poly_instance_free(restored);
  poly_instance_free(inst);
  free(ir2);
  free(weights);
  free(ir);
  PASS();
}

TEST(instance, sgd_momentum_state_is_named_checkpoint_state) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  int64_t numel;
  float *w = poly_instance_param_data(inst, 0, &numel);
  ASSERT_INT_EQ(numel, 4);
  for (int i = 0; i < 4; i++)
    w[i] = 0.5f;

  ASSERT_INT_EQ(
      poly_instance_set_optimizer_ex(
          inst, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, 0.9f, false, false
      ),
      0
  );

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32)};
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss), 0);

  int b_idx = test_find_instance_buf(inst, "optim.sgd.b.w");
  ASSERT_TRUE(b_idx >= 0);
  ASSERT_INT_EQ(poly_instance_buf_role(inst, b_idx), POLY_ROLE_AUX);

  int64_t n_b = 0;
  float *b = poly_instance_buf_data(inst, b_idx, &n_b);
  ASSERT_INT_EQ(n_b, 4);
  ASSERT_TRUE(fabsf(b[0]) > 0.0f);

  /* Make the reuse check strong: imported optimizer state should be consumed
   * as-is, not silently reinitialized to zero on the next train graph build. */
  for (int i = 0; i < 4; i++)
    b[i] = 7.0f;

  int weights_len = 0;
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(weights);
  int n_views = 0;
  char *metadata = NULL;
  PolySafetensorView *views = poly_safetensors_decode(weights, weights_len, &n_views, &metadata);
  ASSERT_NOT_NULL(views);
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "w"));
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "optim.sgd.b.w"));
  test_safetensors_free_views(views, n_views, metadata);

  int model_only_len = 0;
  uint8_t *model_only =
      poly_instance_export_weights_ex(inst, &model_only_len, POLY_EXPORT_WEIGHTS_PARAMS);
  ASSERT_NOT_NULL(model_only);
  n_views = 0;
  metadata = NULL;
  views = poly_safetensors_decode(model_only, model_only_len, &n_views, &metadata);
  ASSERT_NOT_NULL(views);
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "w"));
  ASSERT_FALSE(test_safetensors_has_name(views, n_views, "optim.sgd.b.w"));
  test_safetensors_free_views(views, n_views, metadata);
  free(model_only);

  int ir2_len = 0;
  uint8_t *ir2 = poly_instance_export_ir(inst, &ir2_len);
  ASSERT_NOT_NULL(ir2);
  PolyInstance *restored = poly_instance_from_ir(ir2, ir2_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);
  int rb_idx = test_find_instance_buf(restored, "optim.sgd.b.w");
  ASSERT_TRUE(rb_idx >= 0);
  float *rb = poly_instance_buf_data(restored, rb_idx, NULL);
  ASSERT_FLOAT_EQ(rb[0], 7.0f, 1e-6f);

  ASSERT_INT_EQ(
      poly_instance_set_optimizer_ex(
          restored, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, 0.9f, false, false
      ),
      0
  );
  float source_loss = 0.0f, restored_loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &source_loss), 0);
  ASSERT_INT_EQ(poly_instance_train_step(restored, io, 2, &restored_loss), 0);
  ASSERT_FLOAT_EQ(source_loss, restored_loss, 0.0f);
  float *source_w = poly_instance_param_data(inst, 0, NULL);
  float *restored_w = poly_instance_param_data(restored, 0, NULL);
  ASSERT_NOT_NULL(source_w);
  ASSERT_NOT_NULL(restored_w);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(source_w[i], restored_w[i], 0.0f);
  rb_idx = test_find_instance_buf(restored, "optim.sgd.b.w");
  rb = poly_instance_buf_data(restored, rb_idx, NULL);
  b_idx = test_find_instance_buf(inst, "optim.sgd.b.w");
  b = poly_instance_buf_data(inst, b_idx, NULL);
  ASSERT_NOT_NULL(b);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(b[i], rb[i], 0.0f);
  ASSERT_TRUE(rb[0] > 5.0f);

  poly_instance_free(restored);
  poly_instance_free(inst);
  free(ir2);
  free(weights);
  free(ir);
  PASS();
}

TEST(instance, inline_forward_copies_prefixed_weights) {
  int child_ir_len = 0;
  uint8_t *child_ir = make_train_ir(4, &child_ir_len);
  PolyInstance *child = poly_instance_from_ir(child_ir, child_ir_len, NULL, 0);
  ASSERT_NOT_NULL(child);

  int64_t n;
  float *cw = poly_instance_param_data(child, 0, &n);
  ASSERT_INT_EQ(n, 4);
  for (int i = 0; i < 4; i++)
    cw[i] = (float)(i + 2);

  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *parent = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(parent);
  int64_t s[] = {4};
  PolyTensor *x_tensor = poly_instance_input(parent, "x", POLY_FLOAT32, s, 1);
  ASSERT_NOT_NULL(x_tensor);

  PolyInstanceInlineBinding binds[] = {{.name = "x", .tensor = x_tensor}};
  PolyInstanceInlineOutput outs[2];
  int n_out = 0;
  ASSERT_INT_EQ(
      poly_instance_inline_entrypoint(
          parent, child, "forward", "child.", binds, 1, false, outs, 2, &n_out
      ),
      0
  );
  ASSERT_INT_EQ(n_out, 1);
  ASSERT_STR_EQ(outs[0].name, "output");
  ASSERT_NOT_NULL(outs[0].tensor);
  ASSERT_NOT_NULL(poly_tensor_uop_logical(outs[0].tensor));
  ASSERT_NOT_NULL(poly_tensor_uop_physical(outs[0].tensor));

  PolyTensor *one = poly_tensor_const_float_by_id(
      ctx, 1.0, poly_dtype_id_by_name("float32"), poly_ctx_get_preferred_device(ctx)
  );
  PolyTensor *out_tensor = poly_tensor_alu2(ctx, POLY_OP_ADD, outs[0].tensor, one);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_INT_EQ(poly_instance_output(parent, "output", out_tensor), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(parent, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(parent, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_copy_prefixed_weights(parent, child, "child."), 0);

  float x_data[] = {1, 1, 1, 1};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(parent, io, 1), 0);

  float *out_data = poly_instance_buf_data_named(parent, "output", NULL);
  ASSERT_NOT_NULL(out_data);
  ASSERT_FLOAT_EQ(out_data[0], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[1], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[2], 5.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[3], 6.0f, 1e-5f);

  int child_w_idx = -1;
  for (int i = 0; i < poly_instance_param_count(parent); i++)
    if (strcmp(poly_instance_param_name(parent, i), "child.w") == 0) child_w_idx = i;
  ASSERT_TRUE(child_w_idx >= 0);
  ASSERT_TRUE(!poly_instance_param_trainable(parent, child_w_idx));

  poly_instance_free(parent);
  poly_ctx_destroy(ctx);
  poly_instance_free(child);
  free(child_ir);
  PASS();
}

TEST(instance, inline_frozen_submodel_not_updated_by_parent_train) {
  int child_ir_len = 0;
  uint8_t *child_ir = make_train_ir(4, &child_ir_len);
  PolyInstance *child = poly_instance_from_ir(child_ir, child_ir_len, NULL, 0);
  ASSERT_NOT_NULL(child);
  float *cw = poly_instance_param_data(child, 0, NULL);
  for (int i = 0; i < 4; i++)
    cw[i] = 2.0f;

  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *parent = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(parent);
  int64_t s[] = {4};
  PolyTensor *x_tensor = poly_instance_input(parent, "x", POLY_FLOAT32, s, 1);
  PolyTensor *target_tensor = poly_instance_target(parent, "y", POLY_FLOAT32, s, 1);
  PolyTensor *head_tensor = poly_instance_param(parent, "head", POLY_FLOAT32, s, 1);
  ASSERT_NOT_NULL(x_tensor);
  ASSERT_NOT_NULL(target_tensor);
  ASSERT_NOT_NULL(head_tensor);

  PolyInstanceInlineBinding binds[] = {{.name = "x", .tensor = x_tensor}};
  PolyInstanceInlineOutput outs[2];
  int n_out = 0;
  ASSERT_INT_EQ(
      poly_instance_inline_entrypoint(
          parent, child, "forward", "enc.", binds, 1, false, outs, 2, &n_out
      ),
      0
  );
  ASSERT_INT_EQ(n_out, 1);
  ASSERT_NOT_NULL(outs[0].tensor);
  ASSERT_NOT_NULL(poly_tensor_uop_logical(outs[0].tensor));
  ASSERT_NOT_NULL(poly_tensor_uop_physical(outs[0].tensor));

  PolyTensor *out_tensor =
      poly_tensor_alu2(ctx, POLY_OP_MUL, outs[0].tensor, head_tensor);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_INT_EQ(poly_instance_output(parent, "output", out_tensor), POLY_STATUS_OK);

  PolyTensor *diff = poly_tensor_alu2(ctx, POLY_OP_SUB, out_tensor, target_tensor);
  PolyTensor *sq = poly_tensor_alu2(ctx, POLY_OP_MUL, diff, diff);
  int64_t axes[] = {0};
  PolyTensor *loss_sum = poly_tensor_sum(ctx, sq, axes, 1, false);
  PolyTensor *quarter = poly_tensor_const_float_by_id(
      ctx, 0.25, poly_dtype_id_by_name("float32"), poly_ctx_get_preferred_device(ctx)
  );
  PolyTensor *loss_tensor = poly_tensor_alu2(ctx, POLY_OP_MUL, loss_sum, quarter);
  ASSERT_NOT_NULL(loss_tensor);
  ASSERT_INT_EQ(poly_instance_output(parent, "loss", loss_tensor), POLY_STATUS_OK);

  const char *forward_inputs[] = {"x"};
  const char *forward_outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(
          parent, "forward", forward_inputs, 1, forward_outputs, 1, NULL
      ),
      POLY_STATUS_OK
  );
  const char *loss_inputs[] = {"x", "y"};
  const char *loss_outputs[] = {"loss"};
  PolyEntrypointOptions loss_opts = {.objective = "loss"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(parent, "loss", loss_inputs, 2, loss_outputs, 1, &loss_opts),
      POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(parent, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_copy_prefixed_weights(parent, child, "enc."), 0);

  for (int i = 0; i < poly_instance_param_count(parent); i++) {
    float *p = poly_instance_param_data(parent, i, NULL);
    if (strcmp(poly_instance_param_name(parent, i), "head") == 0) {
      for (int j = 0; j < 4; j++)
        p[j] = 1.0f;
    }
  }

  float x_data[] = {1, 1, 1, 1};
  float y_data[] = {6, 6, 6, 6};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("y", y_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(parent, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f), 0
  );

  float first_loss = 0.0f, last_loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(parent, io, 2, &first_loss), 0);
  for (int step = 0; step < 10; step++)
    ASSERT_INT_EQ(poly_instance_train_step(parent, io, 2, &last_loss), 0);
  ASSERT_TRUE(last_loss < first_loss);

  int enc_idx = -1, head_idx = -1;
  for (int i = 0; i < poly_instance_param_count(parent); i++) {
    if (strcmp(poly_instance_param_name(parent, i), "enc.w") == 0) enc_idx = i;
    if (strcmp(poly_instance_param_name(parent, i), "head") == 0) head_idx = i;
  }
  ASSERT_TRUE(enc_idx >= 0);
  ASSERT_TRUE(head_idx >= 0);
  ASSERT_TRUE(!poly_instance_param_trainable(parent, enc_idx));
  ASSERT_TRUE(poly_instance_param_trainable(parent, head_idx));

  float *enc_w = poly_instance_param_data(parent, enc_idx, NULL);
  float *head_w = poly_instance_param_data(parent, head_idx, NULL);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(enc_w[i], 2.0f, 1e-6f);
  ASSERT_TRUE(head_w[0] > 1.0f);

  poly_instance_free(parent);
  poly_ctx_destroy(ctx);
  poly_instance_free(child);
  free(child_ir);
  PASS();
}

TEST(instance, null_safety) {
  ASSERT_INT_EQ(poly_instance_buf_count(NULL), 0);
  ASSERT_INT_EQ(poly_instance_param_count(NULL), 0);
  ASSERT_TRUE(poly_instance_buf_name(NULL, 0) == NULL);
  ASSERT_TRUE(poly_instance_param_name(NULL, 0) == NULL);
  ASSERT_TRUE(poly_instance_param_data(NULL, 0, NULL) == NULL);
  poly_instance_free(NULL); /* should not crash */
  PASS();
}

/* Phase 5: Backend-aware PolyInstance tests */

TEST(instance, call_basic) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("a", a_data, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("b", b_data, POLY_FLOAT32)};

  /* Use poly_instance_call instead of poly_instance_forward */
  int ret = poly_instance_call(inst, "forward", io, 2);
  ASSERT_INT_EQ(ret, 0);

  int64_t numel;
  float *out = poly_instance_buf_data(inst, 2, &numel);
  ASSERT_NOT_NULL(out);
  ASSERT_TRUE(fabsf(out[0] - 11.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[1] - 22.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[2] - 33.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[3] - 44.0f) < 1e-5f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, set_device_interp) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* Switch to interpreter */
  int ret = poly_instance_set_device(inst, POLY_DEVICE_INTERP);
  ASSERT_INT_EQ(ret, 0);

  float a_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float b_data[] = {100.0f, 200.0f, 300.0f, 400.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("a", a_data, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("b", b_data, POLY_FLOAT32)};

  ret = poly_instance_forward(inst, io, 2);
  ASSERT_INT_EQ(ret, 0);

  int64_t numel;
  float *out = poly_instance_buf_data(inst, 2, &numel);
  ASSERT_NOT_NULL(out);
  ASSERT_TRUE(fabsf(out[0] - 105.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[1] - 206.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[2] - 307.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[3] - 408.0f) < 1e-5f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, cpu_vs_interp_forward) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);

  /* Run on CPU */
  PolyInstance *inst_cpu = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst_cpu);

  float a_data[] = {1.5f, 2.5f, 3.5f, 4.5f};
  float b_data[] = {0.1f, 0.2f, 0.3f, 0.4f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("a", a_data, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("b", b_data, POLY_FLOAT32)};

  ASSERT_INT_EQ(poly_instance_forward(inst_cpu, io, 2), 0);
  int64_t numel;
  float *cpu_out = poly_instance_buf_data(inst_cpu, 2, &numel);

  /* Run on INTERP */
  PolyInstance *inst_interp = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst_interp);
  ASSERT_INT_EQ(poly_instance_set_device(inst_interp, POLY_DEVICE_INTERP), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst_interp, io, 2), 0);
  float *interp_out = poly_instance_buf_data(inst_interp, 2, &numel);

  /* Compare outputs */
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(fabsf(cpu_out[i] - interp_out[i]) < 1e-6f);

  poly_instance_free(inst_cpu);
  poly_instance_free(inst_interp);
  free(ir);
  PASS();
}

TEST(instance, cpu_vs_interp_train) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32)};

  /* CPU training */
  PolyInstance *inst_cpu = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst_cpu);
  {
    int64_t n;
    float *w = poly_instance_param_data(inst_cpu, 0, &n);
    for (int i = 0; i < 4; i++)
      w[i] = 1.0f;
  }
  poly_instance_set_optimizer(inst_cpu, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f);

  float cpu_losses[5];
  for (int s = 0; s < 5; s++) {
    ASSERT_INT_EQ(poly_instance_train_step(inst_cpu, io, 2, &cpu_losses[s]), 0);
  }

  /* INTERP training */
  PolyInstance *inst_interp = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst_interp);
  ASSERT_INT_EQ(poly_instance_set_device(inst_interp, POLY_DEVICE_INTERP), 0);
  {
    int64_t n;
    float *w = poly_instance_param_data(inst_interp, 0, &n);
    for (int i = 0; i < 4; i++)
      w[i] = 1.0f;
  }
  poly_instance_set_optimizer(inst_interp, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f);

  float interp_losses[5];
  for (int s = 0; s < 5; s++) {
    ASSERT_INT_EQ(poly_instance_train_step(inst_interp, io, 2, &interp_losses[s]), 0);
  }

  /* Compare loss trajectories */
  for (int s = 0; s < 5; s++)
    ASSERT_TRUE(fabsf(cpu_losses[s] - interp_losses[s]) < 1e-4f);

  /* Both should decrease */
  ASSERT_TRUE(cpu_losses[4] < cpu_losses[0]);

  poly_instance_free(inst_cpu);
  poly_instance_free(inst_interp);
  free(ir);
  PASS();
}

TEST(instance, set_device_roundtrip) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  PolyCtx *ctx = poly_instance_ctx(inst);
  PolyUOp *initial_a = poly_instance_get_buffer(inst, "a");
  PolyUOp *initial_sink = poly_instance_get_sink(inst, "forward");
  ASSERT_NOT_NULL(ctx);
  ASSERT_NOT_NULL(initial_a);
  ASSERT_NOT_NULL(initial_sink);
  PolyDevice initial_device = poly_uop_device(initial_a);
  ASSERT_TRUE(poly_device_can_execute(initial_device));
  ASSERT_FALSE(poly_tensor_root_has_unplaced_buffer(ctx, initial_sink));

  /* Reapplying the same uniform policy is pointer-idempotent. */
  ASSERT_INT_EQ(poly_instance_set_device(inst, initial_device), 0);
  ASSERT_PTR_EQ(poly_instance_get_buffer(inst, "a"), initial_a);
  ASSERT_PTR_EQ(poly_instance_get_sink(inst, "forward"), initial_sink);

  float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("a", a, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("b", b, POLY_FLOAT32)};
  float expected[] = {11.0f, 22.0f, 33.0f, 44.0f};
  int64_t numel;

  /* Run on CPU */
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 2), 0);
  float *out = poly_instance_buf_data(inst, 2, &numel);
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(fabsf(out[i] - expected[i]) < 1e-5f);

  /* Switch to a distinct executable policy and run. */
  PolyDevice other_device =
      initial_device == POLY_DEVICE_INTERP ? POLY_DEVICE_CPU : POLY_DEVICE_INTERP;
  ASSERT_INT_EQ(poly_instance_set_device(inst, other_device), 0);
  ASSERT_INT_EQ(poly_uop_device(poly_instance_get_buffer(inst, "a")), other_device);
  ASSERT_PTR_NEQ(poly_instance_get_sink(inst, "forward"), initial_sink);
  ASSERT_FALSE(
      poly_tensor_root_has_unplaced_buffer(ctx, poly_instance_get_sink(inst, "forward"))
  );
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 2), 0);
  out = poly_instance_buf_data(inst, 2, &numel);
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(fabsf(out[i] - expected[i]) < 1e-5f);

  /* Returning to the original policy reconstructs the exact original graph. */
  ASSERT_INT_EQ(poly_instance_set_device(inst, initial_device), 0);
  ASSERT_PTR_EQ(poly_instance_get_buffer(inst, "a"), initial_a);
  ASSERT_PTR_EQ(poly_instance_get_sink(inst, "forward"), initial_sink);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 2), 0);
  out = poly_instance_buf_data(inst, 2, &numel);
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(fabsf(out[i] - expected[i]) < 1e-5f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, failed_owner_root_preparation_does_not_publish_placement) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  PolyUOp *old_buffer = poly_instance_get_buffer(inst, "a");
  PolyUOp *old_sink = poly_instance_get_sink(inst, "forward");
  ASSERT_NOT_NULL(old_buffer);
  ASSERT_NOT_NULL(old_sink);
  PolyDevice old_device = poly_uop_device(old_buffer);
  PolyDevice target = old_device == POLY_DEVICE_INTERP ? POLY_DEVICE_CPU : POLY_DEVICE_INTERP;

  poly_instance_test_fail_residency_roots_after(0);
  ASSERT_INT_EQ(poly_instance_set_device(inst, target), -1);
  ASSERT_PTR_EQ(poly_instance_get_buffer(inst, "a"), old_buffer);
  ASSERT_PTR_EQ(poly_instance_get_sink(inst, "forward"), old_sink);
  ASSERT_INT_EQ(poly_uop_device(poly_instance_get_buffer(inst, "a")), old_device);

  float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("a", a, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("b", b, POLY_FLOAT32),
  };
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 2), 0);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, failed_owner_root_preparation_does_not_publish_vag) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
  };
  float loss = 0.0f;
  poly_instance_test_fail_residency_roots_after(0);
  ASSERT_INT_EQ(poly_instance_value_and_grad(inst, "loss", io, 2, &loss), -1);
  ASSERT_FALSE(poly_instance_test_has_vag(inst));

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, failed_owner_root_preparation_preserves_optimizer_and_train_state) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f), 0
  );

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
  };
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss), 0);
  ASSERT_TRUE(poly_instance_test_has_train(inst));

  poly_instance_test_fail_residency_roots_after(0);
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(inst, POLY_OPTIM_ADAM, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f), -1
  );
  ASSERT_INT_EQ(poly_instance_test_optimizer_kind(inst), POLY_OPTIM_SGD);
  ASSERT_TRUE(poly_instance_test_has_train(inst));

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, failed_owner_root_preparation_does_not_publish_train_state) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f), 0
  );

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
  };
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_value_and_grad(inst, "loss", io, 2, &loss), 0);
  ASSERT_TRUE(poly_instance_test_has_vag(inst));
  ASSERT_FALSE(poly_instance_test_has_train(inst));

  poly_instance_test_fail_residency_roots_after(0);
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss), -1);
  ASSERT_FALSE(poly_instance_test_has_train(inst));

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, failed_owner_root_preparation_does_not_publish_optimizer_buffers) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(inst, POLY_OPTIM_ADAM, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f), 0
  );

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
  };
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_value_and_grad(inst, "loss", io, 2, &loss), 0);
  int before = poly_instance_buf_count(inst);

  poly_instance_test_fail_residency_roots_after(0);
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss), -1);
  ASSERT_FALSE(poly_instance_test_has_train(inst));
  ASSERT_INT_EQ(poly_instance_buf_count(inst), before);
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    ASSERT_TRUE(strncmp(poly_instance_buf_name(inst, i), "optim.", 6) != 0);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, failed_owner_root_preparation_preserves_trainable_state) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f), 0
  );
  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
  };
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss), 0);
  int w = test_find_instance_buf(inst, "w");
  ASSERT_TRUE(w >= 0);
  ASSERT_TRUE(poly_instance_buf_trainable(inst, w));
  ASSERT_TRUE(poly_instance_test_has_vag(inst));
  ASSERT_TRUE(poly_instance_test_has_train(inst));

  poly_instance_test_fail_residency_roots_after(0);
  ASSERT_INT_EQ(poly_instance_set_buf_trainable(inst, w, false), -1);
  ASSERT_TRUE(poly_instance_buf_trainable(inst, w));
  ASSERT_TRUE(poly_instance_test_has_vag(inst));
  ASSERT_TRUE(poly_instance_test_has_train(inst));

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, portable_activation_starts_from_logical_and_drops_readonly_copy_history) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  int64_t shape[] = {4};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *to_interp = poly_tensor_to_device(ctx, x, POLY_DEVICE_INTERP);
  PolyTensor *roundtrip = poly_tensor_to_device(ctx, to_interp, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, roundtrip);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(to_interp);
  ASSERT_NOT_NULL(roundtrip);
  ASSERT_NOT_NULL(out);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entries[] = {{
      .name = "forward",
      .inputs = inputs,
      .n_inputs = 1,
      .outputs = outputs,
      .n_outputs = 1,
  }};
  PolyInstance *inst = poly_instance_from_bindings(ctx, bindings, 2, entries, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);

  PolyUOp *cpu_sink = poly_instance_get_sink(inst, "forward");
  ASSERT_NOT_NULL(cpu_sink);
  ASSERT_INT_EQ(instance_count_root_op(ctx, cpu_sink, POLY_OP_COPY), 0);
  ASSERT_INT_EQ(poly_uop_device(poly_instance_get_buffer(inst, "x")), POLY_DEVICE_CPU);

  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_INTERP), 0);
  PolyUOp *interp_sink = poly_instance_get_sink(inst, "forward");
  ASSERT_NOT_NULL(interp_sink);
  ASSERT_PTR_NEQ(interp_sink, cpu_sink);
  ASSERT_INT_EQ(instance_count_root_op(ctx, interp_sink, POLY_OP_COPY), 0);
  ASSERT_INT_EQ(poly_uop_device(poly_instance_get_buffer(inst, "x")), POLY_DEVICE_INTERP);
  ASSERT_FALSE(poly_tensor_root_has_unplaced_buffer(ctx, interp_sink));

  float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  int64_t numel = 0;
  float *result = poly_instance_buf_data_named(inst, "output", &numel);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ((int)numel, 4);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(result[i], 2.0f * x_data[i], 1e-5f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, moved_mutation_portable_activation_fails_without_resource_identity) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  int64_t shape[] = {2};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *v = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *moved = poly_tensor_to_device(ctx, x, POLY_DEVICE_INTERP);
  PolyTensor *moved_v = poly_tensor_to_device(ctx, v, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(v);
  ASSERT_NOT_NULL(moved);
  ASSERT_NOT_NULL(moved_v);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, moved, moved_v), moved);
  PolyTensor *moved_back = poly_tensor_to_device(ctx, moved, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, moved_back);
  ASSERT_NOT_NULL(moved_back);
  ASSERT_NOT_NULL(out);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "v", .role = POLY_ROLE_INPUT, .tensor = v},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = out},
  };
  const char *inputs[] = {"x", "v"};
  const char *outputs[] = {"output"};
  PolyEntrypointSpec entries[] = {{
      .name = "forward",
      .inputs = inputs,
      .n_inputs = 2,
      .outputs = outputs,
      .n_outputs = 1,
  }};
  PolyInstanceError error = {0};
  PolyInstance *inst = poly_instance_from_bindings(
      ctx, bindings, 3, entries, 1, NULL, &error
  );
  /* The portable graph aliases the moved occurrence with x. Until logical
   * storage/effect identity is approved, activation must fail before exposing
   * a runnable Instance. Pinned assign topology is tensor.py:230-257. */
  ASSERT_EQ(inst, NULL);
  ASSERT_TRUE(strstr(error.message, "default placement failed") != NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, module_device_map_restarts_from_logical_program) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  int64_t shape[] = {2};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *w0 = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *w1 = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *to_interp = poly_tensor_to_device(ctx, x, POLY_DEVICE_INTERP);
  PolyTensor *roundtrip = poly_tensor_to_device(ctx, to_interp, POLY_DEVICE_CPU);
  PolyTensor *hidden = poly_tensor_alu2(ctx, POLY_OP_ADD, roundtrip, w0);
  PolyTensor *output = poly_tensor_alu2(ctx, POLY_OP_MUL, hidden, w1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w0);
  ASSERT_NOT_NULL(w1);
  ASSERT_NOT_NULL(to_interp);
  ASSERT_NOT_NULL(roundtrip);
  ASSERT_NOT_NULL(hidden);
  ASSERT_NOT_NULL(output);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "layers.0.weight", .role = POLY_ROLE_PARAM, .tensor = w0},
      {.name = "layers.1.weight", .role = POLY_ROLE_PARAM, .tensor = w1},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = output},
  };
  const char *input_names[] = {"x"};
  const char *output_names[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {{
      .name = "forward",
      .inputs = input_names,
      .n_inputs = 1,
      .outputs = output_names,
      .n_outputs = 1,
  }};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 4, entrypoints, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);

  PolyTensor *module0_inputs[] = {x};
  PolyTensor *module1_inputs[] = {hidden};
  PolyInstanceModuleSpec modules[] = {
      {.name = "layers.0", .inputs = module0_inputs, .n_inputs = 1, .output = hidden},
      {.name = "layers.1", .inputs = module1_inputs, .n_inputs = 1, .output = output},
  };
  ASSERT_INT_EQ(poly_instance_define_modules(inst, modules, 2), 0);

  PolyInstanceDeviceMapEntry map[] = {
      {.module = "layers.0", .device = "CPU"},
      {.module = "layers.1", .device = "CPU:1"},
  };
  ASSERT_INT_EQ(poly_instance_set_device_map(inst, map, 2), 0);
  PolyUOp *sink = poly_instance_get_sink(inst, "forward");
  ASSERT_NOT_NULL(sink);
  /* Portable placement starts from logical roots, so historical read-only
   * transports disappear and only the policy's CPU->CPU:1 boundary remains. */
  ASSERT_INT_EQ(instance_count_root_op(ctx, sink, POLY_OP_COPY), 1);
  ASSERT_INT_EQ(instance_count_root_op(ctx, sink, POLY_OP_STORE), 1);
  ASSERT_STR_EQ(poly_uop_device_name(ctx, poly_instance_get_buffer(inst, "x")), "CPU");
  ASSERT_STR_EQ(
      poly_uop_device_name(ctx, poly_instance_get_buffer(inst, "layers.1.weight")), "CPU:1"
  );
  ASSERT_STR_EQ(poly_uop_device_name(ctx, poly_instance_get_buffer(inst, "output")), "CPU:1");

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, explicit_module_device_map_places_named_regions_atomically) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  int64_t shape[] = {2};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *w0 = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *w1 = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *hidden = poly_tensor_alu2(ctx, POLY_OP_ADD, x, w0);
  PolyTensor *output = poly_tensor_alu2(ctx, POLY_OP_MUL, hidden, w1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w0);
  ASSERT_NOT_NULL(w1);
  ASSERT_NOT_NULL(hidden);
  ASSERT_NOT_NULL(output);

  float w0_data[] = {3.0f, 4.0f};
  float w1_data[] = {2.0f, 3.0f};
  ASSERT_INT_EQ(
      poly_buffer_write(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(w0)),
          w0_data, sizeof(w0_data)
      ),
      0
  );
  ASSERT_INT_EQ(
      poly_buffer_write(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(w1)),
          w1_data, sizeof(w1_data)
      ),
      0
  );

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "layers.0.weight", .role = POLY_ROLE_PARAM, .tensor = w0},
      {.name = "layers.1.weight", .role = POLY_ROLE_PARAM, .tensor = w1},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = output},
  };
  const char *input_names[] = {"x"};
  const char *output_names[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {{
      .name = "forward",
      .inputs = input_names,
      .n_inputs = 1,
      .outputs = output_names,
      .n_outputs = 1,
  }};
  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, 4, entrypoints, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);

  PolyTensor *module0_inputs[] = {x};
  PolyTensor *module1_inputs[] = {hidden};
  PolyInstanceModuleSpec modules[] = {
      {.name = "layers.0", .inputs = module0_inputs, .n_inputs = 1, .output = hidden},
      {.name = "layers.1", .inputs = module1_inputs, .n_inputs = 1, .output = output},
  };
  ASSERT_INT_EQ(poly_instance_define_modules(inst, modules, 2), 0);

  /* Exact logical module boundaries are portable program metadata. Device
   * assignments are not: the imported Instance must accept a newly supplied
   * policy against the restored boundaries. */
  int portable_ir_len = 0;
  uint8_t *portable_ir = poly_instance_export_ir(inst, &portable_ir_len);
  int portable_weights_len = 0;
  uint8_t *portable_weights =
      poly_instance_export_weights(inst, &portable_weights_len);
  ASSERT_NOT_NULL(portable_ir);
  ASSERT_NOT_NULL(portable_weights);
  PolyInstance *restored = poly_instance_from_ir(
      portable_ir, portable_ir_len, portable_weights, portable_weights_len
  );
  ASSERT_NOT_NULL(restored);
  PolyInstanceDeviceMapEntry restored_map[] = {
      {.module = "layers.0", .device = "CPU"},
      {.module = "layers.1", .device = "INTERP"},
  };
  ASSERT_INT_EQ(poly_instance_set_device_map(restored, restored_map, 2), 0);
  float restored_x[] = {1.0f, 2.0f};
  PolyIOBinding restored_io[] = {
      POLY_IO_BINDING_ARRAY("x", restored_x, POLY_FLOAT32),
  };
  ASSERT_INT_EQ(poly_instance_forward(restored, restored_io, 1), 0);
  int64_t restored_numel = 0;
  float *restored_output =
      poly_instance_buf_data_named(restored, "output", &restored_numel);
  ASSERT_NOT_NULL(restored_output);
  ASSERT_INT_EQ(restored_numel, 2);
  ASSERT_FLOAT_EQ(restored_output[0], 8.0f, 1e-6f);
  ASSERT_FLOAT_EQ(restored_output[1], 18.0f, 1e-6f);
  poly_instance_free(restored);
  free(portable_weights);
  free(portable_ir);

  /* Explicit placement is replacement from the immutable logical program, never
   * implicit composition over the current placed graph.  Pinned Tensor.to
   * composes COPY only when invoked on the previous result
   * (tinygrad/tensor.py:327-335); Instance policy order is not such an
   * invocation.  Record direct uniform B before applying module policy A. */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_INTERP), 0);
  PolyUOp *direct_uniform_sink = poly_instance_get_sink(inst, "forward");
  PolyUOp *direct_uniform_bindings[4] = {
      poly_instance_get_buffer(inst, "x"),
      poly_instance_get_buffer(inst, "layers.0.weight"),
      poly_instance_get_buffer(inst, "layers.1.weight"),
      poly_instance_get_buffer(inst, "output"),
  };
  ASSERT_NOT_NULL(direct_uniform_sink);
  ASSERT_INT_EQ(instance_count_root_op(ctx, direct_uniform_sink, POLY_OP_COPY), 0);
  ASSERT_INT_EQ(instance_count_root_op(ctx, direct_uniform_sink, POLY_OP_STORE), 1);

  PolyInstanceDeviceMapEntry forward_map[] = {
      {.module = "layers.1", .device = "INTERP"},
      {.module = "layers.0", .device = "CPU"},
  };
  ASSERT_INT_EQ(poly_instance_set_device_map(inst, forward_map, 2), 0);
  PolyUOp *forward_sink = poly_instance_get_sink(inst, "forward");
  ASSERT_NOT_NULL(forward_sink);
  ASSERT_INT_EQ(instance_count_root_op(ctx, forward_sink, POLY_OP_COPY), 1);
  ASSERT_STR_EQ(poly_uop_device_name(ctx, poly_instance_get_buffer(inst, "x")), "CPU");
  ASSERT_STR_EQ(
      poly_uop_device_name(ctx, poly_instance_get_buffer(inst, "layers.0.weight")), "CPU"
  );
  ASSERT_STR_EQ(
      poly_uop_device_name(ctx, poly_instance_get_buffer(inst, "layers.1.weight")), "INTERP"
  );
  ASSERT_STR_EQ(
      poly_uop_device_name(ctx, poly_instance_get_buffer(inst, "output")), "INTERP"
  );

  /* Pinned tinygrad lowers and executes each CALL through its own runner
   * (engine/realize.py:142-180,244-279).  The aggregate schedule device must
   * not erase the CPU -> INTERP boundary retained by explicit placement. */
  PolyVarBinding *forward_vars = NULL;
  int n_forward_vars = 0;
  PolyUOp *forward_linear =
      poly_linear_effect_sink(ctx, forward_sink, &forward_vars, &n_forward_vars);
  ASSERT_NOT_NULL(forward_linear);
  ASSERT_INT_EQ(forward_linear->n_src, 3);
  PolyUOp *forward_compiled = poly_compile_linear(ctx, forward_linear, -1);
  ASSERT_NOT_NULL(forward_compiled);
  ASSERT_INT_EQ(
      poly_device_from_device_uop(poly_uop_device_uop_cached(
          ctx, poly_test_linear_call_body(forward_compiled, 0), NULL)),
      POLY_DEVICE_CPU
  );
  ASSERT_INT_EQ(
      poly_device_from_device_uop(poly_uop_device_uop_cached(
          ctx, poly_test_linear_call_body(forward_compiled, 1), NULL)),
      POLY_DEVICE_INTERP
  );
  ASSERT_INT_EQ(
      poly_device_from_device_uop(poly_uop_device_uop_cached(
          ctx, poly_test_linear_call_body(forward_compiled, 2), NULL)),
      POLY_DEVICE_INTERP
  );
  free(forward_vars);

  /* A -> B must be pointer-identical to direct B because both compile from
   * the same immutable logical program.  The old mutable-template implementation
   * retained A's CPU:1 COPY here. */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_INTERP), 0);
  ASSERT_PTR_EQ(poly_instance_get_sink(inst, "forward"), direct_uniform_sink);
  ASSERT_PTR_EQ(poly_instance_get_buffer(inst, "x"), direct_uniform_bindings[0]);
  ASSERT_PTR_EQ(
      poly_instance_get_buffer(inst, "layers.0.weight"), direct_uniform_bindings[1]
  );
  ASSERT_PTR_EQ(
      poly_instance_get_buffer(inst, "layers.1.weight"), direct_uniform_bindings[2]
  );
  ASSERT_PTR_EQ(poly_instance_get_buffer(inst, "output"), direct_uniform_bindings[3]);
  ASSERT_INT_EQ(
      instance_count_root_op(ctx, poly_instance_get_sink(inst, "forward"), POLY_OP_COPY), 0
  );
  ASSERT_INT_EQ(
      instance_count_root_op(ctx, poly_instance_get_sink(inst, "forward"), POLY_OP_STORE), 1
  );

  /* B -> A must reconstruct the exact direct A graph, not a second spelling
   * contaminated by B. */
  ASSERT_INT_EQ(poly_instance_set_device_map(inst, forward_map, 2), 0);
  ASSERT_PTR_EQ(poly_instance_get_sink(inst, "forward"), forward_sink);

  float x_data[] = {1.0f, 2.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  int64_t numel = 0;
  float *out = poly_instance_buf_data_named(inst, "output", &numel);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ((int)numel, 2);
  ASSERT_FLOAT_EQ(out[0], 8.0f, 1e-6f);
  ASSERT_FLOAT_EQ(out[1], 18.0f, 1e-6f);

  PolyInstanceDeviceMapEntry incomplete[] = {
      {.module = "layers.0", .device = "CPU"},
  };
  PolyUOp *before_failed_sink = poly_instance_get_sink(inst, "forward");
  PolyUOp *before_failed_bindings[4] = {
      poly_instance_get_buffer(inst, "x"),
      poly_instance_get_buffer(inst, "layers.0.weight"),
      poly_instance_get_buffer(inst, "layers.1.weight"),
      poly_instance_get_buffer(inst, "output"),
  };
  PolyDevice before_failed_preferred = poly_ctx_get_preferred_device(ctx);
  ASSERT_INT_EQ(poly_instance_set_device_map(inst, incomplete, 1), -1);
  ASSERT_PTR_EQ(poly_instance_get_sink(inst, "forward"), before_failed_sink);
  ASSERT_PTR_EQ(poly_instance_get_buffer(inst, "x"), before_failed_bindings[0]);
  ASSERT_PTR_EQ(
      poly_instance_get_buffer(inst, "layers.0.weight"), before_failed_bindings[1]
  );
  ASSERT_PTR_EQ(
      poly_instance_get_buffer(inst, "layers.1.weight"), before_failed_bindings[2]
  );
  ASSERT_PTR_EQ(poly_instance_get_buffer(inst, "output"), before_failed_bindings[3]);
  ASSERT_EQ(poly_ctx_get_preferred_device(ctx), before_failed_preferred);

  PolyInstanceDeviceMapEntry unsupported_accelerator[] = {
      {.module = "layers.0", .device = "CUDA:1"},
      {.module = "layers.1", .device = "CUDA:1"},
  };
  ASSERT_INT_EQ(poly_instance_set_device_map(inst, unsupported_accelerator, 2), -1);
  ASSERT_PTR_EQ(poly_instance_get_sink(inst, "forward"), before_failed_sink);
  ASSERT_PTR_EQ(poly_instance_get_buffer(inst, "x"), before_failed_bindings[0]);
  ASSERT_EQ(poly_ctx_get_preferred_device(ctx), before_failed_preferred);

  PolyInstanceDeviceMapEntry reverse_map[] = {
      {.module = "layers.0", .device = "INTERP"},
      {.module = "layers.1", .device = "CPU"},
  };
  ASSERT_INT_EQ(poly_instance_set_device_map(inst, reverse_map, 2), 0);
  PolyUOp *reverse_sink = poly_instance_get_sink(inst, "forward");
  ASSERT_NOT_NULL(reverse_sink);
  ASSERT_PTR_NEQ(reverse_sink, forward_sink);
  ASSERT_INT_EQ(instance_count_root_op(ctx, reverse_sink, POLY_OP_COPY), 1);
  ASSERT_STR_EQ(poly_uop_device_name(ctx, poly_instance_get_buffer(inst, "x")), "INTERP");
  ASSERT_STR_EQ(poly_uop_device_name(ctx, poly_instance_get_buffer(inst, "output")), "CPU");

  PolyVarBinding *reverse_vars = NULL;
  int n_reverse_vars = 0;
  PolyUOp *reverse_linear =
      poly_linear_effect_sink(ctx, reverse_sink, &reverse_vars, &n_reverse_vars);
  ASSERT_NOT_NULL(reverse_linear);
  ASSERT_INT_EQ(reverse_linear->n_src, 3);
  PolyUOp *reverse_compiled = poly_compile_linear(ctx, reverse_linear, -1);
  ASSERT_NOT_NULL(reverse_compiled);
  ASSERT_INT_EQ(
      poly_device_from_device_uop(poly_uop_device_uop_cached(
          ctx, poly_test_linear_call_body(reverse_compiled, 0), NULL)),
      POLY_DEVICE_INTERP
  );
  ASSERT_INT_EQ(
      poly_device_from_device_uop(poly_uop_device_uop_cached(
          ctx, poly_test_linear_call_body(reverse_compiled, 1), NULL)),
      POLY_DEVICE_CPU
  );
  ASSERT_INT_EQ(
      poly_device_from_device_uop(poly_uop_device_uop_cached(
          ctx, poly_test_linear_call_body(reverse_compiled, 2), NULL)),
      POLY_DEVICE_CPU
  );
  free(reverse_vars);

  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  out = poly_instance_buf_data_named(inst, "output", &numel);
  ASSERT_FLOAT_EQ(out[0], 8.0f, 1e-6f);
  ASSERT_FLOAT_EQ(out[1], 18.0f, 1e-6f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, set_device_unsupported) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* CUDA: fails if not available, succeeds if available -- both are valid */
#ifdef POLY_HAS_CUDA
  if (!poly_cuda_available()) ASSERT_TRUE(poly_instance_set_device(inst, POLY_DEVICE_CUDA) < 0);
#else
  ASSERT_TRUE(poly_instance_set_device(inst, POLY_DEVICE_CUDA) < 0);
#endif

  /* set_device(NULL) */
  ASSERT_TRUE(poly_instance_set_device(NULL, POLY_DEVICE_CPU) < 0);

  poly_instance_free(inst);
  free(ir);
  PASS();
}
