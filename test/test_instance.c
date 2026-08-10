/*
 * test_instance.c -- Tests for PolyInstance runtime
 */

#include "test_harness.h"
#include "../src/instance.h"
#include "../src/codegen.h"
#include "../src/ctx.h"
#include "../src/engine/schedule.h"
#include "../src/ir.h"
#include "../src/frontend.h"
#include "../src/engine/schedule.h"
#include "../src/optim.h"
#include "../src/tensor.h"
#include "../src/device.h"
#include "../src/safetensors.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

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

/* Helper: build IR bytes for a simple add graph */
/* out = a + b, forward entrypoint */
static uint8_t *make_add_ir(int *out_len) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *store = poly_store_val(ctx, out_buf, sum);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIrBufEntry bufs[] = {
      {.name = "a", .role = POLY_IR_ROLE_INPUT, .buffer = a, .shape = {4}, .ndim = 1},
      {.name = "b", .role = POLY_IR_ROLE_INPUT, .buffer = b, .shape = {4}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out_buf, .shape = {4}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, bufs, 3, eps, 1};

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

  PolyTensor *realized[2] = {NULL, NULL};
  ASSERT_INT_EQ(poly_realize_tensors(ctx, targets, 2, realized), 0);
  ASSERT_PTR_EQ(realized[0], momentum);
  ASSERT_PTR_EQ(realized[1], param);
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
      poly_tensor_full_float_by_id(ctx, param_shape, 2, 0.0, f32_id, POLY_DEVICE_CPU);
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
  PolySchedule *schedule = poly_schedule_with_vars(ctx, targets, 2, resolved);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(resolved[0]);
  ASSERT_NOT_NULL(resolved[1]);
  ASSERT_INT_EQ(schedule->template->n_calls, 5);

  const PolyUOp *m_buf = poly_uop_get_buffer_identity(resolved[0]);
  ASSERT_NOT_NULL(m_buf);
  int momentum_slot = -1;
  for (int i = 0; i < schedule->template->n_buf_slots; i++) {
    if (schedule->template->buf_slots[i].buf_uop == m_buf) momentum_slot = i;
  }
  ASSERT_TRUE(momentum_slot >= 0);
  int momentum_writes = 0;
  int momentum_read_writes = 0;
  for (int k = 0; k < schedule->template->n_calls; k++) {
    PolyCallIO *io = &schedule->run->call_io[k];
    ASSERT_NOT_NULL(io->access);
    for (int i = 0; i < io->n_args; i++) {
      if (io->arg_to_slot[i] == momentum_slot && io->access->outs[i]) {
        momentum_writes++;
        if (io->access->ins[i]) momentum_read_writes++;
      }
    }
  }
  /* Pinned tinygrad's exact first lazy step has one output-only state
   * initialization followed by one input/output momentum update. */
  ASSERT_INT_EQ(momentum_writes, 2);
  ASSERT_INT_EQ(momentum_read_writes, 1);

  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
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

  poly_schedule_free(schedule);
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
  PolyTensor *lr_vec_dtype = poly_tensor_empty(
      ctx, poly_dtype_vec(POLY_FLOAT32, 4), scalar_shape, 1, POLY_DEVICE_CPU
  );
  PolyTensor *lr_ptr_dtype = poly_tensor_empty(
      ctx, poly_dtype_ptr(POLY_FLOAT32, 1, POLY_ADDR_GLOBAL), scalar_shape, 1,
      POLY_DEVICE_CPU
  );
  PolyUOp *n = poly_define_var(ctx, "optim_lr_n", 0, 1);
  PolyUOp *symbolic_dims[] = {n};
  PolyUOp *lr_symbolic_uop =
      poly_expand_uop(ctx, poly_buffer_f32(ctx, 1), symbolic_dims, 1);
  PolyUOp *lr_symbolic_physical = poly_expand_uop(
      ctx, poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU), symbolic_dims, 1
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
  ASSERT_NOT_NULL(lr_vec_dtype);
  ASSERT_NOT_NULL(lr_ptr_dtype);
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
          ctx, &cfg, lr_vec_dtype, &param, &grad, 1, NULL, NULL, NULL, NULL, out, 1
      ),
      -1
  );
  ASSERT_INT_EQ(
      poly_optim_build_step(
          ctx, &cfg, lr_ptr_dtype, &param, &grad, 1, NULL, NULL, NULL, NULL, out, 1
      ),
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

  PolyUOp *w = poly_buffer_f32(ctx, n);
  PolyUOp *x = poly_buffer_f32(ctx, n);
  PolyUOp *y = poly_buffer_f32(ctx, n);
  PolyUOp *out_buf = poly_buffer_f32(ctx, n);
  PolyUOp *loss_buf = poly_buffer_f32(ctx, 1);

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
  PolyIrSpec spec = {ctx, bufs, 5, eps, 2};

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

TEST(instance, forward_reuses_entry_schedule_after_ctx_schedule_cache_clear) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  float a1[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b1[] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyIOBinding io1[] = {POLY_IO_BINDING_ARRAY("a", a1, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("b", b1, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io1, 2), 0);

  /* Instance calls keep a per-entrypoint PolySchedule. Clearing the ctx
   * structural LINEAR lookup must not force the second call to rebuild. */
  poly_schedule_cache_clear(poly_instance_ctx(inst));
  ASSERT_INT_EQ((int)poly_schedule_cache_len(poly_instance_ctx(inst)), 0);

  float a2[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float b2[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyIOBinding io2[] = {POLY_IO_BINDING_ARRAY("a", a2, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("b", b2, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io2, 2), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(poly_instance_ctx(inst)), 0);

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

TEST(instance, forward_reuses_entry_runtime_after_ctx_runtime_cache_clear) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);
  PolyCtx *ctx = poly_instance_ctx(inst);
  ASSERT_NOT_NULL(ctx);

  float a1[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b1[] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyIOBinding io1[] = {POLY_IO_BINDING_ARRAY("a", a1, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("b", b1, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io1, 2), 0);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 1);
  PolyCtxStats cached = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &cached), 0);
  ASSERT_INT_EQ(cached.runtime_cache_entries, 1);
  ASSERT_TRUE(cached.runtime_artifact_entries > 0);
  ASSERT_TRUE(cached.compiled_artifact_bytes > 0);

  /* The instance entrypoint plan retains its compiled runner. Evicting the ctx
   * runtime-cache lookup must not invalidate that live plan or force the next
   * call to repopulate the lookup table. Artifact stats should still account
   * for the live retained runner after lookup eviction. */
  poly_runtime_cache_clear(ctx);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 0);
  PolyCtxStats evicted = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &evicted), 0);
  ASSERT_INT_EQ(evicted.runtime_cache_entries, 0);
  ASSERT_INT_EQ(evicted.runtime_artifact_entries, cached.runtime_artifact_entries);
  ASSERT_TRUE(evicted.compiled_artifact_bytes > 0);
  ASSERT_TRUE(evicted.compiled_artifact_bytes < cached.compiled_artifact_bytes);

  float a2[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float b2[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyIOBinding io2[] = {POLY_IO_BINDING_ARRAY("a", a2, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("b", b2, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_forward(inst, io2, 2), 0);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 0);

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

TEST(instance, staged_physical_template_gate_rejects_logical_only_output) {
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

  PolyInstanceError build_error = {0};
  PolyStatus build_rc = poly_instance_build(inst, &build_error);

  ASSERT_INT_EQ(build_rc, POLY_STATUS_INVALID);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_FAILED);
  ASSERT_TRUE(strstr(build_error.message, "complete physical template required") != NULL);

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

  int32_t replacement_x[] = {9, 9, 9};
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

  PolyTensor *w = poly_tensor_empty(
      ctx, POLY_FLOAT32, shape, 1, poly_ctx_get_preferred_device(ctx)
  );
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

  PolyTensor *w = poly_tensor_empty(
      ctx, POLY_FLOAT32, shape, 1, poly_ctx_get_preferred_device(ctx)
  );
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
    ASSERT_INT_EQ(physical->n_src, 2);
    ASSERT_PTR_EQ(logical->src[0], physical->src[0]);
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
  PolyTensor *w = poly_tensor_empty(
      ctx, POLY_FLOAT32, w_shape, 2, poly_ctx_get_preferred_device(ctx)
  );
  ASSERT_NOT_NULL(w);
  poly_tensor_set_requires_grad(w, true);
  poly_tensor_set_provenance(w, POLY_TENSOR_PROVENANCE_PARAM_INIT);

  PolyTensor *w_alias = poly_tensor_create_with_roots(
      ctx, poly_tensor_uop_logical(w), poly_tensor_uop_physical(w),
      POLY_TENSOR_VALUE, poly_ctx_get_preferred_device(ctx)
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

TEST(instance, from_bindings_snapshots_realized_host_parameter) {
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
  ASSERT_TRUE(inst == NULL);
  ASSERT_NOT_NULL(strstr(err.message, "has no current buffer identity"));

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &w, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, w);
  const PolyUOp *w_physical =
      poly_uop_get_buffer_identity(poly_tensor_uop_physical(w));
  const PolyUOp *w_logical =
      poly_uop_get_buffer_identity(poly_tensor_uop_logical(w));
  ASSERT_NOT_NULL(w_physical);
  ASSERT_NOT_NULL(w_logical);
  ASSERT_PTR_NEQ(w_physical, w_logical);

  memset(&err, 0, sizeof(err));
  inst = poly_instance_from_binding_arrays(
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

TEST(instance, staged_build_rejects_realized_output_without_template) {
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
  ASSERT_INT_EQ(poly_buffer_write(ctx, poly_tensor_uop_logical(x), x_initial, sizeof(x_initial)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, poly_tensor_uop_logical(w), w_data, sizeof(w_data)), 0);

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
  PolyInstanceError build_error = {0};
  ASSERT_INT_EQ(poly_instance_build(inst, &build_error), POLY_STATUS_INVALID);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_FAILED);
  ASSERT_TRUE(strstr(build_error.message, "complete physical template required") != NULL);

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
  ASSERT_INT_EQ(poly_instance_train_step(restored, io, 2, &loss), 0);
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
  ASSERT_INT_EQ(poly_instance_train_step(restored, io, 2, &loss), 0);
  rb_idx = test_find_instance_buf(restored, "optim.sgd.b.w");
  rb = poly_instance_buf_data(restored, rb_idx, NULL);
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

  /* Switch to INTERP and run */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_INTERP), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 2), 0);
  out = poly_instance_buf_data(inst, 2, &numel);
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(fabsf(out[i] - expected[i]) < 1e-5f);

  /* Switch back to CPU (should hit exec cache) */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 2), 0);
  out = poly_instance_buf_data(inst, 2, &numel);
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(fabsf(out[i] - expected[i]) < 1e-5f);

  poly_instance_free(inst);
  free(ir);
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
