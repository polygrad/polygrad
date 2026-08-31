/* test_placement.c -- explicit aggregate logical placement. */

#include "test_harness.h"
#include "../src/engine/realize.h"
#include "../src/frontend.h"
#include "../src/frontend_internal.h"
#include "../src/polygrad.h"
#include "../src/tensor.h"

static PolyUOp *placement_buffer(PolyCtx *ctx, int64_t n, PolyDevice device) {
  return device == POLY_DEVICE_AUTO ? poly_test_logical_buffer(ctx, POLY_FLOAT32, n)
                                    : poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, device);
}

static PolyUOp *placement_binding_on_device(PolyCtx *ctx, PolyUOp *logical, PolyDevice device) {
  if (!logical || logical->op != POLY_OP_BUFFER || logical->n_src != 1 || !logical->src[0] ||
      logical->src[0]->op != POLY_OP_UNIQUE || logical->arg.kind != POLY_ARG_INT ||
      logical->src[0]->arg.kind != POLY_ARG_INT)
    return NULL;
  return poly_uop_new_buffer(
      ctx, poly_device_uop(ctx, device), logical->arg.i, logical->dtype, logical->src[0]->arg.i
  );
}

TEST(placement, logical_bindings_reproduce_eager_value_and_instance_sink) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *logical_in = placement_buffer(ctx, 4, POLY_DEVICE_AUTO);
  PolyUOp *logical_out = placement_buffer(ctx, 4, POLY_DEVICE_AUTO);
  PolyUOp *physical_in = placement_buffer(ctx, 4, POLY_DEVICE_INTERP);
  PolyUOp *physical_out = placement_buffer(ctx, 4, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(logical_in);
  ASSERT_NOT_NULL(logical_out);
  ASSERT_NOT_NULL(physical_in);
  ASSERT_NOT_NULL(physical_out);

  PolyUOp *logical_value = poly_add(ctx, logical_in, logical_in);
  int64_t shape[2] = {2, 2};
  logical_value = poly_reshape(ctx, logical_value, shape, 2);
  PolyUOp *eager_value = poly_reshape(ctx, poly_add(ctx, physical_in, physical_in), shape, 2);
  PolyUOp *logical_store = poly_store_val(ctx, logical_out, logical_value);
  PolyUOp *logical_sink = poly_sink1(ctx, logical_store);
  PolyUOp *eager_sink = poly_sink1(ctx, poly_store_val(ctx, physical_out, eager_value));
  ASSERT_NOT_NULL(logical_sink);
  ASSERT_NOT_NULL(eager_sink);

  PolyUOp *logical_roots[2] = {logical_value, logical_sink};
  PolyUOp *from[2] = {logical_in, logical_out};
  PolyUOp *to[2] = {physical_in, physical_out};
  PolyUOp *placed[2] = {NULL, NULL};
  ASSERT_INT_EQ(poly_place_roots(ctx, logical_roots, 2, from, to, 2, placed), 0);
  ASSERT_EQ(placed[0], eager_value);
  ASSERT_EQ(placed[1], eager_sink);
  ASSERT_FALSE(poly_tensor_root_has_unplaced_buffer(ctx, placed[0]));
  ASSERT_FALSE(poly_tensor_root_has_unplaced_buffer(ctx, placed[1]));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(placement, logical_placement_does_not_consume_bound_copy_history) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *logical_base = placement_buffer(ctx, 4, POLY_DEVICE_AUTO);
  PolyUOp *base_cpu = placement_buffer(ctx, 4, POLY_DEVICE_CPU);
  PolyUOp *base_interp = placement_buffer(ctx, 4, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(logical_base);
  ASSERT_NOT_NULL(base_cpu);
  ASSERT_NOT_NULL(base_interp);

  PolyUOp *cuda_device = poly_device_uop(ctx, POLY_DEVICE_CUDA);
  PolyUOp *cpu_device = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *to_cuda = poly_copy_to_device_uop(ctx, base_cpu, cuda_device);
  PolyUOp *roundtrip = poly_copy_to_device_uop(ctx, to_cuda, cpu_device);
  PolyUOp *roundtrip_template = poly_add(ctx, base_cpu, roundtrip);
  ASSERT_NOT_NULL(roundtrip_template);

  PolyUOp *logical_roots[1] = {poly_add(ctx, logical_base, logical_base)};
  PolyUOp *logical_bindings[1] = {logical_base};
  PolyUOp *target_bindings[1] = {base_interp};
  PolyUOp *placed[1] = {NULL};
  ASSERT_INT_EQ(
      poly_place_roots(ctx, logical_roots, 1, logical_bindings, target_bindings, 1, placed), 0
  );

  ASSERT_EQ(placed[0]->op, POLY_OP_ADD);
  ASSERT_EQ(placed[0]->src[0], base_interp);
  ASSERT_EQ(placed[0]->src[1], base_interp);
  ASSERT_FALSE(poly_uop_reachable(ctx, placed[0], roundtrip_template));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(placement, logical_missing_occurrence_evidence_fails_atomically) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *logical = placement_buffer(ctx, 4, POLY_DEVICE_AUTO);
  PolyUOp *physical = placement_buffer(ctx, 4, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(logical);
  ASSERT_NOT_NULL(physical);
  PolyUOp *effect = poly_uop_after(ctx, logical, poly_store_val(ctx, logical, logical));
  PolyUOp *sentinel = poly_const_int(ctx, 77);
  PolyUOp *out[1] = {sentinel};
  PolyUOp *roots[1] = {effect};
  PolyUOp *from[1] = {logical};
  PolyUOp *to[1] = {physical};
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, 1, from, to, 1, out), -1);
  ASSERT_EQ(out[0], sentinel);

  roots[0] = poly_copy_to_device_uop(ctx, logical, poly_device_uop(ctx, POLY_DEVICE_CUDA));
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, 1, from, to, 1, out), -1);
  ASSERT_EQ(out[0], sentinel);

  /* A second logical storage identity requires its own named binding. */
  PolyUOp *unbound = placement_buffer(ctx, 4, POLY_DEVICE_AUTO);
  roots[0] = poly_add(ctx, logical, unbound);
  ASSERT_NOT_NULL(unbound);
  ASSERT_NOT_NULL(roots[0]);
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, 1, from, to, 1, out), -1);
  ASSERT_EQ(out[0], sentinel);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(placement, invalid_binding_shape_or_alias_fails_atomically) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = placement_buffer(ctx, 4, POLY_DEVICE_AUTO);
  PolyUOp *b = placement_buffer(ctx, 4, POLY_DEVICE_AUTO);
  PolyUOp *target = placement_buffer(ctx, 4, POLY_DEVICE_INTERP);
  PolyUOp *wrong_shape = placement_buffer(ctx, 5, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  ASSERT_NOT_NULL(target);
  ASSERT_NOT_NULL(wrong_shape);
  PolyUOp *root = poly_add(ctx, a, b);
  PolyUOp *roots[1] = {root};
  PolyUOp *from[2] = {a, b};
  PolyUOp *sentinel = poly_const_int(ctx, 99);
  PolyUOp *out[1] = {sentinel};

  PolyUOp *wrong_targets[2] = {target, wrong_shape};
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, 1, from, wrong_targets, 2, out), -1);
  ASSERT_EQ(out[0], sentinel);

  PolyUOp *aliased_targets[2] = {target, target};
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, 1, from, aliased_targets, 2, out), -1);
  ASSERT_EQ(out[0], sentinel);

  PolyUOp *partial_from[1] = {a};
  PolyUOp *partial_to[1] = {target};
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, 1, partial_from, partial_to, 1, out), -1);
  ASSERT_EQ(out[0], sentinel);

  /* COPY is physical transport, never a portable logical binding. */
  PolyUOp *interp_device = poly_device_uop(ctx, POLY_DEVICE_INTERP);
  PolyUOp *copy = poly_copy_to_device_uop(ctx, target, interp_device);
  PolyUOp *copy_binding[1] = {copy};
  ASSERT_NOT_NULL(copy);
  PolyUOp *copy_roots[1] = {copy};
  ASSERT_INT_EQ(poly_place_roots(ctx, copy_roots, 1, copy_binding, partial_to, 1, out), -1);
  ASSERT_EQ(out[0], sentinel);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(placement, explicit_module_map_inserts_exact_cross_device_cut) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *lx = placement_buffer(ctx, 2, POLY_DEVICE_AUTO);
  PolyUOp *lw0 = placement_buffer(ctx, 2, POLY_DEVICE_AUTO);
  PolyUOp *lw1 = placement_buffer(ctx, 2, POLY_DEVICE_AUTO);
  PolyUOp *lout = placement_buffer(ctx, 2, POLY_DEVICE_AUTO);
  PolyUOp *module0 = poly_add(ctx, lx, lw0);
  PolyUOp *module1 = poly_mul(ctx, module0, lw1);
  PolyUOp *logical_sink = poly_sink1(ctx, poly_store_val(ctx, lout, module1));
  ASSERT_NOT_NULL(lx);
  ASSERT_NOT_NULL(lw0);
  ASSERT_NOT_NULL(lw1);
  ASSERT_NOT_NULL(lout);
  ASSERT_NOT_NULL(module0);
  ASSERT_NOT_NULL(module1);
  ASSERT_NOT_NULL(logical_sink);

  PolyUOp *logical_bindings[4] = {lx, lw0, lw1, lout};
  PolyUOp *target_bindings[4] = {NULL, NULL, NULL, NULL};
  PolyUOp *module0_inputs[1] = {lx};
  PolyUOp *module1_inputs[1] = {module0};
  PolyPlaceModule modules[2] = {
      {"layers.0", module0, module0_inputs, 1, poly_device_uop(ctx, POLY_DEVICE_CPU)},
      {"layers.1", module1, module1_inputs, 1, poly_device_uop(ctx, POLY_DEVICE_INTERP)},
  };
  ASSERT_NOT_NULL(modules[0].device);
  ASSERT_NOT_NULL(modules[1].device);

  /* Aggregate root order is intentionally consumer-first.  Module order is
   * the declared dataflow order and does not depend on entrypoint root order. */
  PolyUOp *roots[3] = {logical_sink, module1, module0};
  PolyUOp *placed[3] = {NULL, NULL, NULL};
  ASSERT_INT_EQ(
      poly_place_module_map(
          ctx, roots, 3, logical_bindings, 4, modules, 2, target_bindings, placed
      ),
      0
  );

  PolyUOp *px = placement_binding_on_device(ctx, lx, POLY_DEVICE_CPU);
  PolyUOp *pw0 = placement_binding_on_device(ctx, lw0, POLY_DEVICE_CPU);
  PolyUOp *pw1 = placement_binding_on_device(ctx, lw1, POLY_DEVICE_INTERP);
  PolyUOp *pout = placement_binding_on_device(ctx, lout, POLY_DEVICE_INTERP);
  ASSERT_PTR_EQ(target_bindings[0], px);
  ASSERT_PTR_EQ(target_bindings[1], pw0);
  ASSERT_PTR_EQ(target_bindings[2], pw1);
  ASSERT_PTR_EQ(target_bindings[3], pout);
  PolyUOp *expected_module0 = poly_add(ctx, px, pw0);
  PolyUOp *expected_cut = poly_copy_to_device_uop(ctx, expected_module0, modules[1].device);
  PolyUOp *expected_module1 = poly_mul(ctx, expected_cut, pw1);
  PolyUOp *expected_sink = poly_sink1(ctx, poly_store_val(ctx, pout, expected_module1));
  ASSERT_PTR_EQ(placed[0], expected_sink);
  ASSERT_PTR_EQ(placed[1], expected_module1);
  ASSERT_PTR_EQ(placed[2], expected_module0);
  ASSERT_EQ(placed[1]->op, POLY_OP_MUL);
  ASSERT_EQ(placed[1]->src[0]->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(placed[1]->src[0]->src[0], placed[2]);
  ASSERT_INT_EQ(placed[1]->src[0]->arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(placed[1]->src[0]->arg.str, "INTERP");
  ASSERT_EQ(poly_uop_device(placed[1]), POLY_DEVICE_INTERP);

  PolyUOp *ordered_roots[3] = {module0, module1, logical_sink};
  PolyUOp *ordered_placed[3] = {NULL, NULL, NULL};
  ASSERT_INT_EQ(
      poly_place_module_map(
          ctx, ordered_roots, 3, logical_bindings, 4, modules, 2, target_bindings, ordered_placed
      ),
      0
  );
  ASSERT_PTR_EQ(ordered_placed[0], placed[2]);
  ASSERT_PTR_EQ(ordered_placed[1], placed[1]);
  ASSERT_PTR_EQ(ordered_placed[2], placed[0]);

  PolyUOp *reverse_targets[4] = {NULL, NULL, NULL, NULL};
  PolyPlaceModule reverse_devices[2] = {
      {"layers.0", module0, module0_inputs, 1, poly_device_uop(ctx, POLY_DEVICE_INTERP)},
      {"layers.1", module1, module1_inputs, 1, poly_device_uop(ctx, POLY_DEVICE_CPU)},
  };
  PolyUOp *reverse_placed[3] = {NULL, NULL, NULL};
  ASSERT_INT_EQ(
      poly_place_module_map(
          ctx, roots, 3, logical_bindings, 4, reverse_devices, 2, reverse_targets, reverse_placed
      ),
      0
  );
  ASSERT_EQ(reverse_placed[1]->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(reverse_placed[1]->src[0]->arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(reverse_placed[1]->src[0]->arg.str, "CPU");
  ASSERT_EQ(poly_uop_device(reverse_placed[1]), POLY_DEVICE_CPU);
  ASSERT_EQ(poly_uop_device(reverse_targets[3]), POLY_DEVICE_CPU);

  float x_data[2] = {1.0f, 2.0f};
  float w0_data[2] = {3.0f, 4.0f};
  float w1_data[2] = {2.0f, 3.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, px, x_data, sizeof(x_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, pw0, w0_data, sizeof(w0_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, pw1, w1_data, sizeof(w1_data)), 0);
  PolyUOp *realized[2] = {NULL, NULL};
  PolyUOp *value_roots[2] = {placed[1], placed[2]};
  ASSERT_INT_EQ(poly_realize_uops(ctx, value_roots, 2, realized), 0);
  float out[2] = {0.0f, 0.0f};
  ASSERT_INT_EQ(poly_buffer_read(ctx, realized[0], out, sizeof(out)), 0);
  ASSERT_FLOAT_NEAR(out[0], 8.0f, 4, 1e-6f);
  ASSERT_FLOAT_NEAR(out[1], 18.0f, 4, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(placement, explicit_module_map_rejects_ambiguous_regions_atomically) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *lx = placement_buffer(ctx, 2, POLY_DEVICE_AUTO);
  PolyUOp *lw0 = placement_buffer(ctx, 2, POLY_DEVICE_AUTO);
  PolyUOp *lw1 = placement_buffer(ctx, 2, POLY_DEVICE_AUTO);
  PolyUOp *module0 = poly_add(ctx, lx, lw0);
  PolyUOp *module1 = poly_mul(ctx, module0, lw1);
  PolyUOp *logical_bindings[3] = {lx, lw0, lw1};
  PolyUOp *target_bindings[3] = {NULL, NULL, NULL};
  PolyUOp *module0_inputs[1] = {lx};
  PolyUOp *module1_inputs[1] = {module0};
  PolyPlaceModule modules[2] = {
      {"layers.0", module0, module0_inputs, 1, poly_device_uop(ctx, POLY_DEVICE_CPU)},
      {"layers.1", module1, module1_inputs, 1, poly_device_uop(ctx, POLY_DEVICE_INTERP)},
  };
  PolyUOp *roots[1] = {module1};
  PolyUOp *sentinel = poly_const_int(ctx, 73);
  PolyUOp *out[1] = {sentinel};

  PolyPlaceModule missing_cut[2] = {modules[0], modules[1]};
  missing_cut[1].inputs = NULL;
  missing_cut[1].n_inputs = 0;
  ASSERT_INT_EQ(
      poly_place_module_map(
          ctx, roots, 1, logical_bindings, 3, missing_cut, 2, target_bindings, out
      ),
      -1
  );
  ASSERT_PTR_EQ(out[0], sentinel);

  PolyUOp *conflict1 = poly_mul(ctx, module0, lw0);
  PolyUOp *conflict1_inputs[1] = {module0};
  PolyPlaceModule conflicting_owner[2] = {
      modules[0],
      {"layers.1", conflict1, conflict1_inputs, 1, poly_device_uop(ctx, POLY_DEVICE_INTERP)},
  };
  PolyUOp *conflict_roots[1] = {conflict1};
  ASSERT_INT_EQ(
      poly_place_module_map(
          ctx, conflict_roots, 1, logical_bindings, 3, conflicting_owner, 2, target_bindings, out
      ),
      -1
  );
  ASSERT_PTR_EQ(out[0], sentinel);

  PolyPlaceModule reversed[2] = {modules[1], modules[0]};
  ASSERT_INT_EQ(
      poly_place_module_map(ctx, roots, 1, logical_bindings, 3, reversed, 2, target_bindings, out),
      -1
  );
  ASSERT_PTR_EQ(out[0], sentinel);

  poly_ctx_destroy(ctx);
  PASS();
}
