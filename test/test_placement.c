/* test_placement.c -- explicit aggregate logical/template placement. */

#include "test_harness.h"
#include "../src/engine/realize.h"
#include "../src/frontend.h"
#include "../src/frontend_internal.h"
#include "../src/polygrad.h"

static PolyUOp *placement_buffer(PolyCtx *ctx, int64_t n, PolyDevice device) {
  return device == POLY_DEVICE_AUTO ? poly_buffer(ctx, POLY_FLOAT32, n)
                                    : poly_buffer_on_device(ctx, POLY_FLOAT32, n, device);
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
  ASSERT_INT_EQ(poly_place_roots(ctx, logical_roots, NULL, 2, from, NULL, to, 2, placed), 0);
  ASSERT_EQ(placed[0], eager_value);
  ASSERT_EQ(placed[1], eager_sink);
  ASSERT_FALSE(poly_tensor_root_has_unplaced_buffer(ctx, placed[0]));
  ASSERT_FALSE(poly_tensor_root_has_unplaced_buffer(ctx, placed[1]));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(placement, template_bindings_preserve_copy_and_assignment_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *logical_base = placement_buffer(ctx, 4, POLY_DEVICE_AUTO);
  PolyUOp *logical_value = placement_buffer(ctx, 4, POLY_DEVICE_AUTO);
  PolyUOp *base_cpu = placement_buffer(ctx, 4, POLY_DEVICE_CPU);
  PolyUOp *base_interp = placement_buffer(ctx, 4, POLY_DEVICE_INTERP);
  PolyUOp *value_cuda = placement_buffer(ctx, 4, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(logical_base);
  ASSERT_NOT_NULL(logical_value);
  ASSERT_NOT_NULL(base_cpu);
  ASSERT_NOT_NULL(base_interp);
  ASSERT_NOT_NULL(value_cuda);

  PolyUOp *cuda_device = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_CUDA));
  PolyUOp *cpu_device = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_CPU));
  PolyUOp *to_cuda_src[2] = {base_cpu, cuda_device};
  PolyUOp *to_cuda = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, to_cuda_src, 2, poly_arg_none());
  PolyUOp *to_cpu_src[2] = {to_cuda, cpu_device};
  PolyUOp *roundtrip = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, to_cpu_src, 2, poly_arg_none());
  PolyUOp *roundtrip_template = poly_add(ctx, base_cpu, roundtrip);
  PolyUOp *store = poly_store_val(ctx, to_cuda, value_cuda);
  PolyUOp *assignment_template = poly_uop_after(ctx, to_cuda, store);
  ASSERT_NOT_NULL(roundtrip_template);
  ASSERT_NOT_NULL(assignment_template);

  PolyUOp *logical_roots[2] = {
      poly_add(ctx, logical_base, logical_base),
      poly_uop_after(ctx, logical_base, poly_store_val(ctx, logical_base, logical_value)),
  };
  PolyUOp *templates[2] = {roundtrip_template, assignment_template};
  PolyUOp *logical_bindings[1] = {logical_base};
  PolyUOp *template_bindings[1] = {base_cpu};
  PolyUOp *target_bindings[1] = {base_interp};
  PolyUOp *placed[2] = {NULL, NULL};
  ASSERT_INT_EQ(
      poly_place_roots(
          ctx, logical_roots, templates, 2, logical_bindings, template_bindings, target_bindings, 1,
          placed
      ),
      0
  );

  ASSERT_EQ(placed[0]->op, POLY_OP_ADD);
  ASSERT_EQ(placed[0]->src[0], base_interp);
  ASSERT_EQ(placed[0]->src[1]->op, POLY_OP_COPY);
  ASSERT_EQ(placed[0]->src[1]->src[0]->op, POLY_OP_COPY);
  ASSERT_EQ(placed[1]->op, POLY_OP_AFTER);
  ASSERT_EQ(placed[1]->src[0]->op, POLY_OP_COPY);
  ASSERT_EQ(placed[1]->src[1]->op, POLY_OP_STORE);
  ASSERT_EQ(placed[1]->src[1]->src[0], placed[1]->src[0]);

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
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, NULL, 1, from, NULL, to, 1, out), -1);
  ASSERT_EQ(out[0], sentinel);

  PolyUOp *copy_src[2] = {
      logical,
      poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_CUDA)),
  };
  roots[0] = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_src, 2, poly_arg_none());
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, NULL, 1, from, NULL, to, 1, out), -1);
  ASSERT_EQ(out[0], sentinel);

  /* Template mode preserves exact topology, but it still cannot publish a
   * graph with a second caller-visible unplaced storage identity. */
  PolyUOp *unbound = placement_buffer(ctx, 4, POLY_DEVICE_AUTO);
  PolyUOp *template = poly_add(ctx, logical, unbound);
  PolyUOp *templates[1] = {template};
  ASSERT_NOT_NULL(unbound);
  ASSERT_NOT_NULL(template);
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, templates, 1, from, from, to, 1, out), -1);
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
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, NULL, 1, from, NULL, wrong_targets, 2, out), -1);
  ASSERT_EQ(out[0], sentinel);

  PolyUOp *aliased_targets[2] = {target, target};
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, NULL, 1, from, NULL, aliased_targets, 2, out), -1);
  ASSERT_EQ(out[0], sentinel);

  PolyUOp *partial_from[1] = {a};
  PolyUOp *partial_to[1] = {target};
  ASSERT_INT_EQ(poly_place_roots(ctx, roots, NULL, 1, partial_from, NULL, partial_to, 1, out), -1);
  ASSERT_EQ(out[0], sentinel);

  /* COPY is an occurrence inside a template, never a replaceable binding.
   * Substituting it directly would erase pinned `.to()` topology. */
  PolyUOp *interp_device =
      poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_INTERP));
  PolyUOp *copy_src[2] = {target, interp_device};
  PolyUOp *copy = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_src, 2, poly_arg_none());
  PolyUOp *template = poly_add(ctx, copy, copy);
  PolyUOp *templates[1] = {template};
  PolyUOp *copy_binding[1] = {copy};
  ASSERT_NOT_NULL(copy);
  ASSERT_NOT_NULL(template);
  ASSERT_INT_EQ(
      poly_place_roots(ctx, roots, templates, 1, partial_from, copy_binding, partial_to, 1, out), -1
  );
  ASSERT_EQ(out[0], sentinel);

  poly_ctx_destroy(ctx);
  PASS();
}
