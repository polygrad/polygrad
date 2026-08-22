/*
 * test_registry.c -- Tests for the named buffer registry on PolyCtx
 *                    and poly_instance_from_ctx
 */

#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/frontend.h"
#include "../src/instance.h"
#include "../src/ir.h"
#include "../src/ctx.h"
#include "../src/engine/schedule.h"

/* Registration + lookup */

TEST(registry, register_all_roles) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s2[] = {3, 4};
  int64_t s1[] = {10};

  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, s2, 2, "weight");
  PolyUOp *x = poly_input(ctx, POLY_FLOAT32, s1, 1, "x");
  PolyUOp *y = poly_output(ctx, POLY_FLOAT32, s1, 1, "y");
  PolyUOp *t = poly_target(ctx, POLY_FLOAT32, s1, 1, "target");
  PolyUOp *a = poly_aux(ctx, POLY_FLOAT32, s1, 1, "running_mean");

  ASSERT_TRUE(w != NULL);
  ASSERT_TRUE(x != NULL);
  ASSERT_TRUE(y != NULL);
  ASSERT_TRUE(t != NULL);
  ASSERT_TRUE(a != NULL);

  /* All are distinct BUFFER UOps */
  ASSERT_TRUE(w != x);
  ASSERT_TRUE(x != y);
  ASSERT_TRUE(y != t);
  ASSERT_TRUE(t != a);

  /* All are BUFFER ops */
  ASSERT_INT_EQ(w->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(x->op, POLY_OP_BUFFER);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, lookup_by_name) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {5, 3};

  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, s, 2, "layers.%d.weight", 0);
  PolyUOp *b = poly_param(ctx, POLY_FLOAT32, (int64_t[]){3}, 1, "layers.%d.bias", 0);

  /* Lookup returns same pointer */
  ASSERT_EQ(poly_ctx_get(ctx, "layers.%d.weight", 0), w);
  ASSERT_EQ(poly_ctx_get(ctx, "layers.%d.bias", 0), b);

  /* Non-existent returns NULL */
  ASSERT_EQ(poly_ctx_get(ctx, "nonexistent"), NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, formatted_lookup_rejects_truncated_name) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {1};

  char prefix[256];
  memset(prefix, 'x', sizeof(prefix));
  prefix[sizeof(prefix) - 1] = '\0';

  char long_name[320];
  memset(long_name, 'x', sizeof(long_name));
  long_name[sizeof(long_name) - 1] = '\0';

  PolyUOp *short_buf = poly_param(ctx, POLY_FLOAT32, s, 1, "%s", prefix);
  PolyUOp *long_buf = poly_param(ctx, POLY_FLOAT32, s, 1, "%s", long_name);
  ASSERT_NOT_NULL(short_buf);
  ASSERT_NOT_NULL(long_buf);
  ASSERT_NEQ(short_buf, long_buf);

  ASSERT_EQ(poly_ctx_get(ctx, "%s", long_name), NULL);
  ASSERT_EQ(poly_ctx_get_entry(ctx, long_name)->buffer, long_buf);
  ASSERT_EQ(poly_ctx_get(ctx, "%s", prefix), short_buf);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, get_entry) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {4, 8};

  poly_param(ctx, POLY_FLOAT32, s, 2, "weight");

  const PolyRegEntry *e = poly_ctx_get_entry(ctx, "weight");
  ASSERT_TRUE(e != NULL);
  ASSERT_STR_EQ(e->name, "weight");
  ASSERT_INT_EQ(e->role, POLY_ROLE_PARAM);
  ASSERT_INT_EQ(e->ndim, 2);
  ASSERT_INT_EQ(e->shape[0], 4);
  ASSERT_INT_EQ(e->shape[1], 8);
  ASSERT_FALSE(e->is_alias);

  ASSERT_EQ(poly_ctx_get_entry(ctx, "missing"), NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Idempotent re-registration */

TEST(registry, idempotent_rereg) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {3, 4};

  PolyUOp *w1 = poly_param(ctx, POLY_FLOAT32, s, 2, "weight");
  PolyUOp *w2 = poly_param(ctx, POLY_FLOAT32, s, 2, "weight");

  /* Same name + dtype + shape returns same buffer (pointer equality) */
  ASSERT_EQ(w1, w2);
  /* Count stays at 1 */
  ASSERT_INT_EQ(poly_ctx_named_count(ctx), 1);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Mismatch rejection */

TEST(registry, mismatch_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {10};

  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, s, 1, "x");
  ASSERT_TRUE(w != NULL);

  /* Same name, different dtype -> NULL */
  PolyUOp *w2 = poly_param(ctx, POLY_FLOAT64, s, 1, "x");
  ASSERT_EQ(w2, NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, mismatch_shape) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s1[] = {3, 4};
  int64_t s2[] = {4, 3}; /* same numel, different shape */

  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, s1, 2, "weight");
  ASSERT_TRUE(w != NULL);

  PolyUOp *w2 = poly_param(ctx, POLY_FLOAT32, s2, 2, "weight");
  ASSERT_EQ(w2, NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, mismatch_ndim) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s1[] = {12};
  int64_t s2[] = {3, 4};

  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, s1, 1, "weight");
  ASSERT_TRUE(w != NULL);

  PolyUOp *w2 = poly_param(ctx, POLY_FLOAT32, s2, 2, "weight");
  ASSERT_EQ(w2, NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Alias */

TEST(registry, alias_resolves_same_buffer) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {100, 64};

  PolyUOp *emb = poly_param(ctx, POLY_FLOAT32, s, 2, "embedding.weight");
  ASSERT_TRUE(emb != NULL);

  int rc = poly_alias(ctx, "lm_head.weight", "embedding.weight");
  ASSERT_INT_EQ(rc, 0);

  /* Both names resolve to the same UOp */
  ASSERT_EQ(poly_ctx_get(ctx, "embedding.weight"), emb);
  ASSERT_EQ(poly_ctx_get(ctx, "lm_head.weight"), emb);

  /* Alias entry is marked */
  const PolyRegEntry *orig = poly_ctx_get_entry(ctx, "embedding.weight");
  const PolyRegEntry *alias = poly_ctx_get_entry(ctx, "lm_head.weight");
  ASSERT_FALSE(orig->is_alias);
  ASSERT_TRUE(alias->is_alias);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, alias_idempotent) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {10};
  poly_param(ctx, POLY_FLOAT32, s, 1, "w");
  ASSERT_INT_EQ(poly_alias(ctx, "w2", "w"), 0);
  /* Re-aliasing same pair is idempotent */
  ASSERT_INT_EQ(poly_alias(ctx, "w2", "w"), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, alias_nonexistent) {
  PolyCtx *ctx = poly_ctx_new();
  /* Aliasing a non-existent name fails */
  ASSERT_INT_EQ(poly_alias(ctx, "alias", "nonexistent"), -1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, alias_name_conflict) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {10};
  poly_param(ctx, POLY_FLOAT32, s, 1, "a");
  poly_param(ctx, POLY_FLOAT32, s, 1, "b");

  /* "b" already points to a different buffer */
  ASSERT_INT_EQ(poly_alias(ctx, "b", "a"), -1);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Entrypoints */

TEST(registry, entrypoint_register_and_retrieve) {
  PolyCtx *ctx = poly_ctx_new();

  /* Build a trivial SINK (just a store of a const into a buffer) */
  PolyUOp *buf = poly_param(ctx, POLY_FLOAT32, (int64_t[]){4}, 1, "w");
  PolyUOp *val = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *st = poly_store_val(ctx, buf, val);
  PolyUOp *sink = poly_sink1(ctx, st);

  ASSERT_INT_EQ(poly_register_entrypoint(ctx, "forward", sink), 0);

  ASSERT_INT_EQ(poly_ctx_entrypoint_count(ctx), 1);
  ASSERT_STR_EQ(poly_ctx_entrypoint_name(ctx, 0), "forward");
  ASSERT_EQ(poly_ctx_entrypoint_sink(ctx, 0), sink);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, multiple_entrypoints) {
  PolyCtx *ctx = poly_ctx_new();

  /* Just register dummy sinks */
  PolyUOp *s1 = poly_uop0(ctx, POLY_OP_SINK, POLY_VOID, poly_arg_none());
  PolyUOp *s2 = poly_uop0(ctx, POLY_OP_SINK, POLY_VOID, poly_arg_none());

  ASSERT_INT_EQ(poly_register_entrypoint(ctx, "forward", s1), 0);
  ASSERT_INT_EQ(poly_register_entrypoint(ctx, "loss", s2), 0);

  ASSERT_INT_EQ(poly_ctx_entrypoint_count(ctx), 2);
  ASSERT_STR_EQ(poly_ctx_entrypoint_name(ctx, 0), "forward");
  ASSERT_STR_EQ(poly_ctx_entrypoint_name(ctx, 1), "loss");

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, entrypoint_null_rejected) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_INT_EQ(poly_register_entrypoint(ctx, "fwd", NULL), -1);
  ASSERT_INT_EQ(poly_register_entrypoint(ctx, NULL, NULL), -1);
  ASSERT_INT_EQ(poly_ctx_entrypoint_count(ctx), 0);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Enumeration */

TEST(registry, enumeration_count) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {10};

  ASSERT_INT_EQ(poly_ctx_named_count(ctx), 0);

  poly_param(ctx, POLY_FLOAT32, s, 1, "w1");
  poly_param(ctx, POLY_FLOAT32, s, 1, "w2");
  poly_input(ctx, POLY_FLOAT32, s, 1, "x");
  ASSERT_INT_EQ(poly_ctx_named_count(ctx), 3);

  /* Add alias -- counts as an entry */
  poly_alias(ctx, "w1_alias", "w1");
  ASSERT_INT_EQ(poly_ctx_named_count(ctx), 4);

  /* Idempotent re-reg doesn't increase count */
  poly_param(ctx, POLY_FLOAT32, s, 1, "w1");
  ASSERT_INT_EQ(poly_ctx_named_count(ctx), 4);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, enumeration_entries) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {5};

  poly_param(ctx, POLY_FLOAT32, s, 1, "a");
  poly_input(ctx, POLY_FLOAT32, s, 1, "b");
  poly_output(ctx, POLY_FLOAT32, s, 1, "c");

  /* Entries appear in registration order */
  ASSERT_STR_EQ(poly_ctx_named_entry(ctx, 0)->name, "a");
  ASSERT_INT_EQ(poly_ctx_named_entry(ctx, 0)->role, POLY_ROLE_PARAM);
  ASSERT_STR_EQ(poly_ctx_named_entry(ctx, 1)->name, "b");
  ASSERT_INT_EQ(poly_ctx_named_entry(ctx, 1)->role, POLY_ROLE_INPUT);
  ASSERT_STR_EQ(poly_ctx_named_entry(ctx, 2)->name, "c");
  ASSERT_INT_EQ(poly_ctx_named_entry(ctx, 2)->role, POLY_ROLE_OUTPUT);

  /* Out of bounds returns NULL */
  ASSERT_EQ(poly_ctx_named_entry(ctx, -1), NULL);
  ASSERT_EQ(poly_ctx_named_entry(ctx, 3), NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Printf-style naming */

TEST(registry, printf_naming) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {64, 64};

  for (int i = 0; i < 3; i++) {
    poly_param(ctx, POLY_FLOAT32, s, 2, "transformer.h.%d.attn.weight", i);
    poly_param(ctx, POLY_FLOAT32, (int64_t[]){64}, 1, "transformer.h.%d.attn.bias", i);
  }

  ASSERT_INT_EQ(poly_ctx_named_count(ctx), 6);

  /* Verify individual lookups */
  ASSERT_TRUE(poly_ctx_get(ctx, "transformer.h.%d.attn.weight", 1) != NULL);
  ASSERT_TRUE(poly_ctx_get(ctx, "transformer.h.%d.attn.bias", 2) != NULL);

  /* Each layer has distinct buffers */
  PolyUOp *w0 = poly_ctx_get(ctx, "transformer.h.%d.attn.weight", 0);
  PolyUOp *w1 = poly_ctx_get(ctx, "transformer.h.%d.attn.weight", 1);
  ASSERT_TRUE(w0 != w1);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/* Phase 2: poly_instance_from_ctx tests                                */
/* ══════════════════════════════════════════════════════════════════════ */

TEST(registry, instance_from_ctx_basic) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {4};

  /* Register buffers */
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, s, 1, "w");
  PolyUOp *x = poly_input(ctx, POLY_FLOAT32, s, 1, "x");
  PolyUOp *out = poly_output(ctx, POLY_FLOAT32, s, 1, "output");

  /* Build graph: output = w * x */
  PolyUOp *prod =
      poly_alu2(ctx, POLY_OP_MUL, poly_reshape(ctx, w, s, 1), poly_reshape(ctx, x, s, 1));
  PolyUOp *store = poly_store_val(ctx, out, prod);
  PolyUOp *sink = poly_sink1(ctx, store);
  poly_register_entrypoint(ctx, "forward", sink);

  /* Create instance */
  PolyInstance *inst = poly_instance_from_ctx(ctx);
  ASSERT_TRUE(inst != NULL);

  /* Verify buffer enumeration */
  ASSERT_INT_EQ(poly_instance_buf_count(inst), 3);
  ASSERT_STR_EQ(poly_instance_buf_name(inst, 0), "w");
  ASSERT_INT_EQ(poly_instance_buf_role(inst, 0), POLY_ROLE_PARAM);
  ASSERT_STR_EQ(poly_instance_buf_name(inst, 1), "x");
  ASSERT_INT_EQ(poly_instance_buf_role(inst, 1), POLY_ROLE_INPUT);
  ASSERT_STR_EQ(poly_instance_buf_name(inst, 2), "output");
  ASSERT_INT_EQ(poly_instance_buf_role(inst, 2), POLY_ROLE_OUTPUT);

  /* Verify param count */
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "w");

  /* Verify named accessors */
  ASSERT_TRUE(poly_instance_get_buffer(inst, "w") == w);
  ASSERT_TRUE(poly_instance_get_sink(inst, "forward") == sink);
  ASSERT_TRUE(poly_instance_ctx(inst) == ctx);
  ASSERT_INT_EQ(poly_instance_buf_numel_named(inst, "w"), 4);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, instance_from_ctx_execute) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {4};

  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, s, 1, "w");
  PolyUOp *x = poly_input(ctx, POLY_FLOAT32, s, 1, "x");
  PolyUOp *out = poly_output(ctx, POLY_FLOAT32, s, 1, "output");

  PolyUOp *prod =
      poly_alu2(ctx, POLY_OP_MUL, poly_reshape(ctx, w, s, 1), poly_reshape(ctx, x, s, 1));
  PolyUOp *store = poly_store_val(ctx, out, prod);
  PolyUOp *sink = poly_sink1(ctx, store);
  poly_register_entrypoint(ctx, "forward", sink);

  PolyInstance *inst = poly_instance_from_ctx(ctx);
  ASSERT_TRUE(inst != NULL);

  /* Set weights */
  int64_t numel;
  float *w_data = poly_instance_buf_data_named(inst, "w", &numel);
  ASSERT_INT_EQ(numel, 4);
  for (int i = 0; i < 4; i++)
    w_data[i] = (float)(i + 1);

  /* Execute forward */
  float x_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  int rc = poly_instance_call(inst, "forward", io, 1);
  ASSERT_INT_EQ(rc, 0);

  /* Check output */
  float *out_data = poly_instance_buf_data_named(inst, "output", NULL);
  ASSERT_FLOAT_EQ(out_data[0], 2.0f, 1e-5); /* 1*2 */
  ASSERT_FLOAT_EQ(out_data[1], 6.0f, 1e-5); /* 2*3 */
  ASSERT_FLOAT_EQ(out_data[2], 12.0f, 1e-5); /* 3*4 */
  ASSERT_FLOAT_EQ(out_data[3], 20.0f, 1e-5); /* 4*5 */

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, instance_from_ctx_alias_shares_data) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {8};

  PolyUOp *emb = poly_param(ctx, POLY_FLOAT32, s, 1, "embedding");
  poly_alias(ctx, "lm_head", "embedding");

  /* Trivial graph that uses the embedding buffer */
  PolyUOp *store =
      poly_store_val(ctx, emb, poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0)));
  PolyUOp *sink = poly_sink1(ctx, store);
  poly_register_entrypoint(ctx, "init", sink);

  PolyInstance *inst = poly_instance_from_ctx(ctx);
  ASSERT_TRUE(inst != NULL);

  /* Both names in instance */
  ASSERT_INT_EQ(poly_instance_buf_count(inst), 2);

  /* Both resolve to the same data pointer (shared allocation) */
  float *emb_data = poly_instance_buf_data_named(inst, "embedding", NULL);
  float *lm_data = poly_instance_buf_data_named(inst, "lm_head", NULL);
  ASSERT_TRUE(emb_data != NULL);
  ASSERT_EQ(emb_data, lm_data);

  /* Writing through one name is visible through the other */
  emb_data[0] = 42.0f;
  ASSERT_FLOAT_EQ(lm_data[0], 42.0f, 0);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, instance_from_ctx_unreachable_excluded) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {4};

  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, s, 1, "w");
  PolyUOp *x = poly_input(ctx, POLY_FLOAT32, s, 1, "x");
  poly_aux(ctx, POLY_FLOAT32, s, 1, "unused_aux"); /* not in graph */

  PolyUOp *store = poly_store_val(ctx, w, poly_reshape(ctx, x, s, 1));
  PolyUOp *sink = poly_sink1(ctx, store);
  poly_register_entrypoint(ctx, "copy", sink);

  PolyInstance *inst = poly_instance_from_ctx(ctx);
  ASSERT_TRUE(inst != NULL);

  /* Only w and x are reachable; unused_aux is excluded */
  ASSERT_INT_EQ(poly_instance_buf_count(inst), 2);
  ASSERT_TRUE(poly_instance_get_buffer(inst, "w") != NULL);
  ASSERT_TRUE(poly_instance_get_buffer(inst, "x") != NULL);
  ASSERT_TRUE(poly_instance_get_buffer(inst, "unused_aux") == NULL);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, instance_from_ctx_reachability_scan_rewinds_scratch) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {4};

  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, s, 1, "w");
  PolyUOp *x = poly_input(ctx, POLY_FLOAT32, s, 1, "x");
  PolyUOp *out = poly_output(ctx, POLY_FLOAT32, s, 1, "output");

  PolyUOp *mul =
      poly_alu2(ctx, POLY_OP_MUL, poly_reshape(ctx, w, s, 1), poly_reshape(ctx, x, s, 1));
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, mul));
  poly_register_entrypoint(ctx, "forward", sink);

  size_t scratch_before = poly_arena_used(ctx->scratch);
  PolyInstance *inst = poly_instance_from_ctx(ctx);
  ASSERT_TRUE(inst != NULL);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, instance_from_ctx_zero_entrypoints) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t s[] = {4};
  poly_param(ctx, POLY_FLOAT32, s, 1, "w");

  /* No entrypoints registered */
  PolyInstance *inst = poly_instance_from_ctx(ctx);
  ASSERT_EQ(inst, NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(registry, instance_from_ctx_parity_with_ir) {
  /* Build same model via registry API and manual PolyIrSpec, compare results */
  int N = 4;

  /* Registry API fixture. */
  PolyCtx *ctx_a = poly_ctx_new();
  int64_t s[] = {N};
  PolyUOp *wa = poly_param(ctx_a, POLY_FLOAT32, s, 1, "w");
  PolyUOp *xa = poly_input(ctx_a, POLY_FLOAT32, s, 1, "x");
  PolyUOp *oa = poly_output(ctx_a, POLY_FLOAT32, s, 1, "output");
  PolyUOp *prod_a =
      poly_alu2(ctx_a, POLY_OP_MUL, poly_reshape(ctx_a, wa, s, 1), poly_reshape(ctx_a, xa, s, 1));
  PolyUOp *st_a = poly_store_val(ctx_a, oa, prod_a);
  PolyUOp *sink_a = poly_sink1(ctx_a, st_a);
  poly_register_entrypoint(ctx_a, "forward", sink_a);
  PolyInstance *inst_a = poly_instance_from_ctx(ctx_a);
  ASSERT_TRUE(inst_a != NULL);

  /* Manual PolyIrSpec fixture. */
  PolyCtx *ctx_b = poly_ctx_new();
  PolyUOp *wb = poly_buffer_f32(ctx_b, N);
  PolyUOp *xb = poly_buffer_f32(ctx_b, N);
  PolyUOp *ob = poly_buffer_f32(ctx_b, N);
  PolyUOp *prod_b = poly_alu2(ctx_b, POLY_OP_MUL, wb, xb);
  PolyUOp *st_b = poly_store_val(ctx_b, ob, prod_b);
  PolyUOp *sink_b = poly_sink1(ctx_b, st_b);
  PolyIrBufEntry bufs_b[] = {
      {.name = "w", .role = POLY_IR_ROLE_PARAM, .buffer = wb, .shape = {N}, .ndim = 1},
      {.name = "x", .role = POLY_IR_ROLE_INPUT, .buffer = xb, .shape = {N}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = ob, .shape = {N}, .ndim = 1},
  };
  PolyIrEntrypoint eps_b[] = {{.name = "forward", .sink = sink_b}};
  PolyIrSpec spec_b = {ctx_b, bufs_b, 3, eps_b, 1, NULL, 0};
  int ir_len;
  uint8_t *ir = poly_ir_export(&spec_b, &ir_len);
  PolyInstance *inst_b = poly_instance_from_ir(ir, ir_len, NULL, 0);
  free(ir);
  poly_ctx_destroy(ctx_b);
  ASSERT_TRUE(inst_b != NULL);

  /* Same weights + input */
  float *wa_d = poly_instance_buf_data_named(inst_a, "w", NULL);
  float *wb_d = poly_instance_buf_data_named(inst_b, "w", NULL);
  float x_in[] = {2, 3, 4, 5};
  for (int i = 0; i < N; i++) {
    wa_d[i] = (float)(i + 1);
    wb_d[i] = (float)(i + 1);
  }

  PolyIOBinding io_a[] = {POLY_IO_BINDING_ARRAY("x", x_in, POLY_FLOAT32)};
  PolyIOBinding io_b[] = {POLY_IO_BINDING_ARRAY("x", x_in, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_call(inst_a, "forward", io_a, 1), 0);
  ASSERT_INT_EQ(poly_instance_call(inst_b, "forward", io_b, 1), 0);

  float *oa_d = poly_instance_buf_data_named(inst_a, "output", NULL);
  float *ob_d = poly_instance_buf_data_named(inst_b, "output", NULL);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(oa_d[i], ob_d[i], 1e-6);

  poly_instance_free(inst_a);
  poly_instance_free(inst_b);
  poly_ctx_destroy(ctx_a);
  PASS();
}
