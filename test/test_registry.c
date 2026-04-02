/*
 * test_registry.c -- Tests for the named buffer registry on PolyCtx
 */

#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/frontend.h"

/* ── Registration + lookup ─────────────────────────────────────────── */

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

/* ── Idempotent re-registration ────────────────────────────────────── */

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

/* ── Mismatch rejection ────────────────────────────────────────────── */

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
  int64_t s2[] = {4, 3};  /* same numel, different shape */

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

/* ── Alias ─────────────────────────────────────────────────────────── */

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

/* ── Entrypoints ───────────────────────────────────────────────────── */

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

/* ── Enumeration ───────────────────────────────────────────────────── */

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

/* ── Printf-style naming ──────────────────────────────────────────── */

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
