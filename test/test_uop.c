/*
 * test_uop.c — Tests for UOp creation, CSE, and toposort
 */

#include "test_harness.h"
#include "../src/polygrad.h"

/* ── Basic creation ───────────────────────────────────────────────────── */

TEST(uop, create_const) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.14));
  ASSERT_NOT_NULL(c);
  ASSERT_EQ(c->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(c->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(c->n_src, 0);
  ASSERT_EQ(c->arg.kind, POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(c->arg.f, 3.14, 1e-10);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, create_with_sources) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());

  ASSERT_NOT_NULL(add);
  ASSERT_EQ(add->op, POLY_OP_ADD);
  ASSERT_INT_EQ(add->n_src, 2);
  ASSERT_PTR_EQ(add->src[0], a);
  ASSERT_PTR_EQ(add->src[1], b);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, create_int_arg) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  ASSERT_EQ(c->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(c->arg.i, 42);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, create_string_arg) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_str("CPU"));
  ASSERT_EQ(u->arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(u->arg.str, "CPU");
  /* string should be arena-copied, not pointing to the original */
  ASSERT_PTR_NEQ(u->arg.str, "CPU");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, create_int_tuple_arg) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t perm[] = {1, 0, 2};
  PolyArg arg = { .kind = POLY_ARG_INT_TUPLE, .int_tuple = { perm, 3 } };
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *u = poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, a, arg);
  ASSERT_EQ(u->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(u->arg.int_tuple.n, 3);
  ASSERT_INT_EQ(u->arg.int_tuple.vals[0], 1);
  ASSERT_INT_EQ(u->arg.int_tuple.vals[1], 0);
  ASSERT_INT_EQ(u->arg.int_tuple.vals[2], 2);
  /* should be arena-copied */
  ASSERT_PTR_NEQ(u->arg.int_tuple.vals, perm);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── CSE (Common Subexpression Elimination) ───────────────────────────── */

TEST(uop, cse_same_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  ASSERT_PTR_EQ(a, b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cse_different_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  ASSERT_PTR_NEQ(a, b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cse_different_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT64, poly_arg_float(1.0));
  ASSERT_PTR_NEQ(a, b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cse_same_add) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *y = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *add1 = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, y, poly_arg_none());
  PolyUOp *add2 = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, y, poly_arg_none());
  ASSERT_PTR_EQ(add1, add2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cse_different_order) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *y = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *add1 = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, y, poly_arg_none());
  PolyUOp *add2 = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, y, x, poly_arg_none());
  /* Different source order → different UOps (CSE is identity-based) */
  ASSERT_PTR_NEQ(add1, add2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cse_int_tuple) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  int64_t perm1[] = {1, 0};
  int64_t perm2[] = {1, 0};
  PolyArg arg1 = { .kind = POLY_ARG_INT_TUPLE, .int_tuple = { perm1, 2 } };
  PolyArg arg2 = { .kind = POLY_ARG_INT_TUPLE, .int_tuple = { perm2, 2 } };
  PolyUOp *u1 = poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, a, arg1);
  PolyUOp *u2 = poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, a, arg2);
  ASSERT_PTR_EQ(u1, u2);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Toposort ─────────────────────────────────────────────────────────── */

TEST(uop, toposort_single) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  int n;
  PolyUOp **sorted = poly_toposort(ctx, c, &n);
  ASSERT_INT_EQ(n, 1);
  ASSERT_PTR_EQ(sorted[0], c);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, toposort_linear_chain) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *c = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, b, poly_arg_none());

  int n;
  PolyUOp **sorted = poly_toposort(ctx, c, &n);
  ASSERT_INT_EQ(n, 3);
  /* Sources before consumers */
  ASSERT_PTR_EQ(sorted[0], a);
  ASSERT_PTR_EQ(sorted[1], b);
  ASSERT_PTR_EQ(sorted[2], c);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, toposort_diamond) {
  PolyCtx *ctx = poly_ctx_new();
  /*    a
   *   / \
   *  b   c
   *   \ /
   *    d
   */
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *c = poly_uop1(ctx, POLY_OP_SQRT, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *d = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, b, c, poly_arg_none());

  int n;
  PolyUOp **sorted = poly_toposort(ctx, d, &n);
  ASSERT_INT_EQ(n, 4);  /* a, b, c, d — each appears once */
  ASSERT_PTR_EQ(sorted[0], a);  /* a first (leaf) */
  ASSERT_PTR_EQ(sorted[3], d);  /* d last (root) */
  /* b and c can be in either order but both must come before d */
  ASSERT_TRUE((sorted[1] == b && sorted[2] == c) || (sorted[1] == c && sorted[2] == b));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, toposort_shared_subgraph) {
  PolyCtx *ctx = poly_ctx_new();
  /* shared = a + b, root = shared * shared
   * Should not duplicate shared in toposort */
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.0));
  PolyUOp *shared = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, shared, shared, poly_arg_none());

  int n;
  PolyUOp **sorted = poly_toposort(ctx, root, &n);
  ASSERT_INT_EQ(n, 4);  /* a, b, shared, root */
  ASSERT_PTR_EQ(sorted[3], root);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Ops helpers ──────────────────────────────────────────────────────── */

TEST(ops, op_name) {
  ASSERT_STR_EQ(poly_op_name(POLY_OP_ADD), "ADD");
  ASSERT_STR_EQ(poly_op_name(POLY_OP_CONST), "CONST");
  ASSERT_STR_EQ(poly_op_name(POLY_OP_SINK), "SINK");
  ASSERT_STR_EQ(poly_op_name(POLY_OP_RESHAPE), "RESHAPE");
  PASS();
}

TEST(ops, opset) {
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_UNARY, POLY_OP_EXP2));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_UNARY, POLY_OP_NEG));
  ASSERT_FALSE(poly_opset_has(POLY_GROUP_UNARY, POLY_OP_ADD));

  ASSERT_TRUE(poly_opset_has(POLY_GROUP_BINARY, POLY_OP_ADD));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_BINARY, POLY_OP_MUL));
  ASSERT_FALSE(poly_opset_has(POLY_GROUP_BINARY, POLY_OP_NEG));

  ASSERT_TRUE(poly_opset_has(POLY_GROUP_ALU, POLY_OP_ADD));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_ALU, POLY_OP_NEG));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_ALU, POLY_OP_WHERE));
  ASSERT_FALSE(poly_opset_has(POLY_GROUP_ALU, POLY_OP_CONST));

  ASSERT_TRUE(poly_opset_has(POLY_GROUP_MOVEMENT, POLY_OP_RESHAPE));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_MOVEMENT, POLY_OP_PERMUTE));
  ASSERT_FALSE(poly_opset_has(POLY_GROUP_MOVEMENT, POLY_OP_ADD));

  ASSERT_TRUE(poly_opset_has(POLY_GROUP_COMMUTATIVE, POLY_OP_ADD));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_COMMUTATIVE, POLY_OP_MUL));
  ASSERT_FALSE(poly_opset_has(POLY_GROUP_COMMUTATIVE, POLY_OP_SUB));
  PASS();
}

/* ── Pretty-print ─────────────────────────────────────────────────────── */

TEST(uop, print_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.14));
  char *s = poly_uop_str(c);
  ASSERT_NOT_NULL(s);
  /* Should contain "CONST" and "float" and "3.14" */
  ASSERT_TRUE(strstr(s, "CONST") != NULL);
  ASSERT_TRUE(strstr(s, "float") != NULL);
  ASSERT_TRUE(strstr(s, "3.14") != NULL);
  free(s);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Arena basics ─────────────────────────────────────────────────────── */

TEST(arena, alloc_and_destroy) {
  PolyArena *a = poly_arena_new(1024);
  ASSERT_NOT_NULL(a);

  void *p1 = poly_arena_alloc(a, 64, 8);
  ASSERT_NOT_NULL(p1);
  void *p2 = poly_arena_alloc(a, 128, 8);
  ASSERT_NOT_NULL(p2);
  ASSERT_PTR_NEQ(p1, p2);

  ASSERT_TRUE(poly_arena_used(a) >= 192);
  poly_arena_destroy(a);
  PASS();
}

TEST(arena, large_alloc) {
  PolyArena *a = poly_arena_new(64);  /* tiny initial block */
  /* allocate more than initial capacity */
  void *p = poly_arena_alloc(a, 256, 8);
  ASSERT_NOT_NULL(p);
  poly_arena_destroy(a);
  PASS();
}

TEST(arena, reset) {
  PolyArena *a = poly_arena_new(1024);
  poly_arena_alloc(a, 512, 8);
  ASSERT_TRUE(poly_arena_used(a) >= 512);
  poly_arena_reset(a);
  ASSERT_INT_EQ(poly_arena_used(a), 0);
  /* Can allocate again after reset */
  void *p = poly_arena_alloc(a, 64, 8);
  ASSERT_NOT_NULL(p);
  poly_arena_destroy(a);
  PASS();
}

/* ── Hashmap ──────────────────────────────────────────────────────────── */

static bool int_key_eq(const void *a, const void *b) {
  return *(const int*)a == *(const int*)b;
}

TEST(hashmap, basic_set_get) {
  PolyMap *m = poly_map_new(16);
  ASSERT_NOT_NULL(m);

  int key1 = 42;
  int key2 = 99;
  poly_map_set(m, 42, &key1, (void*)(uintptr_t)100, int_key_eq);
  poly_map_set(m, 99, &key2, (void*)(uintptr_t)200, int_key_eq);

  ASSERT_INT_EQ(poly_map_len(m), 2);
  ASSERT_EQ((uintptr_t)poly_map_get(m, 42, &key1, int_key_eq), 100);
  ASSERT_EQ((uintptr_t)poly_map_get(m, 99, &key2, int_key_eq), 200);

  int key3 = 77;
  ASSERT_EQ(poly_map_get(m, 77, &key3, int_key_eq), NULL);

  poly_map_destroy(m);
  PASS();
}

TEST(hashmap, remove) {
  PolyMap *m = poly_map_new(16);
  int key1 = 42;
  poly_map_set(m, 42, &key1, (void*)(uintptr_t)100, int_key_eq);
  ASSERT_INT_EQ(poly_map_len(m), 1);

  poly_map_remove(m, 42, &key1, int_key_eq);
  ASSERT_INT_EQ(poly_map_len(m), 0);
  ASSERT_EQ(poly_map_get(m, 42, &key1, int_key_eq), NULL);

  poly_map_destroy(m);
  PASS();
}

TEST(hashmap, grow) {
  PolyMap *m = poly_map_new(4);
  int keys[100];
  for (int i = 0; i < 100; i++) {
    keys[i] = i;
    poly_map_set(m, (uint32_t)i, &keys[i], (void*)(uintptr_t)(i + 1000), int_key_eq);
  }
  ASSERT_INT_EQ(poly_map_len(m), 100);

  /* verify all values */
  for (int i = 0; i < 100; i++) {
    void *v = poly_map_get(m, (uint32_t)i, &keys[i], int_key_eq);
    ASSERT_EQ((uintptr_t)v, (uintptr_t)(i + 1000));
  }

  poly_map_destroy(m);
  PASS();
}
