/*
 * test_sym.c — Tests for symbolic simplification + ALU constant folding
 */

#include "test_harness.h"
#include "../src/bigint.h"
#include "../src/ctx.h"
#include "../src/pat.h"
#include "../src/tensor.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>

/* Helper: apply symbolic_simple via graph_rewrite */

static PolyUOp *simplify(PolyCtx *ctx, PolyUOp *root) {
  return poly_graph_rewrite(ctx, root, poly_symbolic_simple());
}

static bool integer_const_eq(PolyUOp *u, const char *expected);

/* ALU constant fold tests */

TEST(alu, fold_add_int) {
  PolyArg ops[2] = {poly_arg_int(2), poly_arg_int(3)};
  PolyArg r = poly_exec_alu(POLY_OP_ADD, POLY_INT32, ops, 2, true);
  ASSERT_INT_EQ(r.i, 5);
  PASS();
}

TEST(alu, fold_mul_int) {
  PolyArg ops[2] = {poly_arg_int(4), poly_arg_int(7)};
  PolyArg r = poly_exec_alu(POLY_OP_MUL, POLY_INT32, ops, 2, true);
  ASSERT_INT_EQ(r.i, 28);
  PASS();
}

TEST(alu, fold_neg_float) {
  PolyArg ops[1] = {poly_arg_float(3.14)};
  PolyArg r = poly_exec_alu(POLY_OP_NEG, POLY_FLOAT32, ops, 1, true);
  ASSERT_FLOAT_EQ(r.f, -3.14, 1e-6);
  PASS();
}

TEST(alu, fold_add_float) {
  PolyArg ops[2] = {poly_arg_float(1.5), poly_arg_float(2.5)};
  PolyArg r = poly_exec_alu(POLY_OP_ADD, POLY_FLOAT32, ops, 2, true);
  ASSERT_FLOAT_EQ(r.f, 4.0, 1e-6);
  PASS();
}

TEST(alu, cast_without_output_truncation_matches_host_symbolic_conversion) {
  /* Pinned tinygrad UOp._sym_fxn renders CAST as a host-language conversion
   * (ops.py:1021-1035), so inference does not narrow to the destination width. */
  PolyArg operand = poly_arg_int(260);
  PolyArg host = poly_exec_alu(POLY_OP_CAST, POLY_UINT8, &operand, 1, false);
  PolyArg typed = poly_exec_alu(POLY_OP_CAST, POLY_UINT8, &operand, 1, true);
  ASSERT_TRUE(host.kind == POLY_ARG_INT);
  ASSERT_INT_EQ(host.i, 260);
  ASSERT_TRUE(typed.kind == POLY_ARG_INT);
  ASSERT_INT_EQ(typed.i, 4);
  PASS();
}

TEST(alu, fold_idiv) {
  PolyArg ops[2] = {poly_arg_int(7), poly_arg_int(3)};
  PolyArg r = poly_exec_alu(POLY_OP_IDIV, POLY_INT32, ops, 2, true);
  ASSERT_INT_EQ(r.i, 2);
  PASS();
}

TEST(alu, fold_mod) {
  PolyArg ops[2] = {poly_arg_int(7), poly_arg_int(3)};
  PolyArg r = poly_exec_alu(POLY_OP_MOD, POLY_INT32, ops, 2, true);
  ASSERT_INT_EQ(r.i, 1);
  PASS();
}

TEST(alu, fold_floordiv_floormod_signed_like_tinygrad) {
  const int64_t vals[][4] = {
      {-7, 3, -3, 2}, {-7, -3, 2, -1}, {7, -3, -3, -2}, {7, 3, 2, 1},
      {-1, 4, -1, 3}, {1, -4, -1, -3}, {0, 3, 0, 0},
  };
  for (int i = 0; i < (int)(sizeof(vals) / sizeof(vals[0])); i++) {
    PolyArg ops[2] = {poly_arg_int(vals[i][0]), poly_arg_int(vals[i][1])};
    PolyArg q = poly_exec_alu(POLY_OP_FLOORDIV, POLY_INT32, ops, 2, true);
    PolyArg r = poly_exec_alu(POLY_OP_FLOORMOD, POLY_INT32, ops, 2, true);
    ASSERT_INT_EQ(q.i, vals[i][2]);
    ASSERT_INT_EQ(r.i, vals[i][3]);
  }
  PASS();
}

TEST(alu, fold_cmplt) {
  PolyArg ops[2] = {poly_arg_int(2), poly_arg_int(5)};
  PolyArg r = poly_exec_alu(POLY_OP_CMPLT, POLY_INT32, ops, 2, true);
  ASSERT_TRUE(r.b == true);
  ops[0] = poly_arg_int(5);
  r = poly_exec_alu(POLY_OP_CMPLT, POLY_INT32, ops, 2, true);
  ASSERT_TRUE(r.b == false);
  PASS();
}

TEST(alu, fold_where) {
  PolyArg ops[3] = {poly_arg_bool(true), poly_arg_int(10), poly_arg_int(20)};
  PolyArg r = poly_exec_alu(POLY_OP_WHERE, POLY_INT32, ops, 3, true);
  ASSERT_INT_EQ(r.i, 10);
  ops[0] = poly_arg_bool(false);
  r = poly_exec_alu(POLY_OP_WHERE, POLY_INT32, ops, 3, true);
  ASSERT_INT_EQ(r.i, 20);
  PASS();
}

TEST(alu, raw_bool_neg_matches_pinned_arithmetic_then_bool_truncation) {
  /* Pinned uop/ops.py:1182-1197 maps NEG to operator.neg, then applies
   * bool() only when truncate_output is true. Raw bool NEG is not logical NOT. */
  PolyArg f[1] = {poly_arg_bool(false)};
  PolyArg t[1] = {poly_arg_bool(true)};
  PolyArg f_raw = poly_exec_alu(POLY_OP_NEG, POLY_BOOL, f, 1, false);
  PolyArg t_raw = poly_exec_alu(POLY_OP_NEG, POLY_BOOL, t, 1, false);
  PolyArg f_typed = poly_exec_alu(POLY_OP_NEG, POLY_BOOL, f, 1, true);
  PolyArg t_typed = poly_exec_alu(POLY_OP_NEG, POLY_BOOL, t, 1, true);
  ASSERT_TRUE(f_raw.kind == POLY_ARG_INT && f_raw.i == 0);
  ASSERT_TRUE(t_raw.kind == POLY_ARG_INT && t_raw.i == -1);
  ASSERT_TRUE(f_typed.kind == POLY_ARG_BOOL && !f_typed.b);
  ASSERT_TRUE(t_typed.kind == POLY_ARG_BOOL && t_typed.b);
  PASS();
}

TEST(alu, divmod_zero_and_extreme_floor_remainder_match_pinned_helpers) {
  /* Pinned helpers.py:69-74 defines remainder through x-div(x,y)*y. */
  PolyArg zero_divisor[2] = {poly_arg_int(7), poly_arg_int(0)};
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_IDIV, POLY_INT64, zero_divisor, 2, false).i, 0);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_MOD, POLY_INT64, zero_divisor, 2, false).i, 7);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_FLOORDIV, POLY_INT64, zero_divisor, 2, false).i, 0);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_FLOORMOD, POLY_INT64, zero_divisor, 2, false).i, 7);

  PolyArg extreme[2] = {poly_arg_int(1), poly_arg_int(INT64_MIN)};
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_FLOORDIV, POLY_INT64, extreme, 2, false).i, -1);
  ASSERT_TRUE(poly_exec_alu(POLY_OP_FLOORMOD, POLY_INT64, extreme, 2, false).i == -INT64_MAX);
  PASS();
}

TEST(alu, weak_index_where_preserves_invalid_with_truncation_enabled) {
  /* weakint is absent from pinned dtype.truncate (dtype.py:351-355). */
  PolyArg operands[3] = {poly_arg_bool(false), poly_arg_int(7), poly_arg_invalid()};
  PolyArg raw = poly_exec_alu(POLY_OP_WHERE, POLY_INDEX, operands, 3, false);
  PolyArg typed = poly_exec_alu(POLY_OP_WHERE, POLY_INDEX, operands, 3, true);
  ASSERT_TRUE(raw.kind == POLY_ARG_INVALID);
  ASSERT_TRUE(typed.kind == POLY_ARG_INVALID);
  PASS();
}

TEST(alu, negative_integer_pow_matches_pinned_raw_and_refuses_fixed_width_truncation) {
  /* Pinned safe_pow returns a float before ctypes fixed-width truncation
   * rejects it (uop/ops.py:1178-1197). weakint has no truncation entry. */
  PolyArg finite[2] = {poly_arg_int(2), poly_arg_int(-1)};
  PolyArg raw = poly_exec_alu(POLY_OP_POW, POLY_INT32, finite, 2, false);
  PolyArg typed = poly_exec_alu(POLY_OP_POW, POLY_INT32, finite, 2, true);
  PolyArg weak_typed = poly_exec_alu(POLY_OP_POW, POLY_INDEX, finite, 2, true);
  ASSERT_TRUE(raw.kind == POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(raw.f, 0.5, 0.0);
  ASSERT_TRUE(typed.kind == POLY_ARG_INVALID);
  ASSERT_TRUE(weak_typed.kind == POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(weak_typed.f, 0.5, 0.0);

  PolyArg infinite[2] = {poly_arg_int(0), poly_arg_int(-1)};
  PolyArg inf_raw = poly_exec_alu(POLY_OP_POW, POLY_INT32, infinite, 2, false);
  ASSERT_TRUE(inf_raw.kind == POLY_ARG_FLOAT && isinf(inf_raw.f) && inf_raw.f > 0.0);
  PASS();
}

TEST(alu, raw_integer_operands_match_pinned_python_alu_before_output_truncation) {
  /* Pinned tinygrad uop/ops.py:1182-1197 applies python_alu to stored CONST
   * values first. Nominal signedness/width only truncates the result. */
  PolyArg pow_args[2] = {poly_arg_int(2), poly_arg_int(3)};
  PolyArg cmp_args[2] = {poly_arg_int(130), poly_arg_int(0)};
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_POW, POLY_INT32, pow_args, 2, false).i, 8);
  ASSERT_TRUE(!poly_exec_alu(POLY_OP_CMPLT, POLY_INT8, cmp_args, 2, false).b);

  const PolyOps divmod_ops[] = {
      POLY_OP_CDIV,
      POLY_OP_CMOD,
      POLY_OP_FLOORDIV,
      POLY_OP_FLOORMOD,
  };
  const int64_t raw_expected[] = {-1, -1, -2, 1};
  const int64_t u8_expected[] = {255, 255, 254, 1};
  PolyArg div_args[2] = {poly_arg_int(-3), poly_arg_int(2)};
  for (int i = 0; i < 4; i++) {
    PolyArg raw = poly_exec_alu(divmod_ops[i], POLY_UINT8, div_args, 2, false);
    PolyArg truncated = poly_exec_alu(divmod_ops[i], POLY_UINT8, div_args, 2, true);
    ASSERT_TRUE(raw.kind == POLY_ARG_INT);
    ASSERT_TRUE(truncated.kind == POLY_ARG_INT);
    ASSERT_INT_EQ(raw.i, raw_expected[i]);
    ASSERT_INT_EQ(truncated.i, u8_expected[i]);
  }

  PolyArg where_args[3] = {poly_arg_bool(true), poly_arg_int(-3), poly_arg_int(2)};
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_WHERE, POLY_UINT8, where_args, 3, false).i, -3);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_WHERE, POLY_UINT8, where_args, 3, true).i, 253);
  PASS();
}

TEST(sym, raw_integer_const_folds_match_pinned_python_alu) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *two_i32 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *three_i32 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *pow =
      simplify(ctx, poly_uop2(ctx, POLY_OP_POW, POLY_INT32, two_i32, three_i32, poly_arg_none()));
  ASSERT_NOT_NULL(pow);
  ASSERT_INT_EQ(pow->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(pow->dtype, POLY_INT32));
  ASSERT_INT_EQ(pow->arg.i, 8);

  PolyUOp *raw_i8 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT8, poly_arg_int(130));
  PolyUOp *zero_i8 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT8, poly_arg_int(0));
  PolyUOp *cmp =
      simplify(ctx, poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, raw_i8, zero_i8, poly_arg_none()));
  ASSERT_NOT_NULL(cmp);
  ASSERT_INT_EQ(cmp->op, POLY_OP_CONST);
  ASSERT_TRUE(cmp->arg.kind == POLY_ARG_BOOL);
  ASSERT_TRUE(!cmp->arg.b);

  PolyUOp *neg_three_u8 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(-3));
  PolyUOp *two_u8 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(2));
  PolyUOp *floordiv = simplify(
      ctx, poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_UINT8, neg_three_u8, two_u8, poly_arg_none())
  );
  ASSERT_NOT_NULL(floordiv);
  ASSERT_INT_EQ(floordiv->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(floordiv->dtype, POLY_UINT8));
  ASSERT_INT_EQ(floordiv->arg.i, -2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(alu, compare_int64_preserves_precision) {
  PolyArg ops[2] = {poly_arg_int(9007199254740992LL), poly_arg_int(9007199254740993LL)};
  PolyArg r = poly_exec_alu(POLY_OP_CMPNE, POLY_INT64, ops, 2, true);
  ASSERT_TRUE(r.b == true);
  r = poly_exec_alu(POLY_OP_CMPEQ, POLY_INT64, ops, 2, true);
  ASSERT_TRUE(r.b == false);
  PASS();
}

TEST(alu, compare_uint64_const_args_remain_raw_before_result_truncation) {
  /* A genuinely positive uint64 above INT64_MAX is PG-PARITY-020. Do not
   * reinterpret a stored negative CONST as that missing representation:
   * pinned exec_alu compares the raw Python values (uop/ops.py:1192-1197). */
  PolyArg ops[2] = {poly_arg_int(-3), poly_arg_int(2)};
  PolyArg r = poly_exec_alu(POLY_OP_CMPLT, POLY_UINT64, ops, 2, true);
  ASSERT_TRUE(r.b == true);
  PolyArg rev[2] = {ops[1], ops[0]};
  r = poly_exec_alu(POLY_OP_CMPLT, POLY_UINT64, rev, 2, true);
  ASSERT_TRUE(r.b == false);
  PASS();
}

TEST(sym, vector_compare_uint64_const_fold_uses_raw_args) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType u64x2 = poly_dtype_vec(POLY_UINT64, 2);
  PolyDType b2 = poly_dtype_vec(POLY_BOOL, 2);

  PolyUOp *a0 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(-3));
  PolyUOp *a1 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(4));
  PolyUOp *b0 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(2));
  PolyUOp *b1 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(-5));
  PolyUOp *avec_src[2] = {a0, a1};
  PolyUOp *bvec_src[2] = {b0, b1};
  PolyUOp *avec = poly_uop(ctx, POLY_OP_STACK, u64x2, avec_src, 2, poly_arg_none());
  PolyUOp *bvec = poly_uop(ctx, POLY_OP_STACK, u64x2, bvec_src, 2, poly_arg_none());
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, b2, avec, bvec, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, cmp, poly_symbolic());
  ASSERT_TRUE(r != NULL);
  ASSERT_TRUE(r->op == POLY_OP_STACK);
  ASSERT_INT_EQ(r->n_src, 2);
  ASSERT_TRUE(r->src[0]->op == POLY_OP_CONST && r->src[0]->arg.kind == POLY_ARG_BOOL);
  ASSERT_TRUE(r->src[1]->op == POLY_OP_CONST && r->src[1]->arg.kind == POLY_ARG_BOOL);
  ASSERT_TRUE(r->src[0]->arg.b == true);
  ASSERT_TRUE(r->src[1]->arg.b == false);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, negative_integer_pow_const_folding_normalizes_or_refuses_like_pinned) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *two_i32 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *zero_i32 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *neg_one_i32 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-1));

  /* symbolic.py:30-32 evaluates 0.5, then UOp.const_like/DType.const turns it
   * into the integer CONST 0. */
  PolyUOp *finite =
      simplify(ctx, poly_uop2(ctx, POLY_OP_POW, POLY_INT32, two_i32, neg_one_i32, poly_arg_none()));
  ASSERT_NOT_NULL(finite);
  ASSERT_TRUE(finite->op == POLY_OP_CONST && finite->arg.kind == POLY_ARG_INT);
  ASSERT_INT_EQ(finite->arg.i, 0);

  /* Pinned DType.const raises on inf. C has no exception carrier, so retain
   * the POW graph instead of manufacturing an Invalid CONST. */
  PolyUOp *infinite_root =
      poly_uop2(ctx, POLY_OP_POW, POLY_INT32, zero_i32, neg_one_i32, poly_arg_none());
  PolyUOp *infinite = simplify(ctx, infinite_root);
  ASSERT_NOT_NULL(infinite);
  ASSERT_TRUE(infinite->op == POLY_OP_POW);

  PolyDType i32x2 = poly_dtype_vec(POLY_INT32, 2);
  PolyUOp *two_src[2] = {two_i32, two_i32};
  PolyUOp *neg_src[2] = {neg_one_i32, neg_one_i32};
  PolyUOp *vtwo = poly_uop(ctx, POLY_OP_STACK, i32x2, two_src, 2, poly_arg_none());
  PolyUOp *vneg = poly_uop(ctx, POLY_OP_STACK, i32x2, neg_src, 2, poly_arg_none());
  PolyUOp *fixed_root = poly_uop2(ctx, POLY_OP_POW, i32x2, vtwo, vneg, poly_arg_none());
  PolyUOp *fixed = simplify(ctx, fixed_root);
  ASSERT_NOT_NULL(fixed);
  ASSERT_TRUE(fixed->op == POLY_OP_POW);

  PolyDType weakx2 = poly_dtype_vec(POLY_INDEX, 2);
  PolyUOp *two_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *neg_one_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(-1));
  PolyUOp *two_idx_src[2] = {two_idx, two_idx};
  PolyUOp *neg_idx_src[2] = {neg_one_idx, neg_one_idx};
  PolyUOp *weak_two = poly_uop(ctx, POLY_OP_STACK, weakx2, two_idx_src, 2, poly_arg_none());
  PolyUOp *weak_neg = poly_uop(ctx, POLY_OP_STACK, weakx2, neg_idx_src, 2, poly_arg_none());
  PolyUOp *weak =
      simplify(ctx, poly_uop2(ctx, POLY_OP_POW, weakx2, weak_two, weak_neg, poly_arg_none()));
  ASSERT_NOT_NULL(weak);
  ASSERT_TRUE(weak->op == POLY_OP_STACK && weak->n_src == 2);
  for (int i = 0; i < 2; i++) {
    ASSERT_TRUE(weak->src[i]->op == POLY_OP_CONST);
    ASSERT_TRUE(poly_dtype_is_index(weak->src[i]->dtype));
    ASSERT_TRUE(weak->src[i]->arg.kind == POLY_ARG_INT);
    ASSERT_INT_EQ(weak->src[i]->arg.i, 0);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, uint64_vector_high_bit_intermediate_matches_pinned_exact_fold) {
  /* Pinned uop/ops.py:1192-1197 truncates each vector lane recursively:
   * (INT64_MAX+1)>>1 stays the positive uint64 value 2**62. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType u64x2 = poly_dtype_vec(POLY_UINT64, 2);
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(1));
  PolyUOp *max_src[2] = {max, max};
  PolyUOp *one_src[2] = {one, one};
  PolyUOp *vmax = poly_uop(ctx, POLY_OP_STACK, u64x2, max_src, 2, poly_arg_none());
  PolyUOp *vone = poly_uop(ctx, POLY_OP_STACK, u64x2, one_src, 2, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, u64x2, vmax, vone, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_SHR, u64x2, add, vone, poly_arg_none());
  PolyUOp *folded = simplify(ctx, root);
  ASSERT_NOT_NULL(folded);
  ASSERT_TRUE(folded->op == POLY_OP_STACK);
  ASSERT_INT_EQ(folded->n_src, 2);
  ASSERT_TRUE(folded->src[0] == folded->src[1]);
  ASSERT_TRUE(integer_const_eq(folded->src[0], "4611686018427387904"));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, negative_shift_count_does_not_fold_to_invalid_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *neg = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-1));
  PolyUOp *shl = poly_uop2(ctx, POLY_OP_SHL, POLY_INT32, one, neg, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, shl, poly_symbolic());
  ASSERT_TRUE(r != NULL);
  ASSERT_TRUE(r->op == POLY_OP_SHL);
  ASSERT_TRUE(!(r->op == POLY_OP_CONST && r->arg.kind == POLY_ARG_INVALID));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, vector_negative_shift_count_does_not_fold_to_invalid_stack) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType i32x2 = poly_dtype_vec(POLY_INT32, 2);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *neg = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-1));
  PolyUOp *lhs_src[2] = {one, two};
  PolyUOp *rhs_src[2] = {zero, neg};
  PolyUOp *lhs = poly_uop(ctx, POLY_OP_STACK, i32x2, lhs_src, 2, poly_arg_none());
  PolyUOp *rhs = poly_uop(ctx, POLY_OP_STACK, i32x2, rhs_src, 2, poly_arg_none());
  PolyUOp *shl = poly_uop2(ctx, POLY_OP_SHL, i32x2, lhs, rhs, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, shl, poly_symbolic());
  ASSERT_TRUE(r != NULL);
  ASSERT_TRUE(r->op == POLY_OP_SHL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, vector_float16_const_fold_uses_pinned_lane_truncation) {
  /* tinygrad ops.py:1192-1197 does not forward truncate_output through vector
   * recursion, so fixed-width lanes use the default truncating execution. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType f16x2 = poly_dtype_vec(POLY_FLOAT16, 2);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(1.0));
  PolyUOp *small = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(0.0001));
  PolyUOp *lhs_src[2] = {one, one};
  PolyUOp *rhs_src[2] = {small, small};
  PolyUOp *lhs = poly_uop(ctx, POLY_OP_STACK, f16x2, lhs_src, 2, poly_arg_none());
  PolyUOp *rhs = poly_uop(ctx, POLY_OP_STACK, f16x2, rhs_src, 2, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, f16x2, lhs, rhs, poly_arg_none());

  PolyUOp *folded = poly_graph_rewrite(ctx, add, poly_symbolic_simple());
  ASSERT_NOT_NULL(folded);
  ASSERT_INT_EQ(folded->op, POLY_OP_STACK);
  ASSERT_INT_EQ(folded->n_src, 2);
  for (int i = 0; i < folded->n_src; i++) {
    ASSERT_INT_EQ(folded->src[i]->op, POLY_OP_CONST);
    ASSERT_TRUE(poly_dtype_eq(folded->src[i]->dtype, POLY_FLOAT16));
    ASSERT_TRUE(folded->src[i]->arg.kind == POLY_ARG_FLOAT);
    ASSERT_FLOAT_EQ(folded->src[i]->arg.f, 1.0, 0.0);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

static bool integer_const_eq(PolyUOp *u, const char *expected) {
  if (!u || u->op != POLY_OP_CONST ||
      (u->arg.kind != POLY_ARG_INT && u->arg.kind != POLY_ARG_BIGINT))
    return false;
  char *actual = poly_arg_integer_to_decimal(u->arg);
  bool equal = actual && strcmp(actual, expected) == 0;
  free(actual);
  return equal;
}

TEST(sym, exact_symbolic_integer_results_match_pinned_python_topology) {
  /* Pinned tinygrad uop/symbolic.py:30-32,252,260-262 evaluates these
   * constants as exact Python integers. The approved PolyBigInt carrier must
   * preserve both their values and the resulting rewrite topology. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT64_MAX));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *var = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("x", 0, 1));

  PolyUOp *direct =
      simplify(ctx, poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, max, one, poly_arg_none()));
  ASSERT_NOT_NULL(direct);
  ASSERT_TRUE(integer_const_eq(direct, "9223372036854775808"));

  PolyUOp *assoc = poly_graph_rewrite(
      ctx,
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, var, max, poly_arg_none()), one, poly_arg_none()
      ),
      poly_symbolic()
  );
  ASSERT_NOT_NULL(assoc);
  ASSERT_INT_EQ(assoc->op, POLY_OP_ADD);
  ASSERT_TRUE(assoc->src[0] == var);
  ASSERT_TRUE(integer_const_eq(assoc->src[1], "9223372036854775808"));

  PolyUOp *distributed = poly_graph_rewrite(
      ctx,
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_INDEX, two,
          poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, var, max, poly_arg_none()), poly_arg_none()
      ),
      poly_symbolic()
  );
  ASSERT_NOT_NULL(distributed);
  ASSERT_INT_EQ(distributed->op, POLY_OP_ADD);
  ASSERT_INT_EQ(distributed->src[0]->op, POLY_OP_MUL);
  ASSERT_TRUE(distributed->src[0]->src[0] == var);
  ASSERT_TRUE(integer_const_eq(distributed->src[0]->src[1], "2"));
  ASSERT_TRUE(integer_const_eq(distributed->src[1], "18446744073709551614"));

  PolyDType weakx2 = poly_dtype_vec(POLY_INDEX, 2);
  PolyUOp *max_src[2] = {max, max};
  PolyUOp *one_src[2] = {one, one};
  PolyUOp *vmax = poly_uop(ctx, POLY_OP_STACK, weakx2, max_src, 2, poly_arg_none());
  PolyUOp *vone = poly_uop(ctx, POLY_OP_STACK, weakx2, one_src, 2, poly_arg_none());
  PolyUOp *vector = simplify(ctx, poly_uop2(ctx, POLY_OP_ADD, weakx2, vmax, vone, poly_arg_none()));
  ASSERT_NOT_NULL(vector);
  ASSERT_INT_EQ(vector->op, POLY_OP_STACK);
  ASSERT_INT_EQ(vector->n_src, 2);
  ASSERT_TRUE(vector->src[0] == vector->src[1]);
  ASSERT_TRUE(integer_const_eq(vector->src[0], "9223372036854775808"));

  PolyUOp *min = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT64_MIN));
  PolyUOp *neg = simplify(ctx, poly_uop1(ctx, POLY_OP_NEG, POLY_INDEX, min, poly_arg_none()));
  ASSERT_NOT_NULL(neg);
  ASSERT_TRUE(integer_const_eq(neg, "9223372036854775808"));

  PolyUOp *mulacc =
      simplify(ctx, poly_uop3(ctx, POLY_OP_MULACC, POLY_INDEX, max, two, zero, poly_arg_none()));
  ASSERT_NOT_NULL(mulacc);
  ASSERT_TRUE(integer_const_eq(mulacc, "18446744073709551614"));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, integral_trunc_is_identity_before_rendering) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyDType integral_dtypes[] = {
      POLY_INT32,
      POLY_UINT32,
      POLY_INT64,
      POLY_BOOL,
      POLY_INDEX,
      poly_dtype_vec(POLY_INT32, 4),
      poly_dtype_vec(POLY_BOOL, 4),
      poly_dtype_vec(POLY_INDEX, 4),
  };
  for (int i = 0; i < (int)(sizeof(integral_dtypes) / sizeof(integral_dtypes[0])); i++) {
    PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, integral_dtypes[i], poly_arg_int(i));
    PolyUOp *trunc = poly_uop1(ctx, POLY_OP_TRUNC, integral_dtypes[i], x, poly_arg_none());
    ASSERT_TRUE(simplify(ctx, trunc) == x);
  }

  PolyDType float_dtypes[] = {
      POLY_FLOAT32,
      POLY_FLOAT64,
      poly_dtype_vec(POLY_FLOAT32, 4),
  };
  for (int i = 0; i < (int)(sizeof(float_dtypes) / sizeof(float_dtypes[0])); i++) {
    PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, float_dtypes[i], poly_arg_int(32 + i));
    PolyUOp *trunc = poly_uop1(ctx, POLY_OP_TRUNC, float_dtypes[i], x, poly_arg_none());
    PolyUOp *rewritten = simplify(ctx, trunc);
    ASSERT_NOT_NULL(rewritten);
    ASSERT_INT_EQ(rewritten->op, POLY_OP_TRUNC);
    ASSERT_TRUE(rewritten->src[0] == x);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, integral_trunc_preserves_invalid_gate_ordering) {
  /* Pinned tinygrad/uop/symbolic.py:60-86 propagates Invalid before the
   * line-114 integral TRUNC identity. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_PARAM, POLY_BOOL, poly_arg_int(0));
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_INDEX, poly_arg_int(1));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *masked = poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, gate, x, invalid, poly_arg_none());
  PolyUOp *trunc = poly_uop1(ctx, POLY_OP_TRUNC, POLY_INDEX, masked, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, trunc);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_WHERE);
  ASSERT_TRUE(rewritten->src[0] == gate);
  ASSERT_TRUE(rewritten->src[1] == x);
  ASSERT_TRUE(rewritten->src[2] == invalid);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, invalid_where_propagates_through_binary_before_zero_folding) {
  /*
   * Pinned tinygrad uop/symbolic.py:60-66 keeps Invalid as an index-validity
   * sentinel by lifting it through ALU before ordinary identities such as
   * x*0 -> 0.  This exact topology is consumed later by
   * codegen/late/gater.py:5-17.
   */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_PARAM, POLY_BOOL, poly_arg_int(0));
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_INDEX, poly_arg_int(1));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *masked = poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, gate, x, invalid, poly_arg_none());

  const int64_t factors[] = {2, 0};
  for (int i = 0; i < 2; i++) {
    PolyUOp *factor = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(factors[i]));
    PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, masked, factor, poly_arg_none());
    PolyUOp *r = poly_graph_rewrite(ctx, mul, poly_symbolic());
    ASSERT_NOT_NULL(r);
    ASSERT_INT_EQ(r->op, POLY_OP_WHERE);
    ASSERT_TRUE(r->src[0] == gate);
    ASSERT_INT_EQ(r->src[2]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(r->src[2]->arg.kind, POLY_ARG_INVALID);
    if (factors[i] == 2) {
      ASSERT_INT_EQ(r->src[1]->op, POLY_OP_MUL);
      ASSERT_TRUE(r->src[1]->src[0] == x);
    } else {
      ASSERT_INT_EQ(r->src[1]->op, POLY_OP_CONST);
      ASSERT_INT_EQ(r->src[1]->arg.i, 0);
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(alu, fdiv_zero_zero_is_nan) {
  PolyArg ops[2] = {poly_arg_float(0.0), poly_arg_float(0.0)};
  PolyArg r = poly_exec_alu(POLY_OP_FDIV, POLY_FLOAT32, ops, 2, true);
  ASSERT_TRUE(isnan(r.f));
  PASS();
}

TEST(alu, fold_float16_truncates_output) {
  PolyArg ops[2] = {poly_arg_float(1.0), poly_arg_float(0.0001)};
  PolyArg r = poly_exec_alu(POLY_OP_ADD, POLY_FLOAT16, ops, 2, true);
  ASSERT_FLOAT_EQ(r.f, 1.0, 0.0);
  PASS();
}

TEST(alu, fold_bfloat16_truncates_output) {
  PolyArg ops[2] = {poly_arg_float(1.0), poly_arg_float(0.001)};
  PolyArg r = poly_exec_alu(POLY_OP_ADD, POLY_BFLOAT16, ops, 2, true);
  ASSERT_FLOAT_EQ(r.f, 1.0, 0.0);
  PASS();
}

TEST(sym, associative_float16_const_fold_retains_untruncated_arg) {
  /* Pinned fold_const_alu passes truncate_output=False, and the two-stage
   * associative rule combines c1/c2 through that folder
   * (symbolic.py:26-32,269-271). The CONST keeps host precision even though
   * its graph dtype is half; backend execution owns dtype rounding. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(0));
  PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(1.702));
  PolyUOp *c2 =
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(-1.0 / 0.693147180559945309417));
  PolyUOp *expression = poly_uop2(
      ctx, POLY_OP_MUL, POLY_FLOAT16,
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, x, c1, poly_arg_none()), c2, poly_arg_none()
  );
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(c1);
  ASSERT_NOT_NULL(c2);
  ASSERT_NOT_NULL(expression);

  PolyUOp *rewritten = poly_graph_rewrite(ctx, expression, poly_symbolic());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(rewritten->src[0], x);
  ASSERT_INT_EQ(rewritten->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(rewritten->src[1]->dtype, POLY_FLOAT16));
  double expected = 1.702 * (-1.0 / 0.693147180559945309417);
  ASSERT_DOUBLE_ULP(rewritten->src[1]->arg.f, expected, 0);

  PolyArg operands[2] = {c1->arg, c2->arg};
  PolyArg untruncated = poly_exec_alu(POLY_OP_MUL, POLY_FLOAT16, operands, 2, false);
  ASSERT_DOUBLE_ULP(untruncated.f, expected, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Symbolic simplification tests */

TEST(sym, add_zero) {
  /* x + 0 -> x */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, x, zero, poly_arg_none());

  PolyUOp *r = simplify(ctx, add);
  ASSERT_PTR_EQ(r, x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_zero_reversed) {
  /* 0 + x -> x (commutative) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, zero, x, poly_arg_none());

  PolyUOp *r = simplify(ctx, add);
  ASSERT_PTR_EQ(r, x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, mul_one) {
  /* x * 1 -> x */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, x, one, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  ASSERT_PTR_EQ(r, x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, mul_zero) {
  /* x * 0 -> 0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, x, zero, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, const_fold_add) {
  /* 2 + 3 -> 5 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());

  PolyUOp *r = simplify(ctx, add);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, const_fold_neg) {
  /* NEG(3.0) -> -3.0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.0));
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, c, poly_arg_none());

  PolyUOp *r = simplify(ctx, neg);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_FLOAT_EQ(r->arg.f, -3.0, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, double_neg) {
  /* NEG(NEG(x)) -> x  (use non-CONST source to avoid const-fold firing first) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_FLOAT32, poly_arg_str("v"));
  PolyUOp *neg1 = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, x, poly_arg_none());
  PolyUOp *neg2 = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, neg1, poly_arg_none());

  PolyUOp *r = simplify(ctx, neg2);
  ASSERT_PTR_EQ(r, x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, div_self) {
  /* x // x -> 1 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, x, x, poly_arg_none());

  PolyUOp *r = simplify(ctx, div);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, mod_self) {
  /* x % x -> 0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, x, poly_arg_none());

  PolyUOp *r = simplify(ctx, mod);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, combined_rewrite) {
  /* (a + 0) * 1 -> a  (requires two rules, cascading) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, zero, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, add, one, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  /* Should simplify to just 'a' */
  ASSERT_PTR_EQ(r, a);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, const_fold_mul_float) {
  /* 2.0 * 3.5 -> 7.0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.5));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, a, b, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_FLOAT_EQ(r->arg.f, 7.0, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, where_same_branches) {
  /* WHERE(cond, v, v) -> v */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *v = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(99));
  PolyUOp *wh = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cond, v, v, poly_arg_none());

  PolyUOp *r = simplify(ctx, wh);
  ASSERT_PTR_EQ(r, v);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, where_logical_not_swaps_branches) {
  /* Tinygrad symbolic.py:230-231:
   *   cond.logical_not().where(t, f) -> cond.where(f, t) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_BOOL, poly_arg_define_var("c", 0, 1));
  PolyUOp *not_cond = poly_logical_not(ctx, cond);
  PolyUOp *t = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *f = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *wh = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, not_cond, t, f, poly_arg_none());

  PolyUOp *r = simplify(ctx, wh);
  ASSERT_TRUE(r->op == POLY_OP_WHERE);
  ASSERT_TRUE(r->src[0]->op == POLY_OP_DEFINE_VAR);
  ASSERT_TRUE(r->src[0]->dtype.priority == POLY_BOOL.priority);
  ASSERT_TRUE(r->src[0]->dtype.bitsize == POLY_BOOL.bitsize);
  ASSERT_TRUE(r->src[0]->arg.kind == POLY_ARG_DEFINE_VAR);
  ASSERT_STR_EQ(r->src[0]->arg.define_var.name, "c");
  ASSERT_EQ(r->src[0]->arg.define_var.min_val, 0);
  ASSERT_EQ(r->src[0]->arg.define_var.max_val, 1);
  ASSERT_PTR_EQ(r->src[1], f);
  ASSERT_PTR_EQ(r->src[2], t);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, after_flattens_non_effect_dependencies_like_tinygrad) {
  /* Pinned symbolic.py:306-311 keeps effect dependencies, replaces every
   * other dependency by its direct sources, deduplicates that flattened list
   * in order, and eliminates a one-source AFTER. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *value = poly_uop0(ctx, POLY_OP_NOOP, POLY_INT32, poly_arg_none());
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(7, POLY_AXIS_LOOP));
  PolyUOp *wrapped = poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, range, poly_arg_none());
  PolyUOp *target = poly_uop0(ctx, POLY_OP_NOOP, POLY_INT32, poly_arg_none());
  PolyUOp *stored = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, target,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1)), poly_arg_none()
  );
  PolyUOp *barrier = poly_uop1(ctx, POLY_OP_BARRIER, POLY_VOID, stored, poly_arg_none());
  PolyUOp *end_srcs[2] = {stored, range};
  PolyUOp *ended = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, range, bound, poly_arg_none());
  PolyUOp *srcs[7] = {value, wrapped, stored, add, barrier, ended, wrapped};
  PolyUOp *root = poly_uop_tagged_arg(
      ctx, POLY_OP_AFTER, POLY_INT32, srcs, 7, poly_arg_none(), 19, poly_arg_str("after-canonical")
  );

  PolyUOp *direct = poly_pm_rewrite(poly_symbolic(), ctx, root);
  ASSERT_NOT_NULL(direct);
  ASSERT_INT_EQ(direct->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(direct->n_src, 6);
  PolyUOp *direct_expected[6] = {value, range, stored, bound, barrier, ended};
  for (int i = 0; i < 6; i++)
    ASSERT_PTR_EQ(direct->src[i], direct_expected[i]);
  ASSERT_INT_EQ(direct->tag, 19);
  ASSERT_TRUE(direct->tag_arg.kind == POLY_ARG_STRING);
  ASSERT_STR_EQ(direct->tag_arg.str, "after-canonical");

  /* Recursive graph rewrite first folds ADD(range, bound), so pinned and
   * Polygrad both omit bound from the parent's flattened dependency list. */
  PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_symbolic());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(rewritten->n_src, 5);
  PolyUOp *recursive_expected[5] = {value, range, stored, barrier, ended};
  for (int i = 0; i < 5; i++)
    ASSERT_PTR_EQ(rewritten->src[i], recursive_expected[i]);
  ASSERT_INT_EQ(rewritten->tag, 19);
  ASSERT_TRUE(rewritten->tag_arg.kind == POLY_ARG_STRING);
  ASSERT_STR_EQ(rewritten->tag_arg.str, "after-canonical");

  PolyUOp *singleton =
      poly_uop_tagged(ctx, POLY_OP_AFTER, POLY_INT32, &value, 1, poly_arg_none(), 20);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, singleton, poly_symbolic()), value);

  poly_ctx_destroy(ctx);
  PASS();
}

/* poly_uop_minmax tinygrad parity tests *
 * Every assertion below corresponds to one row of
 * test/parity_scripts/tg_minmax_gt.py output, captured against
 * the pinned tinygrad UOp._min_max. To re-verify after a tinygrad bump:
 *
 *   PYTHONPATH=references/tinygrad_latest \
 *     references/.venv-tinygrad-py311/bin/python test/parity_scripts/tg_minmax_gt.py
 *
 * Then update the literal values below if tinygrad's semantics changed. */

static PolyUOp *mk_const(PolyCtx *ctx, int64_t v) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(v));
}
static PolyUOp *mk_dvar(PolyCtx *ctx, const char *name, int64_t lo, int64_t hi) {
  return poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var(name, lo, hi));
}
static PolyUOp *mk_range(PolyCtx *ctx, int64_t n, int64_t axis_id) {
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  return poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(axis_id, POLY_AXIS_LOOP));
}

typedef struct {
  int64_t ranges[8];
  int n_ranges;
  const char *var_names[8];
  int64_t var_values[8];
  int n_vars;
} SymEvalEnv;

static bool sym_eval_i64(PolyUOp *u, const SymEvalEnv *env, int64_t *out) {
  if (!u || !out) return false;
  switch (u->op) {
  case POLY_OP_CONST:
    if (u->arg.kind == POLY_ARG_BOOL) {
      *out = u->arg.b ? 1 : 0;
      return true;
    }
    if (u->arg.kind != POLY_ARG_INT) return false;
    *out = u->arg.i;
    return true;
  case POLY_OP_RANGE: {
    int64_t axis = poly_range_axis_id(u->arg);
    if (axis < 0 || axis >= env->n_ranges) return false;
    *out = env->ranges[axis];
    return true;
  }
  case POLY_OP_DEFINE_VAR:
    if (u->arg.kind != POLY_ARG_DEFINE_VAR) return false;
    for (int i = 0; i < env->n_vars; i++) {
      if (strcmp(env->var_names[i], u->arg.define_var.name) == 0) {
        *out = env->var_values[i];
        return true;
      }
    }
    return false;
  default:
    break;
  }

  int64_t a = 0, b = 0, c = 0;
  if (u->n_src > 0 && !sym_eval_i64(u->src[0], env, &a)) return false;
  if (u->n_src > 1 && !sym_eval_i64(u->src[1], env, &b)) return false;
  if (u->n_src > 2 && !sym_eval_i64(u->src[2], env, &c)) return false;

  switch (u->op) {
  case POLY_OP_NEG:
    *out = -a;
    return true;
  case POLY_OP_ADD:
    *out = a + b;
    return true;
  case POLY_OP_SUB:
    *out = a - b;
    return true;
  case POLY_OP_MUL:
    *out = a * b;
    return true;
  case POLY_OP_IDIV:
    if (b == 0) return false;
    *out = a / b;
    return true;
  case POLY_OP_MOD:
    if (b == 0) return false;
    *out = a % b;
    return true;
  case POLY_OP_FLOORDIV:
  case POLY_OP_FLOORMOD: {
    if (b == 0) return false;
    PolyArg args[2] = {poly_arg_int(a), poly_arg_int(b)};
    PolyArg result = poly_exec_alu(u->op, u->dtype, args, 2, true);
    if (result.kind != POLY_ARG_INT) return false;
    *out = result.i;
    return true;
  }
  case POLY_OP_CMPLT:
    *out = a < b;
    return true;
  case POLY_OP_CMPNE:
    *out = a != b;
    return true;
  case POLY_OP_CMPEQ:
    *out = a == b;
    return true;
  case POLY_OP_WHERE:
    *out = a ? b : c;
    return true;
  case POLY_OP_MAX:
    *out = a > b ? a : b;
    return true;
  case POLY_OP_AND:
    *out = a & b;
    return true;
  case POLY_OP_OR:
    *out = a | b;
    return true;
  case POLY_OP_XOR:
    *out = a ^ b;
    return true;
  default:
    return false;
  }
}

static int sym_count_ops_in_root(PolyCtx *ctx, PolyUOp *root, PolyOps op) {
  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n);
  int count = 0;
  for (int i = 0; i < n; i++)
    if (topo[i]->op == op) count++;
  return count;
}

static uint32_t sym_fuzz_next(uint32_t *state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

static int64_t sym_fuzz_i64(uint32_t *state, int64_t lo, int64_t hi) {
  return lo + (int64_t)(sym_fuzz_next(state) % (uint32_t)(hi - lo + 1));
}

static PolyUOp *sym_mul_const(PolyCtx *ctx, PolyUOp *x, int64_t c) {
  if (c == 1) return x;
  PolyUOp *cc = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(c));
  return poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, x, cc, poly_arg_none());
}

static PolyUOp *sym_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());
}

static PolyUOp *sym_fuzz_linear(PolyCtx *ctx, PolyUOp **vars, int n_vars, uint32_t *state) {
  PolyUOp *ret =
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(sym_fuzz_i64(state, -15, 15)));
  int n_terms = 1 + (int)(sym_fuzz_next(state) % 4);
  for (int i = 0; i < n_terms; i++) {
    PolyUOp *v = vars[sym_fuzz_next(state) % (uint32_t)n_vars];
    int64_t f = sym_fuzz_i64(state, -12, 12);
    if (f == 0) f = 1;
    ret = sym_add(ctx, ret, sym_mul_const(ctx, v, f));
  }
  return ret;
}

static PolyUOp *sym_fuzz_expr(PolyCtx *ctx, PolyUOp **vars, int n_vars, uint32_t *state) {
  PolyUOp *x = sym_fuzz_linear(ctx, vars, n_vars, state);
  int64_t d = sym_fuzz_i64(state, 2, 19);
  PolyUOp *den = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(d));
  switch (sym_fuzz_next(state) % 7) {
  case 0:
    return poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, den, poly_arg_none());
  case 1:
    return poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, x, den, poly_arg_none());
  case 2: {
    int64_t k = sym_fuzz_i64(state, 2, 6);
    PolyUOp *wide_den = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(d * k));
    PolyUOp *inner = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, wide_den, poly_arg_none());
    return poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, inner, den, poly_arg_none());
  }
  case 3: {
    int64_t k = sym_fuzz_i64(state, 2, 6);
    PolyUOp *wide_den = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(d * k));
    PolyUOp *inner = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, wide_den, poly_arg_none());
    return poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, inner, den, poly_arg_none());
  }
  case 4: {
    PolyUOp *y = sym_fuzz_linear(ctx, vars, n_vars, state);
    PolyUOp *cond =
        poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, vars[0], mk_const(ctx, 4), poly_arg_none());
    return poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cond, x, y, poly_arg_none());
  }
  case 5:
    return poly_uop2(
        ctx, POLY_OP_MOD, POLY_INT32, sym_add(ctx, x, mk_const(ctx, 31)), den, poly_arg_none()
    );
  default:
    return x;
  }
}

TEST(sym, symbolic_fuzzer_integer_rewrite_equivalence) {
  uint32_t seed = 0xC0FFEEu;
  for (int trial = 0; trial < 512; trial++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *vars[4] = {
        mk_range(ctx, 9, 0),
        mk_range(ctx, 7, 1),
        mk_dvar(ctx, "a", 4, 11),
        mk_dvar(ctx, "b", -3, 5),
    };
    PolyUOp *root = sym_fuzz_expr(ctx, vars, 4, &seed);
    PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_symbolic());
    ASSERT_TRUE(rewritten != NULL);

    for (int r0 = 0; r0 < 9; r0 += 2) {
      for (int r1 = 0; r1 < 7; r1 += 3) {
        for (int a = 4; a <= 11; a += 3) {
          for (int b = -3; b <= 5; b += 4) {
            SymEvalEnv env = {0};
            env.ranges[0] = r0;
            env.ranges[1] = r1;
            env.n_ranges = 2;
            env.var_names[0] = "a";
            env.var_values[0] = a;
            env.var_names[1] = "b";
            env.var_values[1] = b;
            env.n_vars = 2;
            int64_t got = 0, want = 0;
            ASSERT_TRUE(sym_eval_i64(root, &env, &want));
            ASSERT_TRUE(sym_eval_i64(rewritten, &env, &got));
            if (got != want) {
              char *root_s = poly_uop_str(root);
              char *rewritten_s = poly_uop_str(rewritten);
              fprintf(stderr, "    root=%s\n    rewritten=%s\n", root_s, rewritten_s);
              fprintf(stderr, "    root tree:\n");
              poly_uop_dump_tree(stderr, root, 0, 10);
              fprintf(stderr, "    rewritten tree:\n");
              poly_uop_dump_tree(stderr, rewritten, 0, 10);
              free(root_s);
              free(rewritten_s);
              FAIL(
                  "trial %d sample r0=%d r1=%d a=%d b=%d got %lld want %lld", trial, r0, r1, a, b,
                  (long long)got, (long long)want
              );
            }
          }
        }
      }
    }
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(sym, add_divmod_recombine_preserves_terms_past_old_cap) {
  /* tinygrad symbolic.fold_add_divmod_recombine uses list(x.split_uop(ADD))
   * with no fixed scratch cap. Polygrad used to collect only the first 32
   * terms, then rebuild after a match, which could drop tail terms. Shape the
   * tree so the div/mod pair appears together only at the root. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(100));
  PolyUOp *tail_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(11));
  PolyUOp *base =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, base_bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *tail =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, tail_bound, poly_arg_range(1, POLY_AXIS_LOOP));
  PolyUOp *c4 = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4));

  PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, base, c4, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, base, c4, poly_arg_none());
  PolyUOp *divmul = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, div, c4, poly_arg_none());

  PolyUOp *left = mod;
  for (int i = 0; i < 30; i++)
    left = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, left, tail, poly_arg_none());

  PolyUOp *right = divmul;
  for (int i = 0; i < 20; i++)
    right = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, right, tail, poly_arg_none());

  PolyUOp *root = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, left, right, poly_arg_none());
  PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_symbolic());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_FLOORMOD), 0);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_FLOORDIV), 0);

  for (int r0 = 0; r0 < 100; r0 += 17) {
    for (int r1 = 0; r1 < 11; r1 += 5) {
      SymEvalEnv env = {0};
      env.ranges[0] = r0;
      env.ranges[1] = r1;
      env.n_ranges = 2;
      int64_t got = 0, want = 0;
      ASSERT_TRUE(sym_eval_i64(root, &env, &want));
      ASSERT_TRUE(sym_eval_i64(rewritten, &env, &got));
      if (got != want) {
        char *root_s = poly_uop_str(root);
        char *rewritten_s = poly_uop_str(rewritten);
        fprintf(stderr, "    root=%s\n    rewritten=%s\n", root_s, rewritten_s);
        free(root_s);
        free(rewritten_s);
        FAIL("sample r0=%d r1=%d got %lld want %lld", r0, r1, (long long)got, (long long)want);
      }
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_divmod_recombine_multiplier_product_overflow_declines) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("base", INT64_MIN, INT64_MAX)
  );
  PolyUOp *q = poly_uop0(ctx, POLY_OP_PARAM, POLY_INDEX, poly_arg_int(0));
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT64_MAX));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3));
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, base, max, poly_arg_none());
  PolyUOp *mod_scaled = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, mod, two, poly_arg_none());
  PolyUOp *q_scaled = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, q, three, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, mod_scaled, q_scaled, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, root);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_ADD);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_divmod_recombine_nested_divisor_product_overflow_declines) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("base", INT64_MIN, INT64_MAX)
  );
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT64_MAX));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3));
  PolyUOp *base_div_max = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, base, max, poly_arg_none());
  PolyUOp *nested_mod =
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, base_div_max, two, poly_arg_none());
  PolyUOp *base_div_three =
      poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, base, three, poly_arg_none());
  PolyUOp *scaled_div =
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, base_div_three, two, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, nested_mod, scaled_div, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, root);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_ADD);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_divmod_recombine_modulus_product_overflow_declines) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("base", INT64_MIN, INT64_MAX)
  );
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT64_MAX));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *base_mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, base, max, poly_arg_none());
  PolyUOp *base_div = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, base, max, poly_arg_none());
  PolyUOp *nested_mod =
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, base_div, two, poly_arg_none());
  PolyUOp *scaled_nested =
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, nested_mod, max, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, base_mod, scaled_nested, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, root);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_ADD);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_divmod_recombine_negative_divisor_stays_add) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("x", -12, 12));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3));
  PolyUOp *negative_two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(-2));
  PolyUOp *negative_six = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(-6));
  PolyUOp *base = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, x, three, poly_arg_none());
  PolyUOp *left = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, base, negative_two, poly_arg_none());
  PolyUOp *right_div =
      poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, x, negative_six, poly_arg_none());
  PolyUOp *right =
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, right_div, negative_two, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, left, right, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, root);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_ADD);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_FLOORMOD), 1);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_FLOORDIV), 2);

  poly_ctx_destroy(ctx);
  PASS();
}
/* check_mm queries minmax and asserts both bounds match the tinygrad
 * ground truth. Implemented as a macro because ASSERT_INT_EQ touches the
 * test-local _passed/_failed counters that only exist inside a TEST. */
#define check_mm(ctx, u, want_lo, want_hi, label)                                                  \
  do {                                                                                             \
    int64_t _mm_lo, _mm_hi;                                                                        \
    poly_uop_minmax((ctx), (u), &_mm_lo, &_mm_hi);                                                 \
    if (_mm_lo != (want_lo) || _mm_hi != (want_hi)) {                                              \
      fprintf(                                                                                     \
          stderr, "%s: got [%lld..%lld], want [%lld..%lld]\n", (label), (long long)_mm_lo,         \
          (long long)_mm_hi, (long long)(want_lo), (long long)(want_hi)                            \
      );                                                                                           \
    }                                                                                              \
    ASSERT_INT_EQ(_mm_lo, (want_lo));                                                              \
    ASSERT_INT_EQ(_mm_hi, (want_hi));                                                              \
  } while (0)

TEST(sym, minmax_const_pos) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_const(ctx, 5), 5, 5, "CONST 5");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_const_neg) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_const(ctx, -2), -2, -2, "CONST -2");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_define_var_pos) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_dvar(ctx, "x", 2, 7), 2, 7, "DEFINE_VAR[2..7]");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_define_var_mixed_sign) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_dvar(ctx, "y", -3, 4), -3, 4, "DEFINE_VAR[-3..4]");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_range_10) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_range(ctx, 10, 0), 0, 9, "RANGE(10)");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_range_1_degenerate) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_range(ctx, 1, 1), 0, 0, "RANGE(1)");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_cast_const_reads_stored_arg_like_tinygrad) {
  /* Pinned tinygrad ops.py:1010-1017 returns CONST.arg directly even when a
   * preceding CAST(CONST) produced a value outside the nominal dtype range. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *source = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(260));
  PolyUOp *cast = simplify(ctx, poly_uop1(ctx, POLY_OP_CAST, POLY_UINT8, source, poly_arg_none()));
  ASSERT_NOT_NULL(cast);
  ASSERT_INT_EQ(cast->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(cast->dtype, POLY_UINT8));
  ASSERT_TRUE(cast->arg.kind == POLY_ARG_INT);
  ASSERT_INT_EQ(cast->arg.i, 260);
  int64_t lo = 0, hi = 0;
  poly_uop_minmax(ctx, cast, &lo, &hi);
  ASSERT_INT_EQ(lo, 260);
  ASSERT_INT_EQ(hi, 260);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ADD / SUB */
TEST(sym, minmax_add_r_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 3), poly_arg_none()
  );
  check_mm(ctx, u, 3, 12, "r+3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_add_r_dvar) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32, mk_range(ctx, 10, 0), mk_dvar(ctx, "x", 2, 7), poly_arg_none()
  );
  check_mm(ctx, u, 2, 16, "r+dv");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_sub_r_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_SUB, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 3), poly_arg_none()
  );
  check_mm(ctx, u, -3, 6, "r-3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_sub_const_r) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_SUB, POLY_INT32, mk_const(ctx, 3), mk_range(ctx, 10, 0), poly_arg_none()
  );
  check_mm(ctx, u, -6, 3, "3-r");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_sub_dv_dvn) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dv = mk_dvar(ctx, "x", 2, 7);
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_SUB, POLY_INT32, dv, dvn, poly_arg_none());
  check_mm(ctx, u, -2, 10, "dv-dvn");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_add_int64_overflow_falls_back_to_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT64, poly_arg_define_var("x", INT64_MAX - 1, INT64_MAX)
  );
  PolyUOp *y =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT64, poly_arg_define_var("y", 1, INT32_MAX));
  PolyUOp *u = poly_uop2(ctx, POLY_OP_ADD, POLY_INT64, x, y, poly_arg_none());
  check_mm(ctx, u, INT64_MIN, INT64_MAX, "i64 add overflow fallback");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_sub_int64_overflow_falls_back_to_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT64, poly_arg_define_var("x", INT64_MIN, INT64_MIN + 1)
  );
  PolyUOp *y =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT64, poly_arg_define_var("y", 1, INT32_MAX));
  PolyUOp *u = poly_uop2(ctx, POLY_OP_SUB, POLY_INT64, x, y, poly_arg_none());
  check_mm(ctx, u, INT64_MIN, INT64_MAX, "i64 sub overflow fallback");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_narrow_integer_wrap_falls_back_to_dtype) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *u8 = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_UINT8, poly_arg_define_var("u8", 250, 251));
  PolyUOp *u8_add = poly_uop2(
      ctx, POLY_OP_ADD, POLY_UINT8, u8, poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(10)),
      poly_arg_none()
  );
  check_mm(ctx, u8_add, 0, UINT8_MAX, "uint8 wrapping add");
  PolyUOp *u8_cmp = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, u8_add,
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(5)), poly_arg_none()
  );
  ASSERT_INT_EQ(simplify(ctx, u8_cmp)->op, POLY_OP_CMPLT);

  PolyUOp *i8 = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT8, poly_arg_define_var("i8", 120, 121));
  PolyUOp *i8_add = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT8, i8, poly_uop0(ctx, POLY_OP_CONST, POLY_INT8, poly_arg_int(10)),
      poly_arg_none()
  );
  check_mm(ctx, i8_add, INT8_MIN, INT8_MAX, "int8 wrapping add");
  PolyUOp *i8_cmp = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, i8_add,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT8, poly_arg_int(0)), poly_arg_none()
  );
  ASSERT_INT_EQ(simplify(ctx, i8_cmp)->op, POLY_OP_CMPLT);

  PolyUOp *u8_sub = poly_uop2(
      ctx, POLY_OP_SUB, POLY_UINT8,
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_UINT8, poly_arg_define_var("u8s", 0, 1)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(2)), poly_arg_none()
  );
  check_mm(ctx, u8_sub, 0, UINT8_MAX, "uint8 wrapping sub");

  PolyUOp *i8_mul = poly_uop2(
      ctx, POLY_OP_MUL, POLY_INT8,
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT8, poly_arg_define_var("i8m", 64, 65)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT8, poly_arg_int(2)), poly_arg_none()
  );
  check_mm(ctx, i8_mul, INT8_MIN, INT8_MAX, "int8 wrapping mul");

  PolyUOp *u8_shl = poly_uop2(
      ctx, POLY_OP_SHL, POLY_UINT8,
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_UINT8, poly_arg_define_var("u8l", 128, 129)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(1)), poly_arg_none()
  );
  check_mm(ctx, u8_shl, 0, UINT8_MAX, "uint8 wrapping shl");

  PolyArg u8_add_args[2] = {poly_arg_int(250), poly_arg_int(10)};
  PolyArg i8_add_args[2] = {poly_arg_int(120), poly_arg_int(10)};
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_ADD, POLY_UINT8, u8_add_args, 2, true).i, 4);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_ADD, POLY_INT8, i8_add_args, 2, true).i, -126);

  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_int64_division_overflow_falls_back_to_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT64, poly_arg_define_var("x", INT64_MIN, INT64_MIN)
  );
  PolyUOp *neg_one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(-1));
  PolyUOp *cdiv_u = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT64, x, neg_one, poly_arg_none());
  PolyUOp *floordiv_u = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT64, x, neg_one, poly_arg_none());
  check_mm(ctx, cdiv_u, INT64_MIN, INT64_MAX, "i64 cdiv overflow fallback");
  check_mm(ctx, floordiv_u, INT64_MIN, INT64_MAX, "i64 floordiv overflow fallback");
  poly_ctx_destroy(ctx);
  PASS();
}

/* MUL (4-corner) */
TEST(sym, minmax_mul_r_3) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MUL, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 3), poly_arg_none()
  );
  check_mm(ctx, u, 0, 27, "r*3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mul_r_neg2) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MUL, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, -2), poly_arg_none()
  );
  check_mm(ctx, u, -18, 0, "r*-2");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mul_dv_dvn_mixed_signs) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dv = mk_dvar(ctx, "x", 2, 7);
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, dv, dvn, poly_arg_none());
  /* corners: 2*-3=-6, 2*4=8, 7*-3=-21, 7*4=28 -> [-21, 28] */
  check_mm(ctx, u, -21, 28, "dv*dvn");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mul_dvn_dvn_squared) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, dvn, dvn, poly_arg_none());
  /* corners: -3*-3=9, -3*4=-12, 4*-3=-12, 4*4=16 -> [-12, 16] */
  check_mm(ctx, u, -12, 16, "dvn*dvn");
  poly_ctx_destroy(ctx);
  PASS();
}

/* IDIV */
TEST(sym, minmax_idiv_r_3) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_IDIV, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 3), poly_arg_none()
  );
  check_mm(ctx, u, 0, 3, "r//3");
  poly_ctx_destroy(ctx);
  PASS();
}

/* MOD */
TEST(sym, minmax_mod_r_3) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MOD, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 3), poly_arg_none()
  );
  check_mm(ctx, u, 0, 2, "r%3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mod_dvn_3_mixed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, dvn, mk_const(ctx, 3), poly_arg_none());
  /* tinygrad: dvn vmin=-3 vmax=4, mod 3 -> middle branch -> [-2, 2] */
  check_mm(ctx, u, -2, 2, "dvn%3");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_floordiv_floormod_dvn_3_mixed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *q = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT32, dvn, mk_const(ctx, 3), poly_arg_none());
  PolyUOp *r = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INT32, dvn, mk_const(ctx, 3), poly_arg_none());
  check_mm(ctx, q, -1, 1, "floor(dvn/3)");
  check_mm(ctx, r, 0, 2, "floor_mod(dvn,3)");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_empty_range_divmod_is_zero_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *empty = mk_range(ctx, 0, 0);
  PolyUOp *three = mk_const(ctx, 3);
  PolyUOp *cdiv = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, empty, three, poly_arg_none());
  PolyUOp *floordiv = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT32, empty, three, poly_arg_none());
  PolyUOp *floormod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INT32, empty, three, poly_arg_none());
  check_mm(ctx, cdiv, 0, 0, "empty CDIV 3");
  check_mm(ctx, floordiv, 0, 0, "empty FLOORDIV 3");
  check_mm(ctx, floormod, 0, 0, "empty FLOORMOD 3");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_negative_int64_divisor_bounds_are_overflow_safe) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *numerator =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT64, poly_arg_define_var("numerator", 0, 1));
  PolyUOp *divisor = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT64, poly_arg_define_var("negative_divisor", INT64_MIN, -1)
  );
  PolyUOp *cmod = poly_uop2(ctx, POLY_OP_MOD, POLY_INT64, numerator, divisor, poly_arg_none());
  PolyUOp *floormod =
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INT64, numerator, divisor, poly_arg_none());
  check_mm(ctx, cmod, 0, INT64_MAX, "CMOD([0,1], [INT64_MIN,-1])");
  check_mm(ctx, floormod, -INT64_MAX, 0, "FLOORMOD([0,1], [INT64_MIN,-1])");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, fold_divmod_uint64_additive_overflow_declines_without_ub) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *max_i64 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(2));
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT64, max_i64, one, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT64, sum, two, poly_arg_none());
  PolyUOp *rewritten = simplify(ctx, div);
  ASSERT_NOT_NULL(rewritten);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, fold_divmod_uint64_congruence_overflow_declines_without_ub) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_UINT64, poly_arg_define_var("base", 0, 4));
  PolyUOp *factor = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX / 3));
  PolyUOp *denominator = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX));
  PolyUOp *numerator = poly_uop2(ctx, POLY_OP_MUL, POLY_UINT64, base, factor, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT64, numerator, denominator, poly_arg_none());
  PolyUOp *rewritten = simplify(ctx, div);
  ASSERT_NOT_NULL(rewritten);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, uint64_minmax_point_does_not_fold_signed_surrogate_bounds) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_UINT64, poly_arg_define_var("x", INT64_MIN, INT64_MIN + 1)
  );
  PolyUOp *denominator = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX));
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT64, x, denominator, poly_arg_none());
  PolyUOp *rewritten = simplify(ctx, div);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_IDIV);

  PolyArg lower_args[2] = {poly_arg_int(INT64_MIN), poly_arg_int(INT64_MAX)};
  PolyArg upper_args[2] = {poly_arg_int(INT64_MIN + 1), poly_arg_int(INT64_MAX)};
  /* These are raw negative CONST args, not representable positive uint64
   * values. Pinned CDIV returns -1, then uint64 truncation is stored as the
   * signed all-ones surrogate until PG-PARITY-020 closes. */
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_IDIV, POLY_UINT64, lower_args, 2, true).i, -1);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_IDIV, POLY_UINT64, upper_args, 2, true).i, -1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, unsigned_div_all_ones_does_not_rewrite_to_negation) {
  const PolyDType unsigned_dtypes[] = {POLY_UINT8, POLY_UINT16, POLY_UINT32, POLY_UINT64};
  const int64_t truncated_neg_one[] = {UINT8_MAX, UINT16_MAX, UINT32_MAX, -1};
  for (int i = 0; i < (int)(sizeof(unsigned_dtypes) / sizeof(unsigned_dtypes[0])); i++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyDType dtype = unsigned_dtypes[i];
    PolyUOp *x = poly_uop0(ctx, POLY_OP_DEFINE_VAR, dtype, poly_arg_define_var("x", 2, 3));
    PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(1));
    PolyUOp *numerator = poly_uop2(ctx, POLY_OP_SHR, dtype, x, one, poly_arg_none());
    PolyUOp *all_ones = poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(-1));
    PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, dtype, numerator, all_ones, poly_arg_none());
    PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, dtype, numerator, all_ones, poly_arg_none());
    PolyUOp *rewritten_div = simplify(ctx, div);
    PolyUOp *rewritten_mod = simplify(ctx, mod);
    ASSERT_NOT_NULL(rewritten_div);
    ASSERT_NOT_NULL(rewritten_mod);
    ASSERT_INT_EQ(rewritten_div->op, POLY_OP_IDIV);
    ASSERT_INT_EQ(rewritten_mod->op, POLY_OP_MOD);

    PolyArg endpoint[2] = {poly_arg_int(1), poly_arg_int(-1)};
    ASSERT_INT_EQ(poly_exec_alu(POLY_OP_IDIV, dtype, endpoint, 2, true).i, truncated_neg_one[i]);
    ASSERT_INT_EQ(poly_exec_alu(POLY_OP_MOD, dtype, endpoint, 2, true).i, 0);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(sym, cdiv_stays_raw_while_floordiv_uses_floor_identities) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("x", 2, 3));
  PolyUOp *neg_one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-1));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));

  PolyUOp *cdiv_neg = poly_uop2(ctx, POLY_OP_CDIV, POLY_INT32, x, neg_one, poly_arg_none());
  PolyUOp *floordiv_neg = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT32, x, neg_one, poly_arg_none());
  PolyUOp *cdiv_four = poly_uop2(ctx, POLY_OP_CDIV, POLY_INT32, x, four, poly_arg_none());
  PolyUOp *floordiv_four = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT32, x, four, poly_arg_none());

  PolyUOp *rewritten_cdiv_neg = simplify(ctx, cdiv_neg);
  PolyUOp *rewritten_floordiv_neg = simplify(ctx, floordiv_neg);
  PolyUOp *rewritten_cdiv_four = simplify(ctx, cdiv_four);
  PolyUOp *rewritten_floordiv_four = simplify(ctx, floordiv_four);
  ASSERT_INT_EQ(rewritten_cdiv_neg->op, POLY_OP_CDIV);
  ASSERT_INT_EQ(rewritten_floordiv_neg->op, POLY_OP_NEG);
  ASSERT_PTR_EQ(rewritten_floordiv_neg->src[0], x);
  ASSERT_INT_EQ(rewritten_cdiv_four->op, POLY_OP_CDIV);
  ASSERT_INT_EQ(rewritten_floordiv_four->op, POLY_OP_CONST);
  ASSERT_INT_EQ(rewritten_floordiv_four->arg.i, 0);

  PolyUOp *min_x = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT64, poly_arg_define_var("min_x", INT64_MIN, INT64_MIN + 1)
  );
  PolyUOp *min_neg_one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(-1));
  PolyUOp *min_floordiv =
      poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT64, min_x, min_neg_one, poly_arg_none());
  ASSERT_INT_EQ(simplify(ctx, min_floordiv)->op, POLY_OP_FLOORDIV);

  poly_ctx_destroy(ctx);
  PASS();
}

/* SHL / SHR */
TEST(sym, minmax_shl_r_2) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_SHL, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 2), poly_arg_none()
  );
  check_mm(ctx, u, 0, 36, "r<<2");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_shl_int64_overflow_falls_back_to_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT64, poly_arg_define_var("x", INT64_MAX / 2, INT64_MAX)
  );
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_SHL, POLY_INT64, x, poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(2)),
      poly_arg_none()
  );
  check_mm(ctx, u, INT64_MIN, INT64_MAX, "i64 shl overflow fallback");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_shr_r_1) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_SHR, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 1), poly_arg_none()
  );
  check_mm(ctx, u, 0, 4, "r>>1");
  poly_ctx_destroy(ctx);
  PASS();
}

/* XOR with -1 (bitwise NOT) */
TEST(sym, minmax_xor_r_neg1) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_XOR, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, -1), poly_arg_none()
  );
  /* ~r where r in [0,9] -> [~9, ~0] = [-10, -1] */
  check_mm(ctx, u, -10, -1, "r xor -1");
  poly_ctx_destroy(ctx);
  PASS();
}

/* AND (int with non-negative const) */
TEST(sym, minmax_and_r_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_AND, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 5), poly_arg_none()
  );
  /* tinygrad: r vmin=0 (>=0), vmax=min(9,5)=5 -> [0, 5] */
  check_mm(ctx, u, 0, 5, "r & 5");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_and_negative_var_nonnegative_mask) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = mk_dvar(ctx, "x", -100, 100);

  /* Port of tinygrad test_uop_vmin_vmax.py:
   * when the mask has no sign bit, x & mask is known non-negative even if x
   * spans negative and positive values. A negative mask falls back to dtype
   * bounds because the sign bit can survive. */
  PolyUOp *mask511 =
      poly_uop2(ctx, POLY_OP_AND, POLY_INT32, x, mk_const(ctx, 511), poly_arg_none());
  check_mm(ctx, mask511, 0, 511, "[-100..100] & 511");

  PolyUOp *mask_all =
      poly_uop2(ctx, POLY_OP_AND, POLY_INT32, x, mk_const(ctx, -1), poly_arg_none());
  check_mm(ctx, mask_all, INT32_MIN, INT32_MAX, "[-100..100] & -1");

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_special_with_define_var_source) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dv = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("i", 1, 10));
  PolyUOp *special = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, dv, poly_arg_str("gidx0"));

  /* tinygrad SPECIAL uses the source bound as an extent, so [1..10] becomes
   * an index-style range [0..9]. */
  check_mm(ctx, special, 0, 9, "SPECIAL(DEFINE_VAR[1..10])");
  poly_ctx_destroy(ctx);
  PASS();
}

/* MAX */
TEST(sym, minmax_max_r_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MAX, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 5), poly_arg_none()
  );
  check_mm(ctx, u, 5, 9, "max(r,5)");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_nested_max_min_clamp) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = mk_dvar(ctx, "x", 0, 10);
  PolyUOp *lo = poly_uop2(ctx, POLY_OP_MAX, POLY_INT32, x, mk_const(ctx, 5), poly_arg_none());
  PolyUOp *neg_one = mk_const(ctx, -1);
  PolyUOp *not_lo = poly_uop2(ctx, POLY_OP_XOR, POLY_INT32, lo, neg_one, poly_arg_none());
  PolyUOp *not_8 =
      poly_uop2(ctx, POLY_OP_XOR, POLY_INT32, mk_const(ctx, 8), neg_one, poly_arg_none());
  PolyUOp *inner = poly_uop2(ctx, POLY_OP_MAX, POLY_INT32, not_lo, not_8, poly_arg_none());
  PolyUOp *clamped = poly_uop2(ctx, POLY_OP_XOR, POLY_INT32, inner, neg_one, poly_arg_none());

  /* Port of tinygrad test_vmin_vmax_nested_min_max:
   * x.maximum(5).minimum(8) renders as (max((max(x, 5)^-1), -9)^-1)
   * at the UOp level, then narrows to [5..8]. */
  check_mm(ctx, clamped, 5, 8, "min(max(x,5),8)");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_max_dv_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MAX, POLY_INT32, mk_dvar(ctx, "x", 2, 7), mk_const(ctx, 5), poly_arg_none()
  );
  check_mm(ctx, u, 5, 7, "max(dv,5)");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_idiv_negative_constant) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *pos = mk_dvar(ctx, "pos", 10, 20);
  PolyUOp *pos_div_neg =
      poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, pos, mk_const(ctx, -2), poly_arg_none());
  check_mm(ctx, pos_div_neg, -10, -5, "[10..20]//-2");

  PolyUOp *neg = mk_dvar(ctx, "neg", -20, -10);
  PolyUOp *neg_div_neg =
      poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, neg, mk_const(ctx, -3), poly_arg_none());
  check_mm(ctx, neg_div_neg, 3, 6, "[-20..-10]//-3");

  poly_ctx_destroy(ctx);
  PASS();
}

/* CMPLT (bool bounds) */
TEST(sym, minmax_cmplt_r_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, 5), poly_arg_none()
  );
  /* (a1<b0) = (9<5)=false=0, (a0<b1) = (0<5)=true=1 -> [0,1] */
  check_mm(ctx, u, 0, 1, "r<5");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_cmplt_r_neg1_static_false) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, -1), poly_arg_none()
  );
  /* (a1<b0) = (9<-1)=false=0, (a0<b1) = (0<-1)=false=0 -> [0,0] */
  check_mm(ctx, u, 0, 0, "r<-1");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_cmplt_r_20_static_true) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, 20), poly_arg_none()
  );
  /* (a1<b0) = (9<20)=true=1, (a0<b1) = (0<20)=true=1 -> [1,1] */
  check_mm(ctx, u, 1, 1, "r<20");
  poly_ctx_destroy(ctx);
  PASS();
}

/* CMPNE */
TEST(sym, minmax_cmpne_r_neg1) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_CMPNE, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, -1), poly_arg_none()
  );
  /* def_ne: (a1<b0)||(b1<a0) = (9<-1)||(-1<0) = false||true = true
   * all_eq: false (a0=0 != a1=9). vmin=1, vmax=1 */
  check_mm(ctx, u, 1, 1, "r!=-1");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_stack_uses_lane_bounds_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType i32x4 = poly_dtype_vec(POLY_INT32, 4);
  PolyUOp *lanes[4] = {
      mk_const(ctx, -2),
      mk_const(ctx, 0),
      mk_const(ctx, 5),
      mk_const(ctx, 10),
  };
  PolyUOp *stack = poly_uop(ctx, POLY_OP_STACK, i32x4, lanes, 4, poly_arg_none());
  check_mm(ctx, stack, -2, 10, "STACK(-2,0,5,10)");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_vector_bool_gep_and_gate_stays_dynamic) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType f32x4 = poly_dtype_vec(POLY_FLOAT32, 4);
  PolyDType b4 = poly_dtype_vec(POLY_BOOL, 4);
  PolyUOp *base = poly_uop0(ctx, POLY_OP_PARAM, f32x4, poly_arg_int(0));
  PolyUOp *exponent = poly_uop0(ctx, POLY_OP_PARAM, f32x4, poly_arg_int(1));
  PolyUOp *zero_lanes[4] = {
      poly_const_float(ctx, 0.0),
      poly_const_float(ctx, 0.0),
      poly_const_float(ctx, 0.0),
      poly_const_float(ctx, 0.0),
  };
  PolyUOp *zero_vec = poly_uop(ctx, POLY_OP_STACK, f32x4, zero_lanes, 4, poly_arg_none());
  PolyUOp *base_eq = poly_uop2(ctx, POLY_OP_CMPEQ, b4, base, zero_vec, poly_arg_none());
  PolyUOp *exp_eq = poly_uop2(ctx, POLY_OP_CMPEQ, b4, exponent, zero_vec, poly_arg_none());
  PolyUOp *base_lane = poly_uop1(ctx, POLY_OP_GEP, POLY_BOOL, base_eq, poly_arg_int(0));
  PolyUOp *exp_lane = poly_uop1(ctx, POLY_OP_GEP, POLY_BOOL, exp_eq, poly_arg_int(0));
  PolyUOp *both = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, base_lane, exp_lane, poly_arg_none());
  PolyUOp *as_int = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, both, poly_arg_none());
  PolyUOp *as_float = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, as_int, poly_arg_none());
  PolyUOp *gate = poly_uop2(
      ctx, POLY_OP_CMPNE, POLY_BOOL, as_float, poly_const_float(ctx, 0.0), poly_arg_none()
  );

  check_mm(ctx, base_eq, 0, 1, "vector CMPEQ");
  check_mm(ctx, base_lane, 0, 1, "GEP(vector CMPEQ)");
  check_mm(ctx, both, 0, 1, "AND of vector comparison lanes");
  check_mm(ctx, gate, 0, 1, "casted bool gate");

  PolyUOp *where = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_FLOAT32, gate, poly_const_float(ctx, 1.0),
      poly_const_float(ctx, 8.0), poly_arg_none()
  );
  PolyUOp *rewritten = simplify(ctx, where);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_WHERE);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_float_comparisons_with_infinity_stay_dynamic) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(0));
  PolyUOp *negative_inf = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(-INFINITY));
  PolyUOp *positive_inf = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(INFINITY));
  PolyUOp *reciprocal = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, x, poly_arg_none());
  PolyUOp *not_negative_inf =
      poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, reciprocal, negative_inf, poly_arg_none());
  PolyUOp *below_positive_inf =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, positive_inf, poly_arg_none());

  PolyUOp *rewritten_ne = simplify(ctx, not_negative_inf);
  PolyUOp *rewritten_lt = simplify(ctx, below_positive_inf);
  ASSERT_NOT_NULL(rewritten_ne);
  ASSERT_NOT_NULL(rewritten_lt);
  ASSERT_INT_EQ(rewritten_ne->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(rewritten_lt->op, POLY_OP_CMPLT);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, reciprocal_self_product_matches_pinned_division_spelling) {
  /* Pinned ElementwiseMixin.div constructs x/x as
   * MUL(x, RECIPROCAL(x)); symbolic.py:136-145 rewrites it to const_like(1). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(0));
  PolyUOp *reciprocal = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT16, x, poly_arg_none());
  PolyUOp *division = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, x, reciprocal, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, division);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_FLOAT16));
  ASSERT_TRUE(rewritten->arg.kind == POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(rewritten->arg.f, 1.0, 0.0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, reciprocal_product_rules_match_pinned_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/symbolic.py:478-480:
   *   x * d       -> 1-d
   *   x * (d*y)   -> y*(1-d)
   *   x * (d+y)   -> (1-d)+x*y
   * where d = reciprocal(1+x). */
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(0));
  PolyUOp *y = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(1));
  PolyUOp *one = poly_const_like_float(ctx, x, 1.0);
  PolyUOp *neg_one = poly_const_like_float(ctx, x, -1.0);
  PolyUOp *den = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT16, one, x, poly_arg_none());
  PolyUOp *d = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT16, den, poly_arg_none());
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(y);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(neg_one);
  ASSERT_NOT_NULL(den);
  ASSERT_NOT_NULL(d);

  PolyUOp *one_minus_d = poly_uop2(
      ctx, POLY_OP_ADD, POLY_FLOAT16, one,
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, d, neg_one, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *expressions[3] = {
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, x, d, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_FLOAT16, x,
          poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, d, y, poly_arg_none()), poly_arg_none()
      ),
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_FLOAT16, x,
          poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT16, d, y, poly_arg_none()), poly_arg_none()
      ),
  };
  PolyUOp *expected[3] = {
      one_minus_d,
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, y, one_minus_d, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_FLOAT16, one_minus_d,
          poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, x, y, poly_arg_none()), poly_arg_none()
      ),
  };

  for (int i = 0; i < 3; i++) {
    PolyUOp *rewritten = poly_graph_rewrite(ctx, expressions[i], poly_symbolic());
    PolyUOp *canonical_expected = poly_graph_rewrite(ctx, expected[i], poly_symbolic());
    ASSERT_NOT_NULL(rewritten);
    ASSERT_NOT_NULL(canonical_expected);
    ASSERT_PTR_EQ(rewritten, canonical_expected);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, move_const_to_end_matches_commutative_outer_orientation) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/symbolic.py:286-287 uses commutative UPat matching:
   *   y op (x op c) -> (x op y) op c
   * The nested same-op subtree can therefore be either outer operand. */
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(0));
  PolyUOp *y = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(1));
  PolyUOp *c = poly_const_like_float(ctx, x, -1.0);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(y);
  ASSERT_NOT_NULL(c);

  PolyOps ops[] = {POLY_OP_ADD, POLY_OP_MUL};
  for (int i = 0; i < 2; i++) {
    PolyUOp *inner = poly_uop2(ctx, ops[i], POLY_FLOAT16, x, c, poly_arg_none());
    PolyUOp *expression = poly_uop2(ctx, ops[i], POLY_FLOAT16, y, inner, poly_arg_none());
    PolyUOp *expected = poly_uop2(
        ctx, ops[i], POLY_FLOAT16, poly_uop2(ctx, ops[i], POLY_FLOAT16, x, y, poly_arg_none()), c,
        poly_arg_none()
    );
    PolyUOp *rewritten = poly_graph_rewrite(ctx, expression, poly_symbolic());
    PolyUOp *canonical_expected = poly_graph_rewrite(ctx, expected, poly_symbolic());
    ASSERT_NOT_NULL(rewritten);
    ASSERT_NOT_NULL(canonical_expected);
    ASSERT_PTR_EQ(rewritten, canonical_expected);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, reciprocal_product_stabilizes_after_commutative_const_move) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Exact half-GELU spelling at the proven first divergence:
   *   x * (((v*d)*d)*-1), d=1/(1+x)
   * Pinned symbolic.py:286-287 first moves -1 to the end; :478-480 then
   * replaces x*(d*y) with y*(1-d). */
  PolyUOp *z = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(0));
  PolyUOp *v = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(1));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT16, z, poly_arg_none());
  PolyUOp *one = poly_const_like_float(ctx, x, 1.0);
  PolyUOp *neg_one = poly_const_like_float(ctx, x, -1.0);
  PolyUOp *d = poly_uop1(
      ctx, POLY_OP_RECIPROCAL, POLY_FLOAT16,
      poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT16, one, x, poly_arg_none()), poly_arg_none()
  );
  ASSERT_NOT_NULL(z);
  ASSERT_NOT_NULL(v);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(neg_one);
  ASSERT_NOT_NULL(d);

  PolyUOp *vd = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, v, d, poly_arg_none());
  PolyUOp *expression = poly_uop2(
      ctx, POLY_OP_MUL, POLY_FLOAT16, x,
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_FLOAT16,
          poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, vd, d, poly_arg_none()), neg_one,
          poly_arg_none()
      ),
      poly_arg_none()
  );
  PolyUOp *one_minus_d = poly_uop2(
      ctx, POLY_OP_ADD, POLY_FLOAT16, one,
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, d, neg_one, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *expected = poly_uop2(
      ctx, POLY_OP_MUL, POLY_FLOAT16,
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, vd, one_minus_d, poly_arg_none()), neg_one,
      poly_arg_none()
  );

  PolyUOp *rewritten = poly_graph_rewrite(ctx, expression, poly_symbolic());
  PolyUOp *canonical_expected = poly_graph_rewrite(ctx, expected, poly_symbolic());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_NOT_NULL(canonical_expected);
  ASSERT_PTR_EQ(rewritten, canonical_expected);

  poly_ctx_destroy(ctx);
  PASS();
}

/* WHERE (int branches) */
TEST(sym, minmax_where_int) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, 5), poly_arg_none()
  );
  PolyUOp *u = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_INT32, cond, mk_const(ctx, 3), mk_const(ctx, 5), poly_arg_none()
  );
  check_mm(ctx, u, 3, 5, "WHERE(r<5,3,5)");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_where_dv_dvn) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, 5), poly_arg_none()
  );
  PolyUOp *dv = mk_dvar(ctx, "x", 2, 7);
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *u = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cond, dv, dvn, poly_arg_none());
  /* min(2,-3)=-3, max(7,4)=7 -> [-3, 7] */
  check_mm(ctx, u, -3, 7, "WHERE(r<5,dv,dvn)");
  poly_ctx_destroy(ctx);
  PASS();
}

/* CAST (monotone narrowing to signed) */
TEST(sym, minmax_cast_dv_to_int16) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop1(ctx, POLY_OP_CAST, POLY_INT16, mk_dvar(ctx, "x", 2, 7), poly_arg_none());
  check_mm(ctx, u, 2, 7, "CAST dv i32->i16");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_cast_r_to_int8) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop1(ctx, POLY_OP_CAST, POLY_INT8, mk_range(ctx, 10, 0), poly_arg_none());
  check_mm(ctx, u, 0, 9, "CAST r i32->i8");
  poly_ctx_destroy(ctx);
  PASS();
}

/* Diamond / chained (memoization stress) */
TEST(sym, minmax_diamond_r3_mul2) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = mk_range(ctx, 10, 0);
  PolyUOp *r3 = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, r, mk_const(ctx, 3), poly_arg_none());
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, r3, mk_const(ctx, 2), poly_arg_none());
  /* (r+3)*2 with r in [0,9]: r+3 in [3,12], *2 in [6,24] */
  check_mm(ctx, u, 6, 24, "(r+3)*2");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_deep_chain_is_iterative) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *expr = mk_range(ctx, 10, 0);
  PolyUOp *one = mk_const(ctx, 1);
  for (int i = 0; i < 12000; i++) {
    expr = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, expr, one, poly_arg_none());
  }
  check_mm(ctx, expr, 12000, 12009, "deep add chain");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_backward_slice_score_uses_rewound_scratch_toposort) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *expr = mk_range(ctx, 10, 0);
  PolyUOp *one = mk_const(ctx, 1);
  for (int i = 0; i < 64; i++)
    expr = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, expr, one, poly_arg_none());

  size_t scratch_before = poly_arena_used(ctx->scratch);
  int64_t lo = 0, hi = 0;
  poly_uop_minmax(ctx, expr, &lo, &hi);
  ASSERT_INT_EQ(lo, 64);
  ASSERT_INT_EQ(hi, 73);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Phase D end-to-end gate * poly_arange returns a *Tensor-level* UOp graph composed of movement ops
 * (RESHAPE / EXPAND / PAD / PERMUTE / SHRINK) over a CONST + REDUCE_AXIS
 * + ADD. Tinygrad's _min_max has no special case for any of these (they
 * live in GroupOp.Movement, not GroupOp.Binary), so the bounds at this
 * level fall through to the dtype default. Tight bounds emerge only after
 * rangeify replaces movement ops with RANGE/INDEX — which is exactly where
 * Phase D's reduce_collapse driver runs.
 *
 * This test is therefore a smoke check: poly_uop_minmax must not crash
 * on a real arange graph and must respect the dtype-bounds fallback. */
TEST(sym, minmax_arange_0_to_5_smoke) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ar = poly_arange(ctx, 0.0, 5.0, 1.0);
  ASSERT_NOT_NULL(ar);
  int64_t lo, hi;
  poly_uop_minmax(ctx, ar, &lo, &hi);
  /* Loose bounds: Tensor-level graph hasn't been rangeified yet. */
  ASSERT_TRUE(lo <= 0);
  ASSERT_TRUE(hi >= 4);
  poly_ctx_destroy(ctx);
  PASS();
}

/* PolyUOpCache lifetime + reuse */
TEST(sym, minmax_cache_reuse_yields_same) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOpCache *c = poly_uop_cache_new();
  ASSERT_NOT_NULL(c);
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MUL, POLY_INT32, mk_dvar(ctx, "x", 2, 7), mk_dvar(ctx, "y", -3, 4),
      poly_arg_none()
  );
  int64_t lo1, hi1, lo2, hi2;
  poly_uop_minmax_ex(ctx, u, c, &lo1, &hi1);
  poly_uop_minmax_ex(ctx, u, c, &lo2, &hi2); /* second query should hit cache */
  ASSERT_INT_EQ(lo1, lo2);
  ASSERT_INT_EQ(hi1, hi2);
  ASSERT_INT_EQ(lo1, -21);
  ASSERT_INT_EQ(hi1, 28);
  poly_uop_cache_destroy(c);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_explicit_cache_does_not_allocate_arena_values) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOpCache *c = poly_uop_cache_new();
  ASSERT_NOT_NULL(c);

  PolyUOp *x = mk_dvar(ctx, "x", 0, 1023);
  PolyUOp *y = mk_dvar(ctx, "y", -7, 11);
  PolyUOp *u = x;
  for (int i = 0; i < 8; i++) {
    PolyUOp *k = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i + 1));
    PolyUOp *a = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, u, k, poly_arg_none());
    PolyUOp *b = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, y, k, poly_arg_none());
    u = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());
  }

  PolyCtxStats before = {0}, after = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &before), 0);

  int64_t lo = 0, hi = 0;
  poly_uop_minmax_ex(ctx, u, c, &lo, &hi);

  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after), 0);
  ASSERT_INT_EQ(after.arena_bytes, before.arena_bytes);
  ASSERT_INT_EQ(lo, -216);
  ASSERT_INT_EQ(hi, 1455);

  poly_uop_cache_destroy(c);
  poly_ctx_destroy(ctx);
  PASS();
}

/* poly_logical_not / bool comparison parity (Phase A P5) *
 * Tinygrad's `logical_not()` (mixin/elementwise.py:25-33) lowers to
 *   CMPNE(CAST(x, bool), CONST(true))
 * which symbolic.py:126 collapses to
 *   CMPNE(x, CONST(true))           when x is already bool.
 *
 * Polygrad's poly_logical_not is the same form (CMPNE-with-true). All
 * comparison helpers (poly_le, poly_ge, poly_eq) return bool via this
 * canonical NOT, mirroring tinygrad's mixin/elementwise.py:240-250.
 *
 * Captured against tinygrad in test/parity_scripts/tg_logical_not_gt.py. */
TEST(sym, logical_not_canonical_form) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = mk_range(ctx, 5, 0);
  PolyUOp *c3 = mk_const(ctx, 3);
  PolyUOp *lt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r, c3, poly_arg_none());
  PolyUOp *not_lt = poly_logical_not(ctx, lt);
  /* Shape: CMPNE(CMPLT(...), CONST(true)) */
  ASSERT_INT_EQ(not_lt->op, POLY_OP_CMPNE);
  ASSERT_TRUE(poly_dtype_eq(not_lt->dtype, POLY_BOOL));
  ASSERT_INT_EQ(not_lt->n_src, 2);
  ASSERT_PTR_EQ(not_lt->src[0], lt);
  ASSERT_INT_EQ(not_lt->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(not_lt->src[1]->dtype, POLY_BOOL));
  ASSERT_TRUE(not_lt->src[1]->arg.b == true);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, poly_le_returns_bool) {
  /* poly_le must return bool (was previously float WHERE(0,1) before P5). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = mk_range(ctx, 5, 0);
  PolyUOp *b = mk_const(ctx, 3);
  PolyUOp *le = poly_le(ctx, a, b);
  ASSERT_TRUE(poly_dtype_eq(le->dtype, POLY_BOOL));
  /* Structure: CMPNE(CMPLT(b, a), CONST(true)) per the (b<a).logical_not() form */
  ASSERT_INT_EQ(le->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(le->src[0]->op, POLY_OP_CMPLT);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, poly_ge_returns_bool) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = mk_range(ctx, 5, 0);
  PolyUOp *b = mk_const(ctx, 3);
  PolyUOp *ge = poly_ge(ctx, a, b);
  ASSERT_TRUE(poly_dtype_eq(ge->dtype, POLY_BOOL));
  ASSERT_INT_EQ(ge->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(ge->src[0]->op, POLY_OP_CMPLT);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, poly_eq_returns_bool) {
  /* poly_eq = (a != b).logical_not() = CMPNE(CMPNE(a,b), true). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = mk_range(ctx, 5, 0);
  PolyUOp *b = mk_const(ctx, 3);
  PolyUOp *eq = poly_eq(ctx, a, b);
  ASSERT_TRUE(poly_dtype_eq(eq->dtype, POLY_BOOL));
  ASSERT_INT_EQ(eq->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(eq->src[0]->op, POLY_OP_CMPNE);
  poly_ctx_destroy(ctx);
  PASS();
}

/* CAST(CONST) constant fold parity (Phase D regression coverage) *
 * Mirrors test/parity_scripts/tg_cast_const_fold_gt.py cases A-E.
 *
 * History: Phase D's pm_reduce_unparented produces
 *   MUL(CONST_float, CAST(CONST_int -> float))
 * which symbolic_simple folds via rule_cast_const + rule_const_fold_binary.
 * The original poly_const_like (src/pat.c) blindly copied the source
 * CONST's PolyArg into a CONST tagged with the new dtype, producing a
 * CONST(dtype=float, arg.kind=INT) — a tagged-union mismatch that the
 * codegen misread as a denormal/zero. The fix: poly_const_like now
 * normalizes the value through poly_arg_float / _int / _bool dispatch,
 * matching tinygrad's DType.const(b) at dtype.py:92-100. These tests
 * lock the fix in place. */

TEST(sym, cast_const_int_to_float) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *cast = poly_cast(ctx, c, POLY_FLOAT32);
  PolyUOp *folded = simplify(ctx, cast);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_FLOAT32));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_FLOAT);
  ASSERT_TRUE(folded->arg.f == 5.0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, vector_stack_cast_remains_explicit_like_tinygrad) {
  /* Pinned symbolic.py:148 folds CAST only over UPat.cvar (scalar CONST).
   * A vector STACK is not a cvar and retains its explicit CAST. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *lanes[2] = {one, two};
  PolyUOp *stack =
      poly_uop(ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INT32, 2), lanes, 2, poly_arg_none());
  PolyUOp *cast =
      poly_uop1(ctx, POLY_OP_CAST, poly_dtype_vec(POLY_FLOAT32, 2), stack, poly_arg_none());
  PolyUOp *rewritten = simplify(ctx, cast);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, poly_dtype_vec(POLY_FLOAT32, 2)));
  ASSERT_PTR_EQ(rewritten->src[0], stack);
  ASSERT_TRUE(poly_dtype_eq(stack->dtype, poly_dtype_vec(POLY_INT32, 2)));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, cast_const_float_to_int) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.7));
  PolyUOp *cast = poly_cast(ctx, c, POLY_INT32);
  PolyUOp *folded = simplify(ctx, cast);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_INT32));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_INT);
  ASSERT_INT_EQ((int)folded->arg.i, 3); /* truncates to 3 */
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, cast_const_large_int_to_uint64_preserves_integer_bits) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t big = 9007199254740993LL;
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(big));
  PolyUOp *cast = poly_cast(ctx, c, POLY_UINT64);
  PolyUOp *folded = simplify(ctx, cast);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_UINT64));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_INT);
  ASSERT_TRUE(folded->arg.i == big);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, cast_const_bool_to_float) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *cast = poly_cast(ctx, c, POLY_FLOAT32);
  PolyUOp *folded = simplify(ctx, cast);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_FLOAT32));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_FLOAT);
  ASSERT_TRUE(folded->arg.f == 1.0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, cast_const_zero_int_to_bool) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *cast = poly_cast(ctx, c, POLY_BOOL);
  PolyUOp *folded = simplify(ctx, cast);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_BOOL));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_BOOL);
  ASSERT_TRUE(folded->arg.b == false);
  poly_ctx_destroy(ctx);
  PASS();
}

/* The exact MUL(CONST_float, CAST(CONST_int)) shape that pm_reduce_unparented
 * produces and that triggered the original expand_reduce_e2e regression. */
TEST(sym, mul_float_cast_int_const_fold) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *one_f = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *five_i = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *cast = poly_cast(ctx, five_i, POLY_FLOAT32);
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, one_f, cast);
  PolyUOp *folded = simplify(ctx, mul);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_FLOAT32));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_FLOAT);
  ASSERT_TRUE(folded->arg.f == 5.0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, double_logical_not_idempotent_via_minmax) {
  /* Tinygrad symbolic.py:91: x.logical_not().logical_not() -> x.
   * Polygrad still doesn't carry a dedicated double-not rewrite, but the
   * bound semantics must match even without structural folding. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = mk_range(ctx, 5, 0);
  PolyUOp *lt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r, mk_const(ctx, 3), poly_arg_none());
  PolyUOp *not_lt = poly_logical_not(ctx, lt);
  PolyUOp *not_not_lt = poly_logical_not(ctx, not_lt);
  int64_t lo_a, hi_a, lo_b, hi_b;
  poly_uop_minmax(ctx, lt, &lo_a, &hi_a);
  poly_uop_minmax(ctx, not_not_lt, &lo_b, &hi_b);
  ASSERT_INT_EQ(lo_a, lo_b);
  ASSERT_INT_EQ(hi_a, hi_b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, weak_index_combines_same_base_terms_with_pinned_dtype_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/symbolic.py:242,247 canonicalizes both x+x and
   * x*c0+x*c1 while retaining the weak-index result dtype. */
  PolyUOp *n = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("n", 1, 8));
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *five = poly_const_int(ctx, 5);
  ASSERT_NOT_NULL(n);
  ASSERT_NOT_NULL(two);
  ASSERT_NOT_NULL(three);
  ASSERT_NOT_NULL(five);

  PolyUOp *add_self = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, n, n, poly_arg_none());
  PolyUOp *expected_two = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, two, poly_arg_none());
  PolyUOp *rewritten_self = poly_graph_rewrite(ctx, add_self, poly_symbolic());
  ASSERT_PTR_EQ(rewritten_self, expected_two);
  ASSERT_TRUE(poly_dtype_eq(rewritten_self->dtype, POLY_INDEX));

  PolyUOp *term_two = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, two, poly_arg_none());
  PolyUOp *term_three = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, three, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, term_two, term_three, poly_arg_none());
  PolyUOp *expected_five = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, five, poly_arg_none());
  PolyUOp *rewritten_sum = poly_graph_rewrite(ctx, sum, poly_symbolic());
  ASSERT_PTR_EQ(rewritten_sum, expected_five);
  ASSERT_EQ(rewritten_sum->op, POLY_OP_MUL);
  ASSERT_INT_EQ(rewritten_sum->n_src, 2);
  ASSERT_PTR_EQ(rewritten_sum->src[0], n);
  ASSERT_PTR_EQ(rewritten_sum->src[1], five);
  ASSERT_TRUE(poly_dtype_eq(rewritten_sum->dtype, POLY_INDEX));

  PolyUOp *y = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("y", 1, 8));
  PolyUOp *assoc_source = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INDEX, poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, y, n, poly_arg_none()),
      n, poly_arg_none()
  );
  PolyUOp *assoc_expected = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INDEX, y,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, two, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *assoc_rewritten = poly_graph_rewrite(ctx, assoc_source, poly_symbolic());
  ASSERT_PTR_EQ(assoc_rewritten, assoc_expected);
  ASSERT_INT_EQ(assoc_rewritten->op, POLY_OP_ADD);
  ASSERT_INT_EQ(assoc_rewritten->src[1]->op, POLY_OP_MUL);
  ASSERT_TRUE(poly_dtype_eq(assoc_rewritten->src[1]->dtype, POLY_INDEX));

  for (int64_t value = 1; value <= 8; value = value == 1 ? 4 : 8) {
    PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(value));
    PolyUOp *bound_y = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(value + 1));
    PolyUOp *from[] = {n, y};
    PolyUOp *to[] = {bound, bound_y};
    PolyUOp *original_value =
        poly_graph_rewrite(ctx, poly_uop_substitute(ctx, sum, from, to, 1), poly_symbolic());
    PolyUOp *rewritten_value = poly_graph_rewrite(
        ctx, poly_uop_substitute(ctx, rewritten_sum, from, to, 1), poly_symbolic()
    );
    int64_t original_i = 0, rewritten_i = 0;
    ASSERT_INT_EQ(poly_uop_const_i64(original_value, &original_i), 0);
    ASSERT_INT_EQ(poly_uop_const_i64(rewritten_value, &rewritten_i), 0);
    ASSERT_INT_EQ(original_i, value * 5);
    ASSERT_INT_EQ(rewritten_i, original_i);

    PolyUOp *assoc_original_value = poly_graph_rewrite(
        ctx, poly_uop_substitute(ctx, assoc_source, from, to, 2), poly_symbolic()
    );
    PolyUOp *assoc_rewritten_value = poly_graph_rewrite(
        ctx, poly_uop_substitute(ctx, assoc_rewritten, from, to, 2), poly_symbolic()
    );
    int64_t assoc_original_i = 0, assoc_rewritten_i = 0;
    ASSERT_INT_EQ(poly_uop_const_i64(assoc_original_value, &assoc_original_i), 0);
    ASSERT_INT_EQ(poly_uop_const_i64(assoc_rewritten_value, &assoc_rewritten_i), 0);
    ASSERT_INT_EQ(assoc_original_i, 3 * value + 1);
    ASSERT_INT_EQ(assoc_rewritten_i, assoc_original_i);
    if (value == 8) break;
  }

  PolyUOp *max_coeff = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT64_MAX));
  PolyUOp *index_one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *overflow_sum = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INDEX,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, max_coeff, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, index_one, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *overflow_rewritten = poly_graph_rewrite(ctx, overflow_sum, poly_symbolic());
  ASSERT_NOT_NULL(overflow_rewritten);
  ASSERT_INT_EQ(overflow_rewritten->op, POLY_OP_ADD);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, lossless_nested_cast_roundtrip_matches_tinygrad) {
  /* Pinned symbolic.py:151-152 removes b.cast(a).cast(b) only when a
   * preserves every value representable by b. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("cast_x", INT32_MIN, INT32_MAX)
  );

  PolyUOp *weak_roundtrip = poly_uop1(
      ctx, POLY_OP_CAST, POLY_INT32, poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, x, poly_arg_none()),
      poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, weak_roundtrip, poly_symbolic()), x);

  PolyUOp *x64 = poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, x, poly_arg_none());
  PolyUOp *wide_weak_roundtrip = poly_uop1(
      ctx, POLY_OP_CAST, POLY_INT64, poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, x64, poly_arg_none()),
      poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, wide_weak_roundtrip, poly_symbolic()), x64);

  PolyUOp *narrow = poly_uop1(ctx, POLY_OP_CAST, POLY_INT16, x, poly_arg_none());
  PolyUOp *lossy_roundtrip = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, narrow, poly_arg_none());
  PolyUOp *lossy_lowered = poly_graph_rewrite(ctx, lossy_roundtrip, poly_symbolic());
  ASSERT_INT_EQ(lossy_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(lossy_lowered->dtype, POLY_INT32));
  ASSERT_INT_EQ(lossy_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(lossy_lowered->src[0]->dtype, POLY_INT16));

  PolyDType int2 = poly_dtype_vec(POLY_INT32, 2);
  PolyDType weak2 = poly_dtype_vec(POLY_INDEX, 2);
  PolyUOp *vector_lanes[2] = {x, x};
  PolyUOp *vector_x = poly_uop(ctx, POLY_OP_STACK, int2, vector_lanes, 2, poly_arg_none());
  PolyUOp *vector_roundtrip = poly_uop1(
      ctx, POLY_OP_CAST, int2, poly_uop1(ctx, POLY_OP_CAST, weak2, vector_x, poly_arg_none()),
      poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, vector_roundtrip, poly_symbolic()), vector_x);

  PolyUOp *bool_x =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_BOOL, poly_arg_define_var("cast_bool", 0, 1));
  PolyUOp *bool_vector_roundtrip = poly_uop1(
      ctx, POLY_OP_CAST, POLY_BOOL, poly_uop1(ctx, POLY_OP_CAST, int2, bool_x, poly_arg_none()),
      poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, bool_vector_roundtrip, poly_symbolic()), bool_x);

  PolyDType int_ptr = poly_dtype_ptr(POLY_INT32, 4, POLY_ADDR_GLOBAL);
  PolyUOp *bool_pointer_roundtrip = poly_uop1(
      ctx, POLY_OP_CAST, POLY_BOOL, poly_uop1(ctx, POLY_OP_CAST, int_ptr, bool_x, poly_arg_none()),
      poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, bool_pointer_roundtrip, poly_symbolic()), bool_x);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, proven_long_binary_narrows_only_with_exact_i32_bounds) {
  /* Pinned tinygrad uop/symbolic.py:297-299 computes signed-long Binary
   * operations in int32 only when the result and both inputs provably fit,
   * then casts the result back to long. This is shared symbolic behavior, not
   * a WebGPU or gather legalization. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *bounded = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT64, poly_arg_define_var("bounded_long", 0, 1)
  );
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(2));
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_INT64, bounded, two, poly_arg_none());
  PolyUOp *lowered = poly_graph_rewrite(ctx, sum, poly_symbolic());
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(lowered->dtype, POLY_INT64));
  ASSERT_INT_EQ(lowered->n_src, 1);
  ASSERT_INT_EQ(lowered->src[0]->op, POLY_OP_ADD);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[0]->n_src, 2);
  ASSERT_INT_EQ(lowered->src[0]->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->src[0]->dtype, POLY_INT32));
  ASSERT_PTR_EQ(lowered->src[0]->src[0]->src[0], bounded);
  ASSERT_INT_EQ(lowered->src[0]->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[0]->src[1]->arg.i, 2);

  PolyUOp *wide = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT64,
      poly_arg_define_var("wide_long", INT32_MAX - 1, INT32_MAX)
  );
  PolyUOp *wide_sum = poly_uop2(ctx, POLY_OP_ADD, POLY_INT64, wide, two, poly_arg_none());
  PolyUOp *wide_lowered = poly_graph_rewrite(ctx, wide_sum, poly_symbolic());
  ASSERT_PTR_EQ(wide_lowered, wide_sum);
  ASSERT_TRUE(poly_dtype_eq(wide_lowered->dtype, POLY_INT64));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, self_comparisons_keep_boolean_result_dtype) {
  /* Pinned symbolic.py:107-108,120-121 casts the operand-like false constant
   * to scalar/vector bool. The result must not retain the integer operand
   * dtype after constant folding. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType uint16x2 = poly_dtype_vec(POLY_UINT16, 2);
  PolyDType boolx2 = poly_dtype_vec(POLY_BOOL, 2);
  PolyUOp *scalar = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_UINT16, poly_arg_define_var("cmp_scalar", 0, 7)
  );
  PolyUOp *vector = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, uint16x2, poly_arg_define_var("cmp_vector", 0, 7)
  );
  const PolyOps ops[] = {POLY_OP_CMPLT, POLY_OP_CMPNE};
  for (int i = 0; i < 2; i++) {
    PolyUOp *scalar_cmp =
        poly_uop2(ctx, ops[i], POLY_BOOL, scalar, scalar, poly_arg_none());
    PolyUOp *scalar_out = poly_graph_rewrite(ctx, scalar_cmp, poly_symbolic());
    ASSERT_NOT_NULL(scalar_out);
    ASSERT_INT_EQ(scalar_out->op, POLY_OP_CONST);
    ASSERT_TRUE(poly_dtype_eq(scalar_out->dtype, POLY_BOOL));
    ASSERT_INT_EQ(scalar_out->arg.kind, POLY_ARG_BOOL);
    ASSERT_TRUE(!scalar_out->arg.b);

    PolyUOp *vector_cmp = poly_uop2(ctx, ops[i], boolx2, vector, vector, poly_arg_none());
    PolyUOp *vector_out = poly_graph_rewrite(ctx, vector_cmp, poly_symbolic());
    ASSERT_NOT_NULL(vector_out);
    ASSERT_INT_EQ(vector_out->op, POLY_OP_CONST);
    ASSERT_TRUE(poly_dtype_eq(vector_out->dtype, boolx2));
    ASSERT_INT_EQ(vector_out->arg.kind, POLY_ARG_BOOL);
    ASSERT_TRUE(!vector_out->arg.b);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, general_nested_cast_composition_matches_tinygrad) {
  /* Pinned symbolic.py:297-301 composes through an intermediate CAST when
   * either can_lossless_cast proves it globally or integer bounds prove this
   * value is not narrowed. Unsafe narrowing must retain both CASTs. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *wide = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT32,
      poly_arg_define_var("compose_wide", -(INT64_C(1) << 30), INT64_C(1) << 30)
  );
  wide = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32, wide,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1)), poly_arg_none()
  );
  PolyUOp *weak_to_uint = poly_uop1(
      ctx, POLY_OP_CAST, POLY_UINT32,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, wide, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *weak_to_uint_lowered = poly_graph_rewrite(ctx, weak_to_uint, poly_symbolic());
  ASSERT_INT_EQ(weak_to_uint_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(weak_to_uint_lowered->dtype, POLY_UINT32));
  ASSERT_PTR_EQ(weak_to_uint_lowered->src[0], wide);

  PolyUOp *small =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("compose_small", 0, 255));
  PolyUOp *safe_narrow = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT32,
      poly_uop1(ctx, POLY_OP_CAST, POLY_UINT8, small, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *safe_narrow_lowered = poly_graph_rewrite(ctx, safe_narrow, poly_symbolic());
  ASSERT_INT_EQ(safe_narrow_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(safe_narrow_lowered->dtype, POLY_FLOAT32));
  ASSERT_PTR_EQ(safe_narrow_lowered->src[0], small);

  PolyUOp *unsafe = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("compose_unsafe", -1, 255)
  );
  PolyUOp *unsafe_narrow = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT32,
      poly_uop1(ctx, POLY_OP_CAST, POLY_UINT8, unsafe, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *unsafe_narrow_lowered = poly_graph_rewrite(ctx, unsafe_narrow, poly_symbolic());
  ASSERT_INT_EQ(unsafe_narrow_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(unsafe_narrow_lowered->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(unsafe_narrow_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(unsafe_narrow_lowered->src[0]->dtype, POLY_UINT8));
  ASSERT_PTR_EQ(unsafe_narrow_lowered->src[0]->src[0], unsafe);

  /* Pinned UPat dtype matching does not scalarize PtrDType to its base, so
   * pointer intermediates are outside the integer-bounds rule. */
  PolyDType int_ptr = poly_dtype_ptr(POLY_INT32, 4, POLY_ADDR_GLOBAL);
  PolyUOp *pointer_intermediate = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT32,
      poly_uop1(ctx, POLY_OP_CAST, int_ptr, small, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *pointer_lowered = poly_graph_rewrite(ctx, pointer_intermediate, poly_symbolic());
  ASSERT_INT_EQ(pointer_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(pointer_lowered->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(pointer_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(pointer_lowered->src[0]->dtype, int_ptr));
  ASSERT_PTR_EQ(pointer_lowered->src[0]->src[0], small);

  /* Pinned Python bounds retain narrowing intermediates for full uint64 and
   * weakint ranges. Polygrad's int64 cache endpoints must not act as proof. */
  PolyUOp *unbounded_u64 = poly_uop0(ctx, POLY_OP_NOOP, POLY_UINT64, poly_arg_none());
  PolyUOp *u64_narrow = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, unbounded_u64, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *u64_narrow_lowered = poly_graph_rewrite(ctx, u64_narrow, poly_symbolic());
  ASSERT_INT_EQ(u64_narrow_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(u64_narrow_lowered->src[0]->dtype, POLY_INT64));

  PolyUOp *unbounded_weak = poly_uop0(ctx, POLY_OP_NOOP, POLY_INDEX, poly_arg_none());
  PolyUOp *weak_narrow = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, unbounded_weak, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *weak_narrow_lowered = poly_graph_rewrite(ctx, weak_narrow, poly_symbolic());
  ASSERT_INT_EQ(weak_narrow_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(weak_narrow_lowered->src[0]->dtype, POLY_INT64));

  /* Pinned `_min_max` keeps arbitrary-precision endpoints through derived
   * uint64/weakint expressions. Sentinel arithmetic must not prove a
   * narrowing intermediate safe. */
  PolyUOp *weak_shr = poly_uop2(
      ctx, POLY_OP_SHR, POLY_INDEX, unbounded_weak,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1)), poly_arg_none()
  );
  PolyUOp *weak_shr_narrow = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, weak_shr, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *weak_shr_lowered = poly_graph_rewrite(ctx, weak_shr_narrow, poly_symbolic());
  ASSERT_INT_EQ(weak_shr_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(weak_shr_lowered->src[0]->dtype, POLY_INT64));

  PolyUOp *u64_shr = poly_uop2(
      ctx, POLY_OP_SHR, POLY_UINT64, unbounded_u64,
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(32)), poly_arg_none()
  );
  PolyUOp *u64_shr_i32 = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, u64_shr, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *u64_shr_i32_lowered = poly_graph_rewrite(ctx, u64_shr_i32, poly_symbolic());
  ASSERT_INT_EQ(u64_shr_i32_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(u64_shr_i32_lowered->src[0]->dtype, POLY_INT32));

  /* The same exact [0, 2**32-1] range does fit int64, so pinned removes
   * that wider intermediate. */
  PolyUOp *u64_shr_i64 = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, u64_shr, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *u64_shr_i64_lowered = poly_graph_rewrite(ctx, u64_shr_i64, poly_symbolic());
  ASSERT_INT_EQ(u64_shr_i64_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(u64_shr_i64_lowered->dtype, POLY_FLOAT64));
  ASSERT_PTR_EQ(u64_shr_i64_lowered->src[0], u64_shr);

  PolyUOp *u64_mask = poly_uop2(
      ctx, POLY_OP_AND, POLY_UINT64, unbounded_u64,
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX)), poly_arg_none()
  );
  PolyUOp *u64_mask_i64 = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, u64_mask, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *u64_mask_lowered = poly_graph_rewrite(ctx, u64_mask_i64, poly_symbolic());
  ASSERT_INT_EQ(u64_mask_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(u64_mask_lowered->dtype, POLY_FLOAT64));
  ASSERT_PTR_EQ(u64_mask_lowered->src[0], u64_mask);

  /* Explicit bounded uint64/weakint variables remain exact and do compose. */
  PolyDType bounded_types[2] = {POLY_UINT64, POLY_INDEX};
  for (int i = 0; i < 2; i++) {
    PolyUOp *bounded = poly_uop0(
        ctx, POLY_OP_DEFINE_VAR, bounded_types[i],
        poly_arg_define_var(i == 0 ? "compose_bounded_u64" : "compose_bounded_weak", 0, 255)
    );
    PolyUOp *bounded_cast = poly_uop1(
        ctx, POLY_OP_CAST, POLY_FLOAT64,
        poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, bounded, poly_arg_none()), poly_arg_none()
    );
    PolyUOp *bounded_lowered = poly_graph_rewrite(ctx, bounded_cast, poly_symbolic());
    ASSERT_INT_EQ(bounded_lowered->op, POLY_OP_CAST);
    ASSERT_TRUE(poly_dtype_eq(bounded_lowered->dtype, POLY_FLOAT64));
    ASSERT_PTR_EQ(bounded_lowered->src[0], bounded);
  }

  PolyDType int2 = poly_dtype_vec(POLY_INT32, 2);
  PolyDType weak2 = poly_dtype_vec(POLY_INDEX, 2);
  PolyDType uint2 = poly_dtype_vec(POLY_UINT32, 2);
  PolyUOp *vector =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, int2, poly_arg_define_var("compose_vector", 0, 255));
  PolyUOp *vector_cast = poly_uop1(
      ctx, POLY_OP_CAST, uint2, poly_uop1(ctx, POLY_OP_CAST, weak2, vector, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *vector_lowered = poly_graph_rewrite(ctx, vector_cast, poly_symbolic());
  ASSERT_INT_EQ(vector_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(vector_lowered->dtype, uint2));
  ASSERT_PTR_EQ(vector_lowered->src[0], vector);

  /* Pinned ops.py:1017-1018 does not treat a vector signed CAST as monotone.
   * Its full int16x2 bounds include negatives, so uint16x2 must be retained. */
  PolyDType short2 = poly_dtype_vec(POLY_INT16, 2);
  PolyDType ushort2 = poly_dtype_vec(POLY_UINT16, 2);
  PolyDType float2 = poly_dtype_vec(POLY_FLOAT32, 2);
  PolyUOp *wide_vector = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, int2, poly_arg_define_var("compose_wide_vector", 0, 100000)
  );
  PolyUOp *short_vector = poly_uop1(ctx, POLY_OP_CAST, short2, wide_vector, poly_arg_none());
  PolyUOp *vector_unsigned = poly_uop1(
      ctx, POLY_OP_CAST, float2,
      poly_uop1(ctx, POLY_OP_CAST, ushort2, short_vector, poly_arg_none()), poly_arg_none()
  );
  ASSERT_TRUE(poly_pm_rewrite(poly_symbolic(), ctx, vector_unsigned) == NULL);
  PolyUOp *vector_unsigned_lowered = poly_graph_rewrite(ctx, vector_unsigned, poly_symbolic());
  ASSERT_TRUE(poly_dtype_eq(vector_unsigned_lowered->dtype, float2));
  ASSERT_INT_EQ(vector_unsigned_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(vector_unsigned_lowered->src[0]->dtype, ushort2));
  ASSERT_PTR_EQ(vector_unsigned_lowered->src[0]->src[0], short_vector);

  /* The retained uint16 conversion is value-material: for lane value 65535,
   * int16 -> uint16 -> float is 65535.0, while deleting uint16 yields -1.0. */
  PolyArg lane = poly_arg_int(65535);
  PolyArg as_short = poly_exec_alu(POLY_OP_CAST, POLY_INT16, &lane, 1, true);
  PolyArg as_ushort = poly_exec_alu(POLY_OP_CAST, POLY_UINT16, &as_short, 1, true);
  PolyArg preserved_value = poly_exec_alu(POLY_OP_CAST, POLY_FLOAT32, &as_ushort, 1, true);
  PolyArg deleted_value = poly_exec_alu(POLY_OP_CAST, POLY_FLOAT32, &as_short, 1, true);
  ASSERT_TRUE(preserved_value.kind == POLY_ARG_FLOAT);
  ASSERT_TRUE(deleted_value.kind == POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(preserved_value.f, 65535.0, 0.0);
  ASSERT_FLOAT_EQ(deleted_value.f, -1.0, 0.0);

  /* Pinned ops.py:980-999 assigns [0,0] directly only to empty floor
   * div/mod numerators. CDIV still uses its corner formula. */
  PolyUOp *empty_range = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INT32, poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-4)),
      poly_arg_range(0, POLY_AXIS_LOOP)
  );
  PolyOps empty_ops[4] = {POLY_OP_CDIV, POLY_OP_CMOD, POLY_OP_FLOORDIV, POLY_OP_FLOORMOD};
  bool empty_composes[4] = {false, true, true, true};
  for (int i = 0; i < 4; i++) {
    PolyUOp *empty_div = poly_uop2(
        ctx, empty_ops[i], POLY_INT32, empty_range,
        poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3)), poly_arg_none()
    );
    PolyUOp *empty_cast = poly_uop1(
        ctx, POLY_OP_CAST, POLY_FLOAT32,
        poly_uop1(ctx, POLY_OP_CAST, POLY_UINT8, empty_div, poly_arg_none()), poly_arg_none()
    );
    PolyUOp *empty_direct = poly_pm_rewrite(poly_symbolic(), ctx, empty_cast);
    if (!empty_composes[i]) {
      ASSERT_TRUE(empty_direct == NULL);
      continue;
    }
    ASSERT_NOT_NULL(empty_direct);
    ASSERT_INT_EQ(empty_direct->op, POLY_OP_CAST);
    ASSERT_TRUE(poly_dtype_eq(empty_direct->dtype, POLY_FLOAT32));
    ASSERT_PTR_EQ(empty_direct->src[0], empty_div);
  }

  poly_ctx_destroy(ctx);
  PASS();
}
