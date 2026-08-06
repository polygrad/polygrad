/*
 * Exact integer-argument tests.
 *
 * Pinned tinygrad keeps UOp integer arguments as Python ints and evaluates
 * symbolic integer ALU with Python semantics before optional dtype
 * truncation (uop/ops.py:1176-1197, dtype.py:92-100).
 */

#include "test_harness.h"
#include "../src/bigint.h"
#include "../src/pat.h"

static bool bigint_decimal_eq(const PolyInt *value, const char *expected) {
  char *actual = poly_int_to_decimal(value);
  bool equal = actual && strcmp(actual, expected) == 0;
  free(actual);
  return equal;
}

static bool bigint_from(PolyInt *value, const char *decimal) {
  poly_int_init(value);
  return poly_int_from_decimal(value, decimal);
}

TEST(bigint, pinned_python_integer_arithmetic) {
  PolyInt a = {0}, b = {0}, result = {0}, quotient = {0}, remainder = {0};

  ASSERT_TRUE(bigint_from(&a, "-1267650600228229401496703205385"));
  ASSERT_TRUE(bigint_from(&b, "67"));
  ASSERT_TRUE(poly_int_shr(&result, &a, 67));
  ASSERT_TRUE(bigint_decimal_eq(&result, "-8589934593"));
  poly_int_free(&result);
  poly_int_free(&b);

  ASSERT_TRUE(bigint_from(&b, "39614081257132168796771975171"));
  ASSERT_TRUE(poly_int_bitwise(&result, POLY_OP_XOR, &a, &b));
  ASSERT_TRUE(bigint_decimal_eq(&result, "-1307264681485361570293475180556"));
  poly_int_free(&result);
  poly_int_free(&a);
  poly_int_free(&b);

  ASSERT_TRUE(bigint_from(&a, "-1267650600228229401496703205383"));
  ASSERT_TRUE(bigint_from(&b, "8589934599"));
  ASSERT_TRUE(poly_int_divmod(&quotient, &remainder, &a, &b, false));
  ASSERT_TRUE(bigint_decimal_eq(&quotient, "-147573952469417328737"));
  ASSERT_TRUE(bigint_decimal_eq(&remainder, "-8589933920"));
  poly_int_free(&quotient);
  poly_int_free(&remainder);

  ASSERT_TRUE(poly_int_divmod(&quotient, &remainder, &a, &b, true));
  ASSERT_TRUE(bigint_decimal_eq(&quotient, "-147573952469417328738"));
  ASSERT_TRUE(bigint_decimal_eq(&remainder, "679"));
  poly_int_free(&quotient);
  poly_int_free(&remainder);
  poly_int_free(&a);
  poly_int_free(&b);

  ASSERT_TRUE(bigint_from(&a, "1048579"));
  ASSERT_TRUE(bigint_from(&b, "5"));
  ASSERT_TRUE(poly_int_pow(&result, &a, &b));
  ASSERT_TRUE(bigint_decimal_eq(&result, "1267668734219286853217504198899"));
  poly_int_free(&result);
  poly_int_free(&a);
  poly_int_free(&b);
  PASS();
}

TEST(bigint, zero_results_release_discarded_limb_storage) {
  PolyInt a = {0}, b = {0}, quotient = {0}, remainder = {0};
  ASSERT_TRUE(bigint_from(&a, "4294967297"));
  ASSERT_TRUE(bigint_from(&b, "4294967297"));
  ASSERT_TRUE(poly_int_divmod(&quotient, &remainder, &a, &b, false));
  ASSERT_TRUE(bigint_decimal_eq(&quotient, "1"));
  ASSERT_INT_EQ((int)remainder.n_limbs, 0);
  ASSERT_TRUE(remainder.limbs == NULL);

  poly_int_free(&quotient);
  poly_int_free(&remainder);
  poly_int_free(&a);
  poly_int_free(&b);
  PASS();
}

TEST(bigint, fixed_width_truncation_matches_pinned_exec_alu) {
  PolyInt value = {0}, truncated = {0};
  ASSERT_TRUE(bigint_from(&value, "-1267650600228229401496703205382"));

  ASSERT_TRUE(poly_int_truncate(&truncated, &value, 64, true));
  ASSERT_TRUE(bigint_decimal_eq(&truncated, "18446744073709551610"));
  poly_int_free(&truncated);

  ASSERT_TRUE(poly_int_truncate(&truncated, &value, 64, false));
  ASSERT_TRUE(bigint_decimal_eq(&truncated, "-6"));
  poly_int_free(&truncated);

  ASSERT_TRUE(poly_int_truncate(&truncated, &value, 32, true));
  ASSERT_TRUE(bigint_decimal_eq(&truncated, "4294967290"));
  poly_int_free(&truncated);

  ASSERT_TRUE(poly_int_truncate(&truncated, &value, 32, false));
  ASSERT_TRUE(bigint_decimal_eq(&truncated, "-6"));
  poly_int_free(&truncated);
  poly_int_free(&value);
  PASS();
}

static PolyUOp *bigint_const(
    PolyCtx *ctx,
    PolyDType dtype,
    const char *decimal
) {
  PolyInt value = {0};
  if (!bigint_from(&value, decimal)) return NULL;
  PolyUOp *out = poly_uop0(ctx, POLY_OP_CONST, dtype, poly_int_as_arg(&value));
  poly_int_free(&value);
  return out;
}

static PolyUOp *stack2(PolyCtx *ctx, PolyDType dtype, int64_t value) {
  PolyUOp *lane =
      poly_uop0(ctx, POLY_OP_CONST, poly_dtype_scalar(dtype), poly_arg_int(value));
  PolyUOp *src[2] = {lane, lane};
  return poly_uop(ctx, POLY_OP_STACK, dtype, src, 2, poly_arg_none());
}

TEST(bigint, uop_identity_and_vector_carriers_match_pinned_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *negative = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(-1));
  PolyUOp *positive = bigint_const(ctx, POLY_UINT64, "18446744073709551615");
  ASSERT_NOT_NULL(positive);
  ASSERT_TRUE(negative != positive);
  ASSERT_EQ(positive->arg.kind, POLY_ARG_BIGINT);
  ASSERT_TRUE(positive == bigint_const(ctx, POLY_UINT64, "18446744073709551615"));

  uint32_t noncanonical_small[] = {42, 0};
  uint32_t noncanonical_zero[] = {0};
  ASSERT_TRUE(
      poly_uop0(
          ctx, POLY_OP_CONST, POLY_INDEX,
          poly_arg_bigint(1, noncanonical_small, 2)
      ) == poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(42))
  );
  ASSERT_TRUE(
      poly_uop0(
          ctx, POLY_OP_CONST, POLY_INDEX,
          poly_arg_bigint(-1, noncanonical_zero, 1)
      ) == poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0))
  );

  PolyDType u64x2 = poly_dtype_vec(POLY_UINT64, 2);
  PolyUOp *product = poly_graph_rewrite(
      ctx,
      poly_uop2(
          ctx, POLY_OP_MUL, u64x2, stack2(ctx, u64x2, 1023),
          stack2(ctx, u64x2, -1), poly_arg_none()
      ),
      poly_symbolic_simple()
  );
  ASSERT_NOT_NULL(product);
  ASSERT_EQ(product->op, POLY_OP_STACK);
  ASSERT_INT_EQ(product->n_src, 2);
  ASSERT_TRUE(product->src[0] == product->src[1]);
  char *product_lane = poly_arg_integer_to_decimal(product->src[0]->arg);
  ASSERT_STR_EQ(product_lane, "18446744073709550593");
  free(product_lane);

  PolyUOp *sum = poly_uop2(
      ctx, POLY_OP_ADD, u64x2, stack2(ctx, u64x2, INT64_MAX),
      stack2(ctx, u64x2, 1), poly_arg_none()
  );
  PolyUOp *shifted = poly_graph_rewrite(
      ctx,
      poly_uop2(
          ctx, POLY_OP_SHR, u64x2, sum, stack2(ctx, u64x2, 1),
          poly_arg_none()
      ),
      poly_symbolic_simple()
  );
  ASSERT_NOT_NULL(shifted);
  ASSERT_EQ(shifted->op, POLY_OP_STACK);
  ASSERT_TRUE(shifted->src[0] == shifted->src[1]);
  char *shifted_lane = poly_arg_integer_to_decimal(shifted->src[0]->arg);
  ASSERT_STR_EQ(shifted_lane, "4611686018427387904");
  free(shifted_lane);

  PolyUOp *huge = bigint_const(ctx, POLY_INDEX, "1267650600228229401496703205376");
  PolyUOp *weak = poly_graph_rewrite(
      ctx,
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX, huge,
          poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(7)),
          poly_arg_none()
      ),
      poly_symbolic_simple()
  );
  ASSERT_NOT_NULL(weak);
  ASSERT_EQ(weak->op, POLY_OP_CONST);
  char *weak_value = poly_arg_integer_to_decimal(weak->arg);
  ASSERT_STR_EQ(weak_value, "1267650600228229401496703205383");
  free(weak_value);

  poly_ctx_destroy(ctx);
  PASS();
}
