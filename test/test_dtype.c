/*
 * test_dtype.c — Tests for the DType system
 */

#include "test_harness.h"
#include "../src/polygrad.h"

TEST(dtype, predefined_types_exist) {
  ASSERT_INT_EQ(POLY_FLOAT32.bitsize, 32);
  ASSERT_INT_EQ(POLY_FLOAT64.bitsize, 64);
  ASSERT_INT_EQ(POLY_INT32.bitsize, 32);
  ASSERT_INT_EQ(POLY_INT64.bitsize, 64);
  ASSERT_INT_EQ(POLY_BOOL.bitsize, 1);
  ASSERT_INT_EQ(POLY_VOID.bitsize, 0);
  PASS();
}

TEST(dtype, equality) {
  ASSERT_TRUE(poly_dtype_eq(POLY_FLOAT32, POLY_FLOAT32));
  ASSERT_TRUE(poly_dtype_eq(POLY_INT32, POLY_INT32));
  ASSERT_FALSE(poly_dtype_eq(POLY_FLOAT32, POLY_INT32));
  ASSERT_FALSE(poly_dtype_eq(POLY_FLOAT32, POLY_FLOAT64));
  PASS();
}

TEST(dtype, classification) {
  ASSERT_TRUE(poly_dtype_is_float(POLY_FLOAT32));
  ASSERT_TRUE(poly_dtype_is_float(POLY_FLOAT64));
  ASSERT_TRUE(poly_dtype_is_float(POLY_FLOAT16));
  ASSERT_TRUE(poly_dtype_is_float(POLY_BFLOAT16));
  ASSERT_FALSE(poly_dtype_is_float(POLY_INT32));
  ASSERT_FALSE(poly_dtype_is_float(POLY_BOOL));

  ASSERT_TRUE(poly_dtype_is_int(POLY_INT32));
  ASSERT_TRUE(poly_dtype_is_int(POLY_UINT64));
  ASSERT_TRUE(poly_dtype_is_int(POLY_INDEX));
  ASSERT_TRUE(poly_dtype_is_index(POLY_INDEX));
  ASSERT_FALSE(poly_dtype_is_int(POLY_FLOAT32));

  ASSERT_TRUE(poly_dtype_is_unsigned(POLY_UINT8));
  ASSERT_TRUE(poly_dtype_is_unsigned(POLY_UINT32));
  ASSERT_FALSE(poly_dtype_is_unsigned(POLY_INT32));

  ASSERT_TRUE(poly_dtype_is_bool(POLY_BOOL));
  ASSERT_FALSE(poly_dtype_is_bool(POLY_INT32));
  PASS();
}

TEST(dtype, weakint_identity_uses_tinygrad_scalar_semantics) {
  PolyDType weak_like = POLY_INDEX;
  weak_like.bitsize = 144;
  weak_like.count = 1;

  ASSERT_FALSE(poly_dtype_eq(weak_like, POLY_INDEX));
  ASSERT_TRUE(poly_dtype_is_index(weak_like));
  ASSERT_TRUE(poly_dtype_is_int(weak_like));
  ASSERT_TRUE(poly_dtype_eq(poly_dtype_scalar(weak_like), POLY_INDEX));
  PASS();
}

TEST(dtype, ffi_roundtrips_internal_weakint_without_renumbering_public_types) {
  /* Pinned UOp.range exposes dtypes.weakint through ordinary UOp dtype
   * reflection (uop/ops.py:563-565); keep existing FFI ids stable and append
   * that internal dtype after float64. */
  ASSERT_INT_EQ(poly_dtype_id_by_name("float64"), 13);
  ASSERT_INT_EQ(poly_dtype_id_by_name("weakint"), 14);
  ASSERT_INT_EQ(poly_dtype_count(), 15);
  PolyDType dt = POLY_VOID;
  ASSERT_TRUE(poly_dtype_by_id(14, &dt));
  ASSERT_TRUE(poly_dtype_eq(dt, POLY_INDEX));
  PASS();
}

TEST(dtype, itemsize) {
  ASSERT_INT_EQ(poly_dtype_itemsize(POLY_FLOAT32), 4);
  ASSERT_INT_EQ(poly_dtype_itemsize(POLY_FLOAT64), 8);
  ASSERT_INT_EQ(poly_dtype_itemsize(POLY_INT32), 4);
  ASSERT_INT_EQ(poly_dtype_itemsize(POLY_INT8), 1);
  ASSERT_INT_EQ(poly_dtype_itemsize(POLY_BOOL), 1);
  ASSERT_INT_EQ(poly_dtype_itemsize(POLY_FLOAT16), 2);
  PASS();
}

TEST(dtype, vec) {
  PolyDType v4 = poly_dtype_vec(POLY_FLOAT32, 4);
  ASSERT_INT_EQ(v4.count, 4);
  ASSERT_INT_EQ(v4.bitsize, 128);
  ASSERT_TRUE(poly_dtype_is_float(v4));

  /* vec(1) returns scalar */
  PolyDType v1 = poly_dtype_vec(POLY_FLOAT32, 1);
  ASSERT_TRUE(poly_dtype_eq(v1, POLY_FLOAT32));

  /* void doesn't vectorize */
  PolyDType vv = poly_dtype_vec(POLY_VOID, 4);
  ASSERT_TRUE(poly_dtype_eq(vv, POLY_VOID));
  PASS();
}

TEST(dtype, scalar) {
  PolyDType v4 = poly_dtype_vec(POLY_FLOAT32, 4);
  PolyDType s = poly_dtype_scalar(v4);
  ASSERT_INT_EQ(s.count, 1);
  ASSERT_INT_EQ(s.bitsize, 32);
  ASSERT_TRUE(poly_dtype_eq(s, POLY_FLOAT32));
  ASSERT_EQ(s.fmt, POLY_FLOAT32.fmt);

  /* scalar of scalar is identity */
  PolyDType s2 = poly_dtype_scalar(POLY_INT64);
  ASSERT_TRUE(poly_dtype_eq(s2, POLY_INT64));
  PASS();
}

TEST(dtype, least_upper_matches_tinygrad_promotion_lattice) {
  PolyDType out;
  ASSERT_TRUE(poly_dtype_least_upper(POLY_INT32, POLY_FLOAT32, &out));
  ASSERT_TRUE(poly_dtype_eq(out, POLY_FLOAT32));
  ASSERT_TRUE(poly_dtype_least_upper(POLY_INT32, POLY_UINT32, &out));
  ASSERT_TRUE(poly_dtype_eq(out, POLY_INT64));
  ASSERT_TRUE(poly_dtype_least_upper(POLY_UINT64, POLY_FLOAT16, &out));
  ASSERT_TRUE(poly_dtype_eq(out, POLY_FLOAT16));
  ASSERT_TRUE(poly_dtype_least_upper(POLY_BOOL, POLY_INT8, &out));
  ASSERT_TRUE(poly_dtype_eq(out, POLY_INT8));
  PASS();
}

TEST(dtype, can_lossless_cast_matches_tinygrad_supported_table) {
  ASSERT_TRUE(poly_dtype_can_lossless_cast(POLY_INT32, POLY_INDEX));
  ASSERT_TRUE(poly_dtype_can_lossless_cast(POLY_INT64, POLY_INDEX));
  ASSERT_FALSE(poly_dtype_can_lossless_cast(POLY_FLOAT32, POLY_INDEX));
  ASSERT_TRUE(poly_dtype_can_lossless_cast(POLY_BOOL, poly_dtype_vec(POLY_INT32, 2)));
  ASSERT_TRUE(
      poly_dtype_can_lossless_cast(POLY_BOOL, poly_dtype_ptr(POLY_INT32, 4, POLY_ADDR_GLOBAL))
  );
  ASSERT_TRUE(poly_dtype_can_lossless_cast(POLY_UINT16, POLY_INT32));
  ASSERT_TRUE(poly_dtype_can_lossless_cast(POLY_INT8, POLY_FLOAT16));
  ASSERT_FALSE(poly_dtype_can_lossless_cast(POLY_INT32, POLY_FLOAT32));
  ASSERT_FALSE(poly_dtype_can_lossless_cast(POLY_INT32, POLY_INT16));
  ASSERT_FALSE(
      poly_dtype_can_lossless_cast(poly_dtype_vec(POLY_INT32, 2), poly_dtype_vec(POLY_INDEX, 2))
  );
  PASS();
}

TEST(dtype, ptr) {
  PolyDType p = poly_dtype_ptr(POLY_FLOAT32, 1024, POLY_ADDR_GLOBAL);
  ASSERT_TRUE(p.is_ptr);
  ASSERT_INT_EQ(p.ptr_size, 1024);
  ASSERT_INT_EQ(p.addrspace, POLY_ADDR_GLOBAL);
  PASS();
}

TEST(dtype, name) {
  ASSERT_STR_EQ(poly_dtype_name(POLY_FLOAT32), "float");
  ASSERT_STR_EQ(poly_dtype_name(POLY_INT32), "int");
  ASSERT_STR_EQ(poly_dtype_name(POLY_BOOL), "bool");
  PASS();
}
