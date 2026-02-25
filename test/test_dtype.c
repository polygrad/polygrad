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
  ASSERT_FALSE(poly_dtype_is_int(POLY_FLOAT32));

  ASSERT_TRUE(poly_dtype_is_unsigned(POLY_UINT8));
  ASSERT_TRUE(poly_dtype_is_unsigned(POLY_UINT32));
  ASSERT_FALSE(poly_dtype_is_unsigned(POLY_INT32));

  ASSERT_TRUE(poly_dtype_is_bool(POLY_BOOL));
  ASSERT_FALSE(poly_dtype_is_bool(POLY_INT32));
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

  /* scalar of scalar is identity */
  PolyDType s2 = poly_dtype_scalar(POLY_INT64);
  ASSERT_TRUE(poly_dtype_eq(s2, POLY_INT64));
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
