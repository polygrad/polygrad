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
  ASSERT_TRUE(poly_dtype_is_int(POLY_WEAKINT));
  ASSERT_TRUE(poly_dtype_is_index(POLY_WEAKINT));
  ASSERT_FALSE(poly_dtype_is_int(POLY_FLOAT32));

  ASSERT_TRUE(poly_dtype_is_unsigned(POLY_UINT8));
  ASSERT_TRUE(poly_dtype_is_unsigned(POLY_UINT32));
  ASSERT_FALSE(poly_dtype_is_unsigned(POLY_INT32));

  ASSERT_TRUE(poly_dtype_is_bool(POLY_BOOL));
  ASSERT_FALSE(poly_dtype_is_bool(POLY_INT32));
  PASS();
}

TEST(dtype, weakint_identity_uses_tinygrad_scalar_semantics) {
  PolyDType weak_like = POLY_WEAKINT;
  weak_like.bitsize = 144;

  ASSERT_FALSE(poly_dtype_eq(weak_like, POLY_WEAKINT));
  ASSERT_TRUE(poly_dtype_is_index(weak_like));
  ASSERT_TRUE(poly_dtype_is_int(weak_like));
  PASS();
}

TEST(dtype, ffi_roundtrips_internal_weakint_without_renumbering_public_types) {
  /* Current weakint and weakfloat are both public UOp scalar kinds. Keep the
   * existing FFI ids stable and append them after float64. */
  ASSERT_INT_EQ(poly_dtype_id_by_name("float64"), 13);
  ASSERT_INT_EQ(poly_dtype_id_by_name("weakint"), 14);
  ASSERT_INT_EQ(poly_dtype_id_by_name("weakfloat"), 15);
  ASSERT_INT_EQ(poly_dtype_count(), 20);
  PolyDType dt = POLY_VOID;
  ASSERT_TRUE(poly_dtype_by_id(14, &dt));
  ASSERT_TRUE(poly_dtype_eq(dt, POLY_WEAKINT));
  ASSERT_TRUE(poly_dtype_by_id(15, &dt));
  ASSERT_TRUE(poly_dtype_eq(dt, POLY_WEAKFLOAT));
  PASS();
}

TEST(dtype, fp8_identities_append_without_renumbering_existing_types) {
  ASSERT_INT_EQ(poly_dtype_id_by_name("fp8e4m3"), 16);
  ASSERT_INT_EQ(poly_dtype_id_by_name("float8_e4m3"), 16);
  ASSERT_INT_EQ(poly_dtype_id_by_name("fp8e5m2"), 17);
  ASSERT_INT_EQ(poly_dtype_id_by_name("float8_e5m2"), 17);
  ASSERT_INT_EQ(poly_dtype_id_by_name("fp8e4m3fnuz"), 18);
  ASSERT_INT_EQ(poly_dtype_id_by_name("float8_e4m3fnuz"), 18);
  ASSERT_INT_EQ(poly_dtype_id_by_name("fp8e5m2fnuz"), 19);
  ASSERT_INT_EQ(poly_dtype_id_by_name("float8_e5m2fnuz"), 19);
  ASSERT_INT_EQ(poly_dtype_count(), 20);
  PASS();
}

TEST(dtype, fp8_storage_conversion_matches_current_tinygrad) {
  const double values[] = {-INFINITY, -1.5, -0.0, 0.0, 0.1, 1.0, 1.5, 448.0,
                           INFINITY, NAN};
  const PolyDType dtypes[] = {
      POLY_FP8E4M3, POLY_FP8E5M2, POLY_FP8E4M3FNUZ, POLY_FP8E5M2FNUZ,
  };
  const uint8_t expected[][10] = {
      {255, 188, 128, 0, 29, 56, 60, 126, 127, 127},
      {252, 190, 128, 0, 46, 60, 62, 95, 124, 127},
      {128, 196, 0, 0, 37, 64, 68, 127, 128, 128},
      {128, 194, 0, 0, 50, 64, 66, 99, 128, 128},
  };
  for (int d = 0; d < 4; d++) {
    ASSERT_TRUE(poly_dtype_is_fp8(dtypes[d]));
    ASSERT_INT_EQ(poly_dtype_is_fp8_fnuz(dtypes[d]), d >= 2);
    for (int i = 0; i < 10; i++)
      ASSERT_INT_EQ(poly_float_to_fp8(values[i], dtypes[d]), expected[d][i]);
  }

  ASSERT_TRUE(isnan(poly_fp8_to_float(0xff, POLY_FP8E4M3)));
  ASSERT_FLOAT_EQ(poly_fp8_to_float(0xbc, POLY_FP8E4M3), -1.5, 0.0);
  ASSERT_FLOAT_EQ(poly_fp8_to_float(0x1d, POLY_FP8E4M3), 0.1015625, 0.0);
  ASSERT_TRUE(isinf(poly_fp8_to_float(0x7c, POLY_FP8E5M2)));
  ASSERT_TRUE(isnan(poly_fp8_to_float(0x80, POLY_FP8E4M3FNUZ)));
  ASSERT_FLOAT_EQ(poly_fp8_to_float(0x63, POLY_FP8E5M2FNUZ), 448.0, 0.0);
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
  ASSERT_TRUE(poly_dtype_can_lossless_cast(POLY_INT32, POLY_WEAKINT));
  ASSERT_TRUE(poly_dtype_can_lossless_cast(POLY_INT64, POLY_WEAKINT));
  ASSERT_FALSE(poly_dtype_can_lossless_cast(POLY_FLOAT32, POLY_WEAKINT));
  ASSERT_TRUE(poly_dtype_can_lossless_cast(POLY_BOOL, POLY_INT32));
  ASSERT_TRUE(poly_dtype_can_lossless_cast(POLY_UINT16, POLY_INT32));
  ASSERT_TRUE(poly_dtype_can_lossless_cast(POLY_INT8, POLY_FLOAT16));
  ASSERT_FALSE(poly_dtype_can_lossless_cast(POLY_INT32, POLY_FLOAT32));
  ASSERT_FALSE(poly_dtype_can_lossless_cast(POLY_INT32, POLY_INT16));
  PASS();
}

TEST(dtype, name) {
  ASSERT_STR_EQ(poly_dtype_name(POLY_FLOAT32), "float");
  ASSERT_STR_EQ(poly_dtype_name(POLY_INT32), "int");
  ASSERT_STR_EQ(poly_dtype_name(POLY_BOOL), "bool");
  PASS();
}
