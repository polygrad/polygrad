/*
 * test_harness.h — Minimal C test framework
 *
 * Usage:
 *   TEST(suite, name) { ... ASSERT_*(...); ... }
 *
 * In main: poly_test_run_all();
 */

#ifndef POLY_TEST_HARNESS_H
#define POLY_TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

typedef void (*TestFn)(int *passed, int *failed);

typedef struct {
  const char *suite;
  const char *name;
  TestFn fn;
} TestEntry;

#define MAX_TESTS 512
extern TestEntry g_tests[MAX_TESTS];
extern int g_n_tests;

#define TEST(suite, name) \
  static void test_##suite##_##name(int *_passed, int *_failed); \
  __attribute__((constructor)) \
  static void register_##suite##_##name(void) { \
    g_tests[g_n_tests++] = (TestEntry){ #suite, #name, test_##suite##_##name }; \
  } \
  static void test_##suite##_##name(int *_passed, int *_failed)

#define PASS() do { (*_passed)++; return; } while(0)

#define FAIL(fmt, ...) do { \
  fprintf(stderr, "    FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
  (*_failed)++; return; \
} while(0)

#define ASSERT_TRUE(expr) do { \
  if (!(expr)) FAIL("expected true: %s", #expr); \
} while(0)

#define ASSERT_FALSE(expr) do { \
  if (expr) FAIL("expected false: %s", #expr); \
} while(0)

#define ASSERT_EQ(a, b) do { \
  if ((a) != (b)) FAIL("%s != %s", #a, #b); \
} while(0)

#define ASSERT_NEQ(a, b) do { \
  if ((a) == (b)) FAIL("%s == %s (expected different)", #a, #b); \
} while(0)

#define ASSERT_INT_EQ(a, b) do { \
  long _a = (long)(a), _b = (long)(b); \
  if (_a != _b) FAIL("%s = %ld, expected %ld", #a, _a, _b); \
} while(0)

#define ASSERT_STR_EQ(a, b) do { \
  const char *_a = (a), *_b = (b); \
  if (strcmp(_a, _b) != 0) FAIL("%s = \"%s\", expected \"%s\"", #a, _a, _b); \
} while(0)

#define ASSERT_FLOAT_EQ(a, b, tol) do { \
  double _a = (a), _b = (b); \
  if (fabs(_a - _b) > (tol)) FAIL("%s = %.8f, expected %.8f (tol=%.1e)", #a, _a, _b, (tol)); \
} while(0)

#define ASSERT_FLOAT_NAN(a) do { \
  double _a = (double)(a); \
  if (!isnan(_a)) FAIL("%s = %.8g, expected NaN", #a, _a); \
} while(0)

#define ASSERT_FLOAT_INF(a, sign) do { \
  double _a = (double)(a); \
  if (!isinf(_a) || ((sign) > 0 && _a < 0) || ((sign) < 0 && _a > 0)) \
    FAIL("%s = %.8g, expected %sinf", #a, _a, (sign) < 0 ? "-" : "+"); \
} while(0)

/* Real ULP distance via bitwise float ordering.  Handles subnormals and
 * near-zero values correctly (unlike relative-error hacks). */
static inline int32_t poly_float_ulp_index(float f) {
  int32_t i;
  memcpy(&i, &f, sizeof(i));
  if (i < 0) i = (int32_t)(0x80000000u - (uint32_t)i);
  return i;
}

#define ASSERT_FLOAT_ULP(a, b, max_ulps) do { \
  float _fa = (float)(a), _fb = (float)(b); \
  if (isnan(_fa) && isnan(_fb)) { /* ok */ } \
  else if (isnan(_fa) || isnan(_fb)) \
    FAIL("%s=%.8g, expected %.8g (NaN mismatch)", #a, (double)_fa, (double)_fb); \
  else if (isinf(_fa) || isinf(_fb)) { \
    if (!(isinf(_fa) && isinf(_fb) && ((_fa > 0) == (_fb > 0)))) \
      FAIL("%s=%.8g, expected %.8g (inf mismatch)", #a, (double)_fa, (double)_fb); \
  } else { \
    int32_t _ia = poly_float_ulp_index(_fa), _ib = poly_float_ulp_index(_fb); \
    int64_t _d = llabs((int64_t)_ia - (int64_t)_ib); \
    if (_d > (max_ulps)) \
      FAIL("%s=%.8g, expected %.8g (%lld ulps, max %d)", \
        #a, (double)_fa, (double)_fb, (long long)_d, (int)(max_ulps)); \
  } \
} while(0)

/* NaN-aware absolute tolerance (use for switchover regions where ULP is noisy). */
#define ASSERT_FLOAT_ABS(a, b, tol) do { \
  float _fa = (float)(a), _fb = (float)(b); \
  if (isnan(_fa) && isnan(_fb)) { /* ok */ } \
  else if (isnan(_fa) || isnan(_fb)) \
    FAIL("%s=%.8g, expected %.8g (NaN mismatch)", #a, (double)_fa, (double)_fb); \
  else if (fabsf(_fa - _fb) > (float)(tol)) \
    FAIL("%s=%.8g, expected %.8g (abs err %.8g, tol %.8g)", \
      #a, (double)_fa, (double)_fb, (double)fabsf(_fa - _fb), (double)(tol)); \
} while(0)

/* Combined ULP + absolute tolerance: passes if EITHER metric is within bounds.
 * Use for sweeps where near-zero values need abs tolerance but normal range
 * needs ULP precision. */
#define ASSERT_FLOAT_NEAR(a, b, max_ulps, abs_tol) do { \
  float _fa = (float)(a), _fb = (float)(b); \
  if (isnan(_fa) && isnan(_fb)) { /* ok */ } \
  else if (isnan(_fa) || isnan(_fb)) \
    FAIL("%s=%.8g, expected %.8g (NaN mismatch)", #a, (double)_fa, (double)_fb); \
  else if (isinf(_fa) && isinf(_fb) && ((_fa > 0) == (_fb > 0))) { /* ok */ } \
  else if (isinf(_fa) || isinf(_fb)) \
    FAIL("%s=%.8g, expected %.8g (inf mismatch)", #a, (double)_fa, (double)_fb); \
  else if (fabsf(_fa - _fb) <= (float)(abs_tol)) { /* within abs tol */ } \
  else { \
    int32_t _ia = poly_float_ulp_index(_fa), _ib = poly_float_ulp_index(_fb); \
    int64_t _d = llabs((int64_t)_ia - (int64_t)_ib); \
    if (_d > (max_ulps)) \
      FAIL("%s=%.8g, expected %.8g (%lld ulps, max %d; abs %.8g, tol %.8g)", \
        #a, (double)_fa, (double)_fb, (long long)_d, (int)(max_ulps), \
        (double)fabsf(_fa - _fb), (double)(abs_tol)); \
  } \
} while(0)

#define ASSERT_PTR_EQ(a, b) do { \
  const void *_a = (a), *_b = (b); \
  if (_a != _b) FAIL("%s = %p, expected %p (same pointer)", #a, _a, _b); \
} while(0)

#define ASSERT_PTR_NEQ(a, b) do { \
  const void *_a = (a), *_b = (b); \
  if (_a == _b) FAIL("%s == %s (expected different pointers)", #a, #b); \
} while(0)

#define ASSERT_NOT_NULL(a) do { \
  if ((a) == NULL) FAIL("%s is NULL", #a); \
} while(0)

static inline int poly_test_run_all(void) {
  int total_passed = 0, total_failed = 0;
  const char *current_suite = "";

  for (int i = 0; i < g_n_tests; i++) {
    if (strcmp(current_suite, g_tests[i].suite) != 0) {
      current_suite = g_tests[i].suite;
      printf("\n  %s:\n", current_suite);
    }

    int passed = 0, failed = 0;
    g_tests[i].fn(&passed, &failed);

    if (failed == 0) {
      printf("    [PASS] %s\n", g_tests[i].name);
      total_passed++;
    } else {
      printf("    [FAIL] %s\n", g_tests[i].name);
      total_failed++;
    }
  }

  printf("\n  Results: %d passed, %d failed, %d total\n\n",
         total_passed, total_failed, total_passed + total_failed);
  return total_failed > 0 ? 1 : 0;
}

#endif /* POLY_TEST_HARNESS_H */
