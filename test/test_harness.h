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
