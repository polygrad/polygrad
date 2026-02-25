/*
 * test_main.c — Test runner for polygrad
 *
 * Usage: polygrad_test [--fast] [suite_filter]
 *   --fast   Skip slow suites (nn)
 *   filter   Only run suites/tests matching substring
 */

#include "test_harness.h"
#include "../src/frontend.h"

/* Global test registry — defined here, declared extern in test_harness.h */
TestEntry g_tests[MAX_TESTS];
int g_n_tests = 0;

static void cleanup_caches(void) {
  poly_cpu_cache_flush();
#ifdef POLY_HAS_CUDA
  poly_cuda_prog_cache_flush();
  poly_cuda_flush_buffers();
#endif
}

static const char *slow_suites[] = { "nn", NULL };

static int is_slow(const char *suite) {
  for (int i = 0; slow_suites[i]; i++)
    if (strcmp(suite, slow_suites[i]) == 0) return 1;
  return 0;
}

int main(int argc, char **argv) {
  atexit(cleanup_caches);
  int fast = 0;
  const char *filter = NULL;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--fast") == 0) fast = 1;
    else filter = argv[i];
  }

  printf("\n  polygrad test suite%s\n", fast ? " (fast)" : "");
  printf("  ================\n");

  if (!fast && !filter)
    return poly_test_run_all();

  /* Filtered run */
  int total_passed = 0, total_failed = 0, skipped = 0;
  const char *current_suite = "";

  for (int i = 0; i < g_n_tests; i++) {
    if (fast && is_slow(g_tests[i].suite)) { skipped++; continue; }
    if (filter && !strstr(g_tests[i].suite, filter) &&
        !strstr(g_tests[i].name, filter)) { skipped++; continue; }

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

  printf("\n  Results: %d passed, %d failed, %d skipped, %d total\n\n",
         total_passed, total_failed, skipped, total_passed + total_failed + skipped);
  return total_failed > 0 ? 1 : 0;
}
