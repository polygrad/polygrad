/*
 * test_main.c — Test runner for polygrad
 *
 * Usage: polygrad_test [--fast] [--common] [--specific SUITE] [suite_filter|suite.test]
 *   --fast            Skip slow suites (nn)
 *   --common          Run the large backend-portable suite. POLY_DEVICE selects
 *                     the execution backend.
 *   --specific SUITE  Run only TEST_BACKEND entries in the exact suite.
 *   --require-no-skips Fail if a selected test skips at runtime.
 *   filter             Match a suite/test substring, or exact suite.test.
 */

#include "test_harness.h"
#include "../src/frontend.h"

/* Global test registry — defined here, declared extern in test_harness.h */
TestEntry g_tests[MAX_TESTS];
int g_n_tests = 0;
int g_current_test_skipped = 0;

static void cleanup_caches(void) {
  poly_cpu_cache_flush();
}

static const char *slow_suites[] = {"nn", NULL};

static int is_slow(const char *suite) {
  for (int i = 0; slow_suites[i]; i++)
    if (strcmp(suite, slow_suites[i]) == 0) return 1;
  return 0;
}

int main(int argc, char **argv) {
  /* Keep crash diagnostics useful when stdout is redirected by Make/CI. */
  setvbuf(stdout, NULL, _IONBF, 0);
  atexit(cleanup_caches);
  int fast = 0;
  int common = 0;
  int require_no_skips = 0;
  const char *specific_suite = NULL;
  const char *filter = NULL;
  const char *filter_name = NULL;
  char filter_suite[128] = {0};

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--fast") == 0)
      fast = 1;
    else if (strcmp(argv[i], "--common") == 0)
      common = 1;
    else if (strcmp(argv[i], "--specific") == 0 && i + 1 < argc)
      specific_suite = argv[++i];
    else if (strcmp(argv[i], "--require-no-skips") == 0)
      require_no_skips = 1;
    else
      filter = argv[i];
  }
  if (filter) {
    const char *dot = strchr(filter, '.');
    if (dot && dot != filter && dot[1] != '\0') {
      size_t n = (size_t)(dot - filter);
      if (n >= sizeof(filter_suite)) n = sizeof(filter_suite) - 1;
      memcpy(filter_suite, filter, n);
      filter_suite[n] = '\0';
      filter_name = dot + 1;
    }
  }

  printf(
      "\n  polygrad test suite%s%s\n", fast ? " (fast)" : "",
      common           ? " (common backend)"
      : specific_suite ? " (backend-specific)"
                       : ""
  );
  printf("  ================\n");

  if (!fast && !common && !specific_suite && !filter && !require_no_skips)
    return poly_test_run_all();

  /* Filtered run */
  int total_passed = 0, total_failed = 0, skipped = 0, runtime_skipped = 0, selected = 0;
  const char *current_suite = "";

  for (int i = 0; i < g_n_tests; i++) {
    if (fast && is_slow(g_tests[i].suite)) {
      skipped++;
      continue;
    }
    if (common && !(g_tests[i].flags & POLY_TEST_COMMON)) {
      skipped++;
      continue;
    }
    if (specific_suite &&
        ((g_tests[i].flags & POLY_TEST_COMMON) || strcmp(g_tests[i].suite, specific_suite) != 0)) {
      skipped++;
      continue;
    }
    if (filter) {
      if (filter_name) {
        if (strcmp(g_tests[i].suite, filter_suite) != 0 ||
            strcmp(g_tests[i].name, filter_name) != 0) {
          skipped++;
          continue;
        }
      } else if (!strstr(g_tests[i].suite, filter) && !strstr(g_tests[i].name, filter)) {
        skipped++;
        continue;
      }
    }

    if (strcmp(current_suite, g_tests[i].suite) != 0) {
      current_suite = g_tests[i].suite;
      printf("\n  %s:\n", current_suite);
    }

    selected++;
    int passed = 0, failed = 0;
    /* Match the full runner: print the active filtered test before entering
     * generated/JIT code so sanitizer crashes name the failing test. */
    printf("    [RUN ] %s\n", g_tests[i].name);
    fflush(stdout);
    g_current_test_skipped = 0;
    g_tests[i].fn(&passed, &failed);

    if (g_current_test_skipped) {
      printf("    [SKIP] %s\n", g_tests[i].name);
      skipped++;
      runtime_skipped++;
    } else if (failed == 0) {
      printf("    [PASS] %s\n", g_tests[i].name);
      total_passed++;
    } else {
      printf("    [FAIL] %s\n", g_tests[i].name);
      total_failed++;
    }
  }

  printf(
      "\n  Results: %d passed, %d failed, %d skipped, %d total\n\n", total_passed, total_failed,
      skipped, total_passed + total_failed + skipped
  );
  if (filter && selected == 0) {
    printf("  ERROR: no tests matched filter '%s'\n\n", filter);
    return 1;
  }
  if (common && selected == 0) {
    printf("  ERROR: no tests are marked common backend\n\n");
    return 1;
  }
  if (specific_suite && selected == 0) {
    printf("  ERROR: no backend-specific tests matched suite '%s'\n\n", specific_suite);
    return 1;
  }
  if (require_no_skips && runtime_skipped > 0) {
    printf("  ERROR: required test route skipped %d selected test(s)\n\n", runtime_skipped);
    return 2;
  }
  return total_failed > 0 ? 1 : 0;
}
