#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/utils.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const char *key;
  char *value;
  bool had_value;
} EnvSave;

static EnvSave save_env(const char *key) {
  const char *cur = getenv(key);
  return (EnvSave){
      .key = key,
      .value = cur ? strdup(cur) : NULL,
      .had_value = cur != NULL,
  };
}

static void restore_env(EnvSave *s) {
  if (!s) return;
  if (s->had_value)
    setenv(s->key, s->value ? s->value : "", 1);
  else
    unsetenv(s->key);
  free(s->value);
  s->value = NULL;
}

TEST(utils, debug_level_precedence) {
  EnvSave debug = save_env("DEBUG");
  EnvSave poly_debug = save_env("POLY_DEBUG");

  unsetenv("DEBUG");
  unsetenv("POLY_DEBUG");
  ASSERT_INT_EQ(poly_debug_level(), 0);

  setenv("DEBUG", "4", 1);
  ASSERT_INT_EQ(poly_debug_level(), 4);

  setenv("POLY_DEBUG", "6", 1);
  ASSERT_INT_EQ(poly_debug_level(), 6);

  restore_env(&poly_debug);
  restore_env(&debug);
  PASS();
}

TEST(utils, poly_free_releases_public_api_allocations) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.0));
  char *s = poly_uop_str(c);
  ASSERT_TRUE(s != NULL);
  ASSERT_TRUE(strstr(s, "CONST") != NULL);
  poly_free(s);
  poly_free(NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(utils, getenv_flag_default_preserves_absent_default) {
  EnvSave flag = save_env("POLY_TEST_FLAG_DEFAULT");

  unsetenv("POLY_TEST_FLAG_DEFAULT");
  ASSERT_TRUE(poly_getenv_flag_default("POLY_TEST_FLAG_DEFAULT", true));
  ASSERT_FALSE(poly_getenv_flag_default("POLY_TEST_FLAG_DEFAULT", false));

  setenv("POLY_TEST_FLAG_DEFAULT", "0", 1);
  ASSERT_FALSE(poly_getenv_flag_default("POLY_TEST_FLAG_DEFAULT", true));

  setenv("POLY_TEST_FLAG_DEFAULT", "1", 1);
  ASSERT_TRUE(poly_getenv_flag_default("POLY_TEST_FLAG_DEFAULT", false));

  setenv("POLY_TEST_FLAG_DEFAULT", "no", 1);
  ASSERT_FALSE(poly_getenv_flag_default("POLY_TEST_FLAG_DEFAULT", true));

  restore_env(&flag);
  PASS();
}

TEST(utils, poly_selftest_runs_portable_interp_path) {
  ASSERT_INT_EQ(poly_selftest(), 0);
  ASSERT_INT_EQ(poly_selftest_device(POLY_DEVICE_AUTO), 0);
  ASSERT_INT_EQ(poly_selftest_device(POLY_DEVICE_INTERP), 0);
  PASS();
}

TEST(utils, poly_selftest_rejects_non_execution_device) {
  ASSERT_INT_EQ(poly_selftest_device(POLY_DEVICE_HOST), -1);
  PASS();
}

TEST(utils, dump_thresholds_follow_tinygrad) {
  EnvSave debug = save_env("DEBUG");
  EnvSave poly_debug = save_env("POLY_DEBUG");
  EnvSave dump = save_env("POLY_DUMP_KERNELS");

  unsetenv("POLY_DEBUG");
  unsetenv("POLY_DUMP_KERNELS");

  setenv("DEBUG", "4", 1);
  ASSERT_TRUE(poly_dump_kernels_enabled());
  ASSERT_FALSE(poly_dump_graph_enabled());
  ASSERT_FALSE(poly_dump_linear_enabled());

  setenv("DEBUG", "5", 1);
  ASSERT_TRUE(poly_dump_kernels_enabled());
  ASSERT_TRUE(poly_dump_graph_enabled());
  ASSERT_FALSE(poly_dump_linear_enabled());

  setenv("DEBUG", "6", 1);
  ASSERT_TRUE(poly_dump_kernels_enabled());
  ASSERT_TRUE(poly_dump_graph_enabled());
  ASSERT_TRUE(poly_dump_linear_enabled());

  restore_env(&dump);
  restore_env(&poly_debug);
  restore_env(&debug);
  PASS();
}

TEST(utils, dump_kernels_override_enables_all) {
  EnvSave debug = save_env("DEBUG");
  EnvSave poly_debug = save_env("POLY_DEBUG");
  EnvSave dump = save_env("POLY_DUMP_KERNELS");

  unsetenv("DEBUG");
  unsetenv("POLY_DEBUG");
  setenv("POLY_DUMP_KERNELS", "1", 1);

  ASSERT_TRUE(poly_dump_kernels_enabled());
  ASSERT_TRUE(poly_dump_graph_enabled());
  ASSERT_TRUE(poly_dump_linear_enabled());

  restore_env(&dump);
  restore_env(&poly_debug);
  restore_env(&debug);
  PASS();
}
