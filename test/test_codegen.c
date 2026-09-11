/*
 * test_codegen.c — Tests for linearizer, C renderer, and CPU runtime
 */

#include "test_harness.h"
#include "../src/codegen/codegen.h"
#include "../src/codegen/decomp/dtype.h"
#include "../src/codegen/late/coalesce.h"
#include "../src/codegen/late/gater.h"
#include "../src/codegen/simplify.h"
#include "../src/renderer/cstyle.h"
#include "../src/bigint.h"
#include "../src/frontend.h"
#include "../src/frontend_internal.h"
#include "../src/engine/schedule.h"
#include "../src/interp.h"
#include "../src/tensor.h" /* poly_sum_reduce */
#include "../src/uop/spec.h"
#include "../src/uop/symbolic.h"
#include "../src/uop/weak.h"
#include "../src/utils.h"

#include <inttypes.h>
#include <unistd.h>
#include <sys/stat.h>

#ifndef __EMSCRIPTEN__
static PolyUOp *beam_action_sink(PolyCtx *ctx, PolyAxisType type, int n) {
  PolyUOp *r = poly_range(ctx, n, 0, type);
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, n, 0);
  PolyUOp *value = poly_cast(ctx, r, POLY_FLOAT32);
  PolyUOp *index = r;
  if (type == POLY_AXIS_REDUCE) {
    PolyUOp *src[] = {value, r};
    value = poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
    index = poly_const_int(ctx, 0);
  }
  PolyUOp *ptr = poly_uop_index(ctx, out, &index, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, ptr, value, poly_arg_none());
  if (type != POLY_AXIS_REDUCE)
    store = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, r, poly_arg_none());
  return poly_test_kernel_sink(ctx, &store, 1, "test");
}

TEST(codegen, beam_compile_timeout_reaps_compiler) {
  char path[] = "temp/beam-timeout-compiler.XXXXXX";
  int fd = mkstemp(path);
  ASSERT_TRUE(fd >= 0);
  FILE *script = fdopen(fd, "w");
  ASSERT_NOT_NULL(script);
  fputs("#!/bin/sh\nsleep 2\nexit 1\n", script);
  fclose(script);
  chmod(path, 0700);
  const char *cc = getenv("CC"), *timeout = getenv("BEAM_TIMEOUT_SEC"),
             *strict = getenv("BEAM_STRICT_MODE");
  char *old_cc = cc ? strdup(cc) : NULL, *old_timeout = timeout ? strdup(timeout) : NULL;
  char *old_strict = strict ? strdup(strict) : NULL;
  setenv("CC", path, 1);
  setenv("BEAM_TIMEOUT_SEC", "1", 1);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_GLOBAL, 8);
  int n_args = 0;
  double start = poly_now_ms();
  double result = poly_test_beam_compile_and_time(
      ctx, sink, (PolyRewriteOpts){.caps = poly_c_renderer_caps(), .device = POLY_DEVICE_CPU}, 1,
      &n_args
  );
  double elapsed = poly_now_ms() - start;
  setenv("BEAM_STRICT_MODE", "1", 1);
  start = poly_now_ms();
  PolyUOp *strict_result = poly_full_rewrite_to_sink_ex(
      ctx, sink,
      (PolyRewriteOpts
      ){.optimize = true,
        .beam_width = 1,
        .caps = poly_c_renderer_caps(),
        .device = POLY_DEVICE_CPU}
  );
  double strict_elapsed = poly_now_ms() - start;
  bool restored = poly_compile_deadline_ms == 0;
  poly_ctx_destroy(ctx);
  if (old_cc)
    setenv("CC", old_cc, 1);
  else
    unsetenv("CC");
  if (old_timeout)
    setenv("BEAM_TIMEOUT_SEC", old_timeout, 1);
  else
    unsetenv("BEAM_TIMEOUT_SEC");
  free(old_cc);
  free(old_timeout);
  if (old_strict)
    setenv("BEAM_STRICT_MODE", old_strict, 1);
  else
    unsetenv("BEAM_STRICT_MODE");
  free(old_strict);
  remove(path);
  ASSERT_TRUE(!isfinite(result));
  ASSERT_TRUE(elapsed < 1800);
  ASSERT_TRUE(!strict_result && strict_elapsed < 1800 && restored);
  PASS();
}

TEST(codegen, beam_compile_deadline_cancels_rewrites_without_poisoning_context) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *value = poly_const_int(ctx, 7);
  poly_compile_deadline_ms = poly_now_ms() - 1;
  bool rejected = !poly_graph_rewrite(ctx, value, poly_symbolic()) &&
                  !poly_graph_walk_rewrite(ctx, value, poly_symbolic(), NULL, NULL, true);
  poly_compile_deadline_ms = 0;
  bool recovered = poly_graph_rewrite(ctx, value, poly_symbolic()) == value &&
                   poly_graph_walk_rewrite(ctx, value, poly_symbolic(), NULL, NULL, true) == value;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rejected && recovered);
  PASS();
}

TEST(codegen, beam_strict_compile_failure_differs_from_budget_rejection) {
  const char *names[] = {"CC", "BEAM_STRICT_MODE", "BEAM_UOPS_MAX", "CACHELEVEL"};
  char *old[4];
  for (int i = 0; i < 4; i++)
    old[i] = getenv(names[i]) ? strdup(getenv(names[i])) : NULL;
  setenv("CC", "false", 1);
  setenv("CACHELEVEL", "0", 1);
  setenv("BEAM_UOPS_MAX", "3000", 1);
  setenv("BEAM_STRICT_MODE", "0", 1);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_WEAK, 8);
  PolyRewriteOpts opts = {
      .optimize = true, .beam_width = 1, .device = POLY_DEVICE_CPU, .caps = poly_c_renderer_caps()};
  bool ordinary = poly_full_rewrite_to_sink_ex(ctx, sink, opts) != NULL;
  setenv("BEAM_STRICT_MODE", "1", 1);
  bool strict = poly_full_rewrite_to_sink_ex(ctx, sink, opts) == NULL;
  setenv("BEAM_UOPS_MAX", "1", 1);
  bool strict_before_budget = poly_full_rewrite_to_sink_ex(ctx, sink, opts) == NULL;
  if (old[0])
    setenv("CC", old[0], 1);
  else
    unsetenv("CC");
  bool budget = poly_full_rewrite_to_sink_ex(ctx, sink, opts) != NULL;
  poly_ctx_destroy(ctx);
  for (int i = 0; i < 4; i++) {
    if (old[i])
      setenv(names[i], old[i], 1);
    else
      unsetenv(names[i]);
    free(old[i]);
  }
  ASSERT_TRUE(ordinary && strict && strict_before_budget && budget);
  PASS();
}

TEST(codegen, beam_actions_match_pinned_catalogue) {
  PolyOpt actions[256];
  int count = poly_test_beam_actions(actions, 256);
  int expected =
      193 + (poly_getenv_flag("BEAM_PADTO") ? 7 : 0) + (poly_getenv_flag("NOLOCALS") ? 1 : 0);
  ASSERT_INT_EQ(count, expected);
  ASSERT_INT_EQ(actions[0].op, POLY_OPT_UPCAST);
  ASSERT_INT_EQ(actions[0].arg, 0);
  ASSERT_INT_EQ(actions[8].arg, 2);
  ASSERT_INT_EQ(actions[48].op, POLY_OPT_UNROLL);
  ASSERT_INT_EQ(actions[63].op, POLY_OPT_LOCAL);
  PASS();
}

static bool scheduler_large_opt(int n_ranges, int n_buffers) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ranges[65], *ends[65];
  for (int i = 0; i < n_ranges; i++)
    ranges[i] = poly_range(ctx, 8, i, POLY_AXIS_GLOBAL);
  int n_ends = n_ranges > n_buffers ? n_ranges : n_buffers;
  for (int i = 0; i < n_ends; i++) {
    PolyUOp *r = ranges[i % n_ranges];
    PolyUOp *buf = poly_test_program_param(ctx, POLY_FLOAT32, 8, i % n_buffers);
    PolyUOp *ptr = poly_uop_index(ctx, buf, &r, 1);
    PolyUOp *store = poly_uop2(
        ctx, POLY_OP_STORE, POLY_VOID, ptr, poly_cast(ctx, r, POLY_FLOAT32), poly_arg_none()
    );
    ends[i] = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, r, poly_arg_none());
  }
  PolyUOp *sink = poly_test_kernel_sink(ctx, ends, n_ends, "test");
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST,
      .has_axis = true,
      .axis = n_ranges - 1,
      .arg_kind = POLY_OPT_ARG_INT,
      .arg = 4};
  PolyUOp *out = poly_test_apply_opt(ctx, sink, (PolyRendererCaps){.device = "CPU"}, opt);
  PolyUOp *r = ranges[n_ranges - 1];
  PolyUOp *remaining = poly_uop1(ctx, POLY_OP_RANGE, r->dtype, poly_const_int(ctx, 2), r->arg);
  PolyUOp *up = poly_range(ctx, 4, n_ranges, POLY_AXIS_UPCAST);
  PolyUOp *replacement = poly_uop2(
      ctx, POLY_OP_ADD, r->dtype,
      poly_uop2(ctx, POLY_OP_MUL, r->dtype, remaining, poly_const_int(ctx, 4), poly_arg_none()), up,
      poly_arg_none()
  );
  PolyUOp *expected = poly_uop_substitute(ctx, sink, &r, &replacement, 1);
  bool ok = out && out->n_src == expected->n_src;
  for (int i = 0; ok && i < out->n_src; i++)
    ok &= out->src[i] == expected->src[i];
  poly_ctx_destroy(ctx);
  return ok;
}

TEST(codegen, scheduler_splits_beyond_64_ranges) {
  ASSERT_TRUE(scheduler_large_opt(65, 1));
  PASS();
}

TEST(codegen, scheduler_splits_with_33_buffers) {
  ASSERT_TRUE(scheduler_large_opt(1, 33));
  PASS();
}

TEST(codegen, scheduler_snapshot_allocation_failure_preserves_parent) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_GLOBAL, 8);
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4};
  bool ok = poly_test_scheduler_copy_rollback(ctx, sink, opt);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(codegen, scheduler_reachability_crosses_word_boundary) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ranges[65];
  for (int i = 0; i < 65; i++)
    ranges[i] = poly_range(ctx, 8, i, POLY_AXIS_GLOBAL);
  PolyUOp *coord = poly_alu2(ctx, POLY_OP_ADD, ranges[0], ranges[64]);
  PolyUOp *buffer = poly_test_program_param(ctx, POLY_FLOAT32, 16, 0);
  PolyUOp *index = poly_uop_index(ctx, buffer, &coord, 1);
  PolyUOp *src[66];
  memcpy(src, ranges, sizeof(ranges));
  src[65] = index;
  PolyUOp *sink = poly_test_kernel_sink(ctx, src, 66, "test");
  bool ok = poly_test_scheduler_reaches(ctx, sink, index, ranges[0]) &&
            poly_test_scheduler_reaches(ctx, sink, index, ranges[64]) &&
            !poly_test_scheduler_reaches(ctx, sink, index, ranges[1]);
  /* backward_slice excludes the root itself, even when that root is RANGE. */
  PolyUOp *direct = poly_uop_index(ctx, buffer, &ranges[64], 1);
  src[65] = direct;
  sink = poly_test_kernel_sink(ctx, src, 66, "test");
  ok &= !poly_test_scheduler_reaches(ctx, sink, direct, ranges[64]);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(codegen, scheduler_includes_buffer_backed_indexes) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 8, 0, POLY_AXIS_GLOBAL);
  PolyUOp *coord = poly_alu2(ctx, POLY_OP_MUL, r, poly_const_int(ctx, 2));
  PolyUOp *buffer = poly_buffer_f32(ctx, 16);
  PolyUOp *index = poly_uop_index(ctx, buffer, &coord, 1);
  PolyUOp *sink = poly_test_kernel_sink(ctx, &index, 1, "test");
  bool ok = poly_test_scheduler_reaches(ctx, sink, index, r);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(codegen, scheduler_heuristic_records_applied_options) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_GLOBAL, 64);
  PolyUOp *out = poly_apply_opts_heuristic_ex(
      ctx, sink, (PolyRendererCaps){.device = "CPU", .max_vec_width = 4}
  );
  const PolyKernelInfo *info =
      out && out->arg.kind == POLY_ARG_KERNEL_INFO ? out->arg.kernel_info : NULL;
  bool ok = info && info->n_applied_opts == 1 && info->applied_opts[0].op == POLY_OPT_UPCAST &&
            info->applied_opts[0].arg == 4;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

static PolyUOp *policy_const(PolyCtx *ctx, int64_t value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(value));
}

static PolyUOp *policy_range(PolyCtx *ctx, int size, int axis, PolyAxisType type) {
  return poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, policy_const(ctx, size), poly_arg_range(axis, type)
  );
}

static bool policy_matches(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyRendererCaps caps,
    const PolyOpt *opts,
    int n
) {
  PolyUOp *expected = sink;
  for (int i = 0; expected && i < n; i++)
    expected = poly_test_apply_opt(ctx, expected, caps, opts[i]);
  return expected && poly_apply_opts_heuristic_ex(ctx, sink, caps) == expected;
}

TEST(codegen, scheduler_policy_masked_upcast_history) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = policy_range(ctx, 3, 0, POLY_AXIS_GLOBAL);
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 3, 0);
  PolyUOp *cond =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r, policy_const(ctx, 2), poly_arg_none());
  PolyUOp *value =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, cond, r, policy_const(ctx, 0), poly_arg_none());
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, out, &r, 1),
      poly_cast(ctx, value, POLY_FLOAT32), poly_arg_none()
  );
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, r, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 0};
  bool ok = policy_matches(ctx, sink, (PolyRendererCaps){.device = "CPU"}, &opt, 1);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(codegen, scheduler_policy_fallback_ignores_vector_cap) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_GLOBAL, 64);
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4};
  bool ok =
      policy_matches(ctx, sink, (PolyRendererCaps){.device = "CPU", .max_vec_width = 8}, &opt, 1);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

static bool policy_thread_case(int lower, bool symbolic, bool legacy_cap) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *g = policy_range(ctx, 1 << 18, 0, POLY_AXIS_WEAK);
  PolyUOp *value = poly_cast(ctx, g, POLY_FLOAT32);
  if (symbolic) {
    PolyUOp *n = poly_uop_variable(ctx, "n", lower, 4, POLY_WEAKINT, 1, false);
    PolyUOp *r =
        poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, n, poly_arg_range(1, POLY_AXIS_REDUCE));
    value = poly_uop2(
        ctx, POLY_OP_REDUCE, POLY_FLOAT32, poly_cast(ctx, r, POLY_FLOAT32), r,
        poly_arg_reduce(POLY_OP_ADD, 0)
    );
  }
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 1 << 18, 0);
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, out, &g, 1), value, poly_arg_none()
  );
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, g, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");
  PolyOpt opts[] = {
      {.op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4},
      {.op = POLY_OPT_THREAD, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 2}};
  PolyRendererCaps caps = {
      .device = "CPU",
      .has_threads = true,
      .global_max = {2, 0, 0},
      .max_threads = legacy_cap ? 2 : 0};
  bool ok = policy_matches(ctx, sink, caps, opts, symbolic && !lower ? 1 : 2);
  poly_ctx_destroy(ctx);
  return ok;
}

TEST(codegen, scheduler_policy_threads_use_global_max) {
  ASSERT_TRUE(policy_thread_case(0, false, false));
  PASS();
}

TEST(codegen, scheduler_policy_threads_prove_symbolic_work) {
  ASSERT_TRUE(policy_thread_case(2, true, true));
  ASSERT_TRUE(policy_thread_case(0, true, true));
  PASS();
}

static bool policy_stride_case(bool big) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = policy_range(ctx, 32, 0, POLY_AXIS_WEAK);
  PolyUOp *y = policy_range(ctx, 32, 1, POLY_AXIS_WEAK);
  PolyUOp *idx = NULL;
  const char *coeffs[] = {"1180591620717411303424", "-1180591620717411303421"};
  for (int i = 0; i < (big ? 2 : 4); i++) {
    PolyUOp *coefficient;
    if (big) {
      PolyInt v = {0};
      if (!poly_int_from_decimal(&v, coeffs[i])) {
        poly_ctx_destroy(ctx);
        return false;
      }
      coefficient = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_int_as_arg(&v));
      poly_int_free(&v);
    } else
      coefficient = policy_const(ctx, i < 2 ? INT64_MAX : -INT64_MAX);
    PolyUOp *term = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, x, coefficient, poly_arg_none());
    idx = idx ? poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, idx, term, poly_arg_none()) : term;
  }
  /* Large intermediates cancel: the actual index is y or 3*x+y (<128). */
  idx = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, idx, y, poly_arg_none());
  PolyUOp *buf = poly_test_program_param(ctx, POLY_FLOAT32, 128, 0);
  PolyUOp *scalar = poly_test_program_param(ctx, POLY_FLOAT32, 1, 1);
  PolyUOp *zero = policy_const(ctx, 0);
  PolyUOp *src[] = {poly_uop_index(ctx, buf, &idx, 1), poly_uop_index(ctx, scalar, &zero, 1)};
  PolyUOp *sink = poly_test_kernel_sink(ctx, src, 2, "test");
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST,
      .has_axis = true,
      .axis = big ? 1 : 0,
      .arg_kind = POLY_OPT_ARG_INT,
      .arg = 4};
  bool ok = policy_matches(ctx, sink, (PolyRendererCaps){.device = "CPU"}, &opt, 1);
  poly_ctx_destroy(ctx);
  return ok;
}

TEST(codegen, scheduler_policy_stride_score_overflow) {
  ASSERT_TRUE(policy_stride_case(false));
  PASS();
}

TEST(codegen, scheduler_policy_stride_score_bigint) {
  ASSERT_TRUE(policy_stride_case(true));
  PASS();
}

TEST(codegen, scheduler_policy_upcast_product_exceeds_host_integer) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *src[17];
  for (int i = 0; i < 16; i++)
    src[i] = policy_range(ctx, 16, i, POLY_AXIS_UPCAST);
  PolyUOp *r = policy_range(ctx, 2, 16, POLY_AXIS_REDUCE);
  src[16] = poly_uop2(
      ctx, POLY_OP_REDUCE, POLY_FLOAT32, poly_cast(ctx, r, POLY_FLOAT32), r,
      poly_arg_reduce(POLY_OP_ADD, 0)
  );
  PolyUOp *sink = poly_test_kernel_sink(ctx, src, 17, "test");
  bool unchanged = policy_matches(ctx, sink, (PolyRendererCaps){.device = "CPU"}, NULL, 0);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(unchanged);
  PASS();
}

TEST(codegen, scheduler_policy_dsp_uses_single_wide_upcast) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = policy_range(ctx, 128, 0, POLY_AXIS_WEAK);
  PolyUOp *y = policy_range(ctx, 32, 1, POLY_AXIS_WEAK);
  PolyUOp *idx =
      poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, x, policy_const(ctx, 32)), y);
  PolyUOp *zero = policy_const(ctx, 0);
  PolyUOp *a = poly_test_program_param(ctx, POLY_FLOAT32, 4096, 0);
  PolyUOp *b = poly_test_program_param(ctx, POLY_FLOAT32, 1, 1);
  PolyUOp *src[] = {poly_uop_index(ctx, a, &idx, 1), poly_uop_index(ctx, b, &zero, 1)};
  PolyUOp *sink = poly_test_kernel_sink(ctx, src, 2, "test");
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 128};
  bool ok = policy_matches(ctx, sink, (PolyRendererCaps){.device = "DSP"}, &opt, 1);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(codegen, scheduler_symbolic_upcast_thresholds) {
  int64_t bounds[][2] = {{64, 128}, {1, 2}, {1, 128}};
  for (int i = 0; i < 3; i++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *size =
        poly_uop_variable(ctx, "up", bounds[i][0], bounds[i][1], POLY_WEAKINT, 1, false);
    PolyUOp *u =
        poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, size, poly_arg_range(0, POLY_AXIS_UPCAST));
    PolyUOp *r = policy_range(ctx, 8, 1, POLY_AXIS_REDUCE);
    PolyUOp *sources[] = {
        u, poly_uop2(
               ctx, POLY_OP_REDUCE, POLY_FLOAT32, poly_cast(ctx, r, POLY_FLOAT32), r,
               poly_arg_reduce(POLY_OP_ADD, 0)
           )};
    PolyUOp *sink = poly_test_kernel_sink(ctx, sources, 2, "test");
    PolyUOp *expected = i == 2 ? NULL : sink;
    if (i == 1)
      expected = poly_test_apply_opt(
          ctx, sink, (PolyRendererCaps){.device = "CPU"},
          (PolyOpt
          ){.op = POLY_OPT_UNROLL,
            .has_axis = true,
            .axis = 0,
            .arg_kind = POLY_OPT_ARG_INT,
            .arg = 0}
      );
    PolyUOp *actual = poly_apply_opts_heuristic_ex(ctx, sink, (PolyRendererCaps){.device = "CPU"});
    bool same = actual == expected && (i != 1 || expected);
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(same);
  }
  PASS();
}

TEST(codegen, scheduler_matvec_requires_integer_output_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_alu2(
      ctx, POLY_OP_MUL, policy_const(ctx, 16),
      poly_uop_variable(ctx, "matvec", 2, 4, POLY_WEAKINT, 1, false)
  );
  PolyUOp *g =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *r = policy_range(ctx, 8, 1, POLY_AXIS_REDUCE);
  PolyUOp *index =
      poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, g, policy_const(ctx, 8)), r);
  PolyUOp *a = poly_test_program_param(ctx, POLY_FLOAT32, 8, 0);
  PolyUOp *b = poly_test_program_param(ctx, POLY_FLOAT32, 512, 1);
  PolyUOp *p = poly_test_program_param(ctx, POLY_FLOAT32, 64, 2);
  PolyUOp *mul =
      poly_alu2(ctx, POLY_OP_MUL, poly_uop_index(ctx, a, &r, 1), poly_uop_index(ctx, b, &index, 1));
  PolyUOp *red =
      poly_uop2(ctx, POLY_OP_REDUCE, POLY_FLOAT32, mul, r, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, p, &g, 1), red, poly_arg_none());
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, g, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");
  PolyOpt opt = {
      .op = POLY_OPT_UNROLL, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 0};
  bool same = policy_matches(
      ctx, sink, (PolyRendererCaps){.device = "CUDA", .has_local = true, .shared_max = 32768}, &opt,
      1
  );
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(same);
  PASS();
}

TEST(codegen, scheduler_beam_proves_symbolic_budgets) {
  PolyAxisType types[] = {POLY_AXIS_UPCAST, POLY_AXIS_LOCAL};
  int bounds[][3][2] = {{{2, 4}, {128, 256}, {1, 128}}, {{2, 4}, {2048, 4096}, {1, 2048}}};
  for (int type = 0; type < 2; type++) {
    for (int span = 0; span < 3; span++) {
      PolyCtx *ctx = poly_ctx_new();
      PolyUOp *g = policy_range(ctx, 8, 0, POLY_AXIS_GLOBAL);
      PolyUOp *n = poly_uop_variable(
          ctx, "budget", bounds[type][span][0], bounds[type][span][1], POLY_WEAKINT, 1, false
      );
      PolyUOp *u = poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, n, poly_arg_range(1, types[type]));
      PolyUOp *sources[] = {g, u};
      PolyUOp *sink = poly_test_kernel_sink(ctx, sources, 2, "test");
      PolyOpt opt = {
          .op = POLY_OPT_UPCAST,
          .has_axis = true,
          .axis = 0,
          .arg_kind = POLY_OPT_ARG_INT,
          .arg = 4};
      int result = poly_test_beam_kernel_action(
          ctx, sink, (PolyRendererCaps){.device = "CUDA", .has_local = true}, opt
      );
      poly_ctx_destroy(ctx);
      ASSERT_INT_EQ(result, span == 0 ? 1 : span == 1 ? 0 : -1);
    }
  }
  PASS();
}

TEST(codegen, scheduler_full_axis_uses_proved_symbolic_maximum) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *n = poly_uop_variable(ctx, "whole", 0, 16, POLY_WEAKINT, 16, false);
  PolyUOp *r = poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, n, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *sink = poly_test_kernel_sink(ctx, &r, 1, "test");
  PolyRendererCaps caps = {.device = "CPU"};
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 16};
  PolyUOp *expected = poly_test_apply_opt(ctx, sink, caps, opt);
  opt.arg = 0;
  PolyUOp *actual = poly_test_apply_opt(ctx, sink, caps, opt);
  bool same = actual && expected && actual->src[0] == expected->src[0] &&
              actual->arg.kernel_info->applied_opts[0].arg == 0;
  /* Nonconstant full_shape is not equal to integer0: do not remove this
   * action as a duplicate before apply_opt proves the full-axis split. */
  same &= poly_test_beam_kernel_action(ctx, sink, caps, opt) == 1;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(same);
  PASS();
}

TEST(codegen, scheduler_negative_relative_axes) {
  PolyOptOps ops[] = {POLY_OPT_UNROLL, POLY_OPT_GROUP, POLY_OPT_GROUPTOP};
  for (int i = 0; i < 3; i++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *r0 = policy_range(ctx, 8, 0, POLY_AXIS_REDUCE),
            *r1 = policy_range(ctx, 12, 1, POLY_AXIS_REDUCE);
    PolyUOp *sources[] = {
        poly_cast(ctx, poly_alu2(ctx, POLY_OP_ADD, r0, r1), POLY_FLOAT32), r0, r1};
    PolyUOp *red =
        poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, sources, 3, poly_arg_reduce(POLY_OP_ADD, 0));
    PolyUOp *sink = poly_test_kernel_sink(ctx, &red, 1, "test");
    PolyRendererCaps caps = {.device = "CUDA", .has_local = true, .shared_max = 32768};
    PolyOpt opt = {
        .op = ops[i], .has_axis = true, .axis = 1, .arg_kind = POLY_OPT_ARG_INT, .arg = 4};
    PolyUOp *expected = poly_test_apply_opt(ctx, sink, caps, opt);
    opt.axis = -1;
    PolyUOp *actual = poly_test_apply_opt(ctx, sink, caps, opt);
    bool same = actual && expected && actual->src[0] == expected->src[0] &&
                actual->arg.kernel_info->applied_opts[0].axis == -1;
    opt.axis = -3;
    same &= !poly_test_apply_opt(ctx, sink, caps, opt);
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(same);
  }
  PASS();
}

TEST(codegen, scheduler_swap_negative_target) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sources[] = {
      policy_range(ctx, 8, 0, POLY_AXIS_GLOBAL), policy_range(ctx, 12, 1, POLY_AXIS_GLOBAL)};
  PolyUOp *sink = poly_test_kernel_sink(ctx, sources, 2, "test");
  PolyOpt opt = {
      .op = POLY_OPT_SWAP, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = -1};
  PolyUOp *actual = poly_test_apply_opt(ctx, sink, (PolyRendererCaps){.device = "CUDA"}, opt);
  bool same = actual && actual->src[0] == policy_range(ctx, 8, 1, POLY_AXIS_GLOBAL) &&
              actual->src[1] == policy_range(ctx, 12, 0, POLY_AXIS_GLOBAL);
  opt.arg = -3;
  same &= !poly_test_apply_opt(ctx, sink, (PolyRendererCaps){.device = "CUDA"}, opt);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(same);
  PASS();
}

TEST(codegen, scheduler_zero_shared_budget_uses_renderer_default) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_REDUCE, 8);
  PolyOpt opt = {
      .op = POLY_OPT_GROUP, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4};
  PolyUOp *actual = poly_test_apply_opt(
      ctx, sink, (PolyRendererCaps){.device = "CUDA", .has_local = true, .shared_max = 0}, opt
  );
  PolyUOp *expected = poly_test_apply_opt(
      ctx, sink, (PolyRendererCaps){.device = "CUDA", .has_local = true, .shared_max = 32768}, opt
  );
  bool same = actual && actual == expected;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(same);
  PASS();
}

TEST(codegen, scheduler_symbolic_shared_memory_admission) {
  int limits[] = {128, 16, 48};
  for (int i = 0; i < 3; i++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *size = poly_uop_variable(ctx, "up", 2, 4, POLY_WEAKINT, 1, false);
    PolyUOp *u =
        poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, size, poly_arg_range(0, POLY_AXIS_UPCAST));
    PolyUOp *r = policy_range(ctx, 8, 1, POLY_AXIS_REDUCE);
    PolyUOp *sources[] = {
        u, poly_uop2(
               ctx, POLY_OP_REDUCE, POLY_FLOAT32, poly_cast(ctx, r, POLY_FLOAT32), r,
               poly_arg_reduce(POLY_OP_ADD, 0)
           )};
    PolyUOp *sink = poly_test_kernel_sink(ctx, sources, 2, "test");
    PolyOpt opt = {
        .op = POLY_OPT_GROUP, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4};
    PolyUOp *actual = poly_test_apply_opt(
        ctx, sink, (PolyRendererCaps){.device = "CUDA", .has_local = true, .shared_max = limits[i]},
        opt
    );
    PolyUOp *part = policy_range(ctx, 2, 1, POLY_AXIS_REDUCE);
    PolyUOp *split = policy_range(ctx, 4, 2, POLY_AXIS_GROUP_REDUCE);
    PolyUOp *idx =
        poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, part, policy_const(ctx, 4)), split);
    PolyUOp *expected = poly_uop_substitute(ctx, sink, &r, &idx, 1);
    bool same = i ? actual == NULL
                  : actual && actual->n_src == expected->n_src &&
                        actual->src[0] == expected->src[0] && actual->src[1] == expected->src[1] &&
                        actual->arg.kernel_info->n_applied_opts == 1;
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(same);
  }
  PASS();
}

TEST(codegen, scheduler_symbolic_group_reduction) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *g = policy_range(ctx, 4, 0, POLY_AXIS_GLOBAL);
  PolyUOp *size = poly_alu2(
      ctx, POLY_OP_MUL, policy_const(ctx, 16),
      poly_uop_variable(ctx, "rsize", 2, 4, POLY_WEAKINT, 1, false)
  );
  PolyUOp *r =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, size, poly_arg_range(1, POLY_AXIS_REDUCE));
  PolyUOp *p = poly_test_program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *value = poly_uop2(
      ctx, POLY_OP_REDUCE, POLY_FLOAT32, poly_cast(ctx, r, POLY_FLOAT32), r,
      poly_arg_reduce(POLY_OP_ADD, 0)
  );
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, p, &g, 1), value, poly_arg_none()
  );
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, g, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");
  PolyOpt opt = {
      .op = POLY_OPT_GROUPTOP,
      .has_axis = true,
      .axis = 0,
      .arg_kind = POLY_OPT_ARG_INT,
      .arg = 16};
  bool same =
      policy_matches(ctx, sink, (PolyRendererCaps){.device = "CUDA", .has_local = true}, &opt, 1);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(same);
  PASS();
}

TEST(codegen, scheduler_local_requires_literal_bound) {
  const int factors[][2] = {{2, 2}, {3, 16}};
  for (int i = 0; i < 2; i++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *g = poly_uop1(
        ctx, POLY_OP_RANGE, POLY_WEAKINT,
        poly_alu2(
            ctx, POLY_OP_MUL, policy_const(ctx, factors[i][0]), policy_const(ctx, factors[i][1])
        ),
        poly_arg_range(0, POLY_AXIS_GLOBAL)
    );
    PolyUOp *u = policy_range(ctx, 4, 1, POLY_AXIS_UPCAST);
    PolyUOp *p = poly_test_program_param(ctx, POLY_FLOAT32, factors[i][0] * factors[i][1] * 4, 0);
    PolyUOp *idx =
        poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, g, policy_const(ctx, 4)), u);
    PolyUOp *store = poly_uop2(
        ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, p, &idx, 1),
        poly_cast(ctx, g, POLY_FLOAT32), poly_arg_none()
    );
    PolyUOp *sources[] = {store, g, u};
    PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, sources, 3, poly_arg_none());
    PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");
    bool same =
        policy_matches(ctx, sink, (PolyRendererCaps){.device = "CUDA", .has_local = true}, NULL, 0);
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(same);
  }
  PASS();
}

TEST(codegen, scheduler_propagates_mandatory_heuristic_failure) {
  const char *value = getenv("NOLOCALS");
  char *saved = value ? strdup(value) : NULL;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *g = policy_range(ctx, 64, 0, POLY_AXIS_GLOBAL);
  PolyUOp *l = policy_range(ctx, 4, 1, POLY_AXIS_LOCAL);
  PolyUOp *u = policy_range(ctx, 4, 2, POLY_AXIS_UPCAST);
  PolyUOp *p = poly_test_program_param(ctx, POLY_FLOAT32, 1024, 0);
  PolyUOp *idx = poly_alu2(
      ctx, POLY_OP_ADD,
      poly_alu2(
          ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, g, policy_const(ctx, 16)),
          poly_alu2(ctx, POLY_OP_MUL, l, policy_const(ctx, 4))
      ),
      u
  );
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, p, &idx, 1),
      poly_cast(ctx, g, POLY_FLOAT32), poly_arg_none()
  );
  PolyUOp *sources[] = {store, g, l, u};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, sources, 4, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");
  setenv("NOLOCALS", "1", 1);
  PolyUOp *result = poly_apply_opts_heuristic_ex(
      ctx, sink, (PolyRendererCaps){.device = "CUDA", .has_local = true}
  );
  if (saved)
    setenv("NOLOCALS", saved, 1);
  else
    unsetenv("NOLOCALS");
  free(saved);
  bool rejected = result == NULL;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rejected);
  PASS();
}

TEST(codegen, scheduler_matvec_requires_symbolic_divisibility) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *g = policy_range(ctx, 16, 0, POLY_AXIS_GLOBAL);
  PolyUOp *size = poly_alu2(
      ctx, POLY_OP_MUL, policy_const(ctx, 3),
      poly_uop_variable(ctx, "rsize", 2, 4, POLY_WEAKINT, 1, false)
  );
  PolyUOp *r =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, size, poly_arg_range(1, POLY_AXIS_REDUCE));
  PolyUOp *idx =
      poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, g, policy_const(ctx, 12)), r);
  PolyUOp *a = poly_test_program_param(ctx, POLY_FLOAT32, 12, 0);
  PolyUOp *b = poly_test_program_param(ctx, POLY_FLOAT32, 192, 1);
  PolyUOp *p = poly_test_program_param(ctx, POLY_FLOAT32, 16, 2);
  PolyUOp *value = poly_uop2(
      ctx, POLY_OP_REDUCE, POLY_FLOAT32,
      poly_alu2(ctx, POLY_OP_MUL, poly_uop_index(ctx, a, &r, 1), poly_uop_index(ctx, b, &idx, 1)),
      r, poly_arg_reduce(POLY_OP_ADD, 0)
  );
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, p, &g, 1), value, poly_arg_none()
  );
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, g, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");
  PolyOpt opts[] = {
      {.op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4},
      {.op = POLY_OPT_LOCAL, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4}};
  bool same =
      policy_matches(ctx, sink, (PolyRendererCaps){.device = "CUDA", .has_local = true}, opts, 2);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(same);
  PASS();
}

TEST(codegen, beam_debug_reports_search_and_candidates) {
  const char *names[] = {"BEAM_DEBUG", "CACHELEVEL", "DEBUG"};
  char *old[3];
  for (int i = 0; i < 3; i++)
    old[i] = getenv(names[i]) ? strdup(getenv(names[i])) : NULL;
  FILE *log = tmpfile();
  ASSERT_NOT_NULL(log);
  fflush(stderr);
  int saved = dup(STDERR_FILENO);
  ASSERT_TRUE(saved >= 0);
  ASSERT_TRUE(dup2(fileno(log), STDERR_FILENO) >= 0);
  setenv("BEAM_DEBUG", "2", 1);
  setenv("CACHELEVEL", "0", 1);
  setenv("DEBUG", "0", 1);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_WEAK, 8);
  PolyUOp *out = poly_full_rewrite_to_sink_ex(
      ctx, sink,
      (PolyRewriteOpts
      ){.optimize = true,
        .beam_width = 1,
        .device = POLY_DEVICE_INTERP,
        .caps = {.device = "PYTHON", .has_int64 = true}}
  );
  bool ok = out != NULL;
  poly_ctx_destroy(ctx);
  fflush(stderr);
  dup2(saved, STDERR_FILENO);
  close(saved);
  rewind(log);
  char line[4096];
  bool start = false, final = false, candidate = false;
  while (fgets(line, sizeof(line), log)) {
    start |= strstr(line, "BEAM_SEARCH:") != NULL;
    final |= strstr(line, "applied_opts=") != NULL;
    candidate |= strstr(line, "compile/") != NULL && strstr(line, "run") != NULL;
  }
  fclose(log);
  for (int i = 0; i < 3; i++) {
    if (old[i])
      setenv(names[i], old[i], 1);
    else
      unsetenv(names[i]);
    free(old[i]);
  }
  ASSERT_TRUE(ok && start && final && candidate);
  PASS();
}

TEST(codegen, scheduler_policy_image_upcasts_before_masked_occupancy) {
  char *old = getenv("IMAGE") ? strdup(getenv("IMAGE")) : NULL;
  setenv("IMAGE", "1", 1);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *g = policy_range(ctx, 3, 0, POLY_AXIS_GLOBAL);
  PolyUOp *r = policy_range(ctx, 8, 1, POLY_AXIS_GLOBAL);
  PolyUOp *idx =
      poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, g, policy_const(ctx, 8)), r);
  PolyUOp *p = poly_test_program_param(ctx, POLY_FLOAT32, 24, 0);
  PolyUOp *value = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_WEAKINT, poly_alu2(ctx, POLY_OP_CMPLT, g, policy_const(ctx, 2)), r,
      policy_const(ctx, 0), poly_arg_none()
  );
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, p, &idx, 1),
      poly_cast(ctx, value, POLY_FLOAT32), poly_arg_none()
  );
  PolyUOp *end =
      poly_uop(ctx, POLY_OP_END, POLY_VOID, (PolyUOp *[]){store, g, r}, 3, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 1, .arg_kind = POLY_OPT_ARG_INT, .arg = 4};
  bool ok = policy_matches(
      ctx, sink, (PolyRendererCaps){.device = "NULL", .arch = "IMAGE_PITCH_ALIGNMENT=1"}, &opt, 1
  );
  poly_ctx_destroy(ctx);
  if (old)
    setenv("IMAGE", old, 1);
  else
    unsetenv("IMAGE");
  free(old);
  ASSERT_TRUE(ok);
  PASS();
}

static bool matvec_heuristic_matches_options(bool first_max, int prefix) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *g = poly_range(ctx, 16, 0, POLY_AXIS_GLOBAL);
  PolyUOp *r = poly_range(ctx, 8, 1, POLY_AXIS_REDUCE);
  PolyUOp *idx = r;
  if (prefix) {
    idx = poly_const_int(ctx, 1);
    for (int i = 1; i < prefix; i++)
      idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, idx, poly_const_int(ctx, 1), poly_arg_none());
    idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, idx, r, poly_arg_none());
  }
  PolyUOp *a = poly_test_program_param(ctx, POLY_FLOAT32, 512, 0);
  PolyUOp *b = poly_test_program_param(ctx, POLY_FLOAT32, 512, 1);
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 512, 2);
  PolyUOp *av = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, a, idx, poly_arg_none());
  PolyUOp *bi = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32, idx,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, g, poly_const_int(ctx, 8), poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *bv = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, b, bi, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, av, bv, poly_arg_none());
  PolyUOp *value =
      poly_uop2(ctx, POLY_OP_REDUCE, POLY_FLOAT32, mul, r, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, out, &g, 1), value, poly_arg_none()
  );
  PolyUOp *ends[2];
  int n = 0;
  if (first_max) {
    PolyUOp *aux = poly_test_program_param(ctx, POLY_FLOAT32, 512, 3);
    PolyUOp *zero = poly_const_int(ctx, 0);
    PolyUOp *mx = poly_uop2(
        ctx, POLY_OP_REDUCE, POLY_FLOAT32, poly_cast(ctx, r, POLY_FLOAT32), r,
        poly_arg_reduce(POLY_OP_MAX, 0)
    );
    ends[n++] = poly_uop2(
        ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, aux, &zero, 1), mx, poly_arg_none()
    );
  }
  ends[n++] = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, g, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, ends, n, "matvec");
  PolyRendererCaps caps = {.device = "CUDA", .has_local = true, .shared_max = 49152};
  PolyOpt opts[] = {
      {.op = first_max ? POLY_OPT_UNROLL : POLY_OPT_GROUP,
       .has_axis = true,
       .axis = 0,
       .arg_kind = POLY_OPT_ARG_INT,
       .arg = first_max ? 0 : 8},
      {.op = POLY_OPT_LOCAL,
       .has_axis = true,
       .axis = 0,
       .arg_kind = POLY_OPT_ARG_INT,
       .arg = first_max ? 16 : 4},
      {.op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4}};
  PolyUOp *expected = sink;
  for (int i = 0; expected && i < (first_max ? 2 : 3); i++)
    expected = poly_test_apply_opt(ctx, expected, caps, opts[i]);
  PolyUOp *actual = poly_apply_opts_heuristic_ex(ctx, sink, caps);
  bool ok = expected && actual == expected;
  poly_ctx_destroy(ctx);
  return ok;
}

TEST(codegen, scheduler_matvec_uses_first_reduction) {
  ASSERT_TRUE(matvec_heuristic_matches_options(false, 0));
  ASSERT_TRUE(matvec_heuristic_matches_options(true, 0));
  PASS();
}

TEST(codegen, scheduler_matvec_visits_all_addends) {
  ASSERT_TRUE(matvec_heuristic_matches_options(false, 256));
  PASS();
}

TEST(codegen, scheduler_copy_survives_parent_and_sibling_disposal) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_GLOBAL, 8);
  bool ok = poly_test_scheduler_copy_lifetime(ctx, sink);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(codegen, scheduler_heuristic_honors_nolocals) {
  const char *env = getenv("NOLOCALS");
  char *old = env ? strdup(env) : NULL;
  setenv("NOLOCALS", "1", 1);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_GLOBAL, 64);
  PolyUOp *out = poly_apply_opts_heuristic_ex(
      ctx, sink, (PolyRendererCaps){.device = "CUDA", .max_vec_width = 4, .has_local = true}
  );
  const PolyKernelInfo *info =
      out && out->arg.kind == POLY_ARG_KERNEL_INFO ? out->arg.kernel_info : NULL;
  bool ok = info && info->dont_use_locals;
  int n = 0;
  PolyUOp **topo = out ? poly_toposort_alloc(ctx, out, &n) : NULL;
  for (int i = 0; topo && i < n; i++)
    if (topo[i]->op == POLY_OP_RANGE && poly_range_axis_type(topo[i]->arg) == POLY_AXIS_LOCAL)
      ok = false;
  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  if (old)
    setenv("NOLOCALS", old, 1);
  else
    unsetenv("NOLOCALS");
  free(old);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(codegen, scheduler_heuristic_does_not_thread_nested_output) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 1 << 20, 0, POLY_AXIS_WEAK);
  PolyUOp *noop = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *inner = poly_uop2(ctx, POLY_OP_END, POLY_VOID, noop, r, poly_arg_none());
  PolyUOp *outer = poly_uop1(ctx, POLY_OP_END, POLY_VOID, inner, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &outer, 1, "test");
  PolyUOp *out = poly_apply_opts_heuristic_ex(ctx, sink, poly_c_renderer_caps());
  bool ok = out != NULL;
  int n = 0;
  PolyUOp **topo = out ? poly_toposort_alloc(ctx, out, &n) : NULL;
  for (int i = 0; topo && i < n; i++)
    if (topo[i]->op == POLY_OP_RANGE && poly_range_axis_type(topo[i]->arg) == POLY_AXIS_THREAD)
      ok = false;
  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(codegen, scheduler_globalizes_only_top_level_outputs) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 8, 0, POLY_AXIS_WEAK);
  PolyUOp *noop = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *inner = poly_uop2(ctx, POLY_OP_END, POLY_VOID, noop, r, poly_arg_none());
  PolyUOp *outer = poly_uop1(ctx, POLY_OP_END, POLY_VOID, inner, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &outer, 1, "test");
  bool correct = poly_test_convert_loop_to_global(ctx, sink) == sink;
  PolyUOp *direct = poly_test_kernel_sink(ctx, &inner, 1, "test");
  PolyUOp *global = poly_range(ctx, 8, 0, POLY_AXIS_GLOBAL);
  correct &= poly_test_convert_loop_to_global(ctx, direct) ==
             poly_uop_substitute(ctx, direct, &r, &global, 1);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(codegen, scheduler_globalizes_all_output_ranges) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ranges[65], *globals[65], *ends[65];
  PolyUOp *noop = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  for (int i = 0; i < 65; i++) {
    ranges[i] = poly_range(ctx, 8, i, POLY_AXIS_WEAK);
    globals[i] = poly_range(ctx, 8, i, POLY_AXIS_GLOBAL);
    ends[i] = poly_uop2(ctx, POLY_OP_END, POLY_VOID, noop, ranges[i], poly_arg_none());
  }
  PolyUOp *sink = poly_test_kernel_sink(ctx, ends, 65, "test");
  bool correct = poly_test_convert_loop_to_global(ctx, sink) ==
                 poly_uop_substitute(ctx, sink, ranges, globals, 65);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(codegen, scheduler_globalizes_complete_range_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t path[20];
  for (int i = 0; i < 20; i++)
    path[i] = i;
  PolyUOp *bound = poly_const_int(ctx, 8);
  PolyUOp *r = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range_ex(0, POLY_AXIS_WEAK, path, 20)
  );
  PolyUOp *g = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range_ex(0, POLY_AXIS_GLOBAL, path, 20)
  );
  PolyUOp *end = poly_uop2(
      ctx, POLY_OP_END, POLY_VOID, poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none()), r,
      poly_arg_none()
  );
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");
  bool correct =
      poly_test_convert_loop_to_global(ctx, sink) == poly_uop_substitute(ctx, sink, &r, &g, 1);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(codegen, scheduler_control_flow_rejects_cyclic_siblings) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r0 = poly_range(ctx, 8, 0, POLY_AXIS_WEAK);
  PolyUOp *r1 = poly_range(ctx, 8, 1, POLY_AXIS_WEAK);
  PolyUOp *n0 = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_int(0));
  PolyUOp *n1 = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_int(1));
  PolyUOp *e0 = poly_uop2(ctx, POLY_OP_END, POLY_VOID, n0, r0, poly_arg_none());
  PolyUOp *e1 = poly_uop2(ctx, POLY_OP_END, POLY_VOID, n1, r1, poly_arg_none());
  PolyUOp *ends[] = {e0, e1};
  PolyUOp *sink = poly_test_kernel_sink(ctx, ends, 2, "test");
  PolyUOp *ordered = poly_uop2(ctx, POLY_OP_RANGE, r1->dtype, r1->src[0], e0, r1->arg);
  PolyUOp *expected = poly_uop_substitute(ctx, sink, &r1, &ordered, 1);
  bool correct = poly_apply_control_flow(ctx, sink) == expected;
  bool complete = false;
  for (int budget = 0; budget < 64; budget++) {
    poly_test_linearizer_alloc_fail_after(budget);
    PolyUOp *result = poly_apply_control_flow(ctx, sink);
    poly_test_linearizer_alloc_fail_after(-1);
    if (result) {
      complete = result == expected;
      break;
    }
  }
  correct &= complete;
  ends[1] = poly_uop2(ctx, POLY_OP_END, POLY_VOID, n1, r0, poly_arg_none());
  sink = poly_test_kernel_sink(ctx, ends, 2, "test");
  correct &= poly_apply_control_flow(ctx, sink) == NULL;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(codegen, scheduler_range_tuple_orders_split_before_type) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t apath = 5, bpath = 0;
  PolyUOp *bound = poly_const_int(ctx, 8);
  PolyUOp *a = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range_ex(0, POLY_AXIS_GLOBAL, &apath, 1)
  );
  PolyUOp *b = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range_ex(0, POLY_AXIS_WEAK, &bpath, 1)
  );
  PolyUOp *src[] = {a, b};
  PolyUOp *sink = poly_test_kernel_sink(ctx, src, 2, "test");
  int n = 0, ai = -1, bi = -1;
  PolyUOp **linear = poly_linearize(ctx, sink, &n);
  for (int i = 0; linear && i < n; i++) {
    if (linear[i] == a) ai = i;
    if (linear[i] == b) bi = i;
  }
  bool correct = bi >= 0 && ai > bi;
  free(linear);
  PolyUOp *body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *end_src[] = {body, a, b};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 3, poly_arg_none());
  PolyUOp *expected = poly_uop2(
      ctx, POLY_OP_END, POLY_VOID, poly_uop2(ctx, POLY_OP_END, POLY_VOID, body, a, poly_arg_none()),
      b, poly_arg_none()
  );
  correct &= poly_graph_rewrite(ctx, end, poly_pm_split_ends()) == expected;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(codegen, beam_scheduler_local_and_group_actions) {
  PolyOptOps ops[] = {POLY_OPT_LOCAL, POLY_OPT_GROUP, POLY_OPT_GROUPTOP, POLY_OPT_THREAD};
  PolyAxisType inputs[] = {POLY_AXIS_GLOBAL, POLY_AXIS_REDUCE, POLY_AXIS_REDUCE, POLY_AXIS_WEAK};
  PolyAxisType outputs[] = {
      POLY_AXIS_LOCAL, POLY_AXIS_GROUP_REDUCE, POLY_AXIS_GROUP_REDUCE, POLY_AXIS_THREAD};
  for (int i = 0; i < 4; i++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyRendererCaps caps = {
        .device = "CUDA",
        .has_local = true,
        .has_threads = true,
        .max_threads = 32,
        .global_max = {32, 1, 1}};
    PolyUOp *sink = beam_action_sink(ctx, inputs[i], 8);
    PolyOpt opt = {
        .op = ops[i], .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4};
    PolyUOp *actual = poly_test_apply_opt(ctx, sink, caps, opt);
    PolyUOp *r = poly_range(ctx, 8, 0, inputs[i]);
    PolyUOp *part = poly_range(ctx, 2, 0, inputs[i]);
    PolyUOp *split = poly_range(ctx, 4, 1, outputs[i]);
    bool top = ops[i] == POLY_OPT_GROUPTOP || ops[i] == POLY_OPT_THREAD;
    PolyUOp *replacement = poly_alu2(
        ctx, POLY_OP_ADD,
        poly_alu2(ctx, POLY_OP_MUL, top ? split : part, poly_const_int(ctx, top ? 2 : 4)),
        top ? part : split
    );
    PolyUOp *expected = poly_uop_substitute(ctx, sink, &r, &replacement, 1);
    bool same = actual && actual->n_src == expected->n_src && actual->src[0] == expected->src[0] &&
                actual->arg.kind == POLY_ARG_KERNEL_INFO &&
                actual->arg.kernel_info->n_applied_opts == 1 &&
                poly_kernel_info_eq(
                    actual->arg.kernel_info,
                    &(PolyKernelInfo){.name = "test", .applied_opts = &opt, .n_applied_opts = 1}
                );
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(same);
  }
  PASS();
}

TEST(codegen, beam_scheduler_full_axis_and_nolocals) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_GLOBAL, 8);
  PolyRendererCaps caps = {.device = "CUDA", .has_local = true};
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 0};
  PolyUOp *up = poly_test_apply_opt(ctx, sink, caps, opt);
  PolyUOp *no = poly_test_apply_opt(ctx, sink, caps, (PolyOpt){.op = POLY_OPT_NOLOCALS});
  bool success =
      up && no && no->arg.kind == POLY_ARG_KERNEL_INFO && no->arg.kernel_info->dont_use_locals;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(success);
  PASS();
}

TEST(codegen, beam_scheduler_excludes_void_and_device_ranges) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_const_int(ctx, 8);
  PolyUOp *loop =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_VOID, bound, poly_arg_range(0, POLY_AXIS_WEAK));
  PolyUOp *device = poly_range(ctx, 8, 1, POLY_AXIS_DEVICE);
  PolyUOp *range = poly_range(ctx, 8, 2, POLY_AXIS_WEAK);
  PolyUOp *src[] = {loop, device, range};
  PolyUOp *sink = poly_test_kernel_sink(ctx, src, 3, "test");
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4};
  PolyUOp *result = poly_test_apply_opt(ctx, sink, (PolyRendererCaps){.device = "CPU"}, opt);
  bool correct = result && result->src[0] == loop && result->src[1] == device &&
                 result->src[2] != range && result->src[2]->op == POLY_OP_ADD;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(codegen, beam_scheduler_padto_guards_loads_and_stores) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 12, 0, POLY_AXIS_GLOBAL);
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 12, 0);
  PolyUOp *in = poly_test_program_param(ctx, POLY_FLOAT32, 12, 1);
  PolyUOp *load = poly_uop_index(ctx, in, &r, 1);
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, out, &r, 1), load, poly_arg_none()
  );
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, r, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");
  PolyUOp *actual = poly_test_apply_opt(
      ctx, sink, (PolyRendererCaps){.device = "CUDA", .has_local = true},
      (PolyOpt
      ){.op = POLY_OPT_PADTO, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 32}
  );
  PolyUOp *padded = poly_range(ctx, 32, 0, POLY_AXIS_GLOBAL);
  PolyUOp *valid = poly_alu2(ctx, POLY_OP_CMPLT, padded, poly_const_int(ctx, 12));
  PolyUOp *mask = poly_alu2(
      ctx, POLY_OP_AND, valid, poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true))
  );
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_BOOL);
  PolyUOp *index =
      poly_uop3(ctx, POLY_OP_WHERE, padded->dtype, mask, padded, invalid, poly_arg_none());
  PolyUOp *gated_load = poly_uop3(
      ctx, POLY_OP_WHERE, load->dtype, valid, poly_uop_index(ctx, in, &index, 1), invalid,
      poly_arg_none()
  );
  PolyUOp *expected_store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, out, &index, 1), gated_load,
      poly_arg_none()
  );
  PolyUOp *expected_end =
      poly_uop2(ctx, POLY_OP_END, POLY_VOID, expected_store, padded, poly_arg_none());
  bool same = actual && actual->src[0] == expected_end;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(same);
  PASS();
}

TEST(codegen, beam_scheduler_splits_symbolic_divisible_bound) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *n = poly_uop_variable(ctx, "n", 2, 8, POLY_WEAKINT, 1, false);
  PolyUOp *four = poly_const_int(ctx, 4);
  PolyUOp *bound = poly_alu2(ctx, POLY_OP_MUL, n, four);
  PolyUOp *r =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_WEAK));
  PolyUOp *sink = poly_test_kernel_sink(ctx, &r, 1, "test");
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4};
  PolyUOp *actual = poly_test_apply_opt(ctx, sink, (PolyRendererCaps){.device = "CPU"}, opt);
  /* UOp.divides keeps n*1 structural until the normal symbolic pass. */
  PolyUOp *remaining = poly_alu2(ctx, POLY_OP_MUL, n, poly_const_int(ctx, 1));
  PolyUOp *part = poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, remaining, r->arg);
  PolyUOp *split = poly_range(ctx, 4, 1, POLY_AXIS_UPCAST);
  PolyUOp *expected = poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, part, four), split);
  bool correct = actual && actual->src[0] == expected;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(codegen, beam_explicit_invalid_option_fails_closed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = beam_action_sink(ctx, POLY_AXIS_WEAK, 8);
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 3};
  PolyKernelInfo info = {
      .name = "test", .has_opts_to_apply = true, .opts_to_apply = &opt, .n_opts_to_apply = 1};
  PolyUOp *sink =
      poly_uop(ctx, POLY_OP_SINK, POLY_VOID, base->src, base->n_src, poly_arg_kernel_info(&info));
  PolyUOp *result = poly_full_rewrite_to_sink_ex(
      ctx, sink, (PolyRewriteOpts){.optimize = true, .caps = poly_c_renderer_caps()}
  );
  bool rejected = result == NULL;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rejected);
  PASS();
}

TEST(codegen, beam_explicit_options_override_search_and_noopt) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = beam_action_sink(ctx, POLY_AXIS_WEAK, 8);
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4};
  int previous = poly_get_noopt();
  poly_set_noopt(1);
  bool correct = true;
  for (int count = 0; count <= 1; count++) {
    PolyKernelInfo info = {
        .name = "test", .has_opts_to_apply = true, .opts_to_apply = &opt, .n_opts_to_apply = count};
    PolyUOp *sink =
        poly_uop(ctx, POLY_OP_SINK, POLY_VOID, base->src, base->n_src, poly_arg_kernel_info(&info));
    PolyUOp *result = poly_full_rewrite_to_sink_ex(
        ctx, sink,
        (PolyRewriteOpts){.optimize = true, .beam_width = 2, .caps = poly_c_renderer_caps()}
    );
    correct &= result && result->tag == 1 && result->arg.kind == POLY_ARG_KERNEL_INFO;
    if (result && result->arg.kind == POLY_ARG_KERNEL_INFO) {
      const PolyKernelInfo *scheduled = result->arg.kernel_info;
      correct &= !scheduled->has_opts_to_apply && scheduled->n_applied_opts == count;
      if (count && scheduled->n_applied_opts == 1)
        correct &= scheduled->applied_opts[0].op == opt.op && scheduled->applied_opts[0].arg == 4;
    }
  }
  poly_set_noopt(previous);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(codegen, beam_scheduler_swap_preserves_connections) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r0 = poly_range(ctx, 2, 0, POLY_AXIS_GLOBAL);
  PolyUOp *r1 = poly_range(ctx, 3, 1, POLY_AXIS_GLOBAL);
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 6, 0);
  PolyUOp *index =
      poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, r0, poly_const_int(ctx, 3)), r1);
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop_index(ctx, out, &index, 1),
      poly_cast(ctx, r0, POLY_FLOAT32), poly_arg_none()
  );
  PolyUOp *src[] = {store, r0, r1};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, src, 3, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");
  PolyUOp *actual = poly_test_apply_opt(
      ctx, sink, (PolyRendererCaps){.device = "CUDA", .has_local = true},
      (PolyOpt
      ){.op = POLY_OPT_SWAP, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 1}
  );
  PolyUOp *from[] = {r0, r1};
  PolyUOp *to[] = {
      poly_range(ctx, 2, 1, POLY_AXIS_GLOBAL), poly_range(ctx, 3, 0, POLY_AXIS_GLOBAL)};
  PolyUOp *expected = poly_uop_substitute(ctx, end, from, to, 2);
  bool same = actual && actual->src[0] == expected;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(same);
  PASS();
}

TEST(codegen, beam_candidate_normalizes_ended_range_expressions) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *r0 = poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *r1 = poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(1, POLY_AXIS_LOOP));
  PolyUOp *idx = poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, r0, two), r1);
  PolyUOp *ptr = poly_uop_index(ctx, out, &idx, 1);
  PolyUOp *store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, ptr, poly_const_float(ctx, 1), poly_arg_none());
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, idx, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "beam_shifted_range");
  int n_args = 0;
  double elapsed = poly_test_beam_compile_and_time(
      ctx, sink, (PolyRewriteOpts){.caps = poly_c_renderer_caps()}, 1, &n_args
  );
  ASSERT_TRUE(isfinite(elapsed));
  ASSERT_INT_EQ(n_args, 1);
  PolyUOp *normalized = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
  ASSERT_NOT_NULL(normalized);
  ASSERT_INT_EQ(normalized->src[0]->n_src, 3);
  ASSERT_TRUE(normalized->src[0]->src[0] == store);
  ASSERT_TRUE(normalized->src[0]->src[1] == r0);
  ASSERT_TRUE(normalized->src[0]->src[2] == r1);
  PolyUOp *lowered = poly_full_rewrite_to_sink_ex(
      ctx, normalized, (PolyRewriteOpts){.caps = poly_c_renderer_caps()}
  );
  int n = 0;
  PolyUOp **lin = poly_do_linearize(ctx, lowered, &n);
  ASSERT_NOT_NULL(lin);
  char *source = poly_render_c(ctx, lin, n, "beam_shifted_values");
  ASSERT_NOT_NULL(source);
  PolyProgram *program = poly_compile_c(source, "beam_shifted_values");
  ASSERT_NOT_NULL(program);
  float values[4] = {0};
  void *args[] = {values};
  poly_program_call(program, args, 1);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(values[i], 1.0, 0.0);
  poly_program_destroy(program);
  free(source);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, beam_scratch_uses_storage_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *param = poly_test_program_param(ctx, POLY_FLOAT64, 8, 0);
  int n_args = 0;
  void **args = poly_test_beam_args_from_ast(ctx, param, &n_args);
  ASSERT_NOT_NULL(args);
  ASSERT_INT_EQ(n_args, 1);
  /* A float32-sized allocation for this f64 parameter overruns here. */
  ((double *)args[0])[7] = 3.0;
  ASSERT_TRUE(((double *)args[0])[7] == 3.0);
  free(args[0]);
  free(args);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, beam_scratch_uses_argument_slot_order) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *params[] = {
      poly_test_program_param(ctx, POLY_FLOAT32, 1, 9),
      poly_test_program_param(ctx, POLY_FLOAT32, 32, 4)};
  PolyUOp *sink = poly_test_kernel_sink(ctx, params, 2, "beam_order");
  int n_args = 0;
  void **args = poly_test_beam_args_from_ast(ctx, sink, &n_args);
  ASSERT_NOT_NULL(args);
  ASSERT_INT_EQ(n_args, 2);
  ((float *)args[0])[31] = 4.0;
  ASSERT_FLOAT_EQ(((float *)args[0])[31], 4.0, 0.0);
  for (int i = 0; i < n_args; i++)
    free(args[i]);
  free(args);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, beam_scratch_samples_scalar_midpoint) {
  PolyCtx *ctx = poly_ctx_new();
  PolyParamArg arg = {
      .slot = 2,
      .addrspace = POLY_ADDR_ALU,
      .name = "n",
      .min_val = -9,
      .max_val = -4,
      .has_minmax = true};
  PolyUOp *variable = poly_uop0(ctx, POLY_OP_PARAM, POLY_INT32, poly_arg_param(&arg));
  int n_args = 0;
  void **args = poly_test_beam_args_from_ast(ctx, variable, &n_args);
  ASSERT_NOT_NULL(args);
  ASSERT_INT_EQ(n_args, 1);
  int value = *(int32_t *)args[0];
  free(args[0]);
  free(args);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(value, -7);
  PASS();
}

TEST(codegen, beam_scratch_uses_c_wrapper_scalar_abi) {
  PolyCtx *ctx = poly_ctx_new();
  PolyParamArg arg = {
      .slot = 0,
      .addrspace = POLY_ADDR_ALU,
      .name = "n",
      .min_val = 4,
      .max_val = 9,
      .has_minmax = true};
  PolyUOp *variable = poly_uop0(ctx, POLY_OP_PARAM, POLY_UINT8, poly_arg_param(&arg));
  int n_args = 0;
  void **args = poly_test_beam_args_from_ast(ctx, variable, &n_args);
  ASSERT_NOT_NULL(args);
  ASSERT_INT_EQ(n_args, 1);
  /* The native C wrapper reads an int, then converts to the declared dtype. */
  int value = *(int *)args[0];
  free(args[0]);
  free(args);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(value, 6);
  PASS();
}

TEST(codegen, beam_scratch_mixed_arguments_and_symbolic_extent) {
  PolyCtx *ctx = poly_ctx_new();
  PolyParamArg var_arg = {
      .slot = 0,
      .addrspace = POLY_ADDR_ALU,
      .name = "n",
      .min_val = 4,
      .max_val = 9,
      .has_minmax = true};
  PolyUOp *variable = poly_uop0(ctx, POLY_OP_PARAM, POLY_INT32, poly_arg_param(&var_arg));
  PolyParamArg buf_arg = {.slot = 7, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *buffer = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT64, variable, poly_arg_param(&buf_arg));
  var_arg.slot = 8;
  var_arg.name = "core_id";
  PolyUOp *core_id = poly_uop0(ctx, POLY_OP_PARAM, POLY_INT32, poly_arg_param(&var_arg));
  PolyUOp *src[] = {variable, buffer, core_id};
  PolyUOp *sink = poly_test_kernel_sink(ctx, src, 3, "beam_mixed");
  int n_args = 0;
  void **args = poly_test_beam_args_from_ast(ctx, sink, &n_args);
  ASSERT_NOT_NULL(args);
  ASSERT_INT_EQ(n_args, 2);
  /* Buffer capacity uses the maximum, not the sampled scalar or topo order. */
  ((double *)args[0])[8] = 7.0;
  ASSERT_TRUE(((double *)args[0])[8] == 7.0);
  ASSERT_INT_EQ(*(int *)args[1], 6);
  free(args[0]);
  free(args[1]);
  free(args);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, beam_scratch_scalar_bounds_and_failed_candidate_cleanup) {
  PolyCtx *ctx = poly_ctx_new();
  PolyParamArg arg = {
      .slot = 1,
      .addrspace = POLY_ADDR_ALU,
      .name = "n",
      .min_val = INT64_MIN,
      .max_val = INT64_MAX,
      .has_minmax = true};
  PolyUOp *variable = poly_uop0(ctx, POLY_OP_PARAM, POLY_INT64, poly_arg_param(&arg));
  int n_args = 0;
  void **args = poly_test_beam_args_from_ast(ctx, variable, &n_args);
  ASSERT_NOT_NULL(args);
  ASSERT_INT_EQ(*(int *)args[0], -1);
  free(args[0]);
  free(args);
  PolyUOp *buffer = poly_test_program_param(ctx, POLY_FLOAT64, 8, 0);
  arg.min_val = arg.max_val = INT64_MAX;
  variable = poly_uop0(ctx, POLY_OP_PARAM, POLY_INT64, poly_arg_param(&arg));
  PolyUOp *src[] = {buffer, variable};
  PolyUOp *sink = poly_test_kernel_sink(ctx, src, 2, "beam_unrepresentable_scalar");
  args = poly_test_beam_args_from_ast(ctx, sink, &n_args);
  ASSERT_TRUE(args == NULL);
  ASSERT_INT_EQ(n_args, 0);
  /* Failure releases the already allocated buffer; a later candidate works. */
  args = poly_test_beam_args_from_ast(ctx, buffer, &n_args);
  ASSERT_NOT_NULL(args);
  free(args[0]);
  free(args);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, beam_parameter_inventory_is_not_fixed_at_64) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *stores[65];
  for (int i = 0; i < 65; i++) {
    PolyUOp *param = poly_test_program_param(ctx, POLY_FLOAT32, 1, i);
    PolyUOp *index = poly_uop_index(ctx, param, &zero, 1);
    stores[i] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, one, poly_arg_none());
  }
  PolyUOp *sink = poly_test_kernel_sink(ctx, stores, 65, "beam_many_params");
  PolyRewriteOpts opts = {.caps = poly_c_renderer_caps()};
  int n_args = 0;
  double elapsed = poly_test_beam_compile_and_time(ctx, sink, opts, 1, &n_args);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_args, 65);
  ASSERT_TRUE(isfinite(elapsed) && elapsed >= 0);
  PASS();
}

TEST(codegen, beam_candidate_uses_selected_backend) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_WEAK, 8);
  int n_args = 0;
  double elapsed = poly_test_beam_compile_and_time(
      ctx, sink,
      (PolyRewriteOpts
      ){.device = POLY_DEVICE_INTERP, .caps = {.device = "PYTHON", .has_int64 = true}},
      1, &n_args
  );
  int device = poly_test_beam_last_device();
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(isfinite(elapsed));
  ASSERT_INT_EQ(n_args, 1);
  ASSERT_INT_EQ(device, POLY_DEVICE_INTERP);
  PASS();
}

TEST(codegen, beam_cache_key_covers_width_and_device) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_WEAK, 8);
  uint64_t base = poly_test_beam_cache_key(ctx, sink, 1, POLY_DEVICE_CPU);
  uint64_t width = poly_test_beam_cache_key(ctx, sink, 2, POLY_DEVICE_CPU);
  uint64_t device = poly_test_beam_cache_key(ctx, sink, 1, POLY_DEVICE_INTERP);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(base != width && base != device);
  PASS();
}

TEST(codegen, beam_cache_key_covers_graph_edges) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_const_int(ctx, 2), *b = poly_const_int(ctx, 3);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *left = poly_alu2(ctx, POLY_OP_MUL, sum, a);
  PolyUOp *right = poly_alu2(ctx, POLY_OP_MUL, sum, b);
  PolyUOp *x = poly_test_kernel_sink(ctx, &left, 1, "test");
  PolyUOp *y = poly_test_kernel_sink(ctx, &right, 1, "test");
  uint64_t xkey = poly_test_beam_cache_key(ctx, x, 1, POLY_DEVICE_CPU);
  uint64_t ykey = poly_test_beam_cache_key(ctx, y, 1, POLY_DEVICE_CPU);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(xkey != ykey);
  PASS();
}

TEST(codegen, beam_binary_identity_is_independent_of_temporary_path) {
  const char *source = "void test_call(void **args) { *(float *)args[0] = 7; }";
  PolyProgram *a = poly_compile_c(source, "test");
  PolyProgram *b = poly_compile_c(source, "test");
  int na = 0, nb = 0;
  uint8_t *ba = a ? poly_program_read_binary(a, &na) : NULL;
  uint8_t *bb = b ? poly_program_read_binary(b, &nb) : NULL;
  bool equal = ba && bb && na > 0 && na == nb && !memcmp(ba, bb, (size_t)na);
  free(ba);
  free(bb);
  poly_program_destroy(a);
  poly_program_destroy(b);
  ASSERT_TRUE(equal);
  PASS();
}

TEST(codegen, beam_cache_replays_long_history_and_rejects_truncation) {
  char directory[] = "temp/beam-cache-test.XXXXXX";
  ASSERT_NOT_NULL(mkdtemp(directory));
  const char *previous = getenv("XDG_CACHE_HOME");
  char *saved = previous ? strdup(previous) : NULL;
  ASSERT_TRUE(!previous || saved);
  setenv("XDG_CACHE_HOME", directory, 1);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_GLOBAL, 128), *result = sink;
  PolyOpt opt = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 2};
  for (int i = 0; result && i < 7; i++)
    result = poly_test_apply_opt(ctx, result, (PolyRendererCaps){.device = "CPU"}, opt);
  char path[600] = {0};
  bool correct = result && result->arg.kernel_info->n_applied_opts == 7 &&
                 poly_test_beam_cache_write(ctx, sink, result, 17, path, sizeof(path)) == 0;
  correct &= poly_test_beam_cache_read(ctx, sink, 17) == result;
  FILE *file = path[0] ? fopen(path, "wb") : NULL;
  if (file) {
    fputc(0, file);
    fclose(file);
  }
  correct &= file != NULL && poly_test_beam_cache_read(ctx, sink, 17) == NULL;
  correct &= sink->arg.kernel_info->n_applied_opts == 0;
  poly_ctx_destroy(ctx);
  if (path[0]) remove(path);
  char subdir[600];
  snprintf(subdir, sizeof(subdir), "%s/polygrad/beam", directory);
  rmdir(subdir);
  snprintf(subdir, sizeof(subdir), "%s/polygrad", directory);
  rmdir(subdir);
  rmdir(directory);
  if (saved)
    setenv("XDG_CACHE_HOME", saved, 1);
  else
    unsetenv("XDG_CACHE_HOME");
  free(saved);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(codegen, beam_time_call_owns_binary_and_program_without_runtime_cache) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = beam_action_sink(ctx, POLY_AXIS_WEAK, 8);
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 8, 0);
  PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, sink, out, poly_arg_none());
  PolyRunner runner;
  ASSERT_INT_EQ(poly_time_call_prepare(ctx, call, POLY_DEVICE_CPU, &runner), 0);
  ASSERT_NOT_NULL(runner.compiled_binary);
  ASSERT_INT_EQ(runner.compiled_binary->op, POLY_OP_BINARY);
  ASSERT_INT_EQ(poly_runtime_cache_len(ctx), 0);
  ASSERT_INT_EQ(poly_to_program_cache_len(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  float values[8] = {0};
  void *args[] = {values};
  double time = poly_time_call(&runner, POLY_DEVICE_CPU, args, 1, NULL, 0, 3, INFINITY, 65536);
  bool correct = isfinite(time) && !runner.wait;
  for (int i = 0; i < 8; i++)
    correct &= values[i] == (float)i;
  poly_time_call_finish(ctx, &runner, POLY_DEVICE_CPU);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}
#endif

TEST(codegen, c_render_parameter_inventory_is_not_fixed_at_64) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *params[65];
  for (int i = 0; i < 65; i++) {
    PolyParamArg arg = {.slot = i, .addrspace = POLY_ADDR_GLOBAL};
    params[i] = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&arg));
  }
  char *source = poly_render_c(ctx, params, 65, "many_params");
  bool accepted = source && strstr(source, "data64");
  free(source);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(accepted);
  PASS();
}

#ifdef POLY_HAS_CUDA
TEST(codegen, cuda_render_parameter_inventory_is_not_fixed_at_64) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *params[65];
  for (int i = 0; i < 65; i++) {
    PolyParamArg arg = {.slot = i, .addrspace = POLY_ADDR_GLOBAL};
    params[i] = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&arg));
  }
  char *source = poly_render_cuda(ctx, params, 65, "many_params", 1);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(strstr(source, "data64"));
  free(source);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, cuda_render_failure_releases_owned_scratch) {
  /* C cleanup for CStyleLanguage._render's rejected WMMA. Include a named
   * parameter so failure must release both parameter strings and scratch. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyParamArg arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&arg));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_WMMA, POLY_FLOAT32, poly_arg_none());
  PolyUOp *ops[] = {param, invalid};
  char *failed = poly_render_cuda(ctx, ops, 2, "rejected", 1);
  bool rejected = !failed;
  free(failed);
  char *retry = poly_render_cuda(ctx, ops, 1, "retry", 1);
  bool recovered = retry && strstr(retry, "__launch_bounds__(1) retry(");
  free(retry);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rejected && recovered);
  PASS();
}
#endif

static uint64_t topology_fnv_bytes(uint64_t h, const void *data, size_t n) {
  const uint8_t *bytes = (const uint8_t *)data;
  for (size_t i = 0; i < n; i++) {
    h ^= bytes[i];
    h *= UINT64_C(1099511628211);
  }
  return h;
}

static uint64_t topology_fnv_u32(uint64_t h, uint32_t value) {
  for (int i = 0; i < 4; i++) {
    uint8_t byte = (uint8_t)(value >> (i * 8));
    h = topology_fnv_bytes(h, &byte, 1);
  }
  return h;
}

static uint64_t topology_fnv_u64(uint64_t h, uint64_t value) {
  for (int i = 0; i < 8; i++) {
    uint8_t byte = (uint8_t)(value >> (i * 8));
    h = topology_fnv_bytes(h, &byte, 1);
  }
  return h;
}

/* Cross-language structural fingerprint used by the pinned direct-UOp
 * transcendental probes. PARAM shape/metadata is intentionally excluded:
 * PG-PARITY-002 tracks that independent vocabulary migration. */
static uint64_t normalized_topology_fingerprint(
    PolyCtx *ctx,
    PolyUOp **topo,
    int n_topo,
    PolyUOp *root
) {
  uint64_t *hashes = calloc((size_t)n_topo, sizeof(*hashes));
  if (!hashes) return 0;
  uint64_t root_hash = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    PolyDType scalar = u->dtype;
    uint8_t category = poly_dtype_eq(scalar, POLY_BOOL)   ? 1
                       : poly_dtype_is_unsigned(scalar)   ? 3
                       : poly_dtype_is_int(scalar)        ? 2
                       : poly_dtype_is_float(scalar)      ? 4
                       : poly_dtype_eq(scalar, POLY_VOID) ? 5
                                                          : 0;
    uint64_t h = UINT64_C(1469598103934665603);
    const char *op_name = poly_op_name(u->op);
    h = topology_fnv_bytes(h, op_name, strlen(op_name) + 1);
    h = topology_fnv_bytes(h, &category, 1);
    uint8_t bits = (uint8_t)scalar.bitsize;
    h = topology_fnv_bytes(h, &bits, 1);
    h = topology_fnv_u32(h, (uint32_t)poly_uop_max_numel(ctx, u));

    uint8_t arg_tag = 0;
    uint64_t arg_value = 0;
    if (u->op == POLY_OP_CONST) {
      if (category == 4 && u->arg.kind == POLY_ARG_FLOAT) {
        arg_tag = 2;
        if (isnan(u->arg.f)) {
          arg_value = scalar.bitsize == 64 ? UINT64_C(0x7ff8000000000000) : UINT64_C(0x7fc00000);
        } else if (scalar.bitsize == 64) {
          memcpy(&arg_value, &u->arg.f, sizeof(arg_value));
        } else {
          float value = (float)u->arg.f;
          uint32_t value_bits = 0;
          memcpy(&value_bits, &value, sizeof(value_bits));
          arg_value = value_bits;
        }
      } else if (category == 1 && u->arg.kind == POLY_ARG_BOOL) {
        arg_tag = 3;
        arg_value = u->arg.b ? 1 : 0;
      } else if (u->arg.kind == POLY_ARG_INT || u->arg.kind == POLY_ARG_BIGINT) {
        arg_tag = 1;
        arg_value = poly_arg_integer_to_u64_mod(u->arg);
        if (scalar.bitsize > 0 && scalar.bitsize < 64)
          arg_value &= (UINT64_C(1) << scalar.bitsize) - 1;
      }
    }
    h = topology_fnv_bytes(h, &arg_tag, 1);
    h = topology_fnv_u64(h, arg_value);

    int n_src = u->op == POLY_OP_PARAM ? 0 : u->n_src;
    h = topology_fnv_u32(h, (uint32_t)n_src);
    for (int s = 0; s < n_src; s++) {
      uint64_t child_hash = 0;
      for (int j = 0; j < i; j++) {
        if (topo[j] == u->src[s]) {
          child_hash = hashes[j];
          break;
        }
      }
      h = topology_fnv_u64(h, child_hash);
    }
    hashes[i] = h;
    if (u == root) root_hash = h;
  }
  free(hashes);
  return root_hash;
}

/* Helper: build c[i] = a[i] OP b[i] kernel IR */

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
  int n; /* loop bound */
} VecKernel;

/* Current Tinygrad final programs use scalar PARAM(dtype, shape, ParamArg),
 * never pointer-typed PARAM(arg=slot). */
static PolyUOp *program_param(PolyCtx *ctx, PolyDType dtype, int64_t numel, int slot) {
  PolyUOp *shape = poly_const_int(ctx, numel);
  PolyParamArg arg = {
      .slot = slot,
      .dtype = dtype,
      .addrspace = POLY_ADDR_GLOBAL,
  };
  return poly_uop1(ctx, POLY_OP_PARAM, dtype, shape, poly_arg_param(&arg));
}

static VecKernel make_vec_binop(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  /* Current UOp.placeholder/rangeify.debuf form: scalar PARAM values carry
   * storage extent in src[0] and ABI identity in ParamArg.slot. */
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(n));
  PolyParamArg args[3] = {
      {.slot = 0, .addrspace = POLY_ADDR_GLOBAL},
      {.slot = 1, .addrspace = POLY_ADDR_GLOBAL},
      {.slot = 2, .addrspace = POLY_ADDR_GLOBAL},
  };
  PolyUOp *p0 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&args[0]));
  PolyUOp *p1 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&args[1]));
  PolyUOp *p2 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&args[2]));

  /* loop: for Lidx0 in range(n) */
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(n));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_WEAK));

  /* index into buffers */
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p2, range, poly_arg_none());

  /* load */
  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());

  /* ALU */
  PolyUOp *alu = poly_uop2(ctx, alu_op, POLY_FLOAT32, load0, load1, poly_arg_none());

  /* store */
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());

  /* end loop + sink */
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  /* Current Tinygrad to_program requires KernelInfo on compilable SINKs. */
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "test");

  return (VecKernel){ctx, sink, n};
}

static int count_lin_ops(PolyUOp **lin, int n, PolyOps op) {
  int c = 0;
  for (int i = 0; i < n; i++)
    if (lin[i]->op == op) c++;
  return c;
}

static int count_reg_storage_ops(PolyUOp **lin, int n) {
  int c = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = lin[i];
    if (u->op == POLY_OP_BUFFER && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
        u->arg.param->addrspace == POLY_ADDR_REG)
      c++;
  }
  return c;
}

static int count_special_named(PolyUOp **lin, int n, const char *name) {
  int c = 0;
  for (int i = 0; i < n; i++) {
    if (lin[i]->op == POLY_OP_SPECIAL && lin[i]->arg.str && strcmp(lin[i]->arg.str, name) == 0) c++;
  }
  return c;
}

static int64_t special_bound_hi_named(PolyUOp **lin, int n, const char *name) {
  for (int i = 0; i < n; i++) {
    if (lin[i]->op != POLY_OP_SPECIAL || !lin[i]->arg.str || strcmp(lin[i]->arg.str, name) != 0 ||
        lin[i]->n_src <= 0)
      continue;
    int64_t lo = 0, hi = 0;
    poly_uop_minmax(NULL, lin[i]->src[0], &lo, &hi);
    return hi;
  }
  return -1;
}

static VecKernel make_vec_copy_with_weak_index_expr(int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = program_param(ctx, POLY_FLOAT32, n, 0);
  PolyUOp *p1 = program_param(ctx, POLY_FLOAT32, n, 1);

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(n));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_LOOP));

  /* tinygrad carries address expressions as weakint through most of codegen,
   * then pm_lower_index_dtype narrows them to a concrete integer dtype. This
   * test mirrors that path; explicit user casts to int64 are intentionally not
   * stripped by tinygrad and are not part of this invariant. */
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *addr = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, range, zero, poly_arg_none());

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, addr, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, range, poly_arg_none());
  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, load0, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  return (VecKernel){ctx, sink, n};
}

static bool subtree_has_i64_dtype(PolyUOp *u) {
  if (!u) return false;
  PolyDType scalar = u->dtype;
  if (poly_dtype_is_int(scalar) && scalar.bitsize == 64) return true;
  for (int i = 0; i < u->n_src; i++)
    if (subtree_has_i64_dtype(u->src[i])) return true;
  return false;
}

static int count_indexes_with_i64_addr(PolyUOp **nodes, int n) {
  int bad = 0;
  for (int i = 0; i < n; i++) {
    if (nodes[i]->op != POLY_OP_INDEX || nodes[i]->n_src < 2) continue;
    if (subtree_has_i64_dtype(nodes[i]->src[1])) bad++;
    if (nodes[i]->n_src >= 3 && subtree_has_i64_dtype(nodes[i]->src[2])) bad++;
  }
  return bad;
}

static PolyUOp *make_shaped_f32_buf(PolyCtx *ctx, const int64_t *shape, int ndim) {
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++)
    numel *= shape[i];
  PolyUOp *buf = poly_buffer_f32(ctx, numel);
  return ndim > 1 ? poly_reshape(ctx, buf, (int64_t *)shape, ndim) : buf;
}

static int wgsl_workgroup_product(const char *wgsl, int dims[3]) {
  dims[0] = dims[1] = dims[2] = 1;
  const char *p = strstr(wgsl, "@workgroup_size(");
  if (!p) return 1;
  p += strlen("@workgroup_size(");
  for (int i = 0; i < 3 && *p; i++) {
    dims[i] = atoi(p);
    const char *comma = strchr(p, ',');
    const char *close = strchr(p, ')');
    if (!comma || (close && close < comma)) break;
    p = comma + 1;
  }
  return dims[0] * dims[1] * dims[2];
}

/* Linearizer tests */

TEST(codegen, linearize_order) {
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);

  ASSERT_TRUE(n >= 6);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_PARAM), 3);

  /* SINK should be last */
  ASSERT_TRUE(lin[n - 1]->op == POLY_OP_SINK);

  /* END should come just before SINK */
  ASSERT_TRUE(lin[n - 2]->op == POLY_OP_END);

  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, linearize_deps) {
  /* Verify: every UOp's sources appear before it in the linearized list */
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < lin[i]->n_src; j++) {
      PolyUOp *src = lin[i]->src[j];
      bool found = false;
      for (int k = 0; k < i; k++) {
        if (lin[k] == src) {
          found = true;
          break;
        }
      }
      ASSERT_TRUE(found);
    }
  }

  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, split_ends_preserves_nested_end_backedge) {
  /* Pinned do_split_ends excludes void backedges from range collection,
   * then reattaches them. An inner END is an effect, not a removable range. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(3, POLY_AXIS_LOOP));
  PolyUOp *body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *outer_body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_str("outer"));
  PolyUOp *inner_srcs[2] = {body, range};
  PolyUOp *inner = poly_uop(ctx, POLY_OP_END, POLY_VOID, inner_srcs, 2, poly_arg_none());
  PolyUOp *outer_srcs[2] = {outer_body, inner};
  PolyUOp *outer = poly_uop(ctx, POLY_OP_END, POLY_VOID, outer_srcs, 2, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, outer, poly_pm_split_ends());
  ASSERT_PTR_EQ(rewritten, outer);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, split_ends_retains_ranges_still_active_in_dependency) {
  /* An arithmetic dependency on RANGE rebuilds exactly END(body, r). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(3, POLY_AXIS_LOOP));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *active = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, range, one, poly_arg_none());
  PolyUOp *body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_str("outer"));
  PolyUOp *end_srcs[2] = {body, active};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, end, poly_pm_split_ends());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_END);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  ASSERT_TRUE(rewritten->src[0] == body);
  ASSERT_TRUE(rewritten->src[1] == range);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, split_ends_preserves_predicate_and_effect_order) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r0 = poly_range(ctx, 2, 0, POLY_AXIS_LOOP);
  PolyUOp *r1 = poly_range(ctx, 3, 1, POLY_AXIS_LOOP);
  PolyUOp *body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *predicate = poly_alu2(ctx, POLY_OP_CMPLT, r0, poly_const_int(ctx, 1));
  PolyUOp *effect =
      poly_uop1(ctx, POLY_OP_NOOP, POLY_VOID, poly_const_int(ctx, 7), poly_arg_none());
  PolyUOp *pred_end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, body, predicate, poly_arg_none());
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, pred_end, poly_pm_split_ends()), pred_end);
  PolyUOp *effect_end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, body, effect, poly_arg_none());
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, effect_end, poly_pm_split_ends()), effect_end);
  PolyUOp *src[] = {body, r1, predicate, r0, effect};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, src, 5, poly_arg_none());
  PolyUOp *inner = poly_uop2(ctx, POLY_OP_END, POLY_VOID, body, r1, poly_arg_none());
  inner = poly_uop2(ctx, POLY_OP_END, POLY_VOID, inner, r0, poly_arg_none());
  PolyUOp *expected_src[] = {inner, predicate, effect};
  PolyUOp *expected = poly_uop(ctx, POLY_OP_END, POLY_VOID, expected_src, 3, poly_arg_none());
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, end, poly_pm_split_ends()), expected);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, split_ends_orders_full_range_arguments) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_const_int(ctx, 3);
  int64_t high[] = {9}, low[] = {1, 0};
  PolyUOp *rhigh = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range_ex(7, POLY_AXIS_LOOP, high, 1)
  );
  PolyUOp *rlow = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range_ex(7, POLY_AXIS_LOOP, low, 2)
  );
  PolyUOp *rshort = poly_range(ctx, 3, 7, POLY_AXIS_LOOP);
  PolyUOp *body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *src[] = {body, rhigh, rlow, rshort};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, src, 4, poly_arg_none());
  PolyUOp *expected = poly_uop2(ctx, POLY_OP_END, POLY_VOID, body, rhigh, poly_arg_none());
  expected = poly_uop2(ctx, POLY_OP_END, POLY_VOID, expected, rlow, poly_arg_none());
  expected = poly_uop2(ctx, POLY_OP_END, POLY_VOID, expected, rshort, poly_arg_none());
  PolyUOp *actual = poly_graph_rewrite(ctx, end, poly_pm_split_ends());
  bool same = actual == expected;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(same);
  PASS();
}

TEST(codegen, split_ends_does_not_merge_sibling_effects) {
  /* tinygrad@2026-08-22/a9069c177a9d codegen/late/linearizer.py:87-95
   * rewrites END only; sibling effects remain separate SINK sources. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(3, POLY_AXIS_LOOP));
  PolyUOp *left = poly_uop0(ctx, POLY_OP_CUSTOM, POLY_VOID, poly_arg_str("left"));
  PolyUOp *right = poly_uop0(ctx, POLY_OP_CUSTOM, POLY_VOID, poly_arg_str("right"));
  PolyUOp *left_end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, left, range, poly_arg_none());
  PolyUOp *right_end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, right, range, poly_arg_none());
  PolyUOp *sink_src[2] = {left_end, right_end};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sink_src, 2, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, sink, poly_pm_split_ends());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_SINK);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  ASSERT_INT_EQ(rewritten->src[0]->op, POLY_OP_END);
  ASSERT_INT_EQ(rewritten->src[1]->op, POLY_OP_END);
  ASSERT_PTR_EQ(rewritten->src[0]->src[0], left);
  ASSERT_PTR_EQ(rewritten->src[1]->src[0], right);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_merge_shared_end) {
  /* Current rangeify PARAM/REDUCE form: two reductions over one RANGE share
   * one merged END while retaining distinct REG placeholders. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *eight = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyParamArg args[3] = {
      {.slot = 0, .addrspace = POLY_ADDR_GLOBAL},
      {.slot = 1, .addrspace = POLY_ADDR_GLOBAL},
      {.slot = 2, .addrspace = POLY_ADDR_GLOBAL},
  };
  PolyUOp *pin = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, eight, poly_arg_param(&args[0]));
  PolyUOp *pout0 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, one, poly_arg_param(&args[1]));
  PolyUOp *pout1 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, one, poly_arg_param(&args[2]));
  PolyUOp *r0 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, eight, poly_arg_range(0, POLY_AXIS_REDUCE));

  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, pin, r0, poly_arg_none());
  PolyUOp *in_ld = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());

  PolyUOp *red0_srcs[2] = {in_ld, r0};
  PolyUOp *red1_srcs[2] = {in_ld, r0};
  PolyUOp *sum =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red0_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *mx =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red1_srcs, 2, poly_arg_reduce(POLY_OP_MAX, 0));

  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *out0_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, pout0, zero, poly_arg_none());
  PolyUOp *out1_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, pout1, zero, poly_arg_none());
  PolyUOp *st0 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out0_idx, sum, poly_arg_none());
  PolyUOp *st1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1_idx, mx, poly_arg_none());
  PolyUOp *stores[2] = {st0, st1};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  int n = 0;
  PolyRewriteOpts opts = {.optimize = false, .caps = poly_c_renderer_caps()};
  PolyUOp **lin = poly_test_full_rewrite_and_linearize_ex(ctx, sink, opts, &n);
  ASSERT_TRUE(n > 0);
  ASSERT_INT_EQ(count_reg_storage_ops(lin, n), 2);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_END), 1);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_merge_applies_independent_groups_atomically) {
  /* Pinned tinygrad/codegen/__init__.py:190-208 collects every
   * mergeable END replacement and applies the complete substitution map once.
   * Two independent reduce-range groups catch the order-dependent failure
   * where rewriting the first group invalidates pointer keys in the second. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *r0 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_REDUCE));
  PolyUOp *r1 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(1, POLY_AXIS_REDUCE));
  PolyUOp *v0 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, r0, poly_arg_none());
  PolyUOp *v1 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, r1, poly_arg_none());
  PolyUOp *red_srcs0[2] = {v0, r0};
  PolyUOp *red_srcs1[2] = {v1, r1};
  PolyUOp *roots[4] = {
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red_srcs0, 2, poly_arg_reduce(POLY_OP_ADD, 0)),
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red_srcs0, 2, poly_arg_reduce(POLY_OP_MAX, 0)),
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red_srcs1, 2, poly_arg_reduce(POLY_OP_ADD, 0)),
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red_srcs1, 2, poly_arg_reduce(POLY_OP_MAX, 0)),
  };
  PolyUOp *sink = poly_uop_sink(ctx, roots, 4);
  PolyUOp *rewritten = poly_apply_pm_reduce(ctx, sink);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  int n_end = 0, n_group = 0, n_range = 0;
  for (int i = 0; i < n_topo; i++) {
    n_end += topo[i]->op == POLY_OP_END;
    n_group += topo[i]->op == POLY_OP_GROUP;
    n_range += topo[i]->op == POLY_OP_RANGE;
    if (topo[i]->op == POLY_OP_END) {
      ASSERT_INT_EQ(topo[i]->n_src, 2);
      ASSERT_INT_EQ(topo[i]->src[0]->op, POLY_OP_GROUP);
      ASSERT_INT_EQ(topo[i]->src[0]->n_src, 2);
      ASSERT_INT_EQ(topo[i]->src[1]->op, POLY_OP_RANGE);
    }
  }
  ASSERT_INT_EQ(n_end, 2);
  ASSERT_INT_EQ(n_group, 2);
  ASSERT_INT_EQ(n_range, 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_merge_cloned_context_preserves_wide_metadata) {
  /* Pinned codegen/__init__.py:190-208 clones a shared reduction RANGE for a
   * second active-range context with one unrestricted
   * e.substitute(dict(zip(r, tr))).  The cloned path must retain every STACK
   * lane and the ancestor tag. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *r =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_REDUCE));
  PolyUOp *c0_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *c1_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *c0 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, c0_bound, poly_arg_range(1, POLY_AXIS_LOOP));
  PolyUOp *c1 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, c1_bound, poly_arg_range(2, POLY_AXIS_LOOP));

  PolyUOp *simple_add = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, r, c0, poly_arg_none());
  PolyUOp *simple_cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, simple_add, poly_arg_none());
  PolyUOp *simple_srcs[2] = {simple_cast, r};
  PolyUOp *simple =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, simple_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0));

  PolyUOp *wide_srcs[70];
  wide_srcs[0] = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, r, c1, poly_arg_none());
  for (int i = 1; i < 70; i++)
    wide_srcs[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(i));
  PolyDType wide_dtype = POLY_WEAKINT;
  PolyUOp *wide = poly_uop_tagged_arg(
      ctx, POLY_OP_STACK, wide_dtype, wide_srcs, 70, poly_arg_none(), 77,
      poly_arg_str("wide-survives")
  );
  PolyUOp *wide_reduce_srcs[2] = {wide, r};
  PolyUOp *wide_reduce = poly_uop(
      ctx, POLY_OP_REDUCE, wide_dtype, wide_reduce_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0)
  );
  PolyUOp *roots[2] = {simple, wide_reduce};
  PolyUOp *out = poly_apply_pm_reduce(ctx, poly_uop_sink(ctx, roots, 2));
  ASSERT_NOT_NULL(out);

  int n_topo = 0, tagged_wide = 0, max_axis = -1;
  PolyUOp **topo = poly_toposort(ctx, out, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_RANGE && u->arg.kind == POLY_ARG_RANGE &&
        poly_range_axis_id(u->arg) > max_axis)
      max_axis = (int)poly_range_axis_id(u->arg);
    if (u->op != POLY_OP_STACK || u->tag != 77) continue;
    tagged_wide++;
    ASSERT_INT_EQ(u->n_src, 70);
    ASSERT_INT_EQ(u->tag_arg.kind, POLY_ARG_STRING);
    ASSERT_STR_EQ(u->tag_arg.str, "wide-survives");
  }
  ASSERT_INT_EQ(tagged_wide, 1);
  ASSERT_TRUE(max_axis >= 3);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Scalar IR loop count is independent of Tensor's maximum shape rank. */
static PolyUOp *reduce_test_ranges(PolyCtx *ctx, int n_reduce, int n_outer) {
  PolyUOp *src[1 + 80];
  PolyUOp *value = NULL;
  for (int i = 0; i < n_reduce + n_outer; i++) {
    PolyUOp *r = poly_range(ctx, 2, i, i < n_reduce ? POLY_AXIS_REDUCE : POLY_AXIS_LOOP);
    if (i < n_reduce) src[i + 1] = r;
    PolyUOp *v = poly_cast(ctx, r, POLY_INT32);
    value = value ? poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, value, v, poly_arg_none()) : v;
  }
  src[0] = value;
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_INT32, src, n_reduce + 1, poly_arg_reduce(POLY_OP_ADD, 0));
  return poly_apply_pm_reduce(ctx, poly_uop_sink(ctx, &red, 1));
}

TEST(codegen, reduce_ranges_have_one_tagged_end) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = reduce_test_ranges(ctx, 2, 0);
  ASSERT_NOT_NULL(out);
  int n = 0, ends = 0;
  PolyUOp **topo = poly_toposort(ctx, out, &n);
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_END) continue;
    ends++;
    ASSERT_INT_EQ(u->n_src, 3);
    ASSERT_INT_EQ(u->src[0]->op, POLY_OP_STORE);
    ASSERT_INT_EQ(u->tag_arg.kind, POLY_ARG_STRING);
    ASSERT_STR_EQ(u->tag_arg.str, "mergeable");
    for (int j = 1; j < 3; j++)
      ASSERT_INT_EQ(poly_range_axis_id(u->src[j]->arg), j - 1);
  }
  ASSERT_INT_EQ(ends, 1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_weak_input_accumulates_in_strong_storage_dtype) {
  PolyDType dtypes[] = {POLY_WEAKINT, POLY_WEAKFLOAT};
  PolyOps ops[] = {POLY_OP_ADD, POLY_OP_MUL, POLY_OP_MAX};
  for (int d = 0; d < 2; d++)
    for (int op = 0; op < 3; op++) {
      PolyCtx *ctx = poly_ctx_new();
      PolyUOp *r = poly_range(ctx, 2, 0, POLY_AXIS_REDUCE);
      PolyUOp *src[] = {poly_cast(ctx, r, dtypes[d]), r};
      PolyUOp *red = poly_uop(ctx, POLY_OP_REDUCE, dtypes[d], src, 2, poly_arg_reduce(ops[op], 0));
      PolyUOp *out = poly_apply_pm_reduce(ctx, poly_uop_sink(ctx, &red, 1));
      ASSERT_NOT_NULL(out);
      int n = 0, updates = 0;
      PolyUOp **topo = poly_toposort(ctx, out, &n);
      for (int i = 0; i < n; i++) {
        PolyUOp *u = topo[i];
        if (u->op != POLY_OP_STORE || u->src[1]->op != ops[op]) continue;
        updates++;
        ASSERT_TRUE(poly_dtype_eq(u->src[1]->dtype, poly_dtype_strong(dtypes[d])));
        ASSERT_TRUE(poly_dtype_eq(u->src[0]->dtype, u->src[1]->dtype));
      }
      ASSERT_INT_EQ(updates, 1);
      poly_ctx_destroy(ctx);
    }
  PASS();
}

TEST(codegen, reduce_loop_count_exceeds_tensor_rank) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = reduce_test_ranges(ctx, POLY_MAX_DIMS + 1, 0);
  ASSERT_NOT_NULL(out);
  int n = 0, ends = 0;
  PolyUOp **topo = poly_toposort(ctx, out, &n);
  for (int i = 0; i < n; i++) {
    if (topo[i]->op != POLY_OP_END) continue;
    ends++;
    ASSERT_INT_EQ(topo[i]->n_src, POLY_MAX_DIMS + 2);
  }
  ASSERT_INT_EQ(ends, 1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_initialization_preserves_all_outer_ranges) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = reduce_test_ranges(ctx, 1, 65);
  ASSERT_NOT_NULL(out);
  int n = 0, initializers = 0;
  PolyUOp **topo = poly_toposort(ctx, out, &n);
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_STORE || u->src[1]->op != POLY_OP_CONST) continue;
    initializers++;
    ASSERT_INT_EQ(u->src[0]->op, POLY_OP_AFTER);
    ASSERT_INT_EQ(u->src[0]->n_src, 66);
    for (int j = 1; j < 66; j++)
      ASSERT_INT_EQ(poly_range_axis_id(u->src[0]->src[j]->arg), j);
  }
  ASSERT_INT_EQ(initializers, 1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_merge_discovers_tagged_ends_without_registration) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 2, 0, POLY_AXIS_REDUCE);
  PolyUOp *v = poly_cast(ctx, r, POLY_INT32);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *roots[2];
  for (int i = 0; i < 2; i++) {
    PolyUOp *body =
        poly_uop2(ctx, i ? POLY_OP_MUL : POLY_OP_ADD, POLY_INT32, v, one, poly_arg_none());
    PolyUOp *src[] = {body, r};
    roots[i] = poly_uop_tagged_arg(
        ctx, POLY_OP_END, POLY_VOID, src, 2, poly_arg_none(), 0, poly_arg_str("mergeable")
    );
  }
  PolyUOp *out = poly_apply_pm_reduce(ctx, poly_uop_sink(ctx, roots, 2));
  ASSERT_NOT_NULL(out);
  int n = 0, ends = 0;
  PolyUOp **topo = poly_toposort(ctx, out, &n);
  for (int i = 0; i < n; i++) {
    if (topo[i]->op != POLY_OP_END) continue;
    ends++;
    ASSERT_INT_EQ(topo[i]->src[0]->op, POLY_OP_GROUP);
    ASSERT_INT_EQ(topo[i]->src[0]->n_src, 2);
    ASSERT_TRUE(topo[i]->src[1] == r);
  }
  ASSERT_INT_EQ(ends, 1);
  ASSERT_TRUE(poly_apply_pm_reduce(ctx, out) == out);
  poly_ctx_destroy(ctx);
  PASS();
}

/* The two active-range sets agree through64 entries and differ at65. */
static PolyUOp *reduce_test_context_ends(PolyCtx *ctx) {
  int64_t extra[] = {91, 92};
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *r = poly_uop_tagged_arg(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, &bound, 1, poly_arg_range_ex(0, POLY_AXIS_REDUCE, extra, 2),
      17, poly_arg_str("range-metadata")
  );
  PolyUOp *shared = poly_cast(ctx, r, POLY_INT32);
  for (int i = 1; i <= 64; i++) {
    PolyUOp *v = poly_cast(ctx, poly_range(ctx, 2, i, POLY_AXIS_LOOP), POLY_INT32);
    shared = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, shared, v, poly_arg_none());
  }
  PolyUOp *ends[2];
  for (int i = 0; i < 2; i++) {
    PolyUOp *v = poly_cast(ctx, poly_range(ctx, 2, 65 + i, POLY_AXIS_LOOP), POLY_INT32);
    PolyUOp *body = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, shared, v, poly_arg_none());
    PolyUOp *src[] = {body, r};
    ends[i] = poly_uop_tagged_arg(
        ctx, POLY_OP_END, POLY_VOID, src, 2, poly_arg_none(), 0, poly_arg_str("mergeable")
    );
  }
  return poly_uop_sink(ctx, ends, 2);
}

TEST(codegen, reduce_merge_preserves_large_contexts_and_range_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_apply_pm_reduce(ctx, reduce_test_context_ends(ctx));
  ASSERT_NOT_NULL(out);
  int n = 0, ends = 0, cloned = 0;
  PolyUOp **topo = poly_toposort(ctx, out, &n);
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_END) continue;
    ends++;
    ASSERT_INT_EQ(u->src[0]->op, POLY_OP_ADD);
    PolyUOp *r = u->src[1];
    ASSERT_INT_EQ(r->tag, 17);
    ASSERT_STR_EQ(r->tag_arg.str, "range-metadata");
    ASSERT_INT_EQ(poly_range_n_extra(r->arg), 2);
    ASSERT_INT_EQ(poly_range_extra(r->arg)[0], 91);
    ASSERT_INT_EQ(poly_range_extra(r->arg)[1], 92);
    int64_t axis = poly_range_axis_id(r->arg);
    ASSERT_TRUE(axis == 0 || axis == 67);
    cloned += axis == 67;
    PolyUOp *active[66];
    ASSERT_INT_EQ(poly_uop_ranges(ctx, u, active, 66), 65);
  }
  ASSERT_INT_EQ(ends, 2);
  ASSERT_INT_EQ(cloned, 1);
  ASSERT_TRUE(poly_apply_pm_reduce(ctx, out) == out);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_scratch_failure_does_not_publish_partial_graph) {
  for (int kind = 0; kind < 2; kind++) {
    int failures = 0;
    bool completed = false;
    for (int allocation = 0; allocation < 32; allocation++) {
      PolyCtx *ctx = poly_ctx_new();
      PolyUOp *sink = kind ? reduce_test_context_ends(ctx) : NULL;
      poly_test_reduce_alloc_fail_after(allocation);
      PolyUOp *out = kind ? poly_apply_pm_reduce(ctx, sink) : reduce_test_ranges(ctx, 2, 0);
      poly_test_reduce_alloc_fail_after(-1);
      if (out)
        completed = true;
      else
        failures++;
      if (sink)
        ASSERT_NOT_NULL(poly_apply_pm_reduce(ctx, sink));
      else
        ASSERT_NOT_NULL(reduce_test_ranges(ctx, 2, 0));
      poly_ctx_destroy(ctx);
      if (completed) break;
    }
    ASSERT_TRUE(completed);
    ASSERT_TRUE(failures > 0);
  }
  PASS();
}

TEST(codegen, reduce_accumulator_uses_tinygrad_placeholder_topology) {
  /* Current codegen/__init__.py:210-220 and uop/ops.py:1138-1150 create
   * placeholder_like directly as scalar storage with flattened shape. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_REDUCE));
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, range, poly_arg_none());
  PolyUOp *reduce_srcs[2] = {value, range};
  PolyUOp *reduce =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_INT32, reduce_srcs, 2, poly_arg_reduce(POLY_OP_MAX, 0));
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *rewritten = poly_apply_pm_reduce(ctx, sink);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  PolyUOp *acc = NULL;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_BUFFER && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
        u->arg.param->addrspace == POLY_ADDR_REG)
      acc = u;
  }
  ASSERT_NOT_NULL(acc);
  ASSERT_TRUE(poly_dtype_eq(acc->dtype, POLY_INT32));
  ASSERT_INT_EQ(acc->arg.kind, POLY_ARG_PARAM);
  ASSERT_NOT_NULL(acc->arg.param);
  ASSERT_INT_EQ(acc->arg.param->slot, 0);
  ASSERT_INT_EQ(acc->arg.param->addrspace, POLY_ADDR_REG);
  ASSERT_INT_EQ(acc->n_src, 1);
  ASSERT_INT_EQ(acc->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(acc->src[0]->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(acc->src[0]->arg.i, 1);

  int accumulator_stores = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_STORE || u->n_src != 2 ||
        !(u->src[0] == acc || (u->src[0]->op == POLY_OP_AFTER && u->src[0]->src[0] == acc)))
      continue;
    accumulator_stores++;
  }
  ASSERT_INT_EQ(accumulator_stores, 2);

  /* Current pm_casted_consts only commits the weak shape literal. */
  PolyRewriteOpts opts = {
      .optimize = false,
  };
  opts.caps = poly_c_renderer_caps();
  PolyUOp *final = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(final);
  int n_final = 0;
  PolyUOp **final_topo = poly_toposort(ctx, final, &n_final);
  PolyUOp *final_acc = NULL;
  for (int i = 0; i < n_final; i++) {
    PolyUOp *u = final_topo[i];
    if (u->op == POLY_OP_BUFFER && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
        u->arg.param->addrspace == POLY_ADDR_REG)
      final_acc = u;
  }
  ASSERT_NOT_NULL(final_acc);
  ASSERT_INT_EQ(final_acc->n_src, 1);
  ASSERT_INT_EQ(final_acc->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(final_acc->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(final_acc->src[0]->n_src, 1);
  ASSERT_INT_EQ(final_acc->src[0]->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(final_acc->src[0]->src[0]->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(final_acc->src[0]->src[0]->arg.i, 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_vector_accumulator_preserves_placeholder_lane_shape) {
  /* Current placeholder_like flattens the shaped STACK to four scalar
   * elements in one REG buffer. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_REDUCE));
  PolyUOp *lanes[4];
  for (int i = 0; i < 4; i++) {
    PolyUOp *off = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(i));
    PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, range, off, poly_arg_none());
    lanes[i] = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, sum, poly_arg_none());
  }
  PolyUOp *value = poly_uop_stack(ctx, lanes, 4);
  PolyUOp *reduce_srcs[2] = {value, range};
  PolyUOp *reduce =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, reduce_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *rewritten =
      poly_apply_pm_reduce(ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none()));
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0, n_acc = 0;
  PolyUOp *acc = NULL;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_BUFFER || u->arg.kind != POLY_ARG_PARAM || !u->arg.param ||
        u->arg.param->addrspace != POLY_ADDR_REG)
      continue;
    n_acc++;
    acc = u;
  }
  ASSERT_INT_EQ(n_acc, 1);
  ASSERT_NOT_NULL(acc);
  ASSERT_TRUE(poly_dtype_eq(acc->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(acc->n_src, 1);
  PolyUOp *shape = acc->src[0];
  ASSERT_INT_EQ(shape->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(shape->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(shape->arg.i, 4);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_matrix_accumulator_restores_placeholder_shape) {
  /* Current codegen/__init__.py:210 and uop/ops.py:1138-1150 preserve the
   * REDUCE result's rank around flattened REG storage. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_REDUCE));
  PolyUOp *lanes[4];
  for (int i = 0; i < 4; i++) {
    PolyUOp *offset = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(i));
    PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, range, poly_arg_none());
    lanes[i] = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, value, offset, poly_arg_none());
  }
  PolyUOp *matrix = poly_reshape(ctx, poly_uop_stack(ctx, lanes, 4), (int64_t[]){2, 2}, 2);
  PolyUOp *reduce_src[2] = {matrix, range};
  PolyUOp *reduce =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, reduce_src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *rewritten =
      poly_apply_pm_reduce(ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none()));
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0, shaped_accumulators = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_RESHAPE || u->n_src < 1 || u->src[0]->op != POLY_OP_BUFFER ||
        u->src[0]->arg.kind != POLY_ARG_PARAM || !u->src[0]->arg.param ||
        u->src[0]->arg.param->addrspace != POLY_ADDR_REG)
      continue;
    shaped_accumulators++;
    ASSERT_INT_EQ(poly_uop_ndim(ctx, u), 2);
    const int64_t *shape = poly_uop_max_shape_dims(ctx, u);
    ASSERT_NOT_NULL(shape);
    ASSERT_INT_EQ(shape[0], 2);
    ASSERT_INT_EQ(shape[1], 2);
  }
  ASSERT_INT_EQ(shaped_accumulators, 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, horizontal_reduce_indexes_shaped_axis_like_tinygrad) {
  /* Current codegen/__init__.py:222-224 indexes every element of the leading
   * reduced shape and folds them with the reduction ALU. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *lanes[4];
  for (int i = 0; i < 4; i++)
    lanes[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(i + 1));
  PolyUOp *stack = poly_uop_stack(ctx, lanes, 4);
  PolyUOp *value = poly_reshape(ctx, stack, (int64_t[]){4, 1}, 2);
  PolyUOp *reduce =
      poly_uop1(ctx, POLY_OP_REDUCE, POLY_FLOAT32, value, poly_arg_reduce(POLY_OP_ADD, 1));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *reshape = poly_uop2(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reduce, one, poly_arg_none());
  ASSERT_NOT_NULL(reshape);
  ASSERT_INT_EQ(reshape->op, POLY_OP_RESHAPE);
  PolyUOp *rewritten =
      poly_apply_pm_reduce(ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reshape, poly_arg_none()));
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0, n_add = 0, n_reduce = 0, n_reshape = 0;
  PolyUOp *remaining_reshape = NULL;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    n_add += topo[i]->op == POLY_OP_ADD;
    n_reduce += topo[i]->op == POLY_OP_REDUCE;
    if (topo[i]->op == POLY_OP_RESHAPE) {
      n_reshape++;
      remaining_reshape = topo[i];
    }
  }
  ASSERT_INT_EQ(n_add, 3);
  ASSERT_INT_EQ(n_reduce, 0);
  ASSERT_INT_EQ(n_reshape, 1);
  ASSERT_PTR_EQ(remaining_reshape, value);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, store_broadcast_is_devectorized_like_current_tinygrad) {
  /* Current pm_expand_broadcast computes the common source shape before
   * devectorizer2 scalarizes STORE((), (1,)) to one scalar STORE. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *dst = program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *target = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, dst, zero, poly_arg_none());
  PolyUOp *a_ptr = program_param(ctx, POLY_FLOAT32, 1, 1);
  PolyUOp *b_ptr = program_param(ctx, POLY_FLOAT32, 1, 2);
  PolyUOp *a_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, a_ptr, zero, poly_arg_none());
  PolyUOp *b_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, b_ptr, zero, poly_arg_none());
  PolyUOp *a = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, a_index, poly_arg_none());
  PolyUOp *b = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, b_index, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *value = poly_uop2(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, add, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, target, value, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  PolyRewriteOpts opts = {
      .optimize = false,
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0, n_reshape = 0, n_store = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    n_reshape += topo[i]->op == POLY_OP_RESHAPE;
    if (topo[i]->op == POLY_OP_STORE) {
      n_store++;
      ASSERT_INT_EQ(topo[i]->n_src, 2);
      ASSERT_TRUE(topo[i]->src[1]->op != POLY_OP_RESHAPE);
    }
  }
  ASSERT_INT_EQ(n_store, 1);
  ASSERT_INT_EQ(n_reshape, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Renderer tests */

TEST(codegen, renderers_size_register_buffer_from_casted_shape_const) {
  /* Current cstyle.py:169-173 renders BUFFER storage with max_numel().  The
   * final pm_casted_consts pass wraps its concrete shape CONST in CAST. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *weak_twelve = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(12));
  PolyUOp *twelve = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, weak_twelve, poly_arg_none());
  PolyParamArg reg_arg = {.slot = 0, .addrspace = POLY_ADDR_REG};
  PolyUOp *reg = poly_uop1(ctx, POLY_OP_BUFFER, POLY_FLOAT32, twelve, poly_arg_param(&reg_arg));
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reg, poly_arg_none());
  PolyUOp *uops[] = {weak_twelve, twelve, reg, sink};

  char *c = poly_render_c(ctx, uops, 4, "reg_extent");
  char *cuda = poly_render_cuda(ctx, uops, 4, "reg_extent", 1);
#ifdef POLY_HAS_HIP
  char *hip = poly_render_hip(ctx, uops, 4, "reg_extent", 1, "gfx1100");
#endif
  char *wgsl = poly_render_wgsl(ctx, uops, 4, "reg_extent");
  ASSERT_NOT_NULL(c);
  ASSERT_NOT_NULL(cuda);
#ifdef POLY_HAS_HIP
  ASSERT_NOT_NULL(hip);
#endif
  ASSERT_NOT_NULL(wgsl);
  ASSERT_NOT_NULL(strstr(c, "float r0[12];"));
  ASSERT_NOT_NULL(strstr(cuda, "float r0[12];"));
#ifdef POLY_HAS_HIP
  ASSERT_NOT_NULL(strstr(hip, "float r0[12];"));
#endif
  ASSERT_NOT_NULL(strstr(wgsl, "var r0: array<f32,12>;"));

  free(c);
  free(cuda);
#ifdef POLY_HAS_HIP
  free(hip);
#endif
  free(wgsl);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_range_names_preserve_full_argument_tuple) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *bound8 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *bound4 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  int64_t outer_extra[] = {0, 1};
  int64_t inner_extra[] = {1};
  PolyUOp *outer = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INT32, bound8, poly_arg_range_ex(1, POLY_AXIS_LOOP, outer_extra, 2)
  );
  PolyUOp *inner = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INT32, bound4, poly_arg_range_ex(1, POLY_AXIS_LOOP, inner_extra, 1)
  );
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *flat = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, outer, four, poly_arg_none()), inner, poly_arg_none()
  );
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, flat, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, one, poly_arg_none());
  PolyUOp *end_inner = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, inner, poly_arg_none());
  PolyUOp *end_outer = poly_uop2(ctx, POLY_OP_END, POLY_VOID, end_inner, outer, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end_outer, poly_arg_none());
  PolyUOp *linear[] = {
      out,  bound8, outer, bound4, inner,     four,      flat->src[0],
      flat, idx,    one,   store,  end_inner, end_outer, sink,
  };

  int n_linear = (int)(sizeof(linear) / sizeof(linear[0]));
  char *sources[4] = {
      poly_render_c(ctx, linear, n_linear, "range_identity"),
      poly_render_cuda(ctx, linear, n_linear, "range_identity", 1),
#ifdef POLY_HAS_HIP
      poly_render_hip(ctx, linear, n_linear, "range_identity", 1, "gfx1100"),
#else
      NULL,
#endif
      poly_render_wgsl(ctx, linear, n_linear, "range_identity"),
  };
  for (int i = 0; i < 4; i++) {
    if (i == 2 && !sources[i]) continue;
    ASSERT_NOT_NULL(sources[i]);
    ASSERT_NOT_NULL(strstr(sources[i], "Lidx1_0_1"));
    ASSERT_NOT_NULL(strstr(sources[i], "Lidx1_1"));
    free(sources[i]);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_sparse_param_slots_use_compact_call_abi) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = program_param(ctx, POLY_FLOAT32, 2, 0);
  PolyUOp *left = program_param(ctx, POLY_FLOAT32, 1, 2);
  PolyUOp *right = program_param(ctx, POLY_FLOAT32, 1, 3);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *one = poly_const_int(ctx, 1);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *out_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, range, poly_arg_none());
  PolyUOp *left_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, left, zero, poly_arg_none());
  PolyUOp *right_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, right, zero, poly_arg_none());
  PolyUOp *left_value = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, left_index, poly_arg_none());
  PolyUOp *right_value = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, right_index, poly_arg_none());
  PolyUOp *gate = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, range, one, poly_arg_none());
  PolyUOp *value =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, gate, left_value, right_value, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_index, value, poly_arg_none());
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, range, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "sparse_param_abi");

  int n_linear = 0;
  PolyUOp **linear = poly_test_full_rewrite_and_linearize(ctx, sink, &n_linear);
  ASSERT_NOT_NULL(linear);
  char *source = poly_render_c(ctx, linear, n_linear, "sparse_param_abi");
  ASSERT_NOT_NULL(source);
  /* tinygrad@2026-08-22/a9069c177a9d ProgramInfo.from_sink passes globals
   * `(0,2,3)` as three compact runtime arguments. */
  ASSERT_NOT_NULL(strstr(source, "(float*)args[0]"));
  ASSERT_NOT_NULL(strstr(source, "(float*)args[1]"));
  ASSERT_NOT_NULL(strstr(source, "(float*)args[2]"));
  ASSERT_TRUE(strstr(source, "args[3]") == NULL);

  free(source);
  free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_vecadd) {
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  const char *old_expand_ssa = getenv("EXPAND_SSA");
  char *saved_expand_ssa = old_expand_ssa ? strdup(old_expand_ssa) : NULL;
  unsetenv("EXPAND_SSA");

  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(k.ctx, lin, n, "vecadd");

  /* Pinned cstyle.py:194,232-237 inlines a single-consumer ALU unless
   * EXPAND_SSA is enabled. */
  ASSERT_NOT_NULL(strstr(src, "void vecadd("));
  ASSERT_NOT_NULL(strstr(src, "float* restrict data0"));
  ASSERT_NOT_NULL(strstr(src, "float* restrict data1"));
  ASSERT_NOT_NULL(strstr(src, "float* restrict data2"));
  ASSERT_NOT_NULL(strstr(src, "for (int Lidx0 = 0; Lidx0 < 10; Lidx0++)"));
  ASSERT_NOT_NULL(strstr(src, "float val0"));
  ASSERT_NOT_NULL(strstr(src, "float val1"));
  ASSERT_TRUE(strstr(src, "float alu0") == NULL);
  /* wrapper function */
  ASSERT_NOT_NULL(strstr(src, "void vecadd_call(void **args)"));

  free(src);

  setenv("EXPAND_SSA", "1", 1);
  src = poly_render_c(k.ctx, lin, n, "vecadd_expanded");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "float alu0"));

  if (saved_expand_ssa)
    setenv("EXPAND_SSA", saved_expand_ssa, 1);
  else
    unsetenv("EXPAND_SSA");
  free(saved_expand_ssa);
  free(lin);
  free(src);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, render_half_single_consumer_chain_matches_pinned_c_expression) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *old_expand_ssa = getenv("EXPAND_SSA");
  char *saved_expand_ssa = old_expand_ssa ? strdup(old_expand_ssa) : NULL;
  unsetenv("EXPAND_SSA");

  PolyUOp *a = poly_test_uop_param(ctx, POLY_FLOAT16, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *b = poly_test_uop_param(ctx, POLY_FLOAT16, -1, 1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_test_uop_param(ctx, POLY_FLOAT16, -1, 2, POLY_ADDR_GLOBAL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *a_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT16, a, zero, poly_arg_none());
  PolyUOp *b_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT16, b, zero, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT16, out, zero, poly_arg_none());
  PolyUOp *a_load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, a_idx, poly_arg_none());
  PolyUOp *b_load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, b_idx, poly_arg_none());
  PolyUOp *scale = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(1.702));
  PolyUOp *inner = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, b_load, scale, poly_arg_none());
  PolyUOp *value = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, a_load, inner, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, value, poly_arg_none());
  PolyUOp *linear[] = {
      a, b, out, zero, a_idx, b_idx, out_idx, a_load, b_load, scale, inner, value, store,
  };

  char *src = poly_render_c(ctx, linear, (int)(sizeof(linear) / sizeof(linear[0])), "half_chain");
  ASSERT_NOT_NULL(src);
  /* Pinned cstyle.py:40-43 casts a larger literal to half, :62-63 removes
   * same-MUL child parentheses, and :232-237 inlines both one-use MULs. */
  ASSERT_TRUE(strstr(src, "__fp16 alu") == NULL);
  ASSERT_NOT_NULL(strstr(src, "((__fp16)(1.702"));
  ASSERT_NOT_NULL(strstr(src, "(val0*val1*"));

  if (saved_expand_ssa)
    setenv("EXPAND_SSA", saved_expand_ssa, 1);
  else
    unsetenv("EXPAND_SSA");
  free(saved_expand_ssa);
  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_vecmul) {
  VecKernel k = make_vec_binop(POLY_OP_MUL, 8);
  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(k.ctx, lin, n, "vecmul");

  /* Default C path follows tinygrad's ClangRenderer float4 upcast:
   * 8 elements become a 2-iteration loop with vec4 loads/stores. */
  ASSERT_NOT_NULL(strstr(src, "void vecmul("));
  ASSERT_NOT_NULL(strstr(src, "Lidx0 < 2"));
  ASSERT_NOT_NULL(strstr(src, "__attribute__((vector_size(16)))"));
  ASSERT_NOT_NULL(strstr(src, "val0[0]"));
  ASSERT_NOT_NULL(strstr(src, "*"));

  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

/* End-to-end tests */

TEST(codegen, render_int64_min_literal_is_portable) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = program_param(ctx, POLY_INT64, 1, 0);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT64, out, zero, poly_arg_none());
  PolyUOp *val = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(INT64_MIN));
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, val, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  int n = 0;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_c(ctx, lin, n, "store_i64_min");
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, "(-9223372036854775807ll - 1ll)") != NULL);
  ASSERT_TRUE(strstr(src, "-9223372036854775808ll") == NULL);

  PolyProgram *prog = poly_compile_c(src, "store_i64_min");
  ASSERT_NOT_NULL(prog);
  int64_t out_data = 0;
  void *args[1] = {&out_data};
  poly_program_call(prog, args, 1);
  ASSERT_TRUE(out_data == INT64_MIN);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, exact_uint64_bigint_const_executes_like_tinygrad) {
  /* Pinned CStyleLanguage truncates uint64 CONST args at rendering
   * (renderer/cstyle.py:37), while the UOp retains the exact Python int. */
  PolyCtx *ctx = poly_ctx_new();
  PolyInt value = {0};
  ASSERT_TRUE(poly_int_from_decimal(&value, "18446744073709550593"));
  PolyUOp *constant = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_int_as_arg(&value));
  poly_int_free(&value);

  PolyUOp *out = program_param(ctx, POLY_UINT64, 1, 0);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT64, out, zero, poly_arg_none());
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, constant, poly_arg_none()));

  int n = 0;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_c(ctx, lin, n, "store_exact_uint64");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "18446744073709550593ull"));
  PolyProgram *prog = poly_compile_c(src, "store_exact_uint64");
  ASSERT_NOT_NULL(prog);

  uint64_t output = 0;
  void *args[1] = {&output};
  poly_program_call(prog, args, 1);
  ASSERT_TRUE(output == UINT64_C(18446744073709550593));
  output = 0;
  ASSERT_INT_EQ(poly_interp_eval(ctx, lin, n, args, 1), 0);
  ASSERT_TRUE(output == UINT64_C(18446744073709550593));

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

#ifdef POLY_TESTING
extern void poly_test_interp_alloc_fail_after(int count);

TEST(codegen, interp_local_allocation_failure_is_not_success) {
  PolyAddrSpace spaces[] = {POLY_ADDR_REG, POLY_ADDR_LOCAL};
  for (int i = 0; i < 2; i++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *reg = poly_test_uop_param(ctx, POLY_INT32, 1, 0, spaces[i]);
    PolyUOp *linear[] = {reg->src[0], reg};
    /* Value table, two index-map arrays, then PythonProgram's bytearray. */
    poly_test_interp_alloc_fail_after(3);
    int rc = poly_interp_eval(ctx, linear, 2, NULL, 0);
    poly_test_interp_alloc_fail_after(-1);
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(rc < 0);
  }
  PASS();
}

TEST(codegen, interp_index_map_allocation_failure_is_clean) {
  for (int fail = 1; fail <= 2; fail++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *one = poly_const_int(ctx, 1);
    PolyUOp *linear[] = {one};
    poly_test_interp_alloc_fail_after(fail);
    int rc = poly_interp_eval(ctx, linear, 1, NULL, 0);
    poly_test_interp_alloc_fail_after(-1);
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(rc < 0);
  }
  PASS();
}

TEST(codegen, interp_local_allocation_byte_product_does_not_overflow_int) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *reg = poly_test_uop_param(ctx, POLY_FLOAT64, INT32_MAX, 0, POLY_ADDR_REG);
  PolyUOp *linear[] = {reg->src[0], reg};
  /* Reject the allocation without requesting gigabytes; byte sizing must
   * reach the allocator without a signed intermediate multiplication. */
  poly_test_interp_alloc_fail_after(3);
  int rc = poly_interp_eval(ctx, linear, 2, NULL, 0);
  poly_test_interp_alloc_fail_after(-1);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rc < 0);
  PASS();
}

TEST(codegen, interp_lane_arena_count_does_not_overflow_int) {
  PolyCtx *ctx = poly_ctx_new();
  int count = INT32_MAX / UINT16_MAX + 1;
  PolyUOp **linear = malloc((size_t)(count + 1) * sizeof(*linear));
  ASSERT_NOT_NULL(linear);
  linear[0] = poly_const_int(ctx, UINT16_MAX);
  for (int i = 0; i < count; i++)
    linear[i + 1] = poly_test_uop_param(ctx, POLY_INT32, UINT16_MAX, i, POLY_ADDR_REG);
  /* The value table fits; fail the oversized arena before requesting its
   * storage. Each UOp and its lane count fit the current representation. */
  poly_test_interp_alloc_fail_after(1);
  int rc = poly_interp_eval(ctx, linear, count + 1, NULL, 0);
  poly_test_interp_alloc_fail_after(-1);
  free(linear);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rc < 0);
  PASS();
}
#endif

TEST(codegen, interp_null_argument_table_is_rejected) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_test_uop_param(ctx, POLY_INT32, 1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *linear[] = {out->src[0], out};
  int rc = poly_interp_eval(ctx, linear, 2, NULL, 1);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rc < 0);
  PASS();
}

TEST(codegen, interp_image_store_rejects_active_invalid_coordinates) {
  /* PythonProgram only masks STORE through IF. An invalid image address
   * is a zero-valued LOAD, but is not a successful active STORE. */
  for (int x = 0; x < 2; x++) {
    for (int active = 0; active < 2; active++) {
      PolyCtx *ctx = poly_ctx_new();
      PolyUOp *one = poly_const_int(ctx, 1), *four = poly_const_int(ctx, 4);
      PolyUOp *dims[] = {one, one, four};
      PolyUOp *shape = poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, dims, 3, poly_arg_none());
      PolyParamArg arg = {.slot = 0, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL};
      PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&arg));
      PolyUOp *y = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
      PolyUOp *xx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(x));
      PolyUOp *index = poly_uop3(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, y, xx, poly_arg_none());
      PolyUOp *scalar = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(7));
      PolyUOp *lanes[] = {scalar, scalar, scalar, scalar};
      PolyUOp *value = poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, lanes, 4, poly_arg_none());
      PolyUOp *flag = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(active != 0));
      PolyUOp *gate = poly_uop1(ctx, POLY_OP_IF, POLY_VOID, flag, poly_arg_none());
      PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, value, poly_arg_none());
      PolyUOp *end = poly_uop1(ctx, POLY_OP_ENDIF, POLY_VOID, gate, poly_arg_none());
      PolyUOp *linear[] = {one,    four,  shape, out,  y,     xx, index,
                           scalar, value, flag,  gate, store, end};
      float data[] = {11, 11, 11, 11};
      void *args[] = {data};
      int rc = poly_interp_eval(ctx, linear, (int)(sizeof(linear) / sizeof(linear[0])), args, 1);
      poly_ctx_destroy(ctx);
      ASSERT_TRUE(x && active ? rc < 0 : rc == 0);
      for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(data[i], !x && active ? 7 : 11, 0);
    }
  }
  PASS();
}

TEST(codegen, interp_loop_local_storage_is_reclaimed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_test_uop_param(ctx, POLY_INT32, 1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *reg = poly_test_uop_param(ctx, POLY_INT32, 1, 0, POLY_ADDR_REG);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *loop =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, three, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *ri = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, reg, zero, poly_arg_none());
  PolyUOp *oi = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, zero, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, loop, poly_arg_none());
  PolyUOp *rs = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, ri, value, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, ri, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oi, load, poly_arg_none());
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, loop, poly_arg_none());
  PolyUOp *linear[] = {out->src[0], out,   three, loop, reg,   zero, ri,
                       oi,          value, rs,    load, store, end};
  int32_t result = -1;
  void *args[] = {&result};
  int rc = poly_interp_eval(ctx, linear, (int)(sizeof(linear) / sizeof(linear[0])), args, 1);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_INT_EQ(result, 2);
#ifdef POLY_TESTING
  for (int fail = 4; fail <= 5; fail++) {
    result = -1;
    poly_test_interp_alloc_fail_after(fail);
    rc = poly_interp_eval(ctx, linear, (int)(sizeof(linear) / sizeof(linear[0])), args, 1);
    poly_test_interp_alloc_fail_after(-1);
    ASSERT_TRUE(rc < 0);
    ASSERT_INT_EQ(result, fail - 4);
  }
#endif
  poly_ctx_destroy(ctx);
  /* LSan must reclaim all three bytearrays, not just the last value-table row. */
  PASS();
}

TEST(codegen, interp_if_masks_nested_stores_not_values) {
  /* PythonProgram keeps evaluating values under IF, but combines execution
   * masks for STORE. ENDIF restores the enclosing mask, not unconditional true. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_test_uop_param(ctx, POLY_INT32, 3, 0, POLY_ADDR_GLOBAL);
  PolyUOp *flags = poly_test_uop_param(ctx, POLY_BOOL, 2, 1, POLY_ADDR_GLOBAL);
  PolyUOp *data = poly_test_uop_param(ctx, POLY_INT32, 1, 2, POLY_ADDR_GLOBAL);
  PolyUOp *offsets[3], *oi[3], *fi[2];
  for (int i = 0; i < 3; i++) {
    offsets[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(i));
    oi[i] = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, offsets[i], poly_arg_none());
    if (i < 2) fi[i] = poly_uop2(ctx, POLY_OP_INDEX, POLY_BOOL, flags, offsets[i], poly_arg_none());
  }
  PolyUOp *di = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, data, offsets[0], poly_arg_none());
  PolyUOp *a = poly_uop1(ctx, POLY_OP_LOAD, POLY_BOOL, fi[0], poly_arg_none());
  PolyUOp *b = poly_uop1(ctx, POLY_OP_LOAD, POLY_BOOL, fi[1], poly_arg_none());
  PolyUOp *outer = poly_uop1(ctx, POLY_OP_IF, POLY_VOID, a, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, di, poly_arg_none());
  PolyUOp *inner = poly_uop1(ctx, POLY_OP_IF, POLY_VOID, b, poly_arg_none());
  PolyUOp *stores[3];
  for (int i = 0; i < 3; i++)
    stores[i] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oi[i], value, poly_arg_none());
  PolyUOp *inner_end = poly_uop1(ctx, POLY_OP_ENDIF, POLY_VOID, inner, poly_arg_none());
  PolyUOp *outer_end = poly_uop1(ctx, POLY_OP_ENDIF, POLY_VOID, outer, poly_arg_none());
  PolyUOp *linear[] = {
      out,   flags,     data,      offsets[0], offsets[1], offsets[2], oi[0], oi[1],
      oi[2], fi[0],     fi[1],     di,         a,          b,          outer, value,
      inner, stores[0], inner_end, stores[1],  outer_end,  stores[2],
  };
  int n = (int)(sizeof(linear) / sizeof(linear[0]));
  for (int av = 0; av < 2; av++) {
    for (int bv = 0; bv < 2; bv++) {
      int32_t output[] = {-1, -1, -1}, input = 17;
      bool conditions[] = {av != 0, bv != 0};
      void *args[] = {output, conditions, &input};
      int rc = poly_interp_eval(ctx, linear, n, args, 3);
      if (rc != 0) {
        poly_ctx_destroy(ctx);
        FAIL("INTERP rejected nested IF/ENDIF");
      }
      ASSERT_INT_EQ(output[0], av && bv ? 17 : -1);
      ASSERT_INT_EQ(output[1], av ? 17 : -1);
      ASSERT_INT_EQ(output[2], 17);
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, interp_float_to_narrow_integer_wraps) {
  /* PythonProgram CAST truncates after converting to int, including negatives. */
  PolyDType types[] = {POLY_INT8, POLY_UINT8, POLY_INT16, POLY_UINT16};
  const int expected[] = {16, 16, -240, 65296};
  for (int i = 0; i < 4; i++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *out = poly_test_uop_param(ctx, POLY_INT32, 1, 0, POLY_ADDR_GLOBAL);
    PolyUOp *zero = poly_const_int(ctx, 0);
    PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, zero, poly_arg_none());
    PolyUOp *value = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(-240.5));
    PolyUOp *narrow = poly_uop1(ctx, POLY_OP_CAST, types[i], value, poly_arg_none());
    PolyUOp *wide = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, narrow, poly_arg_none());
    PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, wide, poly_arg_none());
    PolyUOp *linear[] = {out, zero, index, value, narrow, wide, store};
    int32_t result = 0;
    void *args[] = {&result};
    int rc = poly_interp_eval(ctx, linear, 7, args, 1);
    poly_ctx_destroy(ctx);
    ASSERT_INT_EQ(rc, 0);
    ASSERT_INT_EQ(result, expected[i]);
  }
  PASS();
}

TEST(codegen, interp_bitcast_uint8_to_int8_reinterprets_sign_bit) {
  /* Pinned PythonProgram delegates BITCAST to uop/ops.py:1199-1207, which
   * packs with the uint8 format and unpacks the same byte as int8. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = poly_test_uop_param(ctx, POLY_INT32, 1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *in = poly_test_uop_param(ctx, POLY_UINT8, 1, 1, POLY_ADDR_GLOBAL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, zero, poly_arg_none());
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT8, in, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT8, in_idx, poly_arg_none());
  PolyUOp *signed_byte = poly_uop1(ctx, POLY_OP_BITCAST, POLY_INT8, load, poly_arg_none());
  PolyUOp *wide = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, signed_byte, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, wide, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);
  PolyUOp *linear[] = {
      out, in, zero, out_idx, in_idx, load, signed_byte, wide, store, sink,
  };

  uint8_t input = 159;
  int32_t output = 0;
  void *args[] = {&output, &input};
  ASSERT_INT_EQ(
      poly_interp_eval(ctx, linear, (int)(sizeof(linear) / sizeof(linear[0])), args, 2), 0
  );
  ASSERT_INT_EQ(output, -97);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, interp_shaped_load_store_uses_max_numel) {
  /* Tinygrad PythonProgram iterates shaped LOAD/STORE values by max_numel,
   * not scalar dtype count (runtime/ops_python.py:84-88,133-138). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyParamArg out_arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL};
  PolyParamArg in_arg = {.slot = 1, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&out_arg));
  PolyUOp *in = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&in_arg));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *length = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *out_src[] = {out, zero, length};
  PolyUOp *in_src[] = {in, zero, length};
  PolyUOp *out_view = poly_uop(ctx, POLY_OP_SHRINK, POLY_FLOAT32, out_src, 3, poly_arg_none());
  PolyUOp *in_view = poly_uop(ctx, POLY_OP_SHRINK, POLY_FLOAT32, in_src, 3, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_view, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_view, load, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  ASSERT_INT_EQ(poly_uop_max_numel(ctx, load), 4);
  ASSERT_TRUE(poly_dtype_eq(load->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(store->n_src, 2);
  ASSERT_PTR_EQ(store->src[0], out_view);
  ASSERT_PTR_EQ(store->src[1], load);

  int n_linear = 0;
  PolyUOp **linear = poly_toposort_alloc(ctx, sink, &n_linear);
  ASSERT_NOT_NULL(linear);
  float input[] = {1.0f, 2.0f, 4.0f, 8.0f};
  float output[] = {0.0f, 0.0f, 0.0f, 0.0f};
  void *args[] = {output, input};
  ASSERT_INT_EQ(poly_interp_eval(ctx, linear, n_linear, args, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(output[i], input[i], 0.0);

  poly_toposort_free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, interp_sparse_param_slots_consume_compact_runtime_args) {
  /* Tinygrad 2026-08-22/a9069c177a9d PythonProgram consumes PARAM buffers in
   * linear occurrence order after ProgramInfo.globals selects sparse slots. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *in = program_param(ctx, POLY_FLOAT32, 1, 3);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, zero, poly_arg_none());
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, in, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, load, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n_linear = 0;
  PolyUOp **linear = poly_toposort_alloc(ctx, sink, &n_linear);
  ASSERT_NOT_NULL(linear);
  int64_t slots[2] = {-1, -1};
  int n_params = 0;
  for (int i = 0; i < n_linear; i++) {
    if (linear[i]->op != POLY_OP_PARAM) continue;
    ASSERT_TRUE(n_params < 2);
    slots[n_params++] = poly_program_buffer_slot(linear[i]);
  }
  ASSERT_INT_EQ(n_params, 2);
  ASSERT_INT_EQ(slots[0], 0);
  ASSERT_INT_EQ(slots[1], 3);

  float output = 0.0f, input = 7.25f;
  void *args[2] = {&output, &input};
  ASSERT_INT_EQ(poly_interp_eval(ctx, linear, n_linear, args, 2), 0);
  ASSERT_FLOAT_EQ(output, input, 0.0);

  poly_toposort_free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, raw_bool_neg_matches_pinned_arithmetic_typed_identity) {
  /* Pinned CStyleLanguage renders raw NEG as -x (renderer/cstyle.py:128-130).
   * Standard logical NOT remains CMPNE(x,true); raw bool NEG normalizes back
   * to bool and therefore preserves False/True. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = program_param(ctx, POLY_BOOL, 1, 0);
  PolyUOp *in = program_param(ctx, POLY_BOOL, 1, 1);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_BOOL, out, zero, poly_arg_none());
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_BOOL, in, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_BOOL, in_idx, poly_arg_none());
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_BOOL, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, neg, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n = 0;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_c(ctx, lin, n, "raw_bool_neg");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "-"));
  ASSERT_TRUE(strstr(src, "!") == NULL);

  PolyProgram *prog = poly_compile_c(src, "raw_bool_neg");
  ASSERT_NOT_NULL(prog);
  uint8_t in_false = 0, in_true = 1, out_value = 0;
  void *args[2] = {&out_value, &in_false};
  poly_program_call(prog, args, 2);
  ASSERT_INT_EQ(out_value, 0);
  args[1] = &in_true;
  out_value = 0;
  poly_program_call(prog, args, 2);
  ASSERT_INT_EQ(out_value, 1);

  args[1] = &in_false;
  out_value = 1;
  ASSERT_INT_EQ(poly_interp_eval(ctx, lin, n, args, 2), 0);
  ASSERT_INT_EQ(out_value, 0);
  args[1] = &in_true;
  out_value = 0;
  ASSERT_INT_EQ(poly_interp_eval(ctx, lin, n, args, 2), 0);
  ASSERT_INT_EQ(out_value, 1);

  char *wgsl = poly_render_wgsl(ctx, lin, n, "raw_bool_neg_wgsl");
  ASSERT_NOT_NULL(wgsl);
  ASSERT_TRUE(strstr(wgsl, "(!") == NULL);

  poly_program_destroy(prog);
  free(wgsl);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, e2e_vecadd) {
  /* c[i] = a[i] + b[i] for i in 0..9 */
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(k.ctx, lin, n, "vecadd");

  PolyProgram *prog = poly_compile_c(src, "vecadd");
  ASSERT_NOT_NULL(prog);

  float a[10], b[10], c[10];
  for (int i = 0; i < 10; i++) {
    a[i] = (float)(i + 1); /* 1, 2, ..., 10 */
    b[i] = (float)((i + 1) * 10); /* 10, 20, ..., 100 */
    c[i] = 0.0f;
  }

  void *args[3] = {a, b, c};
  poly_program_call(prog, args, 3);

  for (int i = 0; i < 10; i++) {
    ASSERT_FLOAT_EQ(c[i], a[i] + b[i], 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, e2e_vecmul) {
  /* c[i] = a[i] * b[i] for i in 0..7 */
  VecKernel k = make_vec_binop(POLY_OP_MUL, 8);
  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(k.ctx, lin, n, "vecmul");

  PolyProgram *prog = poly_compile_c(src, "vecmul");
  ASSERT_NOT_NULL(prog);

  float a[8], b[8], c[8];
  for (int i = 0; i < 8; i++) {
    a[i] = (float)(i + 1);
    b[i] = 0.5f;
    c[i] = 0.0f;
  }

  void *args[3] = {a, b, c};
  poly_program_call(prog, args, 3);

  for (int i = 0; i < 8; i++) {
    ASSERT_FLOAT_EQ(c[i], a[i] * 0.5f, 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, e2e_vecsub) {
  /* c[i] = a[i] - b[i] for i in 0..3 */
  VecKernel k = make_vec_binop(POLY_OP_SUB, 4);
  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(k.ctx, lin, n, "vecsub");

  PolyProgram *prog = poly_compile_c(src, "vecsub");
  ASSERT_NOT_NULL(prog);

  float a[4] = {10, 20, 30, 40};
  float b[4] = {1, 2, 3, 4};
  float c[4] = {0};

  void *args[3] = {a, b, c};
  poly_program_call(prog, args, 3);

  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(c[i], a[i] - b[i], 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

/* WGSL renderer tests */

TEST(codegen, render_wgsl_vecadd) {
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_wgsl(k.ctx, lin, n, "vecadd");

  /* preamble: INFINITY uniform at binding(0) */
  ASSERT_NOT_NULL(strstr(src, "fn nan()"));
  ASSERT_NOT_NULL(strstr(src, "var<uniform> INFINITY"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(0)"));

  /* buffer bindings: offset by +1 (binding 0 = INFINITY) */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(1)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data0: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(2)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data1: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(3)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data2: array<f32>"));

  /* compute shader entry point: workgroup_id + local_invocation_id */
  ASSERT_NOT_NULL(strstr(src, "@compute @workgroup_size(1)"));
  ASSERT_NOT_NULL(strstr(src, "fn vecadd("));
  ASSERT_NOT_NULL(strstr(src, "@builtin(workgroup_id) gindex"));
  ASSERT_NOT_NULL(strstr(src, "@builtin(local_invocation_id) lindex"));

  /* loop */
  ASSERT_NOT_NULL(strstr(src, "for (var Lidx0: i32 = 0; Lidx0 < 10; Lidx0++)"));

  /* array indexing (not pointer arithmetic) */
  ASSERT_NOT_NULL(strstr(src, "data0[Lidx0]"));
  ASSERT_NOT_NULL(strstr(src, "data1[Lidx0]"));
  ASSERT_NOT_NULL(strstr(src, "data2[Lidx0]"));

  /* variable declarations with WGSL types */
  ASSERT_NOT_NULL(strstr(src, "var val0: f32"));
  ASSERT_NOT_NULL(strstr(src, "var val1: f32"));
  ASSERT_NOT_NULL(strstr(src, "var alu0: f32"));

  /* no C-style wrapper */
  ASSERT_TRUE(strstr(src, "_call") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, render_wgsl_vecmul) {
  VecKernel k = make_vec_binop(POLY_OP_MUL, 8);
  int n;
  PolyUOp **lin = poly_linearize_webgpu(k.ctx, k.sink, &n);
  char *src = poly_render_wgsl(k.ctx, lin, n, "vecmul");

  ASSERT_NOT_NULL(strstr(src, "fn vecmul("));
  ASSERT_NOT_NULL(strstr(src, "@compute @workgroup_size(2,1,1)"));
  ASSERT_NOT_NULL(strstr(src, "var lidx0: i32 = i32(lindex.x);"));
  /* The tinygrad-style recursive tuplize tiebreak may order vector lanes as
   * offsets first and base lane last. The semantic contract is four scalar
   * stores to the four lanes, not a specific temporary-number pairing. */
  ASSERT_NOT_NULL(strstr(src, "data2[alu0] ="));
  ASSERT_NOT_NULL(strstr(src, "data2[alu1] ="));
  ASSERT_NOT_NULL(strstr(src, "data2[alu2] ="));
  ASSERT_NOT_NULL(strstr(src, "data2[alu3] ="));
  ASSERT_NOT_NULL(strstr(src, "*")); /* multiply operator */
  ASSERT_TRUE(strstr(src, "for (var ridx0") == NULL);
  ASSERT_TRUE(strstr(src, "vec4<f32>") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, webgpu_preserves_native_sin_without_long_shift) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *p1 = program_param(ctx, POLY_FLOAT32, 4, 1);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, range, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());
  PolyUOp *sin_uop = poly_uop1(ctx, POLY_OP_SIN, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, sin_uop, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  /* The four-element loop may be scalarized into four native SINs.  The
   * renderer contract is that SIN survives lowering, not a particular
   * scalarization width or count. */
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_SIN) > 0);
  for (int i = 0; i < n; i++) {
    PolyDType scalar = lin[i]->dtype;
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
  }

  char *src = poly_render_wgsl(ctx, lin, n, "native_sin");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "sin("));
  ASSERT_TRUE(strstr(src, "<<32u") == NULL);
  ASSERT_TRUE(strstr(src, "<< 32u") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, full_rewrite_casts_float_alu_operands_before_decomposition) {
  /* Current tinygrad codegen/__init__.py:252-257,365 keeps the Tensor-stage
   * SIN(float <- int) topology, then legalizes its operand immediately before
   * decompositions.  Final renderers must never depend on an implicit numeric
   * conversion or reinterpret an integer lane as float. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *input = program_param(ctx, POLY_INT32, 1, 0);
  PolyUOp *output = program_param(ctx, POLY_FLOAT32, 1, 1);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, input, zero, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, output, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, in_idx, poly_arg_none());
  PolyUOp *sin_uop = poly_uop1(ctx, POLY_OP_SIN, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, sin_uop, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyRewriteOpts opts = {
      .optimize = false,

      .caps =
          {.has_mulacc = true,
           .has_max = true,
           .has_exp2 = true,
           .has_log2 = true,
           .has_sin = true,
           .has_fdiv = true,
           .has_int64 = true,
           .max_vec_width = 1},
      .device = POLY_DEVICE_INTERP,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0, n_sin = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_SIN) continue;
    n_sin++;
    ASSERT_TRUE(poly_dtype_eq(topo[i]->dtype, POLY_FLOAT32));
    ASSERT_INT_EQ(topo[i]->n_src, 1);
    ASSERT_INT_EQ(topo[i]->src[0]->op, POLY_OP_CAST);
    ASSERT_TRUE(poly_dtype_eq(topo[i]->src[0]->dtype, POLY_FLOAT32));
    ASSERT_INT_EQ(topo[i]->src[0]->n_src, 1);
    ASSERT_TRUE(poly_dtype_eq(topo[i]->src[0]->src[0]->dtype, POLY_INT32));
  }
  ASSERT_INT_EQ(n_sin, 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, webgpu_keeps_reciprocal_without_fdiv_like_pinned_wgsl) {
  /* Pinned WGSLRenderer inherits RECIPROCAL from CStyleLanguage and does not
   * advertise FDIV (renderer/wgsl.py:56-66). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *in = program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *out = program_param(ctx, POLY_FLOAT32, 4, 1);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, in, range, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());
  PolyUOp *reciprocal = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, reciprocal, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, end);

  int n = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_RECIPROCAL) > 0);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_FDIV), 0);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, webgpu_decomposes_bf16_before_wgsl_render) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *in = program_param(ctx, POLY_BFLOAT16, 4, 1);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, range, poly_arg_none());
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_BFLOAT16, in, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_BFLOAT16, in_idx, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, value, poly_arg_none()));

  PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyDType scalar = topo[i]->dtype;
    ASSERT_FALSE(
        scalar.priority == POLY_BFLOAT16.priority && strcmp(scalar.name, POLY_BFLOAT16.name) == 0
    );
  }

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_wgsl(ctx, lin, n_lin, "bf16_to_f32");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "array<atomic<u32>>"));
  ASSERT_TRUE(strstr(src, "data1: array<f32>") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, webgpu_without_shader_f16_emulates_fp8_through_float32) {
  /* Tinygrad 2026-08-22/a9069c177a9d renderer/wgsl.py:104-105 and
   * codegen/decomp/dtype.py:198-206 select f32 when Target.arch omits
   * shader-f16. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *in = program_param(ctx, POLY_FP8E4M3, 4, 1);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, range, poly_arg_none());
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FP8E4M3, in, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FP8E4M3, in_idx, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, value, poly_arg_none()));

  PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0, u8_loads = 0, half_nodes = 0, fp8_nodes = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    u8_loads += topo[i]->op == POLY_OP_LOAD && poly_dtype_eq(topo[i]->dtype, POLY_UINT8);
    half_nodes += poly_dtype_eq(topo[i]->dtype, POLY_FLOAT16);
    fp8_nodes += poly_dtype_is_fp8(topo[i]->dtype);
  }
  ASSERT_INT_EQ(u8_loads, 1);
  ASSERT_INT_EQ(half_nodes, 0);
  ASSERT_INT_EQ(fp8_nodes, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, python311_f16_dtype_decomposition_keeps_native_transcendental) {
  /* Pinned PythonRenderer on Python 3.11 keeps EXP2 in code_for_op but
   * emulates unsupported f16 LOAD/ALU/STORE through f32 and uint16 storage
   * (ops_python.py:203-223; decompositions.py:388-429,532-564). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_test_uop_param(ctx, POLY_FLOAT16, 4, 0, POLY_ADDR_GLOBAL);
  PolyUOp *in = poly_test_uop_param(ctx, POLY_FLOAT16, 4, 1, POLY_ADDR_GLOBAL);
  PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT16, out, idx, poly_arg_none());
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT16, in, idx, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, in_idx, poly_arg_none());
  PolyUOp *exp2 = poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT16, load, poly_arg_none());
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, exp2, poly_arg_none()));

  PolyFloatDecompContext fctx = {.from = POLY_FLOAT16, .to = POLY_FLOAT32};
  PolyUOp *rewritten = poly_graph_rewrite_ctx_ex(ctx, sink, poly_pm_float_decomp(), &fctx, true);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_EXP2), 1);
  int f32_exp2 = 0, u16_load = 0, u16_store_value = 0, residual_f16 = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    PolyDType scalar = u->dtype;
    if (u->op == POLY_OP_EXP2 && poly_dtype_eq(u->dtype, POLY_FLOAT32)) f32_exp2++;
    if (u->op == POLY_OP_LOAD && poly_dtype_eq(u->dtype, POLY_UINT16)) u16_load++;
    if (u->op == POLY_OP_STORE && u->n_src == 2 && poly_dtype_eq(u->src[1]->dtype, POLY_UINT16))
      u16_store_value++;
    if (poly_dtype_is_float(scalar) && scalar.priority == POLY_FLOAT16.priority &&
        scalar.bitsize == 16)
      residual_f16++;
  }
  ASSERT_INT_EQ(f32_exp2, 1);
  ASSERT_INT_EQ(u16_load, 1);
  ASSERT_INT_EQ(u16_store_value, 1);
  ASSERT_INT_EQ(residual_f16, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, float_dtype_decomposition_uses_shaped_load_store_lanes) {
  /* Current f2f_load/f2f_store use max_numel and ordinary INDEX, so a
   * two-element SHRINK becomes STACK(two scalar loads) or GROUP(two scalar
   * stores) (tinygrad/codegen/decomp/dtype.py:129-136). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType storage_dtypes[] = {POLY_BFLOAT16, POLY_FLOAT16};
  for (int d = 0; d < 2; d++) {
    PolyFloatDecompContext fctx = {.from = storage_dtypes[d], .to = POLY_FLOAT32};
    PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
    PolyParamArg arg = {.slot = d, .addrspace = POLY_ADDR_GLOBAL};
    PolyUOp *buf = poly_uop1(ctx, POLY_OP_PARAM, storage_dtypes[d], shape, poly_arg_param(&arg));
    int64_t bounds[1][2] = {{0, 2}};
    PolyUOp *address = poly_shrink(ctx, buf, bounds, 1);
    ASSERT_NOT_NULL(address);

    PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, storage_dtypes[d], address, poly_arg_none());
    PolyUOp *load_out = poly_graph_rewrite_ctx_ex(ctx, load, poly_pm_float_decomp(), &fctx, true);
    ASSERT_NOT_NULL(load_out);
    ASSERT_INT_EQ(load_out->op, POLY_OP_STACK);
    ASSERT_TRUE(poly_dtype_eq(load_out->dtype, POLY_FLOAT32));
    ASSERT_INT_EQ(load_out->n_src, 2);
    for (int i = 0; i < load_out->n_src; i++)
      ASSERT_INT_EQ(poly_uop_max_numel(ctx, load_out->src[i]), 1);

    PolyUOp *values[] = {
        poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0)),
        poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0)),
    };
    PolyUOp *store = poly_uop2(
        ctx, POLY_OP_STORE, POLY_VOID, address, poly_uop_stack(ctx, values, 2), poly_arg_none()
    );
    PolyUOp *store_out = poly_graph_rewrite_ctx_ex(ctx, store, poly_pm_float_decomp(), &fctx, true);
    ASSERT_NOT_NULL(store_out);
    ASSERT_INT_EQ(store_out->op, POLY_OP_GROUP);
    ASSERT_INT_EQ(store_out->n_src, 2);
    for (int i = 0; i < store_out->n_src; i++) {
      ASSERT_INT_EQ(store_out->src[i]->op, POLY_OP_STORE);
      ASSERT_INT_EQ(poly_uop_max_numel(ctx, store_out->src[i]->src[0]), 1);
      ASSERT_INT_EQ(poly_uop_max_numel(ctx, store_out->src[i]->src[1]), 1);
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

static bool float_store_decomp_has_pinned_constants(
    PolyDType storage,
    PolyDType compute,
    int64_t expected,
    int expected_and
) {
  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return false;
  PolyUOp *buf = poly_test_uop_param(ctx, storage, 1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *idx =
      poly_uop2(ctx, POLY_OP_INDEX, storage, buf, poly_const_int(ctx, 0), poly_arg_none());
  PolyUOp *value = poly_uop0(ctx, POLY_OP_CONST, compute, poly_arg_float(1.0));
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, value, poly_arg_none());
  PolyFloatDecompContext fctx = {.from = storage, .to = compute};
  PolyUOp *result = poly_graph_rewrite_ctx_ex(ctx, store, poly_pm_float_decomp(), &fctx, true);
  int n = 0;
  PolyUOp **topo = result ? poly_toposort_alloc(ctx, result, &n) : NULL;
  bool found = false;
  for (int i = 0; topo && i < n; i++)
    if (topo[i]->op == POLY_OP_CONST && poly_dtype_eq(topo[i]->dtype, POLY_WEAKINT) &&
        topo[i]->arg.kind == POLY_ARG_INT && topo[i]->arg.i == expected)
      found = true;
  bool ok = found && count_lin_ops(topo, n, POLY_OP_STORE) == 1 &&
            count_lin_ops(topo, n, POLY_OP_FLOORDIV) == 4 &&
            count_lin_ops(topo, n, POLY_OP_AND) == expected_and;
  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  return ok;
}

TEST(codegen, float_store_decomp_double_sign_mask) {
  /* Pinned dtype.py:f2f uses Python (1 << 63) - 1, not signed C shifting. */
  ASSERT_TRUE(float_store_decomp_has_pinned_constants(POLY_FLOAT32, POLY_FLOAT64, INT64_MAX, 8));
  PASS();
}

TEST(codegen, float_store_decomp_negative_fnuz_bias) {
  /* f16 bias15 minus e5m2fnuz bias16, shifted by two mantissa bits, is -4. */
  ASSERT_TRUE(float_store_decomp_has_pinned_constants(POLY_FP8E5M2FNUZ, POLY_FLOAT16, -4, 7));
  PASS();
}

TEST(codegen, dtype_decomposition_matchers_match_current_row_counts) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/decomp/dtype.py has 12 long,
   * 11 float, and 2 outer dtype-decomposition rows. */
  ASSERT_INT_EQ(poly_pm_rule_count(poly_pm_long_decomp()), 12);
  ASSERT_INT_EQ(poly_pm_rule_count(poly_pm_float_decomp()), 11);
  ASSERT_INT_EQ(poly_pm_rule_count(poly_pm_dtype_decomps()), 2);
  PASS();
}

TEST(codegen, fp8_dtype_decomposition_matches_current_half_storage_topology) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/decomp/dtype.py:198-206
   * emulates unsupported FP8 as half when half is supported. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyParamArg arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL, .dtype = POLY_FP8E4M3};
  PolyUOp *buf = poly_uop1(ctx, POLY_OP_PARAM, POLY_FP8E4M3, shape, poly_arg_param(&arg));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FP8E4M3, buf, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FP8E4M3, idx, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, load, poly_arg_dtype(POLY_FLOAT32));
  PolyUOp *sink = poly_sink1(ctx, cast);
  PolyDTypeDecompsContext dctx = {.caps = {.supports_float16 = true}};
  PolyUOp *rewritten = poly_graph_rewrite_ctx(ctx, sink, poly_pm_dtype_decomps(), &dctx);
  ASSERT_NOT_NULL(rewritten);

  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n);
  ASSERT_NOT_NULL(topo);
  int u8_load = 0, half_bitcast = 0, fp8_nodes = 0;
  for (int i = 0; i < n; i++) {
    u8_load += topo[i]->op == POLY_OP_LOAD && poly_dtype_eq(topo[i]->dtype, POLY_UINT8);
    half_bitcast += topo[i]->op == POLY_OP_BITCAST && poly_dtype_eq(topo[i]->dtype, POLY_FLOAT16);
    fp8_nodes += poly_dtype_is_fp8(topo[i]->dtype);
  }
  ASSERT_INT_EQ(u8_load, 1);
  ASSERT_INT_EQ(half_bitcast, 1);
  ASSERT_INT_EQ(fp8_nodes, 0);
  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, float_narrowing_keeps_sign_outside_underflow_magnitude) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/decomp/dtype.py:126 builds
   * `sign | underflow.where(0, norm)`, preserving negative zero. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyParamArg arg = {.slot = 0, .addrspace = POLY_ADDR_ALU, .dtype = POLY_FLOAT32};
  PolyUOp *f32 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&arg));
  PolyUOp *raw = poly_uop1(ctx, POLY_OP_BITCAST, POLY_INT16, f32, poly_arg_none());
  PolyFloatDecompContext fctx = {.from = POLY_BFLOAT16, .to = POLY_FLOAT32};
  PolyUOp *rewritten = poly_graph_rewrite_ctx_ex(ctx, raw, poly_pm_float_decomp(), &fctx, true);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_BITCAST);
  ASSERT_INT_EQ(rewritten->src[0]->op, POLY_OP_WHERE);
  PolyUOp *finite = rewritten->src[0]->src[2];
  ASSERT_INT_EQ(finite->op, POLY_OP_OR);
  PolyUOp *magnitude = finite->src[0]->op == POLY_OP_WHERE ? finite->src[0] : finite->src[1];
  ASSERT_INT_EQ(magnitude->op, POLY_OP_WHERE);
  ASSERT_INT_EQ(magnitude->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(magnitude->src[1]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(magnitude->src[1]->arg.i, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, long_to_float_decomposition_keeps_weak_constants) {
  /* Tinygrad 2026-08-22/a9069c177a9d dtype.py:34-37 constructs the range
   * checks from Python 0/-1 and the 2**32 scale, hence weak constants. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *value = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_C(0x100000002)));
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, value, poly_arg_dtype(POLY_FLOAT32));
  unsigned char rewrite_ctx = 0;
  PolyUOp *rewritten =
      poly_graph_rewrite_ctx_ex(ctx, cast, poly_pm_long_decomp(), &rewrite_ctx, true);
  ASSERT_NOT_NULL(rewritten);

  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n);
  ASSERT_NOT_NULL(topo);
  bool weak_zero = false, weak_minus_one = false, weak_scale = false;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_CONST) continue;
    if (poly_dtype_eq(u->dtype, POLY_WEAKINT) && u->arg.kind == POLY_ARG_INT) {
      weak_zero |= u->arg.i == 0;
      weak_minus_one |= u->arg.i == -1;
    }
    if (poly_dtype_eq(u->dtype, POLY_WEAKFLOAT) && u->arg.kind == POLY_ARG_FLOAT)
      weak_scale |= u->arg.f == 4294967296.0;
  }
  ASSERT_TRUE(weak_zero);
  ASSERT_TRUE(weak_minus_one);
  ASSERT_TRUE(weak_scale);
  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, float_dtype_decomposition_matches_same_width_bitcast_load) {
  /* Tinygrad 2026-08-22/a9069c177a9d dtype.py:177-179 removes the redundant
   * same-width BITCAST after rewriting a half LOAD to ushort storage. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyParamArg arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL, .dtype = POLY_FLOAT16};
  PolyUOp *buf = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT16, shape, poly_arg_param(&arg));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT16, buf, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, idx, poly_arg_none());
  PolyUOp *bitcast =
      poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT16, load, poly_arg_dtype(POLY_UINT16));
  PolyFloatDecompContext fctx = {.from = POLY_FLOAT16, .to = POLY_FLOAT32};
  PolyUOp *rewritten = poly_graph_rewrite_ctx_ex(ctx, bitcast, poly_pm_float_decomp(), &fctx, true);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_UINT16));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, dtype_reindex_changes_shrink_to_two_source_index) {
  /* Tinygrad 2026-08-22/a9069c177a9d dtype.py:13-17 changes SHRINK to INDEX;
   * the replacement has exactly the storage source and scalar offset. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyParamArg arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL, .dtype = POLY_FLOAT16};
  PolyUOp *buf = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT16, shape, poly_arg_param(&arg));
  int64_t bounds[1][2] = {{0, 2}};
  PolyUOp *shrink = poly_shrink(ctx, buf, bounds, 1);
  ASSERT_NOT_NULL(shrink);
  PolyUOp *reindexed = poly_reindex(ctx, shrink, 1, 1);
  ASSERT_NOT_NULL(reindexed);
  ASSERT_INT_EQ(reindexed->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(reindexed->n_src, 2);
  ASSERT_TRUE(reindexed->src[0] == shrink->src[0]);
  ASSERT_INT_EQ(reindexed->src[1]->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(reindexed->src[1]->src[0], shrink->src[1]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, renderer_float_final_matchers_match_current_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py:276-281,440-444,
   * 522-529: Clang has 10 rows, CUDA 4 FP8 rows, and HIP 7 class rows. */

  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_BFLOAT16, poly_arg_float(1.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_BFLOAT16, poly_arg_float(2.0));
  PolyUOp *const_rewritten = poly_graph_rewrite(ctx, a, poly_hip_renderer_extra_matcher());
  ASSERT_NOT_NULL(const_rewritten);
  ASSERT_INT_EQ(const_rewritten->op, POLY_OP_BITCAST);
  ASSERT_TRUE(poly_dtype_eq(const_rewritten->dtype, POLY_BFLOAT16));
  ASSERT_INT_EQ(const_rewritten->n_src, 1);
  ASSERT_INT_EQ(const_rewritten->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(const_rewritten->src[0]->dtype, POLY_UINT16));

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_BFLOAT16, a, b, poly_arg_none());
  PolyUOp *add_rewritten = poly_graph_rewrite(ctx, add, poly_hip_renderer_extra_matcher());
  ASSERT_NOT_NULL(add_rewritten);
  ASSERT_INT_EQ(add_rewritten->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(add_rewritten->dtype, POLY_BFLOAT16));
  ASSERT_INT_EQ(add_rewritten->n_src, 1);
  ASSERT_INT_EQ(add_rewritten->src[0]->op, POLY_OP_ADD);
  ASSERT_TRUE(poly_dtype_eq(add_rewritten->src[0]->dtype, POLY_FLOAT32));

  PolyUOp *bf16_to_f32 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *bf16_to_f32_rewritten =
      poly_graph_rewrite(ctx, bf16_to_f32, poly_hip_renderer_extra_matcher());
  ASSERT_NOT_NULL(bf16_to_f32_rewritten);
  ASSERT_INT_EQ(bf16_to_f32_rewritten->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(bf16_to_f32_rewritten->dtype, POLY_FLOAT32));
  ASSERT_TRUE(poly_dtype_eq(bf16_to_f32_rewritten->src[0]->dtype, POLY_BFLOAT16));

  PolyUOp *f32 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.25));
  PolyUOp *f32_to_bf16 = poly_uop1(ctx, POLY_OP_CAST, POLY_BFLOAT16, f32, poly_arg_none());
  PolyUOp *f32_to_bf16_rewritten =
      poly_graph_rewrite(ctx, f32_to_bf16, poly_hip_renderer_extra_matcher());
  ASSERT_NOT_NULL(f32_to_bf16_rewritten);
  ASSERT_INT_EQ(f32_to_bf16_rewritten->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(f32_to_bf16_rewritten->dtype, POLY_BFLOAT16));
  ASSERT_TRUE(poly_dtype_eq(f32_to_bf16_rewritten->src[0]->dtype, POLY_FLOAT32));

  PolyUOp *clang_f32_to_bf16 =
      poly_graph_rewrite(ctx, f32_to_bf16, poly_clang_renderer_extra_matcher());
  ASSERT_NOT_NULL(clang_f32_to_bf16);
  ASSERT_INT_EQ(clang_f32_to_bf16->op, POLY_OP_BITCAST);
  ASSERT_TRUE(poly_dtype_eq(clang_f32_to_bf16->dtype, POLY_BFLOAT16));
  int n_clang = 0;
  PolyUOp **clang_topo = poly_toposort(ctx, clang_f32_to_bf16, &n_clang);
  ASSERT_NOT_NULL(clang_topo);
  int neg_count = 0;
  bool weak_minus_one = false, weak_shift = false;
  for (int i = 0; i < n_clang; i++) {
    neg_count += clang_topo[i]->op == POLY_OP_NEG;
    if (clang_topo[i]->op != POLY_OP_CONST || !poly_dtype_eq(clang_topo[i]->dtype, POLY_WEAKINT) ||
        clang_topo[i]->arg.kind != POLY_ARG_INT)
      continue;
    weak_minus_one |= clang_topo[i]->arg.i == -1;
    weak_shift |= clang_topo[i]->arg.i == 16;
  }
  ASSERT_INT_EQ(neg_count, 0);
  ASSERT_TRUE(weak_minus_one);
  ASSERT_TRUE(weak_shift);

  PolyUOp *a_src[] = {a, a, a, a};
  PolyUOp *b_src[] = {b, b, b, b};
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *c_src[] = {zero, zero, zero, zero};
  PolyUOp *a_vec = poly_uop(ctx, POLY_OP_STACK, POLY_BFLOAT16, a_src, 4, poly_arg_none());
  PolyUOp *b_vec = poly_uop(ctx, POLY_OP_STACK, POLY_BFLOAT16, b_src, 4, poly_arg_none());
  PolyUOp *c_vec = poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, c_src, 4, poly_arg_none());
  PolyUOp *wmma_src[] = {a_vec, b_vec, c_vec};
  int bf16_dims[] = {16, 16, 16};
  PolyUOp *wmma = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, wmma_src, 3,
      poly_arg_tensor_core(bf16_dims, POLY_BFLOAT16, "AMD", 64, NULL, NULL, false)
  );
  PolyUOp *wmma_rewritten = poly_graph_rewrite(ctx, wmma, poly_hip_renderer_extra_matcher());
  /* Pinned graph_rewrite rebuilds the WMMA because its BF16 CONST leaves are
   * converted, but preserves the WMMA and BF16 fragment topology. */
  ASSERT_NOT_NULL(wmma_rewritten);
  ASSERT_INT_EQ(wmma_rewritten->op, POLY_OP_WMMA);
  ASSERT_FALSE(wmma_rewritten == wmma);
  ASSERT_FALSE(wmma_rewritten->src[0] == a_vec);
  ASSERT_FALSE(wmma_rewritten->src[1] == b_vec);
  ASSERT_INT_EQ(wmma_rewritten->src[0]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(wmma_rewritten->src[1]->op, POLY_OP_STACK);
  ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[0]->dtype, POLY_BFLOAT16));
  ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[1]->dtype, POLY_BFLOAT16));
  for (int lane = 0; lane < 4; lane++) {
    ASSERT_INT_EQ(wmma_rewritten->src[0]->src[lane]->op, POLY_OP_BITCAST);
    ASSERT_INT_EQ(wmma_rewritten->src[1]->src[lane]->op, POLY_OP_BITCAST);
    ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[0]->src[lane]->dtype, POLY_BFLOAT16));
    ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[1]->src[lane]->dtype, POLY_BFLOAT16));
  }

  PolyUOp *fa = poly_uop0(ctx, POLY_OP_CONST, POLY_FP8E4M3, poly_arg_float(1.0));
  PolyUOp *fb = poly_uop0(ctx, POLY_OP_CONST, POLY_FP8E4M3, poly_arg_float(2.0));
  PolyUOp *fp8_add = poly_uop2(ctx, POLY_OP_ADD, POLY_FP8E4M3, fa, fb, poly_arg_none());
  PolyUOp *cuda_add = poly_graph_rewrite(ctx, fp8_add, poly_cuda_renderer_extra_matcher());
  ASSERT_NOT_NULL(cuda_add);
  ASSERT_INT_EQ(cuda_add->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(cuda_add->dtype, POLY_FP8E4M3));
  ASSERT_INT_EQ(cuda_add->src[0]->op, POLY_OP_ADD);
  ASSERT_TRUE(poly_dtype_eq(cuda_add->src[0]->dtype, POLY_FLOAT32));
  ASSERT_TRUE(poly_dtype_eq(cuda_add->src[0]->src[0]->dtype, POLY_FLOAT32));
  ASSERT_TRUE(poly_dtype_eq(cuda_add->src[0]->src[1]->dtype, POLY_FLOAT32));

  PolyUOp *cross = poly_uop1(ctx, POLY_OP_CAST, POLY_FP8E5M2, fa, poly_arg_none());
  PolyUOp *cuda_cross = poly_graph_rewrite(ctx, cross, poly_cuda_renderer_extra_matcher());
  ASSERT_NOT_NULL(cuda_cross);
  ASSERT_INT_EQ(cuda_cross->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(cuda_cross->dtype, POLY_FP8E5M2));
  ASSERT_INT_EQ(cuda_cross->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(cuda_cross->src[0]->dtype, POLY_FLOAT32));
  ASSERT_TRUE(cuda_cross->src[0]->src[0] == fa);

  PolyUOp *fp8_lanes[] = {fa, fa, fa, fa, fa, fa, fa, fa};
  PolyUOp *fp8_vec = poly_uop_stack(ctx, fp8_lanes, 8);
  PolyUOp *fp8_acc = poly_uop_stack(ctx, (PolyUOp *[]){zero, zero, zero, zero}, 4);
  int fp8_dims[3] = {16, 16, 32};
  PolyUOp *fp8_wmma = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, (PolyUOp *[]){fp8_vec, fp8_vec, fp8_acc}, 3,
      poly_arg_tensor_core(fp8_dims, POLY_FP8E4M3, "AMD", 64, NULL, NULL, false)
  );
  PolyUOp *fp8_wmma_rewritten =
      poly_graph_rewrite(ctx, fp8_wmma, poly_hip_renderer_extra_matcher());
  ASSERT_NOT_NULL(fp8_wmma_rewritten);
  ASSERT_INT_EQ(fp8_wmma_rewritten->op, POLY_OP_WMMA);
  ASSERT_INT_EQ(fp8_wmma_rewritten->src[0]->op, POLY_OP_BITCAST);
  ASSERT_INT_EQ(fp8_wmma_rewritten->src[1]->op, POLY_OP_BITCAST);
  ASSERT_TRUE(poly_dtype_eq(fp8_wmma_rewritten->src[0]->dtype, POLY_UINT64));
  ASSERT_TRUE(poly_dtype_eq(fp8_wmma_rewritten->src[1]->dtype, POLY_UINT64));
  ASSERT_INT_EQ(poly_uop_max_numel(ctx, fp8_wmma_rewritten->src[0]), 1);
  ASSERT_TRUE(fp8_wmma_rewritten->src[2] == fp8_acc);

  ASSERT_INT_EQ(poly_pm_rule_count(poly_clang_renderer_extra_matcher()), 10);
  ASSERT_INT_EQ(poly_pm_rule_count(poly_cuda_renderer_extra_matcher()), 4);
  ASSERT_INT_EQ(poly_pm_rule_count(poly_hip_renderer_extra_matcher()), 7);
  ASSERT_INT_EQ(poly_pm_rule_count(poly_pm_manual_bf16_cast()), 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, tensor_core_arch_tables_match_current_tinygrad) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/opt/tc.py:75-137. */
  int count = -1;
  const PolyTensorCore *tcs = poly_tc_get_cuda(70, &count);
  ASSERT_INT_EQ(count, 0);
  ASSERT_TRUE(tcs == NULL);

  tcs = poly_tc_get_cuda(75, &count);
  ASSERT_NOT_NULL(tcs);
  ASSERT_INT_EQ(count, 2);
  ASSERT_TRUE(poly_dtype_eq(tcs[0].dtype_in, POLY_FLOAT16));
  ASSERT_TRUE(poly_dtype_eq(tcs[0].dtype_out, POLY_FLOAT32));
  ASSERT_TRUE(poly_dtype_eq(tcs[1].dtype_out, POLY_FLOAT16));
  ASSERT_INT_EQ(tcs[0].dims[2], 8);
  ASSERT_STR_EQ(tcs[0].swizzle[0][0][0], "r1");
  ASSERT_STR_EQ(tcs[0].swizzle[0][0][1], "r2");
  ASSERT_STR_EQ(tcs[0].swizzle[0][1][0], "r0");

  tcs = poly_tc_get_cuda(80, &count);
  ASSERT_NOT_NULL(tcs);
  ASSERT_INT_EQ(count, 6);
  ASSERT_INT_EQ(tcs[0].dims[2], 16);
  ASSERT_TRUE(poly_dtype_eq(tcs[1].dtype_in, POLY_BFLOAT16));
  ASSERT_TRUE(poly_dtype_eq(tcs[5].dtype_in, POLY_FLOAT32));

  tcs = poly_tc_get_cuda(89, &count);
  ASSERT_NOT_NULL(tcs);
  ASSERT_INT_EQ(count, 8);
  ASSERT_INT_EQ(tcs[6].dims[2], 32);
  ASSERT_TRUE(poly_dtype_eq(tcs[6].dtype_in, POLY_FP8E4M3));
  ASSERT_TRUE(poly_dtype_eq(tcs[7].dtype_in, POLY_FP8E5M2));
  ASSERT_INT_EQ(tcs[6].elements_per_thread[0], 16);
  ASSERT_INT_EQ(tcs[6].swizzle_len[0][2], 5);

  tcs = poly_tc_get_amd("gfx1100", &count);
  ASSERT_NOT_NULL(tcs);
  ASSERT_INT_EQ(count, 4);
  ASSERT_INT_EQ(tcs[0].threads, 32);
  ASSERT_TRUE(poly_dtype_eq(tcs[3].dtype_in, POLY_INT8));
  ASSERT_TRUE(poly_dtype_eq(tcs[3].dtype_out, POLY_INT32));
  ASSERT_STR_EQ(tcs[0].swizzle[0][0][0], "l4");
  ASSERT_STR_EQ(tcs[0].swizzle[0][0][1], "u0");

  tcs = poly_tc_get_amd("gfx942", &count);
  ASSERT_NOT_NULL(tcs);
  ASSERT_INT_EQ(count, 4);
  ASSERT_TRUE(poly_dtype_eq(tcs[0].dtype_in, POLY_FP8E5M2));
  ASSERT_TRUE(poly_dtype_eq(tcs[1].dtype_in, POLY_FP8E4M3));
  ASSERT_INT_EQ(tcs[0].dims[2], 32);
  ASSERT_INT_EQ(tcs[2].dims[2], 16);

  tcs = poly_tc_get_amd("gfx950", &count);
  ASSERT_NOT_NULL(tcs);
  ASSERT_INT_EQ(count, 8);
  ASSERT_INT_EQ(tcs[0].dims[2], 128);
  ASSERT_TRUE(poly_dtype_eq(tcs[0].dtype_in, POLY_FP8E5M2));
  ASSERT_INT_EQ(tcs[0].elements_per_thread[0], 32);
  ASSERT_INT_EQ(tcs[0].swizzle_len[0][2], 7);

  tcs = poly_tc_get_amd("gfx1200", &count);
  ASSERT_NOT_NULL(tcs);
  ASSERT_INT_EQ(count, 4);
  ASSERT_INT_EQ(tcs[0].threads, 32);
  ASSERT_TRUE(poly_dtype_eq(tcs[3].dtype_in, POLY_BFLOAT16));
  ASSERT_TRUE(poly_dtype_eq(tcs[3].dtype_out, POLY_BFLOAT16));
  PASS();
}

TEST(codegen, rdna3_tensor_core_catalogue_matches_all_pinned_fields) {
  const PolyDType inputs[] = {POLY_FLOAT16, POLY_FLOAT16, POLY_BFLOAT16, POLY_INT8};
  const PolyDType outputs[] = {POLY_FLOAT32, POLY_FLOAT16, POLY_FLOAT32, POLY_INT32};
  const char *opts = "l0l0l0l0l1u1u1u1";
  const char *swizzles[2][3][5] = {
      {{"l4", "u0", "u1", "u2", "l0"}, {"r1", "r2", "r3"}, {"l1", "l2", "l3", "r0"}},
      {{"l0", "l1", "l2", "l3", "l4"}, {"r1", "r2", "r3"}, {"u0", "u1", "u2", "r0"}}};
  const int lengths[] = {5, 3, 4};
  int count = 0;
  const PolyTensorCore *tcs = poly_tc_get_amd("gfx1100", &count);
  ASSERT_INT_EQ(count, 4);
  ASSERT_NOT_NULL(tcs);
  for (int i = 0; i < count; i++) {
    const PolyTensorCore *tc = &tcs[i];
    ASSERT_TRUE(poly_dtype_eq(tc->dtype_in, inputs[i]));
    ASSERT_TRUE(poly_dtype_eq(tc->dtype_out, outputs[i]));
    ASSERT_INT_EQ(tc->threads, 32);
    ASSERT_INT_EQ(tc->n_opts, 8);
    for (int j = 0; j < 3; j++) {
      ASSERT_INT_EQ(tc->dims[j], 16);
      ASSERT_INT_EQ(tc->elements_per_thread[j], j < 2 ? 16 : 8);
    }
    for (int j = 0; j < 8; j++) {
      ASSERT_INT_EQ(tc->opts[j].type, opts[2 * j]);
      ASSERT_INT_EQ(tc->opts[j].dim, opts[2 * j + 1] - '0');
    }
    for (int side = 0; side < 2; side++)
      for (int part = 0; part < 3; part++) {
        ASSERT_INT_EQ(tc->swizzle_len[side][part], lengths[part]);
        for (int j = 0; j < lengths[part]; j++)
          ASSERT_STR_EQ(tc->swizzle[side][part][j], swizzles[side][part][j]);
      }
  }
  PASS();
}

#ifdef POLY_HAS_HIP
TEST(codegen, hip_bf16_dynamic_vector_scalarizes_before_final_matcher) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a_buf = poly_test_uop_param(ctx, POLY_BFLOAT16, 16, 0, POLY_ADDR_GLOBAL);
  PolyUOp *b_buf = poly_test_uop_param(ctx, POLY_BFLOAT16, 16, 1, POLY_ADDR_GLOBAL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *a_addr_src[] = {a_buf, zero, four};
  PolyUOp *b_addr_src[] = {b_buf, zero, four};
  PolyUOp *a_addr = poly_uop(ctx, POLY_OP_SHRINK, POLY_BFLOAT16, a_addr_src, 3, poly_arg_none());
  PolyUOp *b_addr = poly_uop(ctx, POLY_OP_SHRINK, POLY_BFLOAT16, b_addr_src, 3, poly_arg_none());
  PolyUOp *a_vec = poly_uop1(ctx, POLY_OP_LOAD, POLY_BFLOAT16, a_addr, poly_arg_none());
  PolyUOp *b_vec = poly_uop1(ctx, POLY_OP_LOAD, POLY_BFLOAT16, b_addr, poly_arg_none());
  PolyUOp *add_vec = poly_uop2(ctx, POLY_OP_ADD, POLY_BFLOAT16, a_vec, b_vec, poly_arg_none());
  PolyUOp *zero_lane = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *zero_src[] = {zero_lane, zero_lane, zero_lane, zero_lane};
  PolyUOp *zero_vec = poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, zero_src, 4, poly_arg_none());
  PolyUOp *wmma_src[] = {add_vec, add_vec, zero_vec};
  int bf16_dims[] = {16, 16, 16};
  PolyUOp *wmma = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, wmma_src, 3,
      poly_arg_tensor_core(bf16_dims, POLY_BFLOAT16, "AMD", 64, NULL, NULL, false)
  );

  PolyUOp *rewritten = poly_rewrite_hip(ctx, poly_sink1(ctx, wmma));
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  int raw_bf16_consts = 0;
  int encoded_scalar_bf16_lanes = 0;
  int scalar_bf16_loads = 0;
  int vector_bf16_loads = 0;
  int scalar_f32_adds = 0;
  int vector_bf16_alu = 0;
  int vector_bf16_bitcasts = 0;
  int wmma_count = 0;
  PolyUOp *wmma_rewritten = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_CONST && poly_dtype_eq(topo[i]->dtype, POLY_BFLOAT16))
      raw_bf16_consts++;
    if (topo[i]->op == POLY_OP_BITCAST) {
      if (poly_dtype_eq(topo[i]->dtype, POLY_BFLOAT16))
        encoded_scalar_bf16_lanes++;
      else if (poly_dtype_eq(topo[i]->dtype, POLY_BFLOAT16) && poly_uop_max_numel(ctx, topo[i]) > 1)
        vector_bf16_bitcasts++;
    }
    if (topo[i]->op == POLY_OP_LOAD && poly_dtype_eq(topo[i]->dtype, POLY_BFLOAT16)) {
      if (poly_uop_max_numel(ctx, topo[i]) == 1)
        scalar_bf16_loads++;
      else
        vector_bf16_loads++;
    }
    if (topo[i]->op == POLY_OP_ADD && poly_dtype_eq(topo[i]->dtype, POLY_FLOAT32))
      scalar_f32_adds++;
    if (poly_opset_has(POLY_GROUP_ALU, topo[i]->op) &&
        poly_dtype_eq(topo[i]->dtype, POLY_BFLOAT16) && poly_uop_max_numel(ctx, topo[i]) > 1)
      vector_bf16_alu++;
    if (topo[i]->op == POLY_OP_WMMA) {
      wmma_count++;
      wmma_rewritten = topo[i];
    }
  }
  /* tinygrad@2026-08-22/a9069c177a9d codegen/__init__.py:123-156 scalarizes BF16
   * vector memory and ALU before HIPRenderer.extra_matcher encodes each lane
   * (codegen/late/coalesce.py:129-162; renderer/cstyle.py:510-526). */
  ASSERT_INT_EQ(raw_bf16_consts, 0);
  ASSERT_INT_EQ(encoded_scalar_bf16_lanes, 4);
  ASSERT_INT_EQ(scalar_bf16_loads, 8);
  ASSERT_INT_EQ(vector_bf16_loads, 0);
  ASSERT_INT_EQ(scalar_f32_adds, 4);
  ASSERT_INT_EQ(vector_bf16_alu, 0);
  ASSERT_INT_EQ(vector_bf16_bitcasts, 0);
  ASSERT_INT_EQ(wmma_count, 1);
  ASSERT_NOT_NULL(wmma_rewritten);
  ASSERT_INT_EQ(wmma_rewritten->src[0]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(wmma_rewritten->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(wmma_rewritten->src[0]->n_src, 4);
  ASSERT_INT_EQ(wmma_rewritten->src[1]->n_src, 4);
  for (int lane = 0; lane < 4; lane++) {
    ASSERT_INT_EQ(wmma_rewritten->src[0]->src[lane]->op, POLY_OP_BITCAST);
    ASSERT_INT_EQ(wmma_rewritten->src[1]->src[lane]->op, POLY_OP_BITCAST);
    ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[0]->src[lane]->dtype, POLY_BFLOAT16));
    ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[1]->src[lane]->dtype, POLY_BFLOAT16));
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, hip_float_and_half4_memory_use_renderer_vector_width) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType scalar_dtypes[] = {POLY_FLOAT16, POLY_FLOAT32};
  for (int d = 0; d < 2; d++) {
    PolyDType scalar = scalar_dtypes[d];
    PolyUOp *out_buf = poly_test_uop_param(ctx, scalar, 16, d * 3, POLY_ADDR_GLOBAL);
    PolyUOp *a_buf = poly_test_uop_param(ctx, scalar, 16, d * 3 + 1, POLY_ADDR_GLOBAL);
    PolyUOp *b_buf = poly_test_uop_param(ctx, scalar, 16, d * 3 + 2, POLY_ADDR_GLOBAL);
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
    PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
    PolyUOp *out_src[] = {out_buf, zero, four};
    PolyUOp *a_src[] = {a_buf, zero, four};
    PolyUOp *b_src[] = {b_buf, zero, four};
    PolyUOp *out_addr = poly_uop(ctx, POLY_OP_SHRINK, scalar, out_src, 3, poly_arg_none());
    PolyUOp *a_addr = poly_uop(ctx, POLY_OP_SHRINK, scalar, a_src, 3, poly_arg_none());
    PolyUOp *b_addr = poly_uop(ctx, POLY_OP_SHRINK, scalar, b_src, 3, poly_arg_none());
    PolyUOp *a_vec = poly_uop1(ctx, POLY_OP_LOAD, scalar, a_addr, poly_arg_none());
    PolyUOp *b_vec = poly_uop1(ctx, POLY_OP_LOAD, scalar, b_addr, poly_arg_none());
    PolyUOp *add_vec = poly_uop2(ctx, POLY_OP_ADD, scalar, a_vec, b_vec, poly_arg_none());
    PolyUOp *sink = poly_sink1(
        ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_addr, add_vec, poly_arg_none())
    );

    PolyUOp *rewritten = poly_rewrite_hip(ctx, sink);
    ASSERT_NOT_NULL(rewritten);
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
    ASSERT_NOT_NULL(topo);
    int scalar_loads = 0, vector_loads = 0;
    int scalar_adds = 0, vector_adds = 0;
    int scalar_stores = 0, vector_stores = 0;
    for (int i = 0; i < n_topo; i++) {
      if (topo[i]->op == POLY_OP_LOAD && poly_dtype_eq(topo[i]->dtype, scalar)) {
        if (poly_uop_max_numel(ctx, topo[i]) == 1)
          scalar_loads++;
        else if (poly_uop_max_numel(ctx, topo[i]) == 4)
          vector_loads++;
      }
      if (topo[i]->op == POLY_OP_ADD && poly_dtype_eq(topo[i]->dtype, scalar)) {
        if (poly_uop_max_numel(ctx, topo[i]) == 1)
          scalar_adds++;
        else if (poly_uop_max_numel(ctx, topo[i]) == 4)
          vector_adds++;
      }
      if (topo[i]->op == POLY_OP_STORE && topo[i]->n_src >= 2 &&
          poly_dtype_eq(topo[i]->src[1]->dtype, scalar)) {
        if (poly_uop_max_numel(ctx, topo[i]->src[1]) == 1)
          scalar_stores++;
        else if (poly_uop_max_numel(ctx, topo[i]->src[1]) == 4)
          vector_stores++;
      }
    }
    /* tinygrad@2026-08-22/a9069c177a9d HIP inherits supports_float4=True: aligned float16x4
     * and float32x4 memory operations survive while devectorize_alu scalarizes
     * the ADD (renderer/__init__.py:59-64; codegen/__init__.py:123-156;
     * codegen/late/coalesce.py:129-162). */
    ASSERT_INT_EQ(scalar_loads, 0);
    ASSERT_INT_EQ(vector_loads, 2);
    ASSERT_INT_EQ(scalar_adds, 4);
    ASSERT_INT_EQ(vector_adds, 0);
    ASSERT_INT_EQ(scalar_stores, 0);
    ASSERT_INT_EQ(vector_stores, 1);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, hip_bf16_vector_memory_scalarizes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out_buf = poly_test_uop_param(ctx, POLY_BFLOAT16, 16, 0, POLY_ADDR_GLOBAL);
  PolyUOp *in_buf = poly_test_uop_param(ctx, POLY_BFLOAT16, 16, 1, POLY_ADDR_GLOBAL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *out_src[] = {out_buf, zero, four};
  PolyUOp *in_src[] = {in_buf, zero, four};
  PolyUOp *out_addr = poly_uop(ctx, POLY_OP_SHRINK, POLY_BFLOAT16, out_src, 3, poly_arg_none());
  PolyUOp *in_addr = poly_uop(ctx, POLY_OP_SHRINK, POLY_BFLOAT16, in_src, 3, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_LOAD, POLY_BFLOAT16, in_addr, poly_arg_none());
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_addr, value, poly_arg_none()));
  PolyUOp *rewritten = poly_rewrite_hip(ctx, sink);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  int scalar_loads = 0, vector_loads = 0;
  int scalar_stores = 0, vector_stores = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_LOAD && poly_dtype_eq(topo[i]->dtype, POLY_BFLOAT16)) {
      if (poly_uop_max_numel(ctx, topo[i]) == 1)
        scalar_loads++;
      else if (poly_uop_max_numel(ctx, topo[i]) == 4)
        vector_loads++;
    }
    if (topo[i]->op == POLY_OP_STORE && topo[i]->n_src >= 2 &&
        poly_dtype_eq(topo[i]->src[1]->dtype, POLY_BFLOAT16)) {
      if (poly_uop_max_numel(ctx, topo[i]->src[1]) == 1)
        scalar_stores++;
      else if (poly_uop_max_numel(ctx, topo[i]->src[1]) == 4)
        vector_stores++;
    }
  }
  /* tinygrad@2026-08-22/a9069c177a9d memory_coalescing only retains vector
   * memory for float/half/fp8; BF16 takes the scalar fallback
   * (codegen/late/coalesce.py:129-162). */
  ASSERT_INT_EQ(scalar_loads, 4);
  ASSERT_INT_EQ(vector_loads, 0);
  ASSERT_INT_EQ(scalar_stores, 4);
  ASSERT_INT_EQ(vector_stores, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, hip_gated_bf16_load_legalizes_late_zero_alternative) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *buf = poly_test_uop_param(ctx, POLY_BFLOAT16, 1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *gate = poly_test_uop_param(ctx, POLY_BOOL, -1, 1, POLY_ADDR_GLOBAL);
  PolyUOp *offset = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *gated_offset =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, offset, invalid, poly_arg_none());
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_BFLOAT16, buf, gated_offset, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_BFLOAT16, index, poly_arg_none());
  PolyUOp *rewritten = poly_rewrite_hip(ctx, poly_sink1(ctx, load));
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *gated_load = NULL;
  int raw_bf16_consts = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_LOAD && poly_dtype_eq(topo[i]->dtype, POLY_BFLOAT16) &&
        topo[i]->n_src == 3)
      gated_load = topo[i];
    if (topo[i]->op == POLY_OP_CONST && poly_dtype_eq(topo[i]->dtype, POLY_BFLOAT16))
      raw_bf16_consts++;
  }
  /* Pinned codegen/__init__.py:124-137 moves gates before the renderer-final
   * matcher. The BF16 zero alternative introduced by gater.py:14-18 is
   * therefore encoded by HIPRenderer.extra_matcher instead of remaining a raw
   * numeric CONST in uint16 storage. */
  ASSERT_NOT_NULL(gated_load);
  ASSERT_INT_EQ(gated_load->src[1]->op, POLY_OP_BITCAST);
  ASSERT_TRUE(poly_dtype_eq(gated_load->src[1]->dtype, POLY_BFLOAT16));
  ASSERT_INT_EQ(raw_bf16_consts, 0);

  poly_ctx_destroy(ctx);
  PASS();
}
#endif

TEST(codegen, webgpu_decomposes_u64_threefry_buffers_to_u32_lanes) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = program_param(ctx, POLY_UINT64, 1, 0);
  PolyUOp *xbuf = program_param(ctx, POLY_UINT64, 1, 1);
  PolyUOp *kbuf = program_param(ctx, POLY_UINT64, 1, 2);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *oidx = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT64, out, zero, poly_arg_none());
  PolyUOp *xidx = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT64, xbuf, zero, poly_arg_none());
  PolyUOp *kidx = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT64, kbuf, zero, poly_arg_none());
  PolyUOp *x = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT64, xidx, poly_arg_none());
  PolyUOp *key = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT64, kidx, poly_arg_none());
  PolyUOp *value = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT64, x, key, poly_arg_none());
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none()));

  PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  ASSERT_TRUE(n_topo > 0);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_THREEFRY), 0);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_LOAD), 4);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_STORE), 2);

  bool lane_seen[3][2] = {{false}};
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    PolyDType scalar = u->dtype;
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
    if (u->op == POLY_OP_PARAM) {
      ASSERT_INT_EQ(u->dtype.bitsize, 32);
      ASSERT_INT_EQ(poly_uop_max_numel(ctx, u), 2);
    }
    if ((u->op != POLY_OP_LOAD && u->op != POLY_OP_STORE) || u->n_src < 1) continue;
    PolyUOp *idx = u->src[0];
    ASSERT_TRUE(idx->op == POLY_OP_INDEX && idx->n_src >= 2);
    PolyUOp *param = idx->src[0], *offset = idx->src[1];
    ASSERT_TRUE(
        param->op == POLY_OP_PARAM && param->arg.kind == POLY_ARG_PARAM && param->arg.param
    );
    ASSERT_INT_EQ(offset->op, POLY_OP_CAST);
    ASSERT_TRUE(poly_dtype_eq(offset->dtype, POLY_INT32));
    ASSERT_INT_EQ(offset->n_src, 1);
    offset = offset->src[0];
    ASSERT_TRUE(
        offset->op == POLY_OP_CONST && poly_dtype_eq(offset->dtype, POLY_WEAKINT) &&
        offset->arg.kind == POLY_ARG_INT
    );
    ASSERT_TRUE(param->arg.param->slot >= 0 && param->arg.param->slot < 3);
    ASSERT_TRUE(offset->arg.i >= 0 && offset->arg.i < 2);
    lane_seen[param->arg.param->slot][offset->arg.i] = true;
  }
  for (int p = 0; p < 3; p++)
    for (int lane = 0; lane < 2; lane++)
      ASSERT_TRUE(lane_seen[p][lane]);

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_wgsl(ctx, lin, n_lin, "threefry_u64_buffers");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "array<u32>"));
  ASSERT_TRUE(strstr(src, "<<32") == NULL && strstr(src, "<< 32") == NULL);
  ASSERT_TRUE(strstr(src, ">>32") == NULL && strstr(src, ">> 32") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, webgpu_decomposes_exact_uint64_bigint_const_to_u32_lanes) {
  /* Pinned pm_long_decomp selects low/high uint32 lanes from the exact Python
   * CONST arg (uop/decompositions.py:528-529). */
  PolyCtx *ctx = poly_ctx_new();
  PolyInt value = {0};
  ASSERT_TRUE(poly_int_from_decimal(&value, "18446744073709550593"));
  PolyUOp *constant = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_int_as_arg(&value));
  poly_int_free(&value);

  PolyUOp *out = program_param(ctx, POLY_UINT64, 1, 0);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT64, out, zero, poly_arg_none());
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, constant, poly_arg_none()));

  PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n);
  ASSERT_NOT_NULL(topo);
  bool saw_low = false, saw_high = false;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_CAST || !poly_dtype_eq(u->dtype, POLY_UINT32) || u->n_src != 1 ||
        u->src[0]->op != POLY_OP_CONST || !poly_dtype_eq(u->src[0]->dtype, POLY_WEAKINT))
      continue;
    uint64_t lane = poly_arg_integer_to_u64_mod(u->src[0]->arg);
    if (lane == UINT64_C(4294966273)) saw_low = true;
    if (lane == UINT64_C(4294967295)) saw_high = true;
  }
  ASSERT_TRUE(saw_low);
  ASSERT_TRUE(saw_high);
  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, webgpu_decomposes_long_divmod_without_shift32) {
  struct {
    PolyDType dtype;
    PolyOps op;
  } cases[] = {
      {POLY_INT64, POLY_OP_CDIV},
      {POLY_INT64, POLY_OP_CMOD},
      {POLY_UINT64, POLY_OP_CDIV},
      {POLY_UINT64, POLY_OP_CMOD},
  };

  for (int ci = 0; ci < (int)(sizeof(cases) / sizeof(cases[0])); ci++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *out = program_param(ctx, cases[ci].dtype, 1, 0);
    PolyUOp *abuf = program_param(ctx, cases[ci].dtype, 1, 1);
    PolyUOp *bbuf = program_param(ctx, cases[ci].dtype, 1, 2);
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
    PolyUOp *oidx = poly_uop2(ctx, POLY_OP_INDEX, cases[ci].dtype, out, zero, poly_arg_none());
    PolyUOp *aidx = poly_uop2(ctx, POLY_OP_INDEX, cases[ci].dtype, abuf, zero, poly_arg_none());
    PolyUOp *bidx = poly_uop2(ctx, POLY_OP_INDEX, cases[ci].dtype, bbuf, zero, poly_arg_none());
    PolyUOp *a = poly_uop1(ctx, POLY_OP_LOAD, cases[ci].dtype, aidx, poly_arg_none());
    PolyUOp *b = poly_uop1(ctx, POLY_OP_LOAD, cases[ci].dtype, bidx, poly_arg_none());
    PolyUOp *value = poly_uop2(ctx, cases[ci].op, cases[ci].dtype, a, b, poly_arg_none());
    PolyUOp *sink =
        poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none()));

    PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
    ASSERT_NOT_NULL(rewritten);
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
    ASSERT_NOT_NULL(topo);
    ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_LOAD), 4);
    ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_STORE), 2);
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      PolyDType scalar = u->dtype;
      ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
      if ((u->op == POLY_OP_SHL || u->op == POLY_OP_SHR) && u->n_src >= 2 &&
          u->src[1]->op == POLY_OP_CONST && u->src[1]->arg.kind == POLY_ARG_INT)
        ASSERT_TRUE(u->src[1]->arg.i >= 0 && u->src[1]->arg.i < 32);
      if (u->op == POLY_OP_PARAM) {
        ASSERT_INT_EQ(u->dtype.bitsize, 32);
        ASSERT_INT_EQ(poly_uop_max_numel(ctx, u), 2);
      }
    }
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(codegen, webgpu_long_shift_matches_current_word_decomposition) {
  /* Tinygrad 2026-08-22/a9069c177a9d dtype.py:40-48 consumes the count's
   * low word and branches only on low_word >= 32. */
  struct {
    PolyOps op;
    PolyDType dtype;
    uint64_t value;
    uint32_t shift;
    uint64_t expected;
  } cases[] = {
      {POLY_OP_SHR, POLY_INT64, UINT64_C(0x0000000080000000), 1, UINT64_C(0x0000000040000000)},
      {POLY_OP_SHR, POLY_INT64, UINT64_MAX, 32, UINT64_MAX},
      {POLY_OP_SHR, POLY_INT64, UINT64_C(0x8000000000000000), 64, UINT64_C(0xffffffff80000000)},
      {POLY_OP_SHR, POLY_INT64, UINT64_C(0x7fffffffffffffff), 64, UINT64_C(0x000000007fffffff)},
      {POLY_OP_SHL, POLY_INT64, UINT64_C(0x0000000080000000), 1, UINT64_C(0x0000000100000000)},
      {POLY_OP_SHL, POLY_INT64, UINT64_MAX, 64, UINT64_C(0xffffffff00000000)},
      {POLY_OP_SHR, POLY_UINT64, UINT64_MAX, 1, UINT64_C(0x7fffffffffffffff)},
  };

  for (int ci = 0; ci < (int)(sizeof(cases) / sizeof(cases[0])); ci++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *out = program_param(ctx, cases[ci].dtype, 1, 0);
    PolyUOp *in = program_param(ctx, cases[ci].dtype, 1, 1);
    PolyUOp *shift = program_param(ctx, POLY_UINT32, 1, 2);
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
    PolyUOp *oidx = poly_uop2(ctx, POLY_OP_INDEX, cases[ci].dtype, out, zero, poly_arg_none());
    PolyUOp *iidx = poly_uop2(ctx, POLY_OP_INDEX, cases[ci].dtype, in, zero, poly_arg_none());
    PolyUOp *sidx = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT32, shift, zero, poly_arg_none());
    PolyUOp *value = poly_uop2(
        ctx, cases[ci].op, cases[ci].dtype,
        poly_uop1(ctx, POLY_OP_LOAD, cases[ci].dtype, iidx, poly_arg_none()),
        poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, sidx, poly_arg_none()), poly_arg_none()
    );
    PolyUOp *sink =
        poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none()));
    PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
    ASSERT_NOT_NULL(rewritten);

    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
    ASSERT_NOT_NULL(topo);
    for (int i = 0; i < n_topo; i++) {
      PolyDType scalar = topo[i]->dtype;
      ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
    }

    int n_lin = 0;
    PolyUOp **lin = poly_do_linearize(ctx, rewritten, &n_lin);
    ASSERT_NOT_NULL(lin);
    uint32_t out_words[2] = {0, 0};
    uint32_t in_words[2] = {(uint32_t)cases[ci].value, (uint32_t)(cases[ci].value >> 32)};
    uint32_t shift_value = cases[ci].shift;
    void *args[3] = {out_words, in_words, &shift_value};
    ASSERT_INT_EQ(poly_interp_eval(ctx, lin, n_lin, args, 3), 0);
    uint64_t got = (uint64_t)out_words[0] | ((uint64_t)out_words[1] << 32);
    if (got != cases[ci].expected) {
      free(lin);
      poly_ctx_destroy(ctx);
      FAIL(
          "case %d %s value=0x%016" PRIx64 " shift=%" PRIu32 " got=0x%016" PRIx64
          " expected=0x%016" PRIx64,
          ci, poly_op_name(cases[ci].op), cases[ci].value, cases[ci].shift, got, cases[ci].expected
      );
    }

    free(lin);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(codegen, webgpu_long_shift_uses_dynamic_count_low_word) {
  struct {
    uint64_t value;
    uint64_t shift;
    uint64_t expected;
  } cases[] = {
      {UINT64_C(0x0000000080000000), 1, UINT64_C(0x0000000040000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(1) << 32, UINT64_C(0x8000000000000000)},
  };

  for (int ci = 0; ci < (int)(sizeof(cases) / sizeof(cases[0])); ci++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *out = program_param(ctx, POLY_INT64, 1, 0);
    PolyUOp *in = program_param(ctx, POLY_INT64, 1, 1);
    PolyUOp *shift = program_param(ctx, POLY_INT64, 1, 2);
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
    PolyUOp *oidx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT64, out, zero, poly_arg_none());
    PolyUOp *iidx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT64, in, zero, poly_arg_none());
    PolyUOp *sidx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT64, shift, zero, poly_arg_none());
    PolyUOp *value = poly_uop2(
        ctx, POLY_OP_SHR, POLY_INT64,
        poly_uop1(ctx, POLY_OP_LOAD, POLY_INT64, iidx, poly_arg_none()),
        poly_uop1(ctx, POLY_OP_LOAD, POLY_INT64, sidx, poly_arg_none()), poly_arg_none()
    );
    PolyUOp *sink =
        poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none()));
    PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
    ASSERT_NOT_NULL(rewritten);

    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
    ASSERT_NOT_NULL(topo);
    for (int i = 0; i < n_topo; i++) {
      PolyDType scalar = topo[i]->dtype;
      ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
    }

    int n_lin = 0;
    PolyUOp **lin = poly_do_linearize(ctx, rewritten, &n_lin);
    ASSERT_NOT_NULL(lin);
    uint32_t out_words[2] = {0, 0};
    uint32_t in_words[2] = {(uint32_t)cases[ci].value, (uint32_t)(cases[ci].value >> 32)};
    uint32_t shift_words[2] = {(uint32_t)cases[ci].shift, (uint32_t)(cases[ci].shift >> 32)};
    void *args[3] = {out_words, in_words, shift_words};
    ASSERT_INT_EQ(poly_interp_eval(ctx, lin, n_lin, args, 3), 0);
    uint64_t got = (uint64_t)out_words[0] | ((uint64_t)out_words[1] << 32);
    if (got != cases[ci].expected) {
      free(lin);
      poly_ctx_destroy(ctx);
      FAIL(
          "case %d value=0x%016" PRIx64 " shift=0x%016" PRIx64 " got=0x%016" PRIx64
          " expected=0x%016" PRIx64,
          ci, cases[ci].value, cases[ci].shift, got, cases[ci].expected
      );
    }

    free(lin);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(codegen, webgpu_decomposes_raw_mixed_width_long_shift) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = program_param(ctx, POLY_INT64, 1, 0);
  PolyUOp *in = program_param(ctx, POLY_INT32, 1, 1);
  PolyUOp *shift = program_param(ctx, POLY_UINT32, 1, 2);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *input = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_INT32,
      poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, in, zero, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *value = poly_uop2(
      ctx, POLY_OP_SHL, POLY_INT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, input, poly_arg_none()),
      poly_uop1(
          ctx, POLY_OP_LOAD, POLY_UINT32,
          poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT32, shift, zero, poly_arg_none()), poly_arg_none()
      ),
      poly_arg_none()
  );
  PolyUOp *sink = poly_sink1(
      ctx, poly_uop2(
               ctx, POLY_OP_STORE, POLY_VOID,
               poly_uop2(ctx, POLY_OP_INDEX, POLY_INT64, out, zero, poly_arg_none()), value,
               poly_arg_none()
           )
  );
  PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyDType scalar = topo[i]->dtype;
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
  }

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  uint32_t out_words[2] = {0, 0};
  int32_t input_value = 7;
  uint32_t shift_value = 2;
  void *args[3] = {out_words, &input_value, &shift_value};
  ASSERT_INT_EQ(poly_interp_eval(ctx, lin, n_lin, args, 3), 0);
  ASSERT_TRUE(((uint64_t)out_words[0] | ((uint64_t)out_words[1] << 32)) == UINT64_C(28));

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, c_renderer_matches_pinned_clang_transcendental_caps) {
  /* Pinned ClangRenderer removes EXP2, LOG2, and SIN from code_for_op so
   * codegen applies the shared transcendental decompositions
   * (tinygrad/renderer/cstyle.py:246-269). */
  PolyRendererCaps caps = poly_c_renderer_caps();
  ASSERT_FALSE(caps.has_exp2);
  ASSERT_FALSE(caps.has_log2);
  ASSERT_FALSE(caps.has_sin);
  PASS();
}

TEST(codegen, simplifying_pow2_floor_ops_precede_generic_floor_decomposition) {
  /* Current tinygrad codegen/decomp/op.py:get_simplifying_rewrite_patterns
   * lowers signed floor division/modulo by a power of two before the generic
   * floor-to-C rules. Arithmetic SHR and two's-complement AND are exact for
   * negative inputs too. */
  PolyCtx *ctx = poly_ctx_new();
  PolyParamArg arg = {
      .slot = -1,
      .name = "x",
      .min_val = -16,
      .max_val = 16,
      .has_minmax = true,
      .addrspace = POLY_ADDR_ALU,
  };
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_STACK, POLY_VOID, poly_arg_none());
  PolyUOp *x = poly_uop1(ctx, POLY_OP_PARAM, POLY_INT32, shape, poly_arg_param(&arg));
  PolyUOp *eight = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyRendererCaps caps = poly_c_renderer_caps();

  PolyUOp *div = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT32, x, eight, poly_arg_none());
  PolyUOp *div_out = poly_graph_rewrite(ctx, div, poly_get_simplifying_rewrite_patterns(caps));
  ASSERT_INT_EQ(div_out->op, POLY_OP_SHR);
  ASSERT_TRUE(div_out->src[0] == x);
  ASSERT_INT_EQ(div_out->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(div_out->src[1]->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(div_out->src[1]->arg.i, 3);

  PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INT32, x, eight, poly_arg_none());
  PolyUOp *mod_out = poly_graph_rewrite(ctx, mod, poly_get_simplifying_rewrite_patterns(caps));
  ASSERT_INT_EQ(mod_out->op, POLY_OP_AND);
  ASSERT_TRUE(mod_out->src[0] == x);
  ASSERT_INT_EQ(mod_out->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(mod_out->src[1]->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(mod_out->src[1]->arg.i, 7);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, late_demorgan_uses_renderer_or) {
  /* Current tinygrad codegen/decomp/op.py:get_late_rewrite_patterns rewrites
   * logical_not(x) & logical_not(y) to logical_not(x | y). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_STACK, POLY_VOID, poly_arg_none());
  PolyParamArg x_arg = {
      .slot = -1,
      .name = "x",
      .min_val = 0,
      .max_val = 1,
      .has_minmax = true,
      .addrspace = POLY_ADDR_ALU,
  };
  PolyParamArg y_arg = x_arg;
  y_arg.name = "y";
  PolyUOp *x = poly_uop1(ctx, POLY_OP_PARAM, POLY_BOOL, shape, poly_arg_param(&x_arg));
  PolyUOp *y = poly_uop1(ctx, POLY_OP_PARAM, POLY_BOOL, shape, poly_arg_param(&y_arg));
  PolyUOp *true_const = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *not_x = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, x, true_const, poly_arg_none());
  PolyUOp *not_y = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, y, true_const, poly_arg_none());
  PolyUOp *both = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, not_x, not_y, poly_arg_none());

  PolyUOp *out =
      poly_graph_rewrite(ctx, both, poly_get_late_rewrite_patterns(poly_c_renderer_caps()));
  ASSERT_INT_EQ(out->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(out->src[0]->op, POLY_OP_OR);
  ASSERT_TRUE(out->src[0]->src[0] == x);
  ASSERT_TRUE(out->src[0]->src[1] == y);
  ASSERT_TRUE(out->src[1] == true_const);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, late_fdiv_rewrites_follow_renderer_capability) {
  /* tinygrad@2026-08-22/a9069c177a9d codegen/decomp/op.py:134-136 gates both
   * RECIPROCAL(x) -> FDIV(1,x) and a*FDIV(1,b) -> FDIV(a,b) on FDIV being
   * present in the renderer op table. UPat multiplication is commutative. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(1));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *div = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, one, b, poly_arg_none());
  PolyUOp *raw = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, a, div, poly_arg_none());

  PolyRendererCaps without_fdiv = {0};
  PolyUOp *kept = poly_graph_rewrite(ctx, raw, poly_get_late_rewrite_patterns(without_fdiv));
  ASSERT_INT_EQ(kept->op, POLY_OP_MUL);
  ASSERT_INT_EQ(kept->n_src, 2);
  ASSERT_INT_EQ(kept->src[1]->op, POLY_OP_FDIV);

  PolyRendererCaps with_fdiv = {.has_fdiv = true};
  PolyUOp *folded = poly_graph_rewrite(ctx, raw, poly_get_late_rewrite_patterns(with_fdiv));
  ASSERT_INT_EQ(folded->op, POLY_OP_FDIV);
  ASSERT_INT_EQ(folded->n_src, 2);
  ASSERT_TRUE(folded->src[0] == a);
  ASSERT_TRUE(folded->src[1] == b);

  PolyUOp *raw_reversed = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, div, a, poly_arg_none());
  PolyUOp *folded_reversed =
      poly_graph_rewrite(ctx, raw_reversed, poly_get_late_rewrite_patterns(with_fdiv));
  ASSERT_INT_EQ(folded_reversed->op, POLY_OP_FDIV);
  ASSERT_INT_EQ(folded_reversed->n_src, 2);
  ASSERT_TRUE(folded_reversed->src[0] == a);
  ASSERT_TRUE(folded_reversed->src[1] == b);

  PolyUOp *reciprocal = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, b, poly_arg_none());
  PolyUOp *reciprocal_out =
      poly_graph_rewrite(ctx, reciprocal, poly_get_late_rewrite_patterns(with_fdiv));
  ASSERT_INT_EQ(reciprocal_out->op, POLY_OP_FDIV);
  ASSERT_INT_EQ(reciprocal_out->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(reciprocal_out->src[0]->dtype, POLY_WEAKFLOAT));
  ASSERT_TRUE(
      reciprocal_out->src[0]->arg.kind == POLY_ARG_FLOAT && reciprocal_out->src[0]->arg.f == 1.0
  );
  ASSERT_TRUE(reciprocal_out->src[1] == b);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, transcendental_pow2if_dtype_follows_integer_input) {
  /* Pinned tinygrad uop/decompositions.py:29-32 chooses pow2if's float result
   * from q.dtype: int32 -> float32 and int64 -> float64. Its f64
   * payne_hanek_reduction keeps f64 intermediates, while the int32 residual
   * exponent still intentionally creates a float32 pow2 value. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = program_param(ctx, POLY_FLOAT64, 1, 0);
  PolyUOp *in = program_param(ctx, POLY_FLOAT64, 1, 1);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *oidx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT64, out, zero, poly_arg_none());
  PolyUOp *iidx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT64, in, zero, poly_arg_none());
  PolyUOp *value = poly_uop1(
      ctx, POLY_OP_SIN, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, iidx, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none()));
  PolyRewriteOpts opts = {
      .optimize = false,

      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  opts.caps.has_sin = false;
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0, i32_to_f32 = 0, u64_to_f64 = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_BITCAST || u->n_src != 1) continue;
    PolyDType from = u->src[0]->dtype;
    PolyDType to = u->dtype;
    ASSERT_INT_EQ(from.bitsize, to.bitsize);
    if (poly_dtype_eq(from, POLY_INT32) && poly_dtype_eq(to, POLY_FLOAT32)) i32_to_f32++;
    if (poly_dtype_eq(from, POLY_UINT64) && poly_dtype_eq(to, POLY_FLOAT64)) u64_to_f64++;
  }
  ASSERT_TRUE(i32_to_f32 > 0);
  ASSERT_TRUE(u64_to_f64 > 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, bf16_transcendental_widens_to_f32_like_tinygrad) {
  /* Current tinygrad get_transcendental_patterns keeps
   * BF16 outside TRANSCENDENTAL_DTYPES and rewrites it as
   * CAST(BF16, EXP2(CAST(F32, input))) after devectorizer2 scalarizes the
   * shaped PARAM. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyParamArg arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *input = poly_uop1(ctx, POLY_OP_PARAM, POLY_BFLOAT16, shape, poly_arg_param(&arg));
  PolyUOp *raw = poly_uop1(ctx, POLY_OP_EXP2, POLY_BFLOAT16, input, poly_arg_none());
  PolyUOp *lanes = poly_apply_devectorizer2_stage(ctx, raw, (PolyRendererCaps){.max_vec_width = 1});
  ASSERT_NOT_NULL(lanes);
  ASSERT_INT_EQ(lanes->op, POLY_OP_STACK);
  PolyUOp *rewritten =
      poly_graph_rewrite(ctx, lanes, poly_get_transcendental_patterns((PolyRendererCaps){0}));
  ASSERT_NOT_NULL(rewritten);
  ASSERT_EQ(rewritten->op, POLY_OP_STACK);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_BFLOAT16));
  ASSERT_INT_EQ(rewritten->n_src, 1);

  int n_topo = 0;
  int raw_exp2 = 0, bf16_nodes = 0, bf16_to_f32 = 0, f32_to_bf16 = 0;
  int f32_bitcasts = 0, bf16_bitcasts = 0;
  int floordiv_nodes = 0, cdiv_nodes = 0, cmod_nodes = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  ASSERT_INT_EQ(n_topo, 72);
  ASSERT_TRUE(
      normalized_topology_fingerprint(ctx, topo, n_topo, rewritten) == UINT64_C(0x765bb834204fb131)
  );
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_EXP2) raw_exp2++;
    if (u->op == POLY_OP_FLOORDIV) floordiv_nodes++;
    if (u->op == POLY_OP_CDIV) cdiv_nodes++;
    if (u->op == POLY_OP_CMOD) cmod_nodes++;
    if (poly_dtype_eq(u->dtype, POLY_BFLOAT16)) bf16_nodes++;
    if (u->op == POLY_OP_CAST && u->n_src == 1 && poly_dtype_eq(u->src[0]->dtype, POLY_BFLOAT16) &&
        poly_dtype_eq(u->dtype, POLY_FLOAT32))
      bf16_to_f32++;
    if (u->op == POLY_OP_CAST && u->n_src == 1 && poly_dtype_eq(u->src[0]->dtype, POLY_FLOAT32) &&
        poly_dtype_eq(u->dtype, POLY_BFLOAT16))
      f32_to_bf16++;
    if (u->op == POLY_OP_BITCAST && poly_dtype_eq(u->dtype, POLY_FLOAT32)) f32_bitcasts++;
    if (u->op == POLY_OP_BITCAST && poly_dtype_eq(u->dtype, POLY_BFLOAT16)) bf16_bitcasts++;
  }
  ASSERT_INT_EQ(raw_exp2, 0);
  ASSERT_INT_EQ(bf16_nodes, 4);
  ASSERT_INT_EQ(bf16_to_f32, 1);
  ASSERT_INT_EQ(f32_to_bf16, 1);
  ASSERT_INT_EQ(f32_bitcasts, 2);
  ASSERT_INT_EQ(bf16_bitcasts, 0);
  ASSERT_INT_EQ(floordiv_nodes, 1);
  ASSERT_INT_EQ(cdiv_nodes, 0);
  ASSERT_INT_EQ(cmod_nodes, 0);
  free(topo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, shaped_transcendentals_devectorize_before_decomposition) {
  /* Current devectorizer2 turns shaped SIN into scalar lanes before current
   * transcendental decomposition (tinygrad/codegen/__init__.py:148-163,365). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType dtypes[] = {POLY_BFLOAT16, POLY_FLOAT64};
  int expected_nodes[] = {357, 428};
  uint64_t expected_hashes[] = {UINT64_C(0xdc76be45b23dd9e3), UINT64_C(0xc23a158242127c6f)};
  for (int d = 0; d < 2; d++) {
    PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
    PolyParamArg arg = {.slot = d, .addrspace = POLY_ADDR_GLOBAL};
    PolyUOp *input = poly_uop1(ctx, POLY_OP_PARAM, dtypes[d], shape, poly_arg_param(&arg));
    PolyUOp *raw = poly_uop1(ctx, POLY_OP_SIN, dtypes[d], input, poly_arg_none());
    PolyUOp *lanes =
        poly_apply_devectorizer2_stage(ctx, raw, (PolyRendererCaps){.max_vec_width = 1});
    ASSERT_NOT_NULL(lanes);
    ASSERT_INT_EQ(lanes->op, POLY_OP_STACK);
    ASSERT_INT_EQ(lanes->n_src, 2);
    PolyPatternMatcher *transcendental = poly_pm_concat(
        poly_symbolic_simple(), poly_get_transcendental_patterns((PolyRendererCaps){0})
    );
    ASSERT_NOT_NULL(transcendental);
    PolyUOp *rewritten = poly_graph_rewrite(ctx, lanes, transcendental);
    poly_pm_destroy(transcendental);
    ASSERT_NOT_NULL(rewritten);
    ASSERT_INT_EQ(rewritten->op, POLY_OP_STACK);
    ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, dtypes[d]));
    ASSERT_INT_EQ(rewritten->n_src, 2);

    int n_topo = 0, raw_sin = 0;
    PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n_topo);
    ASSERT_NOT_NULL(topo);
    for (int i = 0; i < n_topo; i++) {
      raw_sin += topo[i]->op == POLY_OP_SIN;
    }
    ASSERT_INT_EQ(raw_sin, 0);
    ASSERT_INT_EQ(n_topo, expected_nodes[d]);
    uint64_t fingerprint = normalized_topology_fingerprint(ctx, topo, n_topo, rewritten);
    if (fingerprint != expected_hashes[d])
      FAIL(
          "dtype=%s topology fingerprint=0x%016" PRIx64 " expected=0x%016" PRIx64, dtypes[d].name,
          fingerprint, expected_hashes[d]
      );
    poly_toposort_free(topo);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, partial_reshape_index_matches_tinygrad_mop) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned schedule/rangeify.py:63-77 collapses
   * PARAM(3).RESHAPE(1,3).INDEX(0) to PARAM(3) because the unindexed output
   * suffix exactly matches the input shape. */
  PolyUOp *like = poly_reshape(ctx, poly_buffer_f32(ctx, 3), (int64_t[]){1, 3}, 2);
  PolyUOp *input = poly_uop_placeholder_like(ctx, like, 1);
  PolyUOp *output = poly_uop_placeholder_like(ctx, poly_buffer_f32(ctx, 1), 0);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *partial = poly_uop_index(ctx, input, &zero, 1);
  PolyUOp *value = poly_uop_index(ctx, partial, &two, 1);
  PolyUOp *address = poly_uop_index(ctx, output, &zero, 1);
  PolyUOp *store = poly_uop_store(ctx, address, value);
  PolyUOp *sink = poly_uop_sink(ctx, &store, 1);
  ASSERT_NOT_NULL(sink);

  PolyRewriteOpts opts = {
      .optimize = false,

      .caps = poly_c_renderer_caps(),
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_RESHAPE), 0);

  int n_linear = 0;
  PolyUOp **linear = poly_test_full_rewrite_and_linearize_ex(ctx, sink, opts, &n_linear);
  ASSERT_NOT_NULL(linear);
  char *source = poly_render_c(ctx, linear, n_linear, "partial_reshape_index");
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(strstr(source, "data1+2"));
  PolyProgram *program = poly_compile_c(source, "partial_reshape_index");
  ASSERT_NOT_NULL(program);

  float out = 0.0f;
  float in[3] = {11.0f, 22.0f, 33.0f};
  void *args[2] = {&out, in};
  poly_program_call(program, args, 2);
  ASSERT_FLOAT_EQ(out, 33.0f, 1e-6);

  poly_program_destroy(program);
  free(source);
  free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, wgsl_narrow_cast_and_alu_results) {
  /* PG-DIV-008: WGSL's i32/u32 registers must preserve the UOp's 8/16-bit
   * value before a later widening. Pinned PythonProgram truncates here;
   * pinned WGSLRenderer currently does not. */
  PolyDType types[] = {POLY_INT8, POLY_UINT8, POLY_INT16, POLY_UINT16};
  for (int i = 0; i < 4; i++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *weak = poly_const_int(ctx, 65537);
    PolyUOp *strong = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, weak, poly_arg_none());
    PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, types[i], strong, poly_arg_none());
    PolyUOp *literal = poly_uop1(ctx, POLY_OP_CAST, types[i], weak, poly_arg_none());
    PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, types[i], cast, literal, poly_arg_none());
    PolyUOp *wide = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, add, poly_arg_none());
    PolyUOp *ops[] = {weak, strong, cast, literal, add, wide};
    char *src = poly_render_wgsl(ctx, ops, 6, "narrow_results");
    ASSERT_NOT_NULL(src);
    const char *suffix = i == 0   ? " << 24u) >> 24u)"
                         : i == 1 ? " & 255u)"
                         : i == 2 ? " << 16u) >> 16u)"
                                  : " & 65535u)";
    int count = 0;
    for (const char *p = src; (p = strstr(p, suffix)); p += strlen(suffix))
      count++;
    /* Normal cast, inlined weak cast, and ALU result each normalize. */
    free(src);
    poly_ctx_destroy(ctx);
    ASSERT_INT_EQ(count, 3);
  }
  PASS();
}

TEST(codegen, renderers_inline_current_casted_literals) {
  /* Current tinygrad renderer/cstyle.py:26-47,238-241 renders
   * CAST(strong, CONST(weak/bool)) as the destination-typed literal itself.
   * pm_casted_consts deliberately leaves this topology for every renderer. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(32));
  PolyParamArg arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_INT32, shape, poly_arg_param(&arg));
  PolyUOp *weak_index = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(20));
  PolyUOp *index = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, weak_index, poly_arg_none());
  PolyUOp *address = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, index, poly_arg_none());
  PolyUOp *weak_value = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(7));
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, weak_value, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, address, value, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  int n = 0;
  PolyUOp **uops = poly_toposort(ctx, sink, &n);
  ASSERT_NOT_NULL(uops);

  char *c = poly_render_c(ctx, uops, n, "casted_const");
  ASSERT_NOT_NULL(c);
  ASSERT_NOT_NULL(strstr(c, "data0+20"));
  ASSERT_NOT_NULL(strstr(c, " = 7;"));
  ASSERT_TRUE(strstr(c, "cast0") == NULL);
  free(c);

  char *wgsl = poly_render_wgsl(ctx, uops, n, "casted_const");
  ASSERT_NOT_NULL(wgsl);
  ASSERT_NOT_NULL(strstr(wgsl, "data0[20] = 7;"));
  ASSERT_TRUE(strstr(wgsl, "var cast") == NULL);
  free(wgsl);

#ifdef POLY_HAS_CUDA
  char *cuda = poly_render_cuda(ctx, uops, n, "casted_const", 1);
  ASSERT_NOT_NULL(cuda);
  ASSERT_NOT_NULL(strstr(cuda, "data0+20"));
  ASSERT_NOT_NULL(strstr(cuda, " = 7;"));
  ASSERT_TRUE(strstr(cuda, "cast0") == NULL);
  free(cuda);
#endif

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, partial_reshape_index_maps_nonzero_input_prefix) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned schedule/rangeify.py:68-77 maps
   * (2,3,4)->(6,4)->INDEX(5) to source.INDEX(1,2), retaining shape (4). */
  PolyUOp *like = poly_reshape(ctx, poly_buffer_f32(ctx, 24), (int64_t[]){2, 3, 4}, 3);
  PolyUOp *input = poly_uop_placeholder_like(ctx, like, 1);
  PolyUOp *output = poly_uop_placeholder_like(ctx, poly_buffer_f32(ctx, 1), 0);
  PolyUOp *reshape = poly_reshape(ctx, input, (int64_t[]){6, 4}, 2);
  PolyUOp *five = poly_const_int(ctx, 5);
  PolyUOp *partial = poly_uop_index(ctx, reshape, &five, 1);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *value = poly_uop_index(ctx, partial, &zero, 1);
  PolyUOp *address = poly_uop_index(ctx, output, &zero, 1);
  PolyUOp *store = poly_uop_store(ctx, address, value);
  PolyUOp *sink = poly_uop_sink(ctx, &store, 1);
  ASSERT_NOT_NULL(sink);

  PolyRewriteOpts opts = {
      .optimize = false,

      .caps = poly_c_renderer_caps(),
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  /* Current full_rewrite_to_sink always runs devectorizer2. Its
   * mop_cleanup+pm_mops composition turns partial (1,2), then scalar (0),
   * into flat PARAM index 20. */
  ASSERT_FALSE(poly_uop_reachable(ctx, rewritten, reshape));
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_RESHAPE), 0);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_INDEX), 2);
  PolyUOp *read_load = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_LOAD) continue;
    ASSERT_TRUE(read_load == NULL);
    read_load = topo[i];
  }
  ASSERT_NOT_NULL(read_load);
  ASSERT_INT_EQ(read_load->n_src, 1);
  PolyUOp *read_index = read_load->src[0];
  ASSERT_NOT_NULL(read_index);
  ASSERT_INT_EQ(read_index->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(read_index->n_src, 2);
  ASSERT_INT_EQ(read_index->src[0]->op, POLY_OP_PARAM);
  ASSERT_INT_EQ(read_index->src[0]->arg.kind, POLY_ARG_PARAM);
  ASSERT_NOT_NULL(read_index->src[0]->arg.param);
  ASSERT_INT_EQ(read_index->src[0]->arg.param->slot, 1);
  ASSERT_INT_EQ(read_index->src[1]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(read_index->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(read_index->src[1]->n_src, 1);
  ASSERT_INT_EQ(read_index->src[1]->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(read_index->src[1]->src[0]->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(read_index->src[1]->src[0]->arg.i, 20);

  int n_linear = 0;
  PolyUOp **linear = poly_test_full_rewrite_and_linearize_ex(ctx, sink, opts, &n_linear);
  ASSERT_NOT_NULL(linear);
  char *source = poly_render_c(ctx, linear, n_linear, "partial_reshape_index_prefix");
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(strstr(source, "data1+20"));
  PolyProgram *program = poly_compile_c(source, "partial_reshape_index_prefix");
  ASSERT_NOT_NULL(program);

  float out = 0.0f;
  float in[24];
  for (int i = 0; i < 24; i++)
    in[i] = (float)(100 + i);
  void *args[2] = {&out, in};
  poly_program_call(program, args, 2);
  ASSERT_FLOAT_EQ(out, 120.0f, 1e-6);

  poly_program_destroy(program);
  free(source);
  free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, validates_index_coordinates_before_pointer_concat) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/spec.py:73-77 rejects a direct bool INDEX coordinate
   * before codegen preprocess, while integer WHERE(valid, idx, Invalid)
   * remains a valid coordinate for pm_syntactic_sugar. */
  PolyUOp *input = program_param(ctx, POLY_FLOAT32, 16, 0);
  PolyUOp *output = program_param(ctx, POLY_FLOAT32, 16, 1);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *gate = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, one, poly_arg_none());
  ASSERT_NOT_NULL(gate);

  PolyUOp *bad_inner_indices[] = {zero, gate};
  PolyUOp *bad_inner = poly_uop_index(ctx, input, bad_inner_indices, 2);
  ASSERT_NOT_NULL(bad_inner); /* Construction matches pinned UOp.index. */
  PolyUOp *bad_outer = poly_uop_index(ctx, bad_inner, &one, 1);
  PolyUOp *bad_value = poly_uop_load(ctx, bad_outer);
  PolyUOp *out_index = poly_uop_index(ctx, output, &zero, 1);
  PolyUOp *bad_store = poly_uop_store(ctx, out_index, bad_value);
  PolyUOp *bad_sink = poly_uop_sink(ctx, &bad_store, 1);
  ASSERT_NOT_NULL(bad_sink);
  ASSERT_FALSE(poly_type_verify_tensor(ctx, bad_sink));

  PolyRewriteOpts opts = {
      .optimize = false,

      .caps = poly_c_renderer_caps(),
  };
  ASSERT_TRUE(poly_full_rewrite_to_sink_ex(ctx, bad_sink, opts) == NULL);

  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *valid_coord =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, two, invalid, poly_arg_none());
  PolyUOp *good_index = poly_uop_index(ctx, input, &valid_coord, 1);
  PolyUOp *good_value = poly_uop_load(ctx, good_index);
  PolyUOp *good_store = poly_uop_store(ctx, out_index, good_value);
  PolyUOp *good_sink = poly_uop_sink(ctx, &good_store, 1);
  ASSERT_TRUE(poly_type_verify_tensor(ctx, good_sink));
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, good_sink, opts);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *read_index = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_LOAD && topo[i]->n_src >= 1 && topo[i]->src[0]->op == POLY_OP_INDEX)
      read_index = topo[i]->src[0];
    if (topo[i]->op != POLY_OP_INDEX) continue;
    for (int j = 1; j < topo[i]->n_src; j++)
      ASSERT_TRUE(poly_dtype_is_int(topo[i]->src[j]->dtype));
  }
  /* Current coalescing accepts the flat gated INDEX. Nested INDEX is valid
   * tensor IR but not a supported input to full_rewrite_to_sink. */
  ASSERT_NOT_NULL(read_index);
  ASSERT_INT_EQ(read_index->n_src, 2);
  ASSERT_INT_EQ(read_index->src[0]->op, POLY_OP_PARAM);
  ASSERT_INT_EQ(read_index->src[0]->arg.kind, POLY_ARG_PARAM);
  ASSERT_NOT_NULL(read_index->src[0]->arg.param);
  ASSERT_INT_EQ(read_index->src[0]->arg.param->slot, 0);
  ASSERT_INT_EQ(read_index->src[1]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(read_index->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(read_index->src[1]->n_src, 1);
  ASSERT_INT_EQ(read_index->src[1]->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(read_index->src[1]->src[0]->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(read_index->src[1]->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(read_index->src[1]->src[0]->arg.i, 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, program_spec_rejects_movement_ops_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyParamArg arg = {
      .slot = 0,
      .dtype = POLY_FLOAT32,
      .addrspace = POLY_ADDR_GLOBAL,
  };
  PolyUOp *extent = poly_const_int(ctx, 4);
  PolyUOp *param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, extent, poly_arg_param(&arg));
  PolyUOp *valid = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, param, poly_arg_none());
  ASSERT_TRUE(poly_type_verify_program(ctx, valid));

  PolyUOp *permute =
      poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, param, poly_arg_int_tuple((int64_t[]){0}, 1));
  PolyUOp *invalid = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, permute, poly_arg_none());
  ASSERT_FALSE(poly_type_verify_program(ctx, invalid));

  PolyParamArg count_arg = {
      .slot = 1,
      .dtype = POLY_UINT32,
      .addrspace = POLY_ADDR_GLOBAL,
  };
  PolyParamArg x_arg = {
      .slot = 0,
      .dtype = POLY_INT32,
      .addrspace = POLY_ADDR_GLOBAL,
  };
  PolyUOp *x = poly_uop1(ctx, POLY_OP_PARAM, POLY_INT32, extent, poly_arg_param(&x_arg));
  PolyUOp *count = poly_uop1(ctx, POLY_OP_PARAM, POLY_UINT32, extent, poly_arg_param(&count_arg));
  PolyUOp *shift = poly_uop2(ctx, POLY_OP_SHL, POLY_INT32, x, count, poly_arg_none());
  PolyUOp *shift_sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, shift, poly_arg_none());
  ASSERT_TRUE(poly_type_verify_program(ctx, shift_sink));

  PolyInt large = {0};
  ASSERT_TRUE(poly_int_from_decimal(&large, "8589934593"));
  PolyUOp *weak_bigint = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_int_as_arg(&large));
  poly_int_free(&large);
  PolyUOp *weak_bigint_sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, weak_bigint, poly_arg_none());
  ASSERT_TRUE(poly_type_verify_program(ctx, weak_bigint_sink));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, tensor_spec_accepts_movement_and_rejects_program_if) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *param = program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *reshape = poly_reshape(ctx, param, (int64_t[]){2, 2}, 2);
  PolyUOp *permute = poly_permute(ctx, reshape, (int64_t[]){1, 0}, 2);
  ASSERT_TRUE(poly_type_verify_tensor(ctx, poly_sink1(ctx, permute)));

  PolyUOp *gate = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, zero, poly_arg_none());
  PolyUOp *if_uop = poly_uop2(ctx, POLY_OP_IF, POLY_VOID, gate, index, poly_arg_none());
  ASSERT_FALSE(poly_type_verify_tensor(ctx, poly_sink1(ctx, if_uop)));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, shared_spec_matches_august_matrix) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/spec.py:50-131. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *param = program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *zero_f = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *one_f = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *zero_i = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *one_i = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *false_b = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, zero_i, poly_arg_none());

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, zero_f, one_f, poly_arg_none());
  PolyUOp *after_add = poly_uop1(ctx, POLY_OP_AFTER, POLY_FLOAT32, add, poly_arg_none());
  ASSERT_FALSE(poly_type_verify_tensor(ctx, poly_sink1(ctx, after_add)));
  PolyUOp *after_index = poly_uop1(ctx, POLY_OP_AFTER, POLY_FLOAT32, index, poly_arg_none());
  ASSERT_TRUE(poly_type_verify_tensor(ctx, poly_sink1(ctx, after_index)));

  PolyUOp *barrier_void = poly_uop0(ctx, POLY_OP_BARRIER, POLY_VOID, poly_arg_none());
  PolyUOp *barrier_int = poly_uop0(ctx, POLY_OP_BARRIER, POLY_INT32, poly_arg_none());
  ASSERT_TRUE(poly_type_verify_tensor(ctx, poly_sink1(ctx, barrier_void)));
  ASSERT_FALSE(poly_type_verify_tensor(ctx, poly_sink1(ctx, barrier_int)));

  PolyUOp *load_match =
      poly_uop3(ctx, POLY_OP_LOAD, POLY_FLOAT32, index, zero_f, false_b, poly_arg_none());
  PolyUOp *load_mismatch =
      poly_uop3(ctx, POLY_OP_LOAD, POLY_FLOAT32, index, zero_i, false_b, poly_arg_none());
  PolyUOp *load_nonbool_gate =
      poly_uop3(ctx, POLY_OP_LOAD, POLY_FLOAT32, index, zero_f, zero_i, poly_arg_none());
  ASSERT_TRUE(poly_type_verify_tensor(ctx, poly_sink1(ctx, load_match)));
  ASSERT_FALSE(poly_type_verify_tensor(ctx, poly_sink1(ctx, load_mismatch)));
  ASSERT_FALSE(poly_type_verify_tensor(ctx, poly_sink1(ctx, load_nonbool_gate)));

  PolyUOp *tensor_store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, param, zero_f, poly_arg_none());
  PolyUOp *nonvoid_store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_FLOAT32, param, zero_f, poly_arg_none());
  ASSERT_TRUE(poly_type_verify_tensor(ctx, poly_sink1(ctx, tensor_store)));
  ASSERT_FALSE(poly_type_verify_tensor(ctx, poly_sink1(ctx, nonvoid_store)));

  PolyUOp *cdiv_int = poly_uop2(ctx, POLY_OP_CDIV, POLY_INT32, zero_i, one_i, poly_arg_none());
  PolyUOp *cdiv_float = poly_uop2(ctx, POLY_OP_CDIV, POLY_FLOAT32, zero_f, one_f, poly_arg_none());
  ASSERT_TRUE(poly_type_verify_tensor(ctx, poly_sink1(ctx, cdiv_int)));
  ASSERT_FALSE(poly_type_verify_tensor(ctx, poly_sink1(ctx, cdiv_float)));

  PolyUOp *noop = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *loop = poly_uop1(ctx, POLY_OP_RANGE, POLY_VOID, noop, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *end_srcs[] = {zero_f, loop, false_b};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 3, poly_arg_none());
  ASSERT_TRUE(poly_type_verify_tensor(ctx, poly_sink1(ctx, end)));
  PolyUOp *extra_end_srcs[] = {zero_f, loop, loop, false_b};
  PolyUOp *extra_end = poly_uop(ctx, POLY_OP_END, POLY_VOID, extra_end_srcs, 4, poly_arg_none());
  ASSERT_FALSE(poly_type_verify_tensor(ctx, poly_sink1(ctx, extra_end)));

  PolyUOp *call_body = poly_sink1(ctx, zero_f);
  PolyUOp *nonvoid_call = poly_uop1(ctx, POLY_OP_CALL, POLY_FLOAT32, call_body, poly_arg_none());
  ASSERT_FALSE(poly_type_verify_tensor(ctx, poly_sink1(ctx, nonvoid_call)));

  const char *devices[] = {"CPU", "CPU"};
  PolyParamArg multi_arg = {
      .slot = 1,
      .dtype = POLY_FLOAT32,
      .addrspace = POLY_ADDR_GLOBAL,
      .devices = devices,
      .n_devices = 2,
      .device_is_tuple = true,
  };
  PolyUOp *extent = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *multi = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, extent, poly_arg_param(&multi_arg));
  PolyUOp *unshard_srcs[] = {multi, extent};
  PolyUOp *unshard = poly_uop(
      ctx, POLY_OP_UNSHARD, POLY_FLOAT32, unshard_srcs, 2, poly_arg_int_tuple((int64_t[]){-1}, 1)
  );
  ASSERT_TRUE(poly_type_verify_tensor(ctx, poly_sink1(ctx, unshard)));
  PolyUOp *mselect = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, multi, poly_arg_int(-1));
  ASSERT_TRUE(poly_type_verify_tensor(ctx, poly_sink1(ctx, mselect)));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_unary) {
  /* b[i] = -a[i] for i in 0..5 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = program_param(ctx, POLY_FLOAT32, 6, 0);
  PolyUOp *p1 = program_param(ctx, POLY_FLOAT32, 6, 1);

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(6));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, range, poly_arg_none());

  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, neg, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(ctx, lin, n, "vecneg");

  /* only 2 bindings */
  ASSERT_NOT_NULL(strstr(src, "data0: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "data1: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "fn vecneg("));
  /* NEG renders as (-val) */
  ASSERT_NOT_NULL(strstr(src, "(-val0)"));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_where) {
  /* c[i] = cond ? a[i] : b[i], where cond = (a[i] < 5.0) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *p1 = program_param(ctx, POLY_FLOAT32, 4, 1);
  PolyUOp *p2 = program_param(ctx, POLY_FLOAT32, 4, 2);

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p2, range, poly_arg_none());

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());

  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(5.0));
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, load0, five, poly_arg_none());

  PolyUOp *where_src[3] = {cond, load0, load1};
  PolyUOp *where = poly_uop(ctx, POLY_OP_WHERE, POLY_FLOAT32, where_src, 3, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, where, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(ctx, lin, n, "vecwhere");

  /* WHERE maps to select(false_val, true_val, cond) */
  ASSERT_NOT_NULL(strstr(src, "select("));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_uint32_ops) {
  /* out[i] = where(a[i] < b[i], (a[i] >> 1) + (a[i] % 31), a[i] // b[i]) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = program_param(ctx, POLY_UINT32, 8, 0);
  PolyUOp *p1 = program_param(ctx, POLY_UINT32, 8, 1);
  PolyUOp *p2 = program_param(ctx, POLY_UINT32, 8, 2);

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT32, p2, range, poly_arg_none());

  PolyUOp *la = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx0, poly_arg_none());
  PolyUOp *lb = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx1, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(1));
  PolyUOp *thirty_one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(31));

  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, la, lb, poly_arg_none());
  PolyUOp *rhs = poly_uop2(
      ctx, POLY_OP_ADD, POLY_UINT32,
      poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, la, one, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_MOD, POLY_UINT32, la, thirty_one, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *lhs = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT32, la, lb, poly_arg_none());
  PolyUOp *sel_src[3] = {cond, rhs, lhs};
  PolyUOp *sel = poly_uop(ctx, POLY_OP_WHERE, POLY_UINT32, sel_src, 3, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, sel, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(ctx, lin, n, "vecu32");

  ASSERT_NOT_NULL(strstr(src, "array<u32>"));
  ASSERT_NOT_NULL(strstr(src, "var val0: u32"));
  ASSERT_NOT_NULL(strstr(src, "31u"));
  ASSERT_NOT_NULL(strstr(src, ">>"));
  ASSERT_NOT_NULL(strstr(src, "%"));
  ASSERT_NOT_NULL(strstr(src, "select("));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_uint8_storage_is_packed_like_tinygrad) {
  /* out[i] = in[i] for uint8. WGSL has no byte-addressable storage buffer,
   * so tinygrad packs byte/short storage into atomic<u32> lanes. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = program_param(ctx, POLY_UINT8, 5, 0);
  PolyUOp *p1 = program_param(ctx, POLY_UINT8, 5, 1);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT8, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT8, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT8, idx0, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, load, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(ctx, lin, n, "copy_u8");

  ASSERT_NOT_NULL(strstr(src, "data0: array<atomic<u32>>"));
  ASSERT_NOT_NULL(strstr(src, "data1: array<atomic<u32>>"));
  ASSERT_NOT_NULL(strstr(src, "atomicLoad(&data0[(Lidx0/4)]"));
  ASSERT_NOT_NULL(strstr(src, "atomicAnd(&data1[(Lidx0/4)]"));
  ASSERT_NOT_NULL(strstr(src, "atomicAdd(&data1[(Lidx0/4)]"));
  ASSERT_TRUE(strstr(src, "data1[Lidx0] =") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_bool_storage_is_packed_like_tinygrad) {
  /* WGSL permits scalar bool values but forbids bool in storage buffers.
   * tinygrad's WGSLRenderer packs bool output storage into atomic<u32> byte
   * lanes, while keeping the comparison itself as a scalar bool expression. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = program_param(ctx, POLY_BOOL, 3, 0);
  PolyUOp *a = program_param(ctx, POLY_FLOAT32, 3, 1);
  PolyUOp *b = program_param(ctx, POLY_FLOAT32, 3, 2);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx_out = poly_uop2(ctx, POLY_OP_INDEX, POLY_BOOL, out, range, poly_arg_none());
  PolyUOp *idx_a = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, a, range, poly_arg_none());
  PolyUOp *idx_b = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, b, range, poly_arg_none());
  PolyUOp *load_a = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx_a, poly_arg_none());
  PolyUOp *load_b = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx_b, poly_arg_none());
  PolyUOp *eq = poly_uop2(ctx, POLY_OP_CMPEQ, POLY_BOOL, load_a, load_b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx_out, eq, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(ctx, lin, n, "eq_bool");

  ASSERT_NOT_NULL(strstr(src, "data0: array<atomic<u32>>"));
  ASSERT_TRUE(strstr(src, "array<bool>") == NULL);
  ASSERT_NOT_NULL(strstr(src, "var alu0: bool"));
  ASSERT_NOT_NULL(strstr(src, "atomicAdd(&data0[(Lidx0/4)]"));
  ASSERT_NOT_NULL(strstr(src, "select(0u, 1u, alu0)"));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Pinned renderer/wgsl.py:is_packed,_packed_size,buf_map and render_load.
 * Exercise declaration and both accesses together, including an AFTER owner:
 * LOAD.addrspace itself is ALU, not the backing storage's address space. */
static bool check_wgsl_storage_address_space(PolyAddrSpace space) {
  PolyDType dtypes[] = {POLY_BOOL,  POLY_UINT8,   POLY_INT8,   POLY_UINT16,
                        POLY_INT16, POLY_FLOAT16, POLY_FLOAT32};
  const char *types[] = {"bool", "u32", "i32", "u32", "i32", "f16", "f32"};
  int64_t sizes[] = {1, 2, 3, 4, 5, 7, 8, 9};
  for (int d = 0; d < 7; d++) {
    for (int s = 0; s < 8; s++) {
      for (int wrapped = 0; wrapped < 2; wrapped++) {
        PolyCtx *ctx = poly_ctx_new();
        if (!ctx) return false;
        PolyDType dt = dtypes[d];
        int64_t size = sizes[s];
        PolyParamArg arg = {.slot = 0, .dtype = dt, .addrspace = space};
        PolyUOp *extent = poly_const_int(ctx, size);
        PolyUOp *buffer = poly_uop1(
            ctx, space == POLY_ADDR_GLOBAL ? POLY_OP_PARAM : POLY_OP_BUFFER, dt, extent,
            poly_arg_param(&arg)
        );
        PolyUOp *offset = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(size - 1));
        PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, dt, buffer, offset, poly_arg_none());
        PolyArg value_arg = poly_dtype_is_bool(dt)       ? poly_arg_bool(true)
                            : poly_dtype_is_float(dt)    ? poly_arg_float(1.5)
                            : poly_dtype_is_unsigned(dt) ? poly_arg_int(42)
                                                         : poly_arg_int(-7);
        PolyUOp *value = poly_uop0(ctx, POLY_OP_CONST, dt, value_arg);
        PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, value, poly_arg_none());
        PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, dt, buffer, store, poly_arg_none());
        PolyUOp *read_idx =
            wrapped ? poly_uop2(ctx, POLY_OP_INDEX, dt, after, offset, poly_arg_none()) : idx;
        PolyUOp *gate = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
        PolyUOp *load_src[] = {read_idx, value, gate};
        PolyUOp *load = poly_uop(ctx, POLY_OP_LOAD, dt, load_src, 3, poly_arg_none());
        PolyUOp *write_idx = read_idx;
        PolyUOp *copy = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, write_idx, load, poly_arg_none());
        PolyUOp *uops[] = {extent, buffer,   offset, idx,  value, store,
                           after,  read_idx, gate,   load, copy};
        /* The direct case shares its INDEX, so list that UOp only once. */
        if (!wrapped) {
          memmove(&uops[7], &uops[8], 3 * sizeof(*uops));
        }
        char *source = poly_render_wgsl(ctx, uops, wrapped ? 11 : 10, "storage_access");
        const char *name = space == POLY_ADDR_GLOBAL  ? "data0"
                           : space == POLY_ADDR_LOCAL ? "smem0"
                                                      : "r0";
        bool packed = d < 5 && space != POLY_ADDR_REG;
        int elems = packed ? 4 / poly_dtype_itemsize(dt) : 1;
        int64_t count = size / elems + (size % elems != 0);
        char declaration[128], load_expr[128], store_expr[128];
        if (space == POLY_ADDR_GLOBAL)
          snprintf(
              declaration, sizeof(declaration), "%s: array<%s>;", name,
              packed ? "atomic<u32>" : types[d]
          );
        else
          snprintf(
              declaration, sizeof(declaration), "%s: array<%s,%lld>;", name,
              packed ? "atomic<u32>" : types[d], (long long)count
          );
        if (packed) {
          snprintf(
              load_expr, sizeof(load_expr), "atomicLoad(&%s[(%lld/%d)]", name,
              (long long)(size - 1), elems
          );
          snprintf(
              store_expr, sizeof(store_expr), "atomicAnd(&%s[(%lld/%d)]", name,
              (long long)(size - 1), elems
          );
        } else {
          snprintf(load_expr, sizeof(load_expr), "%s[%lld]", name, (long long)(size - 1));
          snprintf(store_expr, sizeof(store_expr), "%s[%lld] =", name, (long long)(size - 1));
        }
        bool ok = source && strstr(source, declaration) && strstr(source, load_expr) &&
                  strstr(source, store_expr) && strstr(source, "select(") &&
                  (packed || !strstr(source, "atomic"));
        if (packed && (d == 2 || d == 4))
          ok = ok && strstr(source, d == 2 ? "<<24)>>24" : "<<16)>>16");
        if (ok && space == POLY_ADDR_LOCAL) {
          const char *local = strstr(source, "var<workgroup>");
          const char *compute = strstr(source, "@compute");
          ok = local && compute && local < compute;
        }
        if (!ok)
          fprintf(
              stderr, "WGSL storage space=%d dtype=%s size=%lld after=%d\n%s\n", space,
              poly_dtype_name(dt), (long long)size, wrapped, source ? source : "NULL"
          );
        free(source);
        poly_ctx_destroy(ctx);
        if (!ok) return false;
      }
    }
  }
  return true;
}

TEST(codegen, render_wgsl_reg_storage_stays_unpacked) {
  ASSERT_TRUE(check_wgsl_storage_address_space(POLY_ADDR_REG));
  PASS();
}

TEST(codegen, render_wgsl_local_storage_matches_packed_access) {
  ASSERT_TRUE(check_wgsl_storage_address_space(POLY_ADDR_LOCAL));
  PASS();
}

TEST(codegen, render_wgsl_global_storage_matches_packed_access) {
  ASSERT_TRUE(check_wgsl_storage_address_space(POLY_ADDR_GLOBAL));
  PASS();
}

TEST(codegen, render_wgsl_reduce) {
  /* Current rangeify graph for out[0] = sum(a[0..9]). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *ten = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(10));
  PolyParamArg args[2] = {
      {.slot = 0, .addrspace = POLY_ADDR_GLOBAL},
      {.slot = 1, .addrspace = POLY_ADDR_GLOBAL},
  };
  PolyUOp *p0 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, ten, poly_arg_param(&args[0]));
  PolyUOp *p1 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, one, poly_arg_param(&args[1]));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, ten, poly_arg_range(0, POLY_AXIS_REDUCE));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *reduce_srcs[2] = {load, range};
  PolyUOp *reduce =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, reduce_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, zero, poly_arg_none());
  PolyUOp *store_out = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, reduce, poly_arg_none());

  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store_out, poly_arg_none());

  int n_lin;
  PolyRewriteOpts opts = {.optimize = false, .caps = poly_c_renderer_caps()};
  PolyUOp **lin = poly_test_full_rewrite_and_linearize_ex(ctx, sink, opts, &n_lin);
  char *src = poly_render_wgsl(ctx, lin, n_lin, "reduce_sum");

  /* Current WGSLRenderer renders REG BUFFERs as one-element arrays. */
  ASSERT_NOT_NULL(strstr(src, "var r0: array<f32,1>;"));
  ASSERT_NOT_NULL(strstr(src, "r0[0] = 0.0"));
  /* loop present */
  ASSERT_NOT_NULL(strstr(src, "for (var Ridx0: i32 = 0;"));
  /* accumulator store (not array write) */
  ASSERT_NOT_NULL(strstr(src, "r0[0] = "));
  /* output buffer store */
  ASSERT_NOT_NULL(strstr(src, "data1[0]"));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_alu_param) {
  /* Current WGSL renders GLOBAL PARAMs as storage and ALU PARAMs as uniforms. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = program_param(ctx, POLY_FLOAT32, 16, 0);
  PolyUOp *p1 = program_param(ctx, POLY_FLOAT32, 16, 1);
  PolyUOp *N = poly_uop_param(ctx, 2, poly_uop_variable(ctx, "N", 1, 16, POLY_INT32, 1, false));

  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, N, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, range, poly_arg_none());

  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *cast_n = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, N, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, load, cast_n, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, add, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n_lin);
  char *src = poly_render_wgsl(ctx, lin, n_lin, "var_kernel");

  /* binding(0) = INFINITY uniform (always) */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(0)\nvar<uniform> INFINITY"));

  /* binding(1) = data0 storage buffer */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(1)\nvar<storage,read_write> data0: array<f32>"));

  /* binding(2) = data1 storage buffer */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(2)\nvar<storage,read_write> data1: array<f32>"));

  /* binding(3) = scalar ALU PARAM */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(3)\nvar<uniform> data2: i32"));

  ASSERT_NOT_NULL(strstr(src, "f32(data2)"));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_param_bindings_follow_encounter_order) {
  /* Match tinygrad WGSL binding assignment:
   * bindings are sequential in PARAM encounter order, not PARAM.arg.
   * Sparse PARAM ids used to leak into @binding(N), which mismatched runtime binding order. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p7 = program_param(ctx, POLY_FLOAT32, 8, 7);
  PolyUOp *p2 = program_param(ctx, POLY_FLOAT32, 8, 2);
  PolyUOp *p9 = program_param(ctx, POLY_FLOAT32, 8, 9);

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx7 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p7, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p2, range, poly_arg_none());
  PolyUOp *idx9 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p9, range, poly_arg_none());

  PolyUOp *lhs = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx2, poly_arg_none());
  PolyUOp *rhs = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx9, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, lhs, rhs, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx7, sum, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin = 0;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n_lin);
  char *src = poly_render_wgsl(ctx, lin, n_lin, "param_order");

  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(1)\nvar<storage,read_write>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(2)\nvar<storage,read_write>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(3)\nvar<storage,read_write>"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data7: array<f32>;"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data2: array<f32>;"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data9: array<f32>;"));
  ASSERT_TRUE(
      strstr(src, "@group(0) @binding(8)\nvar<storage,read_write> data7: array<f32>") == NULL
  );
  ASSERT_TRUE(
      strstr(src, "@group(0) @binding(10)\nvar<storage,read_write> data9: array<f32>") == NULL
  );

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_bindings_grow_past_old_fixed_cap) {
  PolyCtx *ctx = poly_ctx_new();
  enum { N_BINDINGS = 70 };
  PolyUOp **uops = calloc(N_BINDINGS, sizeof(PolyUOp *));
  ASSERT_NOT_NULL(uops);

  for (int i = 0; i < N_BINDINGS; i++)
    uops[i] = program_param(ctx, POLY_FLOAT32, 1, i);

  char *src = poly_render_wgsl(ctx, uops, N_BINDINGS, "many_bindings");
  ASSERT_NOT_NULL(src);

  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(65)\nvar<storage,read_write> data64"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(70)\nvar<storage,read_write> data69"));

  free(src);
  free(uops);
  poly_ctx_destroy(ctx);
  PASS();
}

/* WebGPU GPU linearizer output tests */

TEST(codegen, linearize_webgpu_vecadd_emits_gpudims) {
  /* Verify GPU linearizer produces SPECIAL ops and correct WGSL builtins */
  VecKernel k = make_vec_binop(POLY_OP_ADD, 1024);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(k.ctx, k.sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n_lin > 0);

  /* GPU linearizer should produce at least gidx0 SPECIAL op */
  ASSERT_TRUE(count_special_named(lin, n_lin, "gidx0") >= 1);

  /* Render to WGSL and verify GPU-specific patterns */
  char *wgsl = poly_render_wgsl(k.ctx, lin, n_lin, "vecadd_gpu");
  ASSERT_NOT_NULL(wgsl);
  ASSERT_NOT_NULL(strstr(wgsl, "i32(gindex."));
  ASSERT_NOT_NULL(strstr(wgsl, "@compute @workgroup_size("));
  /* Builtins: workgroup_id + local_invocation_id */
  ASSERT_NOT_NULL(strstr(wgsl, "workgroup_id"));
  ASSERT_NOT_NULL(strstr(wgsl, "local_invocation_id"));

  free(wgsl);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, linearize_webgpu_vecadd_splits_large_global_dispatch) {
  /* End-to-end through the WebGPU linearizer: the backend caps must flow into
   * add_gpudims, so oversized logical dispatches produce multiple hardware
   * workgroup_id SPECIALs instead of an illegal x-dimension. */
  VecKernel k = make_vec_binop(POLY_OP_ADD, 16777216);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(k.ctx, k.sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  ASSERT_INT_EQ(count_special_named(lin, n_lin, "gidx0"), 1);
  ASSERT_INT_EQ(count_special_named(lin, n_lin, "gidx1"), 1);
  ASSERT_INT_EQ((int)special_bound_hi_named(lin, n_lin, "gidx0"), 32768);
  ASSERT_INT_EQ((int)special_bound_hi_named(lin, n_lin, "gidx1"), 4);

  char *wgsl = poly_render_wgsl(k.ctx, lin, n_lin, "vecadd_split_gpu");
  ASSERT_NOT_NULL(wgsl);
  ASSERT_NOT_NULL(strstr(wgsl, "i32(gindex.x)"));
  ASSERT_NOT_NULL(strstr(wgsl, "i32(gindex.y)"));

  free(wgsl);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, add_gpudims_same_axis_ranges_share_special) {
  /* tinygrad gpudims groups ranges by axis tuple, not UOp identity. Distinct
   * RANGE nodes for the same axis, including weakint/int variants, must map to
   * one GPU builtin instead of rendering duplicate gidx/lidx declarations. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b16_w = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(16));
  PolyUOp *b16_i = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(16));
  PolyUOp *rw =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, b16_w, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *ri =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, b16_i, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *srcs[2] = {rw, ri};
  PolyUOp *sink = poly_test_kernel_sink(ctx, srcs, 2, "same_axis");
  PolyUOp *rewritten = poly_add_gpudims(ctx, sink);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_SPECIAL), 1);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_RANGE), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_threads_uses_core_id_param) {
  /* tinygrad/codegen/gpudims.py:60 lowers THREAD through the shared
   * add_gpudims path to PARAM(core_id).cast(weakint), never SPECIAL. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_THREAD));
  PolyUOp *sink = poly_test_kernel_sink(ctx, &range, 1, "threads");
  PolyRendererCaps caps = {.has_threads = true, .max_threads = 8};

  PolyUOp *rewritten = poly_add_gpudims_ex(ctx, sink, caps);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->src[0]->op, POLY_OP_CAST);
  ASSERT_INT_EQ(rewritten->src[0]->src[0]->op, POLY_OP_PARAM);
  ASSERT_TRUE(poly_uop_is_alu_param(rewritten->src[0]->src[0]));
  ASSERT_STR_EQ(poly_uop_expr(rewritten->src[0]->src[0]), "core_id");
  ASSERT_INT_EQ(rewritten->src[0]->src[0]->arg.param->min_val, 0);
  ASSERT_INT_EQ(rewritten->src[0]->src[0]->arg.param->max_val, 7);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_SPECIAL), 0);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_RANGE), 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_preserves_end_with_special_sources_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *c5 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(5));
  PolyUOp *c4 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *global =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, c5, poly_arg_range(1, POLY_AXIS_GLOBAL));
  PolyUOp *local =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, c4, poly_arg_range(3, POLY_AXIS_LOCAL));
  PolyUOp *out = poly_test_uop_param(ctx, POLY_FLOAT32, 20, 0, POLY_ADDR_GLOBAL);
  PolyUOp *coord = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, global, c4, poly_arg_none()), local, poly_arg_none()
  );
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, coord, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, local, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, value, poly_arg_none());
  PolyUOp *end_srcs[3] = {store, global, local};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 3, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "gpudims_end");
  PolyRendererCaps caps = {
      .has_local = true,
      .global_max = {INT32_MAX, 65535, 65535},
      .local_max = {1024, 1024, 64},
  };

  PolyUOp *rewritten = poly_add_gpudims_ex(ctx, sink, caps);
  ASSERT_NOT_NULL(rewritten);
  /* tinygrad/codegen/gpudims.py:58-105 uses s.substitute(subs), preserving
   * the END and substituting both range occurrences in-place. */
  ASSERT_INT_EQ(rewritten->op, POLY_OP_SINK);
  ASSERT_INT_EQ(rewritten->n_src, 1);
  ASSERT_INT_EQ(rewritten->src[0]->op, POLY_OP_END);
  ASSERT_INT_EQ(rewritten->src[0]->n_src, 3);
  ASSERT_INT_EQ(rewritten->src[0]->src[0]->op, POLY_OP_STORE);
  ASSERT_INT_EQ(rewritten->src[0]->src[1]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(rewritten->src[0]->src[1]->arg.str, "gidx0");
  ASSERT_INT_EQ(rewritten->src[0]->src[2]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(rewritten->src[0]->src[2]->arg.str, "lidx0");

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_reverse_group_preserves_logical_axis_order) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/gpudims.py:28-56 groups the
   * reversed (3,9) pair, reconstructs it as (gidx0//9,gidx0%9), then reverses
   * the result: (gidx2,gidx1,gidx0%9,gidx0//9). */
  PolyCtx *ctx = poly_ctx_new();
  const int64_t bounds[4] = {32, 2, 9, 3};
  PolyUOp *ranges[4];
  for (int i = 0; i < 4; i++) {
    PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(bounds[i]));
    ranges[i] =
        poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(3 + i, POLY_AXIS_GLOBAL));
  }
  PolyUOp *sink = poly_test_kernel_sink(ctx, ranges, 4, "reverse_group");
  PolyRendererCaps caps = {.global_max = {INT32_MAX, 65535, 65535}};
  PolyUOp *rewritten = poly_add_gpudims_ex(ctx, sink, caps);

  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->n_src, 4);
  ASSERT_INT_EQ(rewritten->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(rewritten->src[0]->arg.str, "gidx2");
  ASSERT_INT_EQ(rewritten->src[1]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(rewritten->src[1]->arg.str, "gidx1");

  ASSERT_INT_EQ(rewritten->src[2]->op, POLY_OP_FLOORMOD);
  ASSERT_INT_EQ(rewritten->src[2]->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(rewritten->src[2]->src[0]->arg.str, "gidx0");
  ASSERT_INT_EQ(rewritten->src[2]->src[0]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(rewritten->src[2]->src[0]->src[0]->arg.i, 27);
  ASSERT_INT_EQ(rewritten->src[2]->src[1]->arg.i, 9);

  ASSERT_INT_EQ(rewritten->src[3]->op, POLY_OP_FLOORDIV);
  ASSERT_TRUE(rewritten->src[3]->src[0] == rewritten->src[2]->src[0]);
  ASSERT_INT_EQ(rewritten->src[3]->src[1]->arg.i, 9);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx0"), 27);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx1"), 2);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx2"), 32);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_reverse_group_handles_nonprefix_and_repeated_merges) {
  /* Pinned test/null/test_gpudims.py:73-79 covers reverse grouping when the
   * leftmost pair cannot merge. Keep the ordered-domain origins on that
   * non-prefix contraction as well. */
  PolyCtx *ctx = poly_ctx_new();
  const int64_t nonprefix_bounds[4] = {2, 3, 4, 5};
  PolyUOp *nonprefix_ranges[4];
  for (int i = 0; i < 4; i++) {
    PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(nonprefix_bounds[i]));
    nonprefix_ranges[i] =
        poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(i, POLY_AXIS_GLOBAL));
  }
  PolyUOp *nonprefix_sink = poly_test_kernel_sink(ctx, nonprefix_ranges, 4, "nonprefix");
  PolyRendererCaps nonprefix_caps = {.global_max = {16, 16, 16}};
  PolyUOp *nonprefix = poly_add_gpudims_ex(ctx, nonprefix_sink, nonprefix_caps);
  ASSERT_INT_EQ(nonprefix->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(nonprefix->src[0]->arg.str, "gidx2");
  ASSERT_INT_EQ(nonprefix->src[1]->op, POLY_OP_FLOORMOD);
  ASSERT_INT_EQ(nonprefix->src[1]->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(nonprefix->src[1]->src[0]->arg.str, "gidx1");
  ASSERT_INT_EQ(nonprefix->src[1]->src[1]->arg.i, 3);
  ASSERT_INT_EQ(nonprefix->src[2]->op, POLY_OP_FLOORDIV);
  ASSERT_TRUE(nonprefix->src[2]->src[0] == nonprefix->src[1]->src[0]);
  ASSERT_INT_EQ(nonprefix->src[2]->src[1]->arg.i, 3);
  ASSERT_INT_EQ(nonprefix->src[3]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(nonprefix->src[3]->arg.str, "gidx0");
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, nonprefix, &n_topo);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx0"), 5);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx1"), 12);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx2"), 2);
  poly_ctx_destroy(ctx);

  /* Two ordered-domain merges must preserve the complete contraction order,
   * not only the first adjacent pair used by the HLB regression above. */
  ctx = poly_ctx_new();
  const int64_t repeated_bounds[5] = {6, 5, 4, 3, 2};
  PolyUOp *repeated_ranges[5];
  for (int i = 0; i < 5; i++) {
    PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(repeated_bounds[i]));
    repeated_ranges[i] =
        poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(i, POLY_AXIS_GLOBAL));
  }
  PolyUOp *repeated_sink = poly_test_kernel_sink(ctx, repeated_ranges, 5, "repeated");
  PolyRendererCaps repeated_caps = {.global_max = {100, 100, 100}};
  PolyUOp *repeated = poly_add_gpudims_ex(ctx, repeated_sink, repeated_caps);
  ASSERT_INT_EQ(repeated->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(repeated->src[0]->arg.str, "gidx2");
  ASSERT_INT_EQ(repeated->src[1]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(repeated->src[1]->arg.str, "gidx1");
  ASSERT_INT_EQ(repeated->src[2]->op, POLY_OP_FLOORMOD);
  ASSERT_INT_EQ(repeated->src[2]->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(repeated->src[2]->src[0]->arg.str, "gidx0");
  ASSERT_INT_EQ(repeated->src[2]->src[0]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(repeated->src[2]->src[0]->src[0]->arg.i, 24);
  ASSERT_INT_EQ(repeated->src[2]->src[1]->arg.i, 4);
  ASSERT_INT_EQ(repeated->src[3]->op, POLY_OP_FLOORMOD);
  ASSERT_INT_EQ(repeated->src[3]->src[0]->op, POLY_OP_FLOORDIV);
  ASSERT_TRUE(repeated->src[3]->src[0]->src[0] == repeated->src[2]->src[0]);
  ASSERT_INT_EQ(repeated->src[3]->src[0]->src[1]->arg.i, 4);
  ASSERT_INT_EQ(repeated->src[3]->src[1]->arg.i, 3);
  ASSERT_INT_EQ(repeated->src[4]->op, POLY_OP_FLOORDIV);
  ASSERT_TRUE(repeated->src[4]->src[0] == repeated->src[2]->src[0]);
  ASSERT_INT_EQ(repeated->src[4]->src[1]->arg.i, 12);
  n_topo = 0;
  topo = poly_toposort(ctx, repeated, &n_topo);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx0"), 24);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx1"), 5);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx2"), 6);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_rewrites_large_source_nodes) {
  /* C port guard: tinygrad substitution handles arbitrary source tuple sizes.
   * Polygrad must not cap rewritten source arrays at the small stack scratch
   * size, or model-scale SINK/END nodes can read past initialized sources. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(16));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *srcs[72];
  srcs[0] = range;
  for (int i = 1; i < 72; i++)
    srcs[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
  PolyUOp *sink = poly_test_kernel_sink(ctx, srcs, 72, "large_srcs");
  PolyUOp *rewritten = poly_add_gpudims(ctx, sink);

  ASSERT_INT_EQ(rewritten->n_src, 72);
  ASSERT_TRUE(rewritten->src[0]->op == POLY_OP_SPECIAL);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_SPECIAL), 1);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_RANGE), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_webgpu_splits_oversized_global_dim) {
  /* tinygrad gpudims.py legalizes backend launch dimensions before rendering.
   * WebGPU caps workgroup_id.x at 65535, so a logical 1D launch of 73728
   * workgroups must split into two hardware SPECIAL dimensions while the
   * replacement expression reconstructs the original logical index. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(73728));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *sink = poly_test_kernel_sink(ctx, &range, 1, "split_global");

  PolyRendererCaps caps = {.global_max = {65535, 65535, 65535}, .local_max = {256, 256, 64}};
  PolyUOp *rewritten = poly_add_gpudims_ex(ctx, sink, caps);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_INT_EQ(count_special_named(topo, n_topo, "gidx0"), 1);
  ASSERT_INT_EQ(count_special_named(topo, n_topo, "gidx1"), 1);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx0"), 36864);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx1"), 2);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_RANGE), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, expander2_upcast_range_uses_shaped_scalar_stack_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(3, POLY_AXIS_UPCAST));
  PolyUOp *out = poly_apply_expander2(ctx, range);

  /* Current tinygrad codegen/__init__.py:52-54 constructs a scalar-dtype
   * STACK and applies an identity reshape. UOp CSE removes that reshape, so a
   * single expanded axis is the STACK itself. */
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(out->op, POLY_OP_STACK);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, out), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, out)[0], 4);
  ASSERT_TRUE(poly_dtype_eq(out->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(out->n_src, 4);
  for (int i = 0; i < 4; i++) {
    ASSERT_INT_EQ(out->src[i]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(out->src[i]->arg.kind, POLY_ARG_INT);
    ASSERT_INT_EQ(out->src[i]->arg.i, i);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, expander2_wmma_uses_movement_topology_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *half_rows[2];
  for (int row = 0; row < 2; row++) {
    PolyUOp *lanes[3];
    for (int col = 0; col < 3; col++)
      lanes[col] =
          poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float((double)(row * 3 + col + 1)));
    half_rows[row] = poly_uop_stack(ctx, lanes, 3);
  }
  PolyUOp *a = poly_uop_stack(ctx, half_rows, 2);
  PolyUOp *b = a;
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *acc = poly_uop_stack(ctx, (PolyUOp *[]){zero, zero}, 2);
  int dims[3] = {2, 2, 2};
  int64_t axes_data[3][1][2] = {{{0, 2}}, {{0, 2}}, {{0, 2}}};
  int64_t(*axes[3])[2] = {axes_data[0], axes_data[1], axes_data[2]};
  int n_axes[3] = {1, 1, 1};
  PolyUOp *wmma = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, (PolyUOp *[]){a, b, acc}, 3,
      poly_arg_tensor_core(dims, POLY_FLOAT16, "AMD", 64, axes, n_axes, true)
  );
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_UPCAST));
  PolyUOp *sink =
      poly_uop(ctx, POLY_OP_SINK, POLY_VOID, (PolyUOp *[]){wmma, range}, 2, poly_arg_none());
  PolyUOp *out = poly_apply_expander2(ctx, sink);
  ASSERT_NOT_NULL(out);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, out, &n_topo);
  PolyUOp *expanded_wmma = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_WMMA) expanded_wmma = topo[i];
  }
  ASSERT_NOT_NULL(expanded_wmma);
  ASSERT_TRUE(!expanded_wmma->arg.tensor_core.has_upcast_axes);
  ASSERT_INT_EQ(expanded_wmma->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(expanded_wmma->src[1]->op, POLY_OP_PERMUTE);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, wmma_add_is_absorbed_into_accumulator_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = poly_uop_stack(
      ctx,
      (PolyUOp *[]){
          poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(1.0)),
          poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(2.0)),
      },
      2
  );
  PolyUOp *b = poly_uop_stack(
      ctx,
      (PolyUOp *[]){
          poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(3.0)),
          poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(4.0)),
      },
      2
  );
  PolyUOp *acc = poly_uop_stack(
      ctx,
      (PolyUOp *[]){
          poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(5.0)),
          poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(6.0)),
      },
      2
  );
  PolyUOp *add = poly_uop_stack(
      ctx,
      (PolyUOp *[]){
          poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(7.0)),
          poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(8.0)),
      },
      2
  );
  int dims[3] = {2, 2, 2};
  int n_axes[3] = {0, 0, 0};
  PolyUOp *wmma = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, (PolyUOp *[]){a, b, acc}, 3,
      poly_arg_tensor_core(dims, POLY_FLOAT16, "AMD", 64, NULL, n_axes, false)
  );
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, wmma, add, poly_arg_none());
  PolyUOp *out =
      poly_apply_pm_reduce(ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, sum, poly_arg_none()));
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(out->src[0]->op, POLY_OP_WMMA);
  ASSERT_INT_EQ(out->src[0]->src[2]->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(out->src[0]->src[2]->src[0], acc);
  ASSERT_PTR_EQ(out->src[0]->src[2]->src[1], add);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, wmma_outer_broadcast_builds_indexed_fragments_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *half_vals[6], *float_vals[6];
  for (int i = 0; i < 6; i++) {
    half_vals[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float((double)i + 1.0));
    float_vals[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  }
  PolyUOp *a_flat = poly_uop_stack(ctx, half_vals, 2);
  PolyUOp *b_flat = poly_uop_stack(ctx, half_vals, 6);
  PolyUOp *acc_flat = poly_uop_stack(ctx, float_vals, 6);
  PolyUOp *a = poly_reshape(ctx, a_flat, (int64_t[]){1, 2}, 2);
  PolyUOp *b = poly_reshape(ctx, b_flat, (int64_t[]){3, 2}, 2);
  PolyUOp *acc = poly_reshape(ctx, acc_flat, (int64_t[]){3, 2}, 2);
  int dims[3] = {2, 2, 2};
  int n_axes[3] = {0, 0, 0};
  PolyUOp *wmma = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, (PolyUOp *[]){a, b, acc}, 3,
      poly_arg_tensor_core(dims, POLY_FLOAT16, "AMD", 64, NULL, n_axes, false)
  );
  PolyUOp *out = poly_apply_expand_broadcast_stage(
      ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, wmma, poly_arg_none())
  );
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(out->src[0]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(out->src[0]->n_src, 3);
  for (int i = 0; i < 3; i++) {
    ASSERT_INT_EQ(out->src[0]->src[i]->op, POLY_OP_WMMA);
    ASSERT_INT_EQ(out->src[0]->src[i]->src[0]->op, POLY_OP_INDEX);
    ASSERT_INT_EQ(out->src[0]->src[i]->src[1]->op, POLY_OP_INDEX);
    ASSERT_INT_EQ(out->src[0]->src[i]->src[2]->op, POLY_OP_INDEX);
  }
  ASSERT_INT_EQ(poly_uop_ndim(ctx, out->src[0]), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, out->src[0])[0], 3);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, out->src[0])[1], 2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, devectorizer2_mops_removes_index_of_expanded_scalar) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Current schedule/rangeify.py:41-47 and codegen/__init__.py:145-153:
   * indexing the injected axis of EXPAND(scalar, (3,)) maps to INDEX(scalar),
   * then devectorizer2 removes the empty INDEX. */
  PolyUOp *scalar = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(7));
  PolyUOp *expanded = poly_expand(ctx, scalar, (int64_t[]){3}, 1);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *indexed = poly_uop_index(ctx, expanded, &one, 1);
  ASSERT_NOT_NULL(indexed);
  ASSERT_INT_EQ(indexed->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(indexed->src[0]->op, POLY_OP_EXPAND);

  PolyUOp *rewritten =
      poly_apply_devectorizer2_stage(ctx, indexed, (PolyRendererCaps){.max_vec_width = 1});
  ASSERT_PTR_EQ(rewritten, scalar);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, devectorizer2_normalizes_wmma_sources_to_stack) {
  /* Current do_stack_wmma indexes every non-STACK source into scalar lanes
   * before renderer lowering (tinygrad/codegen/__init__.py:132-140). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a_lanes[] = {
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(1.0)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(2.0)),
  };
  PolyUOp *b_lanes[] = {
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(3.0)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(4.0)),
  };
  PolyUOp *c_lanes[] = {
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0)),
  };
  PolyUOp *stacks[] = {
      poly_uop_stack(ctx, a_lanes, 2),
      poly_uop_stack(ctx, b_lanes, 2),
      poly_uop_stack(ctx, c_lanes, 2),
  };
  PolyUOp *src[3];
  int64_t perm[] = {1, 0};
  for (int i = 0; i < 3; i++) {
    PolyUOp *matrix = poly_reshape(ctx, stacks[i], (int64_t[]){1, 2}, 2);
    src[i] = poly_reshape(ctx, poly_permute(ctx, matrix, perm, 2), (int64_t[]){2}, 1);
    ASSERT_INT_EQ(src[i]->op, POLY_OP_RESHAPE);
  }
  int dims[] = {2, 2, 2}, n_axes[] = {0, 0, 0};
  PolyUOp *wmma = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, src, 3,
      poly_arg_tensor_core(dims, POLY_FLOAT16, "AMD", 64, NULL, n_axes, false)
  );
  PolyUOp *out = poly_apply_devectorizer2_stage(ctx, wmma, (PolyRendererCaps){.max_vec_width = 1});
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(out->op, POLY_OP_WMMA);
  for (int i = 0; i < 3; i++) {
    ASSERT_INT_EQ(out->src[i]->op, POLY_OP_STACK);
    ASSERT_INT_EQ(out->src[i]->n_src, 2);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, devectorizer2_composes_complete_pm_mops_like_current_tinygrad) {
  /* Current devectorizer2 composes schedule/rangeify.py::pm_mops, so movement
   * is moved outside AFTER before later movement cleanup and scalarization. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *base = poly_test_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *movement = poly_reshape(ctx, base, (int64_t[]){2, 3}, 2);
  PolyUOp *effect = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *after_src[] = {movement, effect};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, movement->dtype, after_src, 2, poly_arg_none());

  PolyUOp *rewritten =
      poly_apply_devectorizer2_stage(ctx, after, (PolyRendererCaps){.max_vec_width = 1});
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  ASSERT_INT_EQ(rewritten->src[0]->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(rewritten->src[0]->src[0], base);
  ASSERT_PTR_EQ(rewritten->src[0]->src[1], effect);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, devectorizer2_indexes_reshaped_stack_into_distinct_memory_lanes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Current tinygrad codegen/__init__.py:149-153 first moves RESHAPE out of
   * INDEX(PARAM, ...), then expands STACK coordinates into distinct scalar
   * INDEX nodes.  This is the address topology consumed by memory_coalescing. */
  PolyUOp *buf = poly_test_uop_param(ctx, POLY_FLOAT32, 4, 0, POLY_ADDR_GLOBAL);
  PolyUOp *coords[4];
  for (int i = 0; i < 4; i++)
    coords[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(i));
  PolyUOp *stack = poly_uop_stack(ctx, coords, 4);
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *reshaped = poly_reshape_uop(ctx, stack, &shape, 1);
  PolyUOp *indexed = poly_uop_index(ctx, buf, &reshaped, 1);
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, indexed, poly_arg_none());

  PolyUOp *rewritten =
      poly_apply_devectorizer2_stage(ctx, load, (PolyRendererCaps){.max_vec_width = 1});
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  bool seen[4] = {false, false, false, false};
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_INDEX || u->n_src != 2 || u->src[0] != buf ||
        u->src[1]->op != POLY_OP_CONST || u->src[1]->arg.kind != POLY_ARG_INT)
      continue;
    if (u->src[1]->arg.i >= 0 && u->src[1]->arg.i < 4) seen[u->src[1]->arg.i] = true;
  }
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(seen[i]);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_local_preserves_group_range_replacement_metadata) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:170-184 builds the
   * final REDUCE loop with x.replace(arg=...),
   * so dtype, bound source, tag, and tag_arg stay identical to the original
   * GROUP_REDUCE range and the final INDEX consumes that exact occurrence. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(16));
  PolyUOp *group_srcs[1] = {bound};
  PolyUOp *group = poly_uop_tagged_arg(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, group_srcs, 1, poly_arg_range(7, POLY_AXIS_GROUP_REDUCE),
      23, poly_arg_str("group-range")
  );
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, group, poly_arg_none());
  PolyUOp *reduce_srcs[2] = {value, group};
  PolyUOp *reduce =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, reduce_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *grouped = poly_apply_pm_reduce(ctx, sink);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, grouped, &n_topo);
  PolyUOp *final_range = NULL;
  int n_reduce = 0, n_stage = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    n_reduce += u->op == POLY_OP_REDUCE;
    n_stage += u->op == POLY_OP_STAGE;
    if (u->op == POLY_OP_STAGE) {
      char *text = poly_uop_str(u);
      ASSERT_NOT_NULL(text);
      bool has_local_id = strstr(text, "BufferizeOpts(device=7,") != NULL;
      free(text);
      ASSERT_TRUE(has_local_id);
    }
    if (u->op == POLY_OP_RANGE && poly_range_axis_type(u->arg) == POLY_AXIS_REDUCE &&
        poly_range_axis_id(u->arg) == 107) {
      final_range = u;
    }
  }
  ASSERT_INT_EQ(n_reduce, 0);
  ASSERT_INT_EQ(n_stage, 1);
  ASSERT_NOT_NULL(final_range);
  ASSERT_TRUE(poly_dtype_eq(final_range->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(final_range->n_src, 1);
  ASSERT_PTR_EQ(final_range->src[0], bound);
  ASSERT_INT_EQ(final_range->tag, 23);
  ASSERT_TRUE(final_range->tag_arg.kind == POLY_ARG_STRING);
  ASSERT_STR_EQ(final_range->tag_arg.str, "group-range");

  bool direct_index_coordinate = false;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_INDEX && u->n_src >= 2 && u->src[1] == final_range) {
      direct_index_coordinate = true;
      break;
    }
  }
  ASSERT_TRUE(direct_index_coordinate);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, grouped_reduce_keeps_non_group_loops_beyond_tensor_rank) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *src[POLY_MAX_DIMS + 3];
  src[1] = poly_range(ctx, 2, 0, POLY_AXIS_GROUP_REDUCE);
  PolyUOp *value = poly_cast(ctx, src[1], POLY_INT32);
  for (int i = 2; i < POLY_MAX_DIMS + 3; i++) {
    src[i] = poly_range(ctx, 2, i - 1, POLY_AXIS_REDUCE);
    value = poly_uop2(
        ctx, POLY_OP_ADD, POLY_INT32, value, poly_cast(ctx, src[i], POLY_INT32), poly_arg_none()
    );
  }
  src[0] = value;
  PolyUOp *red = poly_uop(
      ctx, POLY_OP_REDUCE, POLY_INT32, src, POLY_MAX_DIMS + 3, poly_arg_reduce(POLY_OP_ADD, 0)
  );
  PolyUOp *out = poly_apply_pm_reduce(ctx, poly_uop_sink(ctx, &red, 1));
  ASSERT_NOT_NULL(out);
  int n = 0, stages = 0, partial_ends = 0;
  PolyUOp **topo = poly_toposort(ctx, out, &n);
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_STAGE) {
      stages++;
      ASSERT_INT_EQ(u->n_src, 2);
      ASSERT_PTR_EQ(u->src[1], src[1]);
    }
    if (u->op == POLY_OP_END && u->n_src == POLY_MAX_DIMS + 2) partial_ends++;
  }
  ASSERT_INT_EQ(stages, 1);
  ASSERT_INT_EQ(partial_ends, 1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, grouped_reduce_consumes_horizontal_axes_only_once) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *group = poly_range(ctx, 4, 7, POLY_AXIS_GROUP_REDUCE);
  PolyUOp *x = poly_cast(ctx, group, POLY_FLOAT32);
  PolyUOp *lanes[] = {
      poly_alu2(ctx, POLY_OP_ADD, x, poly_const_float(ctx, 1)),
      poly_alu2(ctx, POLY_OP_ADD, x, poly_const_float(ctx, 2))};
  PolyUOp *stack = poly_uop_stack(ctx, lanes, 2);
  PolyUOp *reduce_src[] = {stack, group};
  PolyUOp *reduce =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, reduce_src, 2, poly_arg_reduce(POLY_OP_ADD, 1));
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *idx = poly_uop_index(ctx, out, &zero, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, reduce, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "horizontal_group");
  PolyUOp *lowered = poly_apply_pm_reduce(ctx, sink);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, lowered, &n_topo);
  ASSERT_NOT_NULL(topo);
  int reductions = count_lin_ops(topo, n_topo, POLY_OP_REDUCE);
  poly_toposort_free(topo);
  ASSERT_INT_EQ(reductions, 0);
  int n = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  char *source = poly_render_wgsl(ctx, lin, n, "horizontal_group");
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(strstr(source, "workgroupBarrier"));
  free(source);
#ifdef POLY_HAS_CUDA
  if (poly_cuda_available()) {
    source = poly_render_cuda(ctx, lin, n, "horizontal_group", 4);
    ASSERT_NOT_NULL(source);
    PolyCudaProgram *program = poly_compile_cuda(source, "horizontal_group");
    ASSERT_NOT_NULL(program);
    unsigned long long device = poly_cuda_alloc(sizeof(float));
    ASSERT_TRUE(device != 0);
    void *args[] = {&device};
    ASSERT_INT_EQ(poly_cuda_launch(program, args, 1, 1, 1, 1, 4, 1, 1), 0);
    ASSERT_INT_EQ(poly_cuda_sync(), 0);
    float result = 0;
    ASSERT_INT_EQ(poly_cuda_copy_dtoh(&result, device, sizeof(result)), 0);
    poly_cuda_free(device);
    poly_cuda_program_destroy(program);
    free(source);
    ASSERT_FLOAT_EQ(result, 24.0, 0.0);
  }
#endif
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_local_is_unbounded_and_preserves_ancestor_metadata) {
  /* Tinygrad 2026-08-22/a9069c177a9d pm_reduce_local is one unrestricted
   * graph_rewrite. More than 32 matches or 64 ancestor sources must not alter
   * source order, metadata, or nested-rewrite closure. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(16));
  PolyUOp *group =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(7, POLY_AXIS_GROUP_REDUCE));
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, group, poly_arg_none());
  PolyUOp *reduce_srcs[2] = {value, group};
  PolyUOp *reduce =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, reduce_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0));

  PolyUOp *wide_srcs[70];
  wide_srcs[0] = reduce;
  for (int i = 1; i < 70; i++)
    wide_srcs[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
  PolyUOp *wide = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, wide_srcs, 70, poly_arg_none());
  PolyUOp *wide_out = poly_apply_pm_reduce(ctx, wide);
  ASSERT_NOT_NULL(wide_out);
  ASSERT_INT_EQ(wide_out->n_src, 70);
  for (int i = 1; i < 70; i++)
    ASSERT_PTR_EQ(wide_out->src[i], wide_srcs[i]);

  PolyUOp *many_reduces[33];
  for (int i = 0; i < 33; i++) {
    PolyUOp *range = poly_uop1(
        ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(1000 + i, POLY_AXIS_GROUP_REDUCE)
    );
    PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, range, poly_arg_none());
    PolyUOp *srcs[2] = {cast, range};
    many_reduces[i] = poly_uop_tagged_arg(
        ctx, POLY_OP_REDUCE, POLY_FLOAT32, srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0),
        (uint64_t)(100 + i), poly_arg_none()
    );
  }
  PolyUOp *many = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, many_reduces, 33, poly_arg_none());
  PolyUOp *many_out = poly_apply_pm_reduce(ctx, many);
  ASSERT_NOT_NULL(many_out);
  ASSERT_INT_EQ(many_out->n_src, 33);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, many_out, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_REDUCE) continue;
    for (int j = 1; j < u->n_src; j++) {
      ASSERT_FALSE(
          u->src[j]->op == POLY_OP_RANGE &&
          poly_range_axis_type(u->src[j]->arg) == POLY_AXIS_GROUP_REDUCE
      );
    }
  }

  PolyUOp *wrapped = poly_uop_tagged_arg(
      ctx, POLY_OP_NEG, POLY_FLOAT32, &reduce, 1, poly_arg_none(), 77, poly_arg_str("must-survive")
  );
  PolyUOp *tagged = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, wrapped, poly_arg_none());
  PolyUOp *tagged_out = poly_apply_pm_reduce(ctx, tagged);
  ASSERT_NOT_NULL(tagged_out);
  ASSERT_INT_EQ(tagged_out->n_src, 1);
  ASSERT_INT_EQ(tagged_out->src[0]->tag, 77);
  ASSERT_INT_EQ(tagged_out->src[0]->tag_arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(tagged_out->src[0]->tag_arg.str, "must-survive");

  /* graph_rewrite visits children before rebuilding/matching their parent.
   * The aggregate substitution must therefore also rewrite a grouped child
   * reachable only through the replacement constructed for a grouped parent. */
  PolyUOp *inner_range = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(2000, POLY_AXIS_GROUP_REDUCE)
  );
  PolyUOp *inner_cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, inner_range, poly_arg_none());
  PolyUOp *inner_srcs[2] = {inner_cast, inner_range};
  PolyUOp *inner =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, inner_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *outer_range = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(2001, POLY_AXIS_GROUP_REDUCE)
  );
  PolyUOp *outer_cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, outer_range, poly_arg_none());
  PolyUOp *outer_value =
      poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, inner, outer_cast, poly_arg_none());
  PolyUOp *outer_srcs[2] = {outer_value, outer_range};
  PolyUOp *outer =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, outer_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *nested = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, outer, poly_arg_none());
  PolyUOp *nested_out = poly_apply_pm_reduce(ctx, nested);
  ASSERT_NOT_NULL(nested_out);
  int n_nested = 0;
  PolyUOp **nested_topo = poly_toposort(ctx, nested_out, &n_nested);
  for (int i = 0; i < n_nested; i++) {
    PolyUOp *u = nested_topo[i];
    if (u->op != POLY_OP_REDUCE) continue;
    for (int j = 1; j < u->n_src; j++) {
      ASSERT_FALSE(
          u->src[j]->op == POLY_OP_RANGE &&
          poly_range_axis_type(u->src[j]->arg) == POLY_AXIS_GROUP_REDUCE
      );
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_group_reduce_after_source_becomes_lidx) {
  /* tinygrad gpudims.py substitutes GROUP_REDUCE axes as local workitem
   * indices. It only skips AxisType.REDUCE. Polygrad must not infer "serial
   * reduce" from accumulator AFTER source lists, because group-reduce ranges can
   * appear there as input-context ranges and still need to become lidxN. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(256));
  PolyUOp *group_range = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(1000, POLY_AXIS_GROUP_REDUCE)
  );
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyParamArg reg_arg = {.slot = 0, .addrspace = POLY_ADDR_REG};
  PolyUOp *acc = poly_uop1(ctx, POLY_OP_BUFFER, POLY_FLOAT32, one, poly_arg_param(&reg_arg));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, acc, zero, poly_arg_none());
  PolyUOp *init = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, idx0,
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0)), poly_arg_none()
  );
  PolyUOp *after_srcs[3] = {acc, init, group_range};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, POLY_FLOAT32, after_srcs, 3, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, after, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &load, 1, "group_reduce");

  PolyUOp *rewritten = poly_add_gpudims(ctx, sink);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);

  ASSERT_INT_EQ(count_special_named(topo, n_topo, "lidx0"), 1);
  for (int i = 0; i < n_topo; i++) {
    ASSERT_FALSE(
        topo[i]->op == POLY_OP_RANGE && poly_arg_is_range(topo[i]->arg) &&
        poly_range_axis_type(topo[i]->arg) == POLY_AXIS_GROUP_REDUCE
    );
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, gpu_output_gate_matches_pinned_gpudims_gater_and_linear_cleanup) {
  /* Pinned tinygrad:
   *   gpudims.py:92-99       -> INDEX(buf, WHERE(gate, idx, Invalid))
   *   late/gater.py:15-17    -> STORE(INDEX(buf, idx), data, gate)
   *   codegen/__init__.py:152-174 -> IF / STORE / ENDIF lines. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(32));
  PolyUOp *local =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_GROUP_REDUCE));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, zero, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, local, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, value, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "gated_store");

  PolyRendererCaps caps = {
      .has_mulacc = true,
      .has_int64 = true,
      .has_local = true,
      .global_max = {2147483647, 65535, 65535},
      .local_max = {1024, 1024, 64},
  };
  PolyUOp *gpudims = poly_add_gpudims_ex(ctx, sink, caps);
  ASSERT_NOT_NULL(gpudims);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, gpudims, &n_topo);
  PolyUOp *gpudims_store = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_STORE) gpudims_store = topo[i];
  ASSERT_NOT_NULL(gpudims_store);
  ASSERT_INT_EQ(gpudims_store->n_src, 2);
  PolyUOp *gpudims_idx = poly_as_index(gpudims_store->src[0]);
  ASSERT_NOT_NULL(gpudims_idx);
  ASSERT_INT_EQ(gpudims_idx->n_src, 2);
  ASSERT_INT_EQ(gpudims_idx->src[1]->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_is_int(gpudims_idx->src[1]->dtype));
  ASSERT_TRUE(poly_dtype_eq(gpudims_idx->src[1]->src[0]->dtype, POLY_BOOL));
  ASSERT_INT_EQ(gpudims_idx->src[1]->src[2]->arg.kind, POLY_ARG_INVALID);
  PolyUOp *gpudims_gate = gpudims_idx->src[1]->src[0];
  ASSERT_INT_EQ(gpudims_gate->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(gpudims_gate->n_src, 2);
  ASSERT_INT_EQ(gpudims_gate->src[0]->op, POLY_OP_CMPNE);
  ASSERT_TRUE(poly_dtype_eq(gpudims_gate->src[1]->dtype, POLY_BOOL));
  ASSERT_INT_EQ(gpudims_gate->src[1]->arg.kind, POLY_ARG_BOOL);
  ASSERT_TRUE(gpudims_gate->src[1]->arg.b);
  PolyUOp *gpudims_not_zero = gpudims_gate->src[0];
  ASSERT_INT_EQ(gpudims_not_zero->n_src, 2);
  ASSERT_TRUE(poly_dtype_is_index(gpudims_not_zero->src[0]->dtype));
  ASSERT_TRUE(poly_dtype_is_index(gpudims_not_zero->src[1]->dtype));
  ASSERT_INT_EQ(gpudims_not_zero->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(gpudims_not_zero->src[1]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(gpudims_not_zero->src[1]->arg.i, 0);
  ASSERT_TRUE(poly_type_verify_tensor(ctx, gpudims));

  PolyRewriteOpts opts = {
      .optimize = false,

      .caps = caps,
      .device = POLY_DEVICE_CUDA,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);
  topo = poly_toposort(ctx, rewritten, &n_topo);
  PolyUOp *final_store = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (poly_dtype_is_index(topo[i]->dtype)) {
      ASSERT_INT_EQ(topo[i]->op, POLY_OP_CONST);
      ASSERT_INT_EQ(topo[i]->arg.kind, POLY_ARG_INT);
    }
    ASSERT_FALSE(topo[i]->op == POLY_OP_CONST && topo[i]->arg.kind == POLY_ARG_INVALID);
    if (topo[i]->op == POLY_OP_STORE) final_store = topo[i];
    if (topo[i]->op != POLY_OP_INDEX) continue;
    for (int j = 1; j < topo[i]->n_src; j++)
      ASSERT_TRUE(poly_dtype_is_int(topo[i]->src[j]->dtype));
  }
  ASSERT_NOT_NULL(final_store);
  ASSERT_INT_EQ(final_store->n_src, 3);
  ASSERT_TRUE(poly_dtype_eq(final_store->src[2]->dtype, POLY_BOOL));
  PolyUOp *final_idx = poly_as_index(final_store->src[0]);
  ASSERT_NOT_NULL(final_idx);
  ASSERT_INT_EQ(final_idx->n_src, 2);
  ASSERT_TRUE(poly_type_verify_program(ctx, rewritten));

  int n_linear = 0;
  PolyUOp **linear = poly_do_linearize(ctx, rewritten, &n_linear);
  ASSERT_NOT_NULL(linear);
  int if_count = 0, endif_count = 0, store_count = 0;
  for (int i = 0; i < n_linear; i++) {
    if (linear[i]->op == POLY_OP_IF) {
      if_count++;
      ASSERT_TRUE(i + 2 < n_linear);
      ASSERT_INT_EQ(linear[i + 1]->op, POLY_OP_STORE);
      ASSERT_INT_EQ(linear[i + 1]->n_src, 2);
      ASSERT_INT_EQ(linear[i + 2]->op, POLY_OP_ENDIF);
      ASSERT_PTR_EQ(linear[i + 2]->src[0], linear[i]);
    }
    if (linear[i]->op == POLY_OP_ENDIF) endif_count++;
    if (linear[i]->op == POLY_OP_STORE) {
      store_count++;
      ASSERT_INT_EQ(linear[i]->n_src, 2);
    }
  }
  ASSERT_INT_EQ(if_count, 1);
  ASSERT_INT_EQ(endif_count, 1);
  ASSERT_INT_EQ(store_count, 1);

  free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, casted_invalid_index_reaches_gater_and_control_flow) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:74-78 and
   * codegen/late/gater.py:5-17 preserve a casted validity predicate until it
   * becomes STORE(gate), then IF/STORE/ENDIF in the linear program. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = program_param(ctx, POLY_FLOAT32, 2, 0);
  PolyUOp *bound = poly_const_int(ctx, 2);
  PolyUOp *index = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_WEAKINT, bound, poly_arg_str("gidx0"));
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *gate = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, index, zero, poly_arg_none());
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *masked =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, index, invalid, poly_arg_none());
  PolyUOp *casted = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, masked, poly_arg_none());
  PolyUOp *out_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, casted, poly_arg_none());
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_index, one, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "casted_invalid_gate");

  PolyRewriteOpts opts = {
      .optimize = false,
      .caps =
          {
              .has_mulacc = true,
              .has_int64 = true,
              .has_local = true,
              .global_max = {2147483647, 65535, 65535},
              .local_max = {1024, 1024, 64},
          },
      .device = POLY_DEVICE_CUDA,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  PolyUOp *final_store = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_STORE) final_store = topo[i];
  ASSERT_NOT_NULL(final_store);
  ASSERT_INT_EQ(final_store->n_src, 3);
  ASSERT_TRUE(poly_dtype_eq(final_store->src[2]->dtype, POLY_BOOL));

  int n_linear = 0;
  PolyUOp **linear = poly_do_linearize(ctx, rewritten, &n_linear);
  ASSERT_NOT_NULL(linear);
  int if_count = 0, store_count = 0, endif_count = 0;
  for (int i = 0; i < n_linear; i++) {
    if_count += linear[i]->op == POLY_OP_IF;
    store_count += linear[i]->op == POLY_OP_STORE;
    endif_count += linear[i]->op == POLY_OP_ENDIF;
  }
  ASSERT_INT_EQ(if_count, 1);
  ASSERT_INT_EQ(store_count, 1);
  ASSERT_INT_EQ(endif_count, 1);

  free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, move_where_on_load_preserves_cast_and_existing_valid) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:398-416 moves
   * the outer range clause into the existing invalid coordinate and keeps
   * or_casted around the rebuilt INDEX. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *buf = program_param(ctx, POLY_INT32, 4, 0);
  PolyUOp *range = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, poly_const_int(ctx, 4), poly_arg_range(0, POLY_AXIS_LOOP)
  );
  PolyUOp *old_gate =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, range, poly_const_int(ctx, 3), poly_arg_none());
  PolyUOp *coord = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_WEAKINT, old_gate, range,
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid()), poly_arg_none()
  );
  PolyUOp *outer_gate =
      poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, range, poly_const_int(ctx, 0), poly_arg_none());
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, buf, coord, poly_arg_none());
  PolyUOp *casted = poly_cast(ctx, index, POLY_FLOAT32);
  PolyUOp *where = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_FLOAT32, outer_gate, casted, poly_const_float(ctx, 0.0),
      poly_arg_none()
  );
  PolyPatternMatcher *stage = poly_pm_concat(poly_sym(), poly_pm_move_where_on_load());
  PolyUOp *rewritten = poly_graph_rewrite(ctx, where, stage);
  poly_pm_destroy(stage);

  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_CAST);
  ASSERT_INT_EQ(rewritten->n_src, 1);
  ASSERT_INT_EQ(rewritten->src[0]->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(rewritten->src[0]->n_src, 2);
  PolyUOp *new_coord = rewritten->src[0]->src[1];
  ASSERT_INT_EQ(new_coord->op, POLY_OP_WHERE);
  ASSERT_INT_EQ(new_coord->n_src, 3);
  ASSERT_INT_EQ(new_coord->src[0]->op, POLY_OP_AND);
  ASSERT_PTR_EQ(new_coord->src[0]->src[0], old_gate);
  ASSERT_PTR_EQ(new_coord->src[0]->src[1], outer_gate);
  ASSERT_PTR_EQ(new_coord->src[1], range);
  ASSERT_INT_EQ(new_coord->src[2]->arg.kind, POLY_ARG_INVALID);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, late_gater_merges_where_into_existing_load_gate) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/late/gater.py:5-21 turns
   * gate.where(LOAD(idx, zero, gate), alt) into LOAD(idx, alt, gate). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *buf = program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, buf, zero, poly_arg_none());
  PolyUOp *var =
      poly_uop1(ctx, POLY_OP_SPECIAL, POLY_WEAKINT, poly_const_int(ctx, 2), poly_arg_str("gidx0"));
  PolyUOp *gate = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, var, zero, poly_arg_none());
  PolyUOp *load_src[3] = {idx, poly_const_float(ctx, 0.0), gate};
  PolyUOp *load = poly_uop(ctx, POLY_OP_LOAD, POLY_FLOAT32, load_src, 3, poly_arg_none());
  PolyUOp *alt = poly_const_float(ctx, 7.0);
  PolyUOp *where = poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, gate, load, alt, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, where, poly_pm_move_gates_from_index());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_LOAD);
  ASSERT_INT_EQ(rewritten->n_src, 3);
  ASSERT_PTR_EQ(rewritten->src[0], idx);
  ASSERT_INT_EQ(rewritten->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(rewritten->src[1]->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(rewritten->src[1]->arg.kind, POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(rewritten->src[1]->arg.f, 7.0, 0.0);
  ASSERT_PTR_EQ(rewritten->src[2], gate);

  PolyUOp *not_gate = poly_uop2(
      ctx, POLY_OP_CMPNE, POLY_BOOL, gate, poly_const_typed(ctx, POLY_BOOL, 1.0), poly_arg_none()
  );
  PolyUOp *reverse_load_src[3] = {idx, poly_const_float(ctx, 0.0), not_gate};
  PolyUOp *reverse_load =
      poly_uop(ctx, POLY_OP_LOAD, POLY_FLOAT32, reverse_load_src, 3, poly_arg_none());
  PolyUOp *reverse =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, gate, alt, reverse_load, poly_arg_none());
  rewritten = poly_graph_rewrite(ctx, reverse, poly_pm_move_gates_from_index());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_LOAD);
  ASSERT_INT_EQ(rewritten->n_src, 3);
  ASSERT_PTR_EQ(rewritten->src[0], idx);
  ASSERT_INT_EQ(rewritten->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(rewritten->src[1]->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(rewritten->src[1]->arg.kind, POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(rewritten->src[1]->arg.f, 7.0, 0.0);
  ASSERT_PTR_EQ(rewritten->src[2], not_gate);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, late_gater_only_moves_the_matched_coordinate) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/late/gater.py:14-17 matches
   * the first gated coordinate of an arbitrary-length INDEX. A three-axis
   * INDEX keeps the other two coordinates gated. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *buf = program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *special =
      poly_uop1(ctx, POLY_OP_SPECIAL, POLY_WEAKINT, poly_const_int(ctx, 2), poly_arg_str("gidx0"));
  PolyUOp *gate =
      poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, special, poly_const_int(ctx, 0), poly_arg_none());
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *coords[3];
  for (int i = 0; i < 3; i++) {
    PolyUOp *coord = poly_const_int(ctx, i);
    coords[i] = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, coord, invalid, poly_arg_none());
  }
  PolyUOp *index_src[4] = {buf, coords[0], coords[1], coords[2]};
  PolyUOp *index = poly_uop(ctx, POLY_OP_INDEX, POLY_FLOAT32, index_src, 4, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, index, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, load, poly_pm_move_gates_from_index());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_LOAD);
  ASSERT_INT_EQ(rewritten->n_src, 3);
  ASSERT_PTR_EQ(rewritten->src[2], gate);
  ASSERT_INT_EQ(rewritten->src[0]->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(rewritten->src[0]->n_src, 4);
  ASSERT_INT_EQ(rewritten->src[0]->src[1]->op, POLY_OP_CONST);
  ASSERT_PTR_EQ(rewritten->src[0]->src[2], coords[1]);
  ASSERT_PTR_EQ(rewritten->src[0]->src[3], coords[2]);

  PolyUOp *image_index_src[3] = {buf, coords[0], coords[1]};
  PolyUOp *image_index =
      poly_uop(ctx, POLY_OP_INDEX, POLY_FLOAT32, image_index_src, 3, poly_arg_none());
  PolyUOp *image_load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, image_index, poly_arg_none());
  rewritten = poly_graph_rewrite(ctx, image_load, poly_pm_move_gates_from_index());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->n_src, 3);
  ASSERT_PTR_EQ(rewritten->src[2], gate);
  ASSERT_INT_EQ(rewritten->src[0]->n_src, 3);
  ASSERT_INT_EQ(rewritten->src[0]->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(rewritten->src[0]->src[2]->op, POLY_OP_CONST);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, late_gater_has_no_index_source_cap) {
  /* Tinygrad's allow_any_len row handles a 65-coordinate INDEX and moves only
   * its matched first coordinate; Polygrad must not impose a fixed C cap. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *buf = program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *special =
      poly_uop1(ctx, POLY_OP_SPECIAL, POLY_WEAKINT, poly_const_int(ctx, 2), poly_arg_str("gidx0"));
  PolyUOp *gate =
      poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, special, poly_const_int(ctx, 0), poly_arg_none());
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *index_src[66] = {buf};
  for (int i = 0; i < 65; i++) {
    index_src[i + 1] = poly_uop3(
        ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, poly_const_int(ctx, i), invalid, poly_arg_none()
    );
  }
  PolyUOp *index = poly_uop(ctx, POLY_OP_INDEX, POLY_FLOAT32, index_src, 66, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, index, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, load, poly_pm_move_gates_from_index());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->n_src, 3);
  ASSERT_PTR_EQ(rewritten->src[2], gate);
  ASSERT_INT_EQ(rewritten->src[0]->n_src, 66);
  ASSERT_INT_EQ(rewritten->src[0]->src[1]->op, POLY_OP_CONST);
  for (int i = 2; i < 66; i++)
    ASSERT_PTR_EQ(rewritten->src[0]->src[i], index_src[i]);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, gpu_output_gate_tracks_each_missing_local_range) {
  /* Pinned tinygrad codegen/gpudims.py:92-99 gates the exact set difference
   * `local_dims - idx.ranges`. An output index can contain one ordinary LOCAL
   * range while omitting a GROUP_REDUCE range; only the omitted occurrence
   * may appear in the validity predicate. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c5 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(5));
  PolyUOp *c4 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *c8 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *global =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, c5, poly_arg_range(1, POLY_AXIS_GLOBAL));
  PolyUOp *group =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, c8, poly_arg_range(2, POLY_AXIS_GROUP_REDUCE));
  PolyUOp *local =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, c4, poly_arg_range(3, POLY_AXIS_LOCAL));
  PolyUOp *out = poly_test_uop_param(ctx, POLY_FLOAT32, 20, 0, POLY_ADDR_GLOBAL);
  PolyUOp *coord = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, global, c4, poly_arg_none()), local, poly_arg_none()
  );
  PolyUOp *target = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, coord, poly_arg_none());
  PolyUOp *value = poly_uop2(
      ctx, POLY_OP_ADD, POLY_FLOAT32,
      poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, local, poly_arg_none()),
      poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, group, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, target, value, poly_arg_none());
  PolyUOp *end_srcs[4] = {store, global, group, local};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 4, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "missing_local");
  PolyRendererCaps caps = {
      .has_local = true,
      .global_max = {INT32_MAX, 65535, 65535},
      .local_max = {1024, 1024, 64},
  };

  PolyUOp *rewritten = poly_add_gpudims_ex(ctx, sink, caps);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_CMPNE), 2);

  PolyUOp *where = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_WHERE) where = topo[i];
  ASSERT_NOT_NULL(where);
  int n_gate = 0;
  PolyUOp **gate_topo = poly_toposort(ctx, where->src[0], &n_gate);
  ASSERT_INT_EQ(count_special_named(gate_topo, n_gate, "lidx0"), 1);
  ASSERT_INT_EQ(count_special_named(gate_topo, n_gate, "lidx1"), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, gpu_output_gate_accepts_element_typed_index_like_tinygrad) {
  /* Current tinygrad/codegen/gpudims.py:92 checks idx.src[0].addrspace.
   * Scheduled storage uses shaped PARAM<element>, not a pointer DType. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(16));
  PolyUOp *local =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_GROUP_REDUCE));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyParamArg out_arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_INT32, one, poly_arg_param(&out_arg));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, zero, poly_arg_none());
  ASSERT_TRUE(poly_program_memory_is(idx->src[0], POLY_ADDR_GLOBAL));
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, local, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, value, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "gate");
  PolyRendererCaps caps = {
      .has_local = true,
      .global_max = {INT32_MAX, 65535, 65535},
      .local_max = {1024, 1024, 64},
  };

  PolyUOp *rewritten = poly_add_gpudims_ex(ctx, sink, caps);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_CMPNE), 2);
  PolyUOp *where = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_WHERE) where = topo[i];
  ASSERT_NOT_NULL(where);
  ASSERT_INT_EQ(where->n_src, 3);
  ASSERT_INT_EQ(where->src[2]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(where->src[2]->arg.kind, POLY_ARG_INVALID);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_index_dtype_preserves_rank10_index_sources) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 0, POLY_ADDR_GLOBAL);

  PolyUOp *idx_srcs[11];
  idx_srcs[0] = buf;
  for (int i = 1; i < 11; i++)
    idx_srcs[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));

  PolyUOp *idx = poly_uop(ctx, POLY_OP_INDEX, POLY_FLOAT32, idx_srcs, 11, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &load, 1, "rank10_index");

  PolyUOp *rewritten = poly_apply_post_index_symbolic_stage(ctx, sink);
  ASSERT_NOT_NULL(rewritten);

  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n);
  bool saw_rank10_index = false;
  for (int i = 0; i < n; i++) {
    if (topo[i]->op != POLY_OP_INDEX || topo[i]->n_src != 11) continue;
    saw_rank10_index = true;
    for (int j = 1; j < topo[i]->n_src; j++)
      ASSERT_TRUE(poly_dtype_eq(topo[i]->src[j]->dtype, POLY_INT32));
  }
  ASSERT_TRUE(saw_rank10_index);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_index_dtype_memoizes_repeated_weak_source) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/weak.py:60-68 uses the
   * lower-index rewrite's ctx dict to lower one repeated weak source once. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *shared = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, one, two, poly_arg_none());
  PolyUOp *src[64];
  for (int i = 0; i < 64; i++)
    src[i] = shared;
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, src, 64, poly_arg_none());

  PolyPatternMatcher *lower = poly_pm_concat(poly_symbolic_simple(), poly_pm_lower_index_dtype());
  PolyMap *lower_cache = poly_map_new(16);
  ASSERT_NOT_NULL(lower);
  ASSERT_NOT_NULL(lower_cache);
  PolyUOp *lowered = poly_graph_rewrite_ctx(ctx, sink, lower, lower_cache);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(poly_map_len(lower_cache), 1);
  ASSERT_INT_EQ(lowered->op, POLY_OP_SINK);
  ASSERT_INT_EQ(lowered->n_src, 64);
  ASSERT_INT_EQ(lowered->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(lowered->src[0]->arg.i, 3);
  for (int i = 1; i < lowered->n_src; i++)
    ASSERT_PTR_EQ(lowered->src[i], lowered->src[0]);

  poly_map_destroy(lower_cache);
  poly_pm_destroy(lower);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, weak_cast_const_survives_symbolic_until_consumer_lowering) {
  /* Current tinygrad symbolic.py:145-147 restricts CAST(CONST) folding to
   * concrete dtypes.all. weak.py:54-76 preserves a lone weak CAST and lets a
   * concrete consumer absorb its inner value during pm_lower_index_dtype. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *weak = poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, four, poly_arg_none());

  PolyUOp *symbolic = poly_graph_rewrite(ctx, weak, poly_symbolic_simple());
  ASSERT_PTR_EQ(symbolic, weak);
  ASSERT_INT_EQ(symbolic->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_is_index(symbolic->dtype));
  ASSERT_PTR_EQ(symbolic->src[0], four);

  PolyUOp *noop = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *end_src[2] = {noop, weak};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_SINK);
  ASSERT_INT_EQ(lowered->src[0]->op, POLY_OP_END);
  ASSERT_INT_EQ(lowered->src[0]->n_src, 2);
  ASSERT_PTR_EQ(lowered->src[0]->src[0], noop);
  ASSERT_PTR_EQ(lowered->src[0]->src[1], four);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, default_dtype_lowering_keeps_integer_bound_policy) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *f = poly_graph_rewrite(ctx, poly_const_float(ctx, 1.25), poly_pm_lower_weak());
  PolyUOp *i = poly_graph_rewrite(ctx, poly_const_int(ctx, 1), poly_pm_lower_weak());
  ASSERT_NOT_NULL(f);
  ASSERT_INT_EQ(f->op, POLY_OP_CAST);
  ASSERT_INT_EQ(f->n_src, 1);
  ASSERT_PTR_EQ(
      f->src[0], poly_uop_const(ctx, poly_arg_float(1.25), poly_dtype_strong(POLY_WEAKFLOAT))
  );
  ASSERT_NOT_NULL(i);
  ASSERT_INT_EQ(i->op, POLY_OP_CAST);
  ASSERT_PTR_EQ(i->src[0], poly_uop_const(ctx, poly_arg_int(1), POLY_INT32));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_weak_const_creates_untagged_literal) {
  /* pm_lower_weak creates a fresh UOp.const, not a tagged replacement. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType types[] = {POLY_WEAKINT, POLY_WEAKFLOAT};
  PolyArg values[] = {poly_arg_int(7), poly_arg_float(0.6931471825)};
  for (int i = 0; i < 2; i++) {
    PolyUOp *root = poly_uop_tagged_arg(
        ctx, POLY_OP_CONST, types[i], NULL, 0, values[i], 91, poly_arg_str("literal")
    );
    PolyUOp *out = poly_graph_rewrite(ctx, root, poly_pm_lower_weak());
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(out->op, POLY_OP_CAST);
    ASSERT_INT_EQ(out->tag, 0);
    ASSERT_INT_EQ(out->tag_arg.kind, POLY_ARG_NONE);
    ASSERT_INT_EQ(out->src[0]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(out->src[0]->tag, 0);
    ASSERT_INT_EQ(out->src[0]->tag_arg.kind, POLY_ARG_NONE);
    ASSERT_PTR_EQ(out->src[0], poly_uop_const(ctx, values[i], poly_dtype_strong(types[i])));
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_weak_alu_resource_preserves_tag) {
  /* The PARAM/BUFFER rule replaces ParamArg.dtype and preserves UOp.tag. */
  PolyCtx *ctx = poly_ctx_new();
  PolyOps ops[] = {POLY_OP_PARAM, POLY_OP_BUFFER};
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyParamArg arg = {.slot = 0, .dtype = POLY_WEAKINT, .addrspace = POLY_ADDR_ALU};
  for (int i = 0; i < 2; i++) {
    PolyUOp *root = poly_uop_tagged_arg(
        ctx, ops[i], POLY_WEAKINT, &shape, 1, poly_arg_param(&arg), 92, poly_arg_str("resource")
    );
    PolyUOp *out = poly_graph_rewrite(ctx, root, poly_pm_lower_weak());
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(out->op, POLY_OP_CAST);
    ASSERT_INT_EQ(out->tag, 0);
    ASSERT_INT_EQ(out->tag_arg.kind, POLY_ARG_NONE);
    PolyUOp *resource = out->src[0];
    ASSERT_INT_EQ(resource->op, ops[i]);
    ASSERT_INT_EQ(resource->tag, 92);
    ASSERT_INT_EQ(resource->tag_arg.kind, POLY_ARG_STRING);
    ASSERT_STR_EQ(resource->tag_arg.str, "resource");
    ASSERT_INT_EQ(resource->arg.kind, POLY_ARG_PARAM);
    ASSERT_INT_EQ(resource->arg.param->slot, arg.slot);
    ASSERT_INT_EQ(resource->arg.param->addrspace, POLY_ADDR_ALU);
    ASSERT_TRUE(poly_dtype_eq(resource->arg.param->dtype, resource->dtype));
    ASSERT_TRUE(poly_dtype_eq(resource->dtype, POLY_INT64));
    ASSERT_INT_EQ(resource->n_src, 1);
    ASSERT_PTR_EQ(resource->src[0], shape);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_gated_index_preserves_tag_and_width_boundary) {
  /* pm_lower_index_dtype narrows through n-1 <= INT32_MAX and replaces
   * INDEX/SHRINK without erasing their identity tags or SHRINK extent. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *shape = poly_shape_to_shape_arg(ctx, NULL, 0);
  PolyParamArg arg = {.slot = 1, .dtype = POLY_INT64, .addrspace = POLY_ADDR_ALU};
  PolyUOp *coord = poly_uop1(ctx, POLY_OP_PARAM, POLY_INT64, shape, poly_arg_param(&arg));
  PolyUOp *four = poly_uop_const(ctx, poly_arg_int(4), POLY_INT64);
  PolyUOp *gate = poly_alu2(ctx, POLY_OP_CMPLT, coord, four);
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_BOOL);
  PolyUOp *valid = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT64, gate, coord, invalid, poly_arg_none());
  int64_t sizes[] = {8, (int64_t)INT32_MAX + 1, (int64_t)INT32_MAX + 2};
  PolyOps ops[] = {POLY_OP_INDEX, POLY_OP_SHRINK};
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      PolyUOp *buffer = program_param(ctx, POLY_FLOAT32, sizes[j], 0);
      PolyUOp *sources[] = {buffer, valid, four};
      int n_src = i ? 3 : 2;
      PolyUOp *root = poly_uop_tagged_arg(
          ctx, ops[i], POLY_FLOAT32, sources, n_src, poly_arg_none(), 93, poly_arg_str("address")
      );
      ASSERT_NOT_NULL(root);
      PolyUOp *out = poly_graph_rewrite(ctx, root, poly_pm_lower_index_dtype());
      ASSERT_NOT_NULL(out);
      ASSERT_INT_EQ(out->op, ops[i]);
      ASSERT_INT_EQ(out->n_src, n_src);
      if (i) ASSERT_PTR_EQ(out->src[2], four);
      ASSERT_INT_EQ(out->tag, 93);
      ASSERT_INT_EQ(out->tag_arg.kind, POLY_ARG_STRING);
      ASSERT_STR_EQ(out->tag_arg.str, "address");
      ASSERT_INT_EQ(out->src[0]->arg.param->slot, 0);
      ASSERT_INT_EQ(out->src[1]->op, POLY_OP_WHERE);
      PolyUOp *lowered_coord = out->src[1]->src[1];
      ASSERT_TRUE(poly_dtype_eq(lowered_coord->dtype, j < 2 ? POLY_INT32 : POLY_INT64));
      if (j < 2) {
        ASSERT_INT_EQ(lowered_coord->op, POLY_OP_CAST);
        ASSERT_PTR_EQ(lowered_coord->src[0], coord);
      } else {
        ASSERT_PTR_EQ(lowered_coord, coord);
      }
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, weak_float_commit_uses_dtype_const_rounding) {
  /* Current tinygrad uop/weak.py:14-16 commits a weak CONST through
   * UOp.const(value, dtype). DType.const rounds float32 before CSE, so two
   * mathematically different doubles with the same float32 value coalesce. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *rounded =
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float((double)(float)0.6931471825));
  PolyUOp *weak = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(0.6931471825));
  PolyUOp *root = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, rounded, weak, poly_arg_none());
  PolyUOp *lowered = poly_graph_rewrite(ctx, root, poly_pm_commit_weak());
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(lowered->src[0], rounded);
  ASSERT_PTR_EQ(lowered->src[1], rounded);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, linearizer_allocation_failure_rejects_partial_order) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = policy_const(ctx, 9), *b = policy_const(ctx, 2);
  PolyUOp *sources[] = {a, b};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sources, 2, poly_arg_none());
  bool complete = false;
  for (int budget = 0; budget < 64; budget++) {
    poly_test_linearizer_alloc_fail_after(budget);
    int count = -1;
    PolyUOp **linear = poly_linearize(ctx, sink, &count);
    poly_test_linearizer_alloc_fail_after(-1);
    if (linear) {
      complete = count == 3 && linear[0] == b && linear[1] == a && linear[2] == sink;
      free(linear);
      break;
    }
    ASSERT_INT_EQ(count, 0);
  }
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(complete);
  PASS();
}

TEST(codegen, linearize_preserves_exact_large_run_counts) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_range(ctx, INT64_C(1) << 40, 0, POLY_AXIS_LOOP);
  PolyUOp *b = poly_range(ctx, INT64_C(1) << 30, 1, POLY_AXIS_LOOP);
  PolyUOp *c = poly_range(ctx, INT64_C(1) << 41, 2, POLY_AXIS_LOOP);
  PolyUOp *d = poly_range(ctx, INT64_C(1) << 30, 3, POLY_AXIS_LOOP);
  PolyUOp *small = poly_alu2(ctx, POLY_OP_MUL, a, b);
  PolyUOp *large = poly_alu2(ctx, POLY_OP_ADD, c, d);
  PolyUOp *sources[] = {large, small};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sources, 2, poly_arg_none());
  int count = 0, small_index = -1, large_index = -1;
  PolyUOp **linear = poly_linearize(ctx, sink, &count);
  for (int i = 0; linear && i < count; i++) {
    if (linear[i] == small) small_index = i;
    if (linear[i] == large) large_index = i;
  }
  free(linear);
  poly_ctx_destroy(ctx);
  /* 2^70 precedes 2^71 even though the opcode tie-break prefers ADD. */
  ASSERT_TRUE(small_index >= 0 && small_index < large_index);
  PASS();
}

TEST(codegen, linearize_honors_tuple_order) {
  const char *value = getenv("TUPLE_ORDER");
  char *saved = value ? strdup(value) : NULL;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *first = policy_const(ctx, 9), *second = policy_const(ctx, 2);
  PolyUOp *sources[] = {first, second};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sources, 2, poly_arg_none());
  bool matches = true;
  for (int enabled = 0; enabled <= 1; enabled++) {
    setenv("TUPLE_ORDER", enabled ? "1" : "0", 1);
    int count = 0;
    PolyUOp **linear = poly_linearize(ctx, sink, &count);
    matches &= linear && count == 3 && linear[0] == (enabled ? second : first) &&
               linear[1] == (enabled ? first : second) && linear[2] == sink;
    free(linear);
  }
  if (saved)
    setenv("TUPLE_ORDER", saved, 1);
  else
    unsetenv("TUPLE_ORDER");
  free(saved);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(matches);
  PASS();
}

TEST(codegen, linearize_orders_mixed_numeric_args_like_python_tuples) {
  /* Current tinygrad codegen/late/linearizer.py:36 orders UOp.tuplize keys
   * with Python numeric comparison, so weakfloat 0.0 sorts before weakint 4. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *i4 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, four, poly_arg_dtype(POLY_FLOAT32));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(0.0));
  PolyUOp *f0 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, zero, poly_arg_dtype(POLY_FLOAT32));
  PolyUOp *sink_src[2] = {i4, f0};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sink_src, 2, poly_arg_none());

  int n = 0;
  PolyUOp **linear = poly_linearize(ctx, sink, &n);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(n, 5);
  PolyOps expected[] = {
      POLY_OP_CONST, POLY_OP_CAST, POLY_OP_CONST, POLY_OP_CAST, POLY_OP_SINK,
  };
  for (int i = 0; i < n; i++)
    ASSERT_INT_EQ(linear[i]->op, expected[i]);
  ASSERT_PTR_EQ(linear[0], zero);
  ASSERT_PTR_EQ(linear[2], four);
  free(linear);

  PolyUOp *nan = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(NAN));
  PolyUOp *fnan = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, nan, poly_arg_dtype(POLY_FLOAT32));
  PolyUOp *twenty_three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(23));
  PolyUOp *i23 =
      poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, twenty_three, poly_arg_dtype(POLY_FLOAT32));
  PolyUOp *nan_src[2] = {fnan, i23};
  sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, nan_src, 2, poly_arg_none());
  linear = poly_linearize(ctx, sink, &n);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(n, 5);
  ASSERT_PTR_EQ(linear[0], nan);
  ASSERT_PTR_EQ(linear[2], twenty_three);
  free(linear);

  PolyInt huge_value = {0};
  ASSERT_TRUE(poly_int_from_decimal(&huge_value, "9007199254740993"));
  PolyUOp *huge = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_int_as_arg(&huge_value));
  poly_int_free(&huge_value);
  PolyUOp *fhuge = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, huge, poly_arg_dtype(POLY_FLOAT32));
  PolyUOp *rounded_huge =
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(9007199254740992.0));
  PolyUOp *frounded =
      poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, rounded_huge, poly_arg_dtype(POLY_FLOAT32));
  PolyUOp *huge_src[2] = {fhuge, frounded};
  sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, huge_src, 2, poly_arg_none());
  linear = poly_linearize(ctx, sink, &n);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(n, 5);
  ASSERT_PTR_EQ(linear[0], rounded_huge);
  ASSERT_PTR_EQ(linear[2], huge);

  free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, linearize_orders_paramarg_sources_by_current_dataclass_order) {
  /* Current tinygrad uop/ops.py:22-35 makes ParamArg an ordered dataclass;
   * nested PARAMs therefore order INDEX tuplize keys by slot. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *size = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, four, poly_arg_dtype(POLY_INT32));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *index = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, zero, poly_arg_dtype(POLY_INT32));
  PolyParamArg p2_arg = {
      .slot = 2,
      .dtype = POLY_FLOAT32,
      .addrspace = POLY_ADDR_GLOBAL,
      .device = "CPU",
  };
  PolyParamArg p1_arg = p2_arg;
  p1_arg.slot = 1;
  PolyUOp *p2 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, size, poly_arg_param(&p2_arg));
  PolyUOp *p1 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, size, poly_arg_param(&p1_arg));
  PolyUOp *x2 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p2, index, poly_arg_none());
  PolyUOp *x1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, index, poly_arg_none());
  PolyUOp *sink_src[2] = {x2, x1};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sink_src, 2, poly_arg_none());

  int n = 0;
  PolyUOp **linear = poly_linearize(ctx, sink, &n);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(n, 9);
  ASSERT_PTR_EQ(linear[2], p1);
  ASSERT_PTR_EQ(linear[3], p2);
  ASSERT_PTR_EQ(linear[6], x1);
  ASSERT_PTR_EQ(linear[7], x2);

  free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, placeholder_like_uses_current_shaped_param_topology) {
  /* Current tinygrad uop/ops.py:1138-1150: GLOBAL placeholders are scalar
   * value-typed PARAMs with a flattened shape source and ParamArg metadata;
   * rank greater than one is restored by an outer RESHAPE. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *like = poly_reshape(ctx, poly_buffer_f32(ctx, 6), (int64_t[]){2, 3}, 2);
  PolyUOp *out = poly_uop_placeholder_like(ctx, like, 7);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(out->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(out->n_src, 2);
  PolyUOp *param = out->src[0];
  ASSERT_INT_EQ(param->op, POLY_OP_PARAM);
  ASSERT_TRUE(poly_dtype_eq(param->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(param->n_src, 1);
  ASSERT_INT_EQ(param->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_is_index(param->src[0]->dtype));
  ASSERT_INT_EQ(param->src[0]->arg.i, 6);
  ASSERT_INT_EQ(param->arg.kind, POLY_ARG_PARAM);
  ASSERT_NOT_NULL(param->arg.param);
  ASSERT_INT_EQ(param->arg.param->slot, 7);
  ASSERT_INT_EQ(param->arg.param->addrspace, POLY_ADDR_GLOBAL);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, out), 2);
  const int64_t *dims = poly_uop_max_shape_dims(ctx, out);
  ASSERT_NOT_NULL(dims);
  ASSERT_INT_EQ(dims[0], 2);
  ASSERT_INT_EQ(dims[1], 3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_loaded_weak_index_uses_overflow_bounds) {
  /* Pinned tinygrad uop/ops.py:1655 lowers from u.overflows(int32), even when
   * the weak expression contains a LOAD. int32 bounds plus one require long;
   * forcing all loaded expressions to int32 silently wraps the address. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *buf = poly_test_uop_param(ctx, POLY_INT32, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, buf, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, index, poly_arg_none());
  PolyUOp *weak_load = poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, load, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *expr = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, weak_load, one, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, expr, poly_arg_none());

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_SINK);
  ASSERT_INT_EQ(lowered->n_src, 1);
  PolyUOp *root = lowered->src[0];
  ASSERT_INT_EQ(root->op, POLY_OP_ADD);
  ASSERT_TRUE(poly_dtype_eq(root->dtype, POLY_INT64));
  ASSERT_INT_EQ(root->n_src, 2);
  ASSERT_INT_EQ(root->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(root->src[0]->dtype, POLY_INT64));
  ASSERT_INT_EQ(root->src[0]->n_src, 1);
  ASSERT_INT_EQ(root->src[0]->src[0]->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(root->src[0]->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(root->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(root->src[1]->dtype, POLY_INT64));
  ASSERT_INT_EQ(root->src[1]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(root->src[1]->arg.i, 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, valid_gated_loaded_index_narrows_like_tinygrad) {
  /* Current uop/weak.py:74-81 narrows a gated long coordinate when the
   * indexed buffer's max_numel fits int32. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *indices = program_param(ctx, POLY_INT32, 1024, 0);
  PolyUOp *table = program_param(ctx, POLY_INT32, 1024, 1);
  PolyUOp *large_table = program_param(ctx, POLY_INT32, (int64_t)INT32_MAX + 64, 3);
  PolyUOp *out = program_param(ctx, POLY_INT32, 1024, 2);
  PolyUOp *lane = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *indices_at_lane =
      poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, indices, lane, poly_arg_none());
  PolyUOp *loaded = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, indices_at_lane, poly_arg_none());
  PolyUOp *weak_loaded = poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, loaded, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *true_uop = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *below_zero =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak_loaded, zero, poly_arg_none());
  PolyUOp *at_least_zero =
      poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, below_zero, true_uop, poly_arg_none());
  PolyUOp *below_two = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak_loaded, two, poly_arg_none());
  PolyUOp *gate = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, at_least_zero, below_two, poly_arg_none());
  PolyUOp *shifted = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, weak_loaded, two, poly_arg_none());
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *gated =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, shifted, invalid, poly_arg_none());
  PolyUOp *table_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, table, gated, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, table_index, poly_arg_none());
  PolyUOp *out_zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *out_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, out_zero, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_index, value, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink);
  ASSERT_NOT_NULL(lowered);
  PolyUOp *lowered_coord = lowered->src[0]->src[1]->src[0]->src[1];
  ASSERT_INT_EQ(lowered_coord->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->n_src, 3);
  ASSERT_INT_EQ(lowered_coord->src[1]->op, POLY_OP_ADD);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->src[1]->src[0]->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->src[1]->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->src[1]->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->src[1]->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->src[1]->src[1]->arg.i, 2);
  int n_lowered = 0;
  PolyUOp **lowered_topo = poly_toposort(ctx, lowered, &n_lowered);
  ASSERT_NOT_NULL(lowered_topo);
  for (int i = 0; i < n_lowered; i++) {
    PolyDType scalar = lowered_topo[i]->dtype;
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
  }

  PolyUOp *webgpu = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(webgpu);
  int n_webgpu = 0;
  PolyUOp **webgpu_topo = poly_toposort(ctx, webgpu, &n_webgpu);
  ASSERT_NOT_NULL(webgpu_topo);
  for (int i = 0; i < n_webgpu; i++) {
    PolyDType scalar = webgpu_topo[i]->dtype;
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
  }
  ASSERT_TRUE(poly_type_verify_program(ctx, webgpu));

  /* The same coordinate remains long for a buffer larger than int32. */
  PolyUOp *i32max = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(INT32_MAX));
  PolyUOp *wide_upper =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak_loaded, i32max, poly_arg_none());
  PolyUOp *wide_gate =
      poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, at_least_zero, wide_upper, poly_arg_none());
  PolyUOp *wide_gated =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, wide_gate, shifted, invalid, poly_arg_none());
  PolyUOp *wide_index =
      poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, large_table, wide_gated, poly_arg_none());
  PolyUOp *wide_load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, wide_index, poly_arg_none());
  PolyUOp *wide_store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_index, wide_load, poly_arg_none());
  PolyUOp *wide_sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, wide_store, poly_arg_none());
  PolyUOp *wide_lowered = poly_apply_post_index_symbolic_stage(ctx, wide_sink);
  ASSERT_NOT_NULL(wide_lowered);
  PolyUOp *wide_coord = wide_lowered->src[0]->src[1]->src[0]->src[1];
  ASSERT_INT_EQ(wide_coord->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(wide_coord->dtype, POLY_INT64));
  ASSERT_INT_EQ(wide_coord->src[1]->op, POLY_OP_ADD);
  ASSERT_TRUE(poly_dtype_eq(wide_coord->src[1]->dtype, POLY_INT64));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, valid_index_simplifies_inside_devectorizer_like_tinygrad) {
  /* Pinned tinygrad codegen/__init__.py:105-110 includes
   * load_store_indexing in both the devectorizer and lower-index rewrites.
   * devectorizer.py:39-42 therefore reduces x%8 to x under 0<=x<8 before
   * weak-index dtype lowering; checking only the final graph misses that
   * stage-order contract and can produce materially worse kernels. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *buf = poly_test_uop_param(ctx, POLY_INT32, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *input_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, buf, zero, poly_arg_none());
  PolyUOp *loaded = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, input_index, poly_arg_none());
  PolyUOp *weak_loaded = poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, loaded, poly_arg_none());
  PolyUOp *width = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *truth = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *negative = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak_loaded, zero, poly_arg_none());
  PolyUOp *nonnegative = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, negative, truth, poly_arg_none());
  PolyUOp *below_width =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak_loaded, width, poly_arg_none());
  PolyUOp *valid =
      poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, nonnegative, below_width, poly_arg_none());
  PolyUOp *bounded_mod =
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, weak_loaded, width, poly_arg_none());
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *coord =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, valid, bounded_mod, invalid, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, buf, coord, poly_arg_none());

  PolyRendererCaps caps = {.max_vec_width = 1};
  PolyUOp *devec = poly_apply_devectorizer2_stage(ctx, root, caps);
  ASSERT_NOT_NULL(devec);
  ASSERT_INT_EQ(devec->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(devec->n_src, 2);
  PolyUOp *devec_coord = devec->src[1];
  ASSERT_INT_EQ(devec_coord->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(devec_coord->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(devec_coord->n_src, 3);
  ASSERT_INT_EQ(devec_coord->src[1]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(devec_coord->src[1]->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(devec_coord->src[1]->n_src, 1);
  ASSERT_PTR_EQ(devec_coord->src[1]->src[0], loaded);

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, devec);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_INDEX);
  PolyUOp *lowered_coord = lowered->src[1];
  ASSERT_INT_EQ(lowered_coord->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->src[1]->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->src[1]->n_src, 1);
  PolyUOp *lowered_input_index = lowered_coord->src[1]->src[0];
  ASSERT_INT_EQ(lowered_input_index->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(lowered_input_index->n_src, 2);
  ASSERT_PTR_EQ(lowered_input_index->src[0], buf);
  ASSERT_INT_EQ(lowered_input_index->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered_input_index->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_input_index->src[1]->arg.i, 0);

  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, lowered, &n);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n; i++) {
    ASSERT_TRUE(topo[i]->op != POLY_OP_FLOORMOD);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, image_valid_index_simplifies_two_coordinates_like_tinygrad) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/late/coalesce.py:43-61
   * removes a shared image bounds gate from both coordinates when image
   * out-of-range behavior already supplies the invalid value. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape_src[3] = {poly_const_int(ctx, 2), poly_const_int(ctx, 3), poly_const_int(ctx, 4)};
  PolyUOp *shape = poly_shape_to_shape_arg(ctx, shape_src, 3);
  PolyParamArg image_arg = {.slot = 0, .dtype = POLY_FLOAT16, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *image = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT16, shape, poly_arg_param(&image_arg));
  PolyUOp *y = poly_uop_variable(ctx, "y", 0, 3, POLY_WEAKINT, 1, true);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *x = poly_const_int(ctx, 0);
  PolyUOp *valid = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, y, two, poly_arg_none());
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_BOOL);
  PolyUOp *gated_y =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, valid, y, invalid, poly_arg_none());
  PolyUOp *gated_x =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, valid, x, invalid, poly_arg_none());
  PolyUOp *coords[2] = {gated_y, gated_x};
  PolyUOp *index = poly_uop_index(ctx, image, coords, 2);

  PolyRendererCaps caps = {.device = "PYTHON", .max_vec_width = 4};
  PolyUOp *rewritten = poly_apply_devectorizer2_stage(ctx, index, caps);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(rewritten->n_src, 3);
  ASSERT_PTR_EQ(rewritten->src[1], y);
  ASSERT_PTR_EQ(rewritten->src[2], x);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_FLOAT32));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, simplify_add_image_matches_current_four_rule_topology) {
  /* Tinygrad 2026-08-22/a9069c177a9d
   * codegen/late/coalesce.py:95-101 has exactly four ordered rows. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_pm_rule_count(poly_pm_simplify_add_image()), 4);

  PolyUOp *shape_src[3] = {poly_const_int(ctx, 2), poly_const_int(ctx, 3), poly_const_int(ctx, 4)};
  PolyUOp *shape = poly_shape_to_shape_arg(ctx, shape_src, 3);
  PolyParamArg image_arg = {.slot = 0, .dtype = POLY_FLOAT16, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *image = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT16, shape, poly_arg_param(&image_arg));
  PolyUOp *coords[2] = {poly_const_int(ctx, 0), poly_const_int(ctx, 1)};
  PolyUOp *index = poly_uop_index(ctx, image, coords, 2);

  PolyUOp *half_load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, index, poly_arg_none());
  PolyUOp *load_out = poly_graph_rewrite(ctx, half_load, poly_pm_simplify_add_image());
  ASSERT_INT_EQ(load_out->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(load_out->dtype, POLY_FLOAT16));
  ASSERT_INT_EQ(load_out->src[0]->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(load_out->src[0]->dtype, POLY_FLOAT32));
  ASSERT_PTR_EQ(load_out->src[0]->src[0], index);

  PolyUOp *half_value = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(1.0));
  PolyUOp *half_store = poly_store_val(ctx, index, half_value);
  PolyUOp *store_out = poly_graph_rewrite(ctx, half_store, poly_pm_simplify_add_image());
  ASSERT_INT_EQ(store_out->op, POLY_OP_STORE);
  ASSERT_INT_EQ(store_out->src[1]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(store_out->src[1]->dtype, POLY_FLOAT32));
  ASSERT_PTR_EQ(store_out->src[1]->src[0], half_value);

  PolyUOp *value = program_param(ctx, POLY_FLOAT32, 1, 1);
  PolyUOp *half = poly_cast(ctx, value, POLY_FLOAT16);
  PolyUOp *roundtrip = poly_cast(ctx, half, POLY_FLOAT32);
  PolyUOp *roundtrip_out = poly_graph_rewrite(ctx, roundtrip, poly_pm_simplify_add_image());
  ASSERT_PTR_EQ(roundtrip_out, value);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, python_image_transform_executes_float_vector_like_tinygrad) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/late/coalesce.py:73-101
   * maps a coalesced four-lane SHRINK to one image pixel.  A scalar RANGE
   * alone does not create this SHRINK; memory coalescing must expose it. */
  const char *saved_image = getenv("IMAGE");
  char *saved_image_copy = saved_image ? strdup(saved_image) : NULL;
  setenv("IMAGE", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape = poly_const_int(ctx, 8);
  PolyParamArg out_arg = {.slot = 0, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL};
  PolyParamArg in_arg = {.slot = 1, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&out_arg));
  PolyUOp *in = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&in_arg));
  PolyUOp *offset = poly_const_int(ctx, 0), *lanes = poly_const_int(ctx, 4);
  PolyUOp *out_shrink_src[3] = {out, offset, lanes};
  PolyUOp *in_shrink_src[3] = {in, offset, lanes};
  PolyUOp *out_shrink =
      poly_uop(ctx, POLY_OP_SHRINK, POLY_FLOAT32, out_shrink_src, 3, poly_arg_none());
  PolyUOp *in_shrink =
      poly_uop(ctx, POLY_OP_SHRINK, POLY_FLOAT32, in_shrink_src, 3, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_shrink, poly_arg_none());
  PolyUOp *store = poly_store_val(ctx, out_shrink, load);
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "image_copy4");
  PolyRewriteOpts opts = {
      .optimize = false,
      .caps =
          {
              .device = "PYTHON",
              .arch = "IMAGE_PITCH_ALIGNMENT=1",
              .has_mulacc = true,
              .has_max = true,
              .has_exp2 = true,
              .has_log2 = true,
              .has_sin = true,
              .has_int64 = true,
              .max_vec_width = 4,
          },
      .device = POLY_DEVICE_INTERP,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_TRUE(poly_type_verify_program(ctx, rewritten));

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  int image_params = 0, image_indexes = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_PARAM && poly_uop_is_image_shape(ctx, topo[i])) image_params++;
    if (topo[i]->op == POLY_OP_INDEX && topo[i]->n_src == 3 &&
        poly_uop_is_image_shape(ctx, topo[i]->src[0]))
      image_indexes++;
  }
  ASSERT_INT_EQ(image_params, 2);
  ASSERT_INT_EQ(image_indexes, 2);

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  float input[8] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};
  float output[8] = {0};
  void *args[2] = {output, input};
  ASSERT_INT_EQ(poly_interp_eval(ctx, lin, n_lin, args, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(output[i], input[i], 1e-6);
  for (int i = 4; i < 8; i++)
    ASSERT_FLOAT_EQ(output[i], 0.0f, 1e-6);

  free(lin);
  poly_ctx_destroy(ctx);
  if (saved_image_copy) {
    setenv("IMAGE", saved_image_copy, 1);
    free(saved_image_copy);
  } else {
    unsetenv("IMAGE");
  }
  PASS();
}

TEST(codegen, post_index_strong_point_after_matches_current_symbolic) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:256-258 folds a
   * strong-float point-valued AFTER to its value before AFTER canonicalization. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, range, one, poly_arg_none());
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *coord =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, cond, range, invalid, poly_arg_none());
  PolyUOp *buf = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, buf, coord, poly_arg_none());
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, index,
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0)), poly_arg_none()
  );
  PolyUOp *after = poly_uop2(
      ctx, POLY_OP_AFTER, POLY_FLOAT32,
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0)), store, poly_arg_none()
  );
  PolyUOp *outer = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_FLOAT32, cond, after,
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0)), poly_arg_none()
  );
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, outer, poly_arg_none());

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink);
  ASSERT_NOT_NULL(lowered);
  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, lowered, &n);
  ASSERT_NOT_NULL(topo);
  ASSERT_INT_EQ(n, 8);
  ASSERT_INT_EQ(count_lin_ops(topo, n, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_lin_ops(topo, n, POLY_OP_STORE), 0);
  ASSERT_INT_EQ(count_lin_ops(topo, n, POLY_OP_AFTER), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_weak_preserves_invalid_base) {
  /* uop/weak.py:lower_weak_node commits concrete operands, but leaves
   * invalid bases untouched even when hidden beneath movement or DETACH. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_BOOL);
  int64_t shape[] = {2};
  PolyUOp *expanded = poly_expand(ctx, invalid, shape, 1);
  PolyUOp *variants[] = {invalid, expanded, poly_alu1(ctx, POLY_OP_DETACH, expanded)};
  PolyDType strong_types[] = {POLY_FLOAT32, POLY_INT32};
  PolyDType weak_types[] = {POLY_WEAKFLOAT, POLY_WEAKINT};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      PolyUOp *strong = poly_uop_const(ctx, poly_arg_int(3), strong_types[j]);
      if (i) strong = poly_expand(ctx, strong, shape, 1);
      PolyUOp *weak = poly_cast(ctx, strong, weak_types[j]);
      PolyUOp *root = poly_alu2(ctx, POLY_OP_ADD, weak, variants[i]);
      PolyUOp *lowered = poly_pm_rewrite(poly_pm_lower_weak(), ctx, root);
      ASSERT_NOT_NULL(lowered);
      ASSERT_INT_EQ(lowered->op, POLY_OP_CAST);
      ASSERT_INT_EQ(lowered->n_src, 1);
      ASSERT_TRUE(poly_dtype_eq(lowered->dtype, weak_types[j]));
      PolyUOp *add = lowered->src[0];
      ASSERT_INT_EQ(add->op, POLY_OP_ADD);
      ASSERT_INT_EQ(add->n_src, 2);
      ASSERT_PTR_EQ(add->src[1], variants[i]);
      ASSERT_TRUE(poly_dtype_eq(add->src[1]->dtype, POLY_BOOL));
      /* Expanded weakint bounds are unknown, so the pinned rule widens. */
      PolyDType concrete = i && j ? POLY_INT64 : strong_types[j];
      ASSERT_TRUE(poly_dtype_eq(add->dtype, concrete));
      ASSERT_PTR_EQ(add->src[0], poly_cast(ctx, strong, concrete));
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_weak_comparison_legalizes_mixed_integer_operands) {
  /* Pinned tinygrad's GroupOp.Binary rule includes comparisons
   * (uop/ops.py:1657-1659): the result remains bool, but both weak-index
   * operands are cast to their least-upper concrete integer dtype. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *buf32 = program_param(ctx, POLY_INT32, 1, 0);
  PolyUOp *buf64 = program_param(ctx, POLY_INT64, 1, 1);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx32 = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, buf32, zero, poly_arg_none());
  PolyUOp *idx64 = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT64, buf64, zero, poly_arg_none());
  PolyUOp *load32 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, idx32, poly_arg_none());
  PolyUOp *load64 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT64, idx64, poly_arg_none());
  PolyUOp *weak32 = poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, load32, poly_arg_none());
  PolyUOp *weak64 = poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, load64, poly_arg_none());
  PolyUOp *compare = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak32, weak64, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, compare, poly_arg_none());

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_SINK);
  ASSERT_INT_EQ(lowered->n_src, 1);
  PolyUOp *root = lowered->src[0];
  ASSERT_INT_EQ(root->op, POLY_OP_CMPLT);
  ASSERT_TRUE(poly_dtype_eq(root->dtype, POLY_BOOL));
  ASSERT_INT_EQ(root->n_src, 2);
  ASSERT_INT_EQ(root->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(root->src[0]->dtype, POLY_INT64));
  PolyUOp *lowered_load32 = root->src[0]->src[0];
  ASSERT_INT_EQ(lowered_load32->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(lowered_load32->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_load32->src[0]->op, POLY_OP_INDEX);
  PolyUOp *lowered_buf32 = lowered_load32->src[0]->src[0];
  ASSERT_INT_EQ(lowered_buf32->op, POLY_OP_PARAM);
  ASSERT_TRUE(poly_dtype_eq(lowered_buf32->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_buf32->arg.kind, POLY_ARG_PARAM);
  ASSERT_INT_EQ(lowered_buf32->arg.param->slot, 0);
  ASSERT_INT_EQ(lowered_buf32->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered_buf32->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(root->src[1]->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(root->src[1]->dtype, POLY_INT64));
  ASSERT_INT_EQ(root->src[1]->src[0]->src[0]->arg.param->slot, 1);

  /* The weak wrappers are the provenance for this legalization. Pinned
   * pm_lower_index_dtype leaves a raw mixed concrete comparison untouched so
   * validation can reject it; do not normalize arbitrary malformed IR. */
  PolyUOp *raw_compare = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, load32, load64, poly_arg_none());
  PolyUOp *raw_sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, raw_compare, poly_arg_none());
  PolyUOp *raw_lowered = poly_apply_post_index_symbolic_stage(ctx, raw_sink);
  ASSERT_NOT_NULL(raw_lowered);
  ASSERT_INT_EQ(raw_lowered->src[0]->op, POLY_OP_CMPLT);
  ASSERT_TRUE(poly_dtype_eq(raw_lowered->src[0]->dtype, POLY_BOOL));
  ASSERT_INT_EQ(raw_lowered->src[0]->src[0]->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(raw_lowered->src[0]->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(raw_lowered->src[0]->src[1]->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(raw_lowered->src[0]->src[1]->dtype, POLY_INT64));

  PolyUOp *out = program_param(ctx, POLY_INT32, 1, 2);
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, zero, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, compare, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, value, poly_arg_none());
  PolyUOp *webgpu =
      poly_rewrite_webgpu(ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none()));
  ASSERT_NOT_NULL(webgpu);
  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, webgpu, &n);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n; i++) {
    PolyDType scalar = topo[i]->dtype;
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
  }
  ASSERT_TRUE(poly_type_verify_program(ctx, webgpu));

  poly_ctx_destroy(ctx);
  PASS();
}

/* Current tinygrad codegen/__init__.py:378-379 composes pm_commit_weak and
 * pm_cast_weak before renderer.extra_matcher in one fixed-point matcher.  A
 * renderer rule may therefore mint a weak literal and rely on the matcher
 * restarting at pm_commit_weak before the graph reaches the renderer. */
static PolyUOp *test_renderer_mint_weak_from_neg(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!root || root->op != POLY_OP_NEG || root->n_src != 1 ||
      !poly_dtype_eq(root->dtype, POLY_FLOAT32))
    return NULL;
  PolyUOp *half = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(0.5));
  return poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, root->src[0], half, poly_arg_none());
}

TEST(codegen, final_renderer_matcher_recommits_weak_values) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *in = program_param(ctx, POLY_FLOAT32, 1, 1);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, range, poly_arg_none());
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, in, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, neg, poly_arg_none());
  PolyUOp *end_src[] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  PolyUPat *neg_pat = poly_upat_op(POLY_OP_NEG, NULL, 0, "u");
  PolyNamedRule rules[] = {POLY_RULE(neg_pat, test_renderer_mint_weak_from_neg)};
  PolyPatternMatcher *extra = poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  PolyRewriteOpts opts = {
      .optimize = false,

      .caps = poly_c_renderer_caps(),
      .extra_matcher = extra,
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  int weak_literals = 0;
  PolyUOp *add = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (poly_dtype_is_weak(topo[i]->dtype)) {
      weak_literals++;
      ASSERT_INT_EQ(topo[i]->op, POLY_OP_CONST);
      bool casted = false;
      for (int j = 0; j < n_topo; j++)
        if (topo[j]->op == POLY_OP_CAST && topo[j]->n_src == 1 && topo[j]->src[0] == topo[i] &&
            !poly_dtype_is_weak(topo[j]->dtype))
          casted = true;
      ASSERT_TRUE(casted);
    }
    if (topo[i]->op == POLY_OP_ADD && poly_dtype_eq(topo[i]->dtype, POLY_FLOAT32)) add = topo[i];
  }
  ASSERT_NOT_NULL(add);
  ASSERT_TRUE(weak_literals >= 1);
  ASSERT_INT_EQ(add->n_src, 2);
  ASSERT_INT_EQ(add->src[1]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(add->src[1]->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(add->src[1]->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(add->src[1]->src[0]->dtype, POLY_WEAKFLOAT));

  poly_pm_destroy(extra);
  poly_upat_free(neg_pat);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, weak_stack_rederives_dtype_from_committed_lanes) {
  /* Current STACK keeps its lane count in shape and re-derives only the
   * promoted scalar dtype (tinygrad/uop/ops.py:149-152). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *lanes[4];
  for (int i = 0; i < 4; i++)
    lanes[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(0.0));
  PolyUOp *stack = poly_uop_stack(ctx, lanes, 4);
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, stack, poly_arg_none());
  ASSERT_NOT_NULL(cast);

  PolyUOp *rewritten = poly_apply_post_index_symbolic_stage(ctx, cast);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_STACK);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(poly_uop_max_numel(ctx, rewritten), 4);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++)
    ASSERT_TRUE(!poly_dtype_is_weak(topo[i]->dtype));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_index_dtype_matches_pinned_range_bound_patterns) {
  /* Pinned pm_lower_index_dtype matches a weak-wrapped RANGE bound regardless
   * of the RANGE's current dtype. Weak bounds choose int/long by overflow; an
   * explicitly concrete RANGE<long> with a concrete bound is unchanged. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *narrow_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(16));
  PolyUOp *boundary_bound =
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(INT64_C(1) << 31));
  PolyUOp *wide_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(INT64_C(1) << 40));
  PolyUOp *long_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(16));
  PolyUOp *wrapped_int_bound = poly_uop1(
      ctx, POLY_OP_CAST, POLY_WEAKINT, poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(16)),
      poly_arg_none()
  );
  PolyUOp *ranges[5] = {
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, narrow_bound, poly_arg_range(0, POLY_AXIS_LOOP)),
      poly_uop1(
          ctx, POLY_OP_RANGE, POLY_WEAKINT, boundary_bound, poly_arg_range(1, POLY_AXIS_LOOP)
      ),
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, wide_bound, poly_arg_range(2, POLY_AXIS_LOOP)),
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT64, long_bound, poly_arg_range(3, POLY_AXIS_LOOP)),
      poly_uop1(
          ctx, POLY_OP_RANGE, POLY_INT32, wrapped_int_bound, poly_arg_range(4, POLY_AXIS_LOOP)
      ),
  };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, ranges, 5, poly_arg_none());

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_SINK);
  ASSERT_INT_EQ(lowered->n_src, 5);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->dtype, POLY_INT32));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->src[0]->dtype, POLY_INT32));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[1]->dtype, POLY_INT64));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[1]->src[0]->dtype, POLY_INT64));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[2]->dtype, POLY_INT64));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[2]->src[0]->dtype, POLY_INT64));
  ASSERT_PTR_EQ(lowered->src[3], ranges[3]);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[3]->dtype, POLY_INT64));
  ASSERT_PTR_EQ(lowered->src[3]->src[0], long_bound);
  ASSERT_INT_EQ(lowered->src[4]->op, POLY_OP_RANGE);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[4]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[4]->n_src, 1);
  ASSERT_INT_EQ(lowered->src[4]->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[4]->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[4]->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(lowered->src[4]->src[0]->arg.i, 16);

  poly_ctx_destroy(ctx);
  PASS();
}

/* August pm_lower_index_dtype coverage lives in the weak-lowering and gated-index tests above. */

TEST(codegen, linearize_webgpu_reduce_emits_tinygrad_sized_shared_barrier) {
  /* tinygrad emits shared memory for this reduce, but with one local axis:
   * @workgroup_size(16). The regression is growing an extra local reduce axis
   * and producing @workgroup_size(16,256,1). */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4096);
  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *sum = poly_sum_reduce(ctx, a, 0, 0);
  PolyUOp *store = poly_store_val(ctx, out, sum);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);

  bool found = false;
  bool cuda_found = false;
  for (int i = 0; i < linear_schedule->n_src; i++) {
    int n_lin = 0;
    PolyUOp **lin =
        poly_linearize_webgpu(ctx, poly_test_linear_call_body(linear_schedule, i), &n_lin);
    ASSERT_NOT_NULL(lin);
    char *wgsl = poly_render_wgsl(ctx, lin, n_lin, "reduce_webgpu");
    ASSERT_NOT_NULL(wgsl);
    int dims[3];
    int prod = wgsl_workgroup_product(wgsl, dims);
    ASSERT_TRUE(prod <= 256);
    ASSERT_INT_EQ(dims[0], 16);
    ASSERT_INT_EQ(dims[1], 1);
    ASSERT_INT_EQ(dims[2], 1);
    if (strstr(wgsl, "var<workgroup>") && strstr(wgsl, "workgroupBarrier();")) found = true;
    free(wgsl);
    /* Renderer-only parity: the same tinygrad-shaped LOCAL BUFFER must be
     * accepted by CUDA without probing or opening a CUDA runtime. */
    char *cuda = poly_render_cuda(ctx, lin, n_lin, "reduce_cuda", 256);
    ASSERT_NOT_NULL(cuda);
    if (strstr(cuda, "__shared__") && !strstr(cuda, "(null)")) cuda_found = true;
    free(cuda);
    free(lin);
  }

  poly_ctx_destroy(ctx);
  ASSERT_TRUE(found);
  ASSERT_TRUE(cuda_found);
  PASS();
}

TEST(codegen, linearize_webgpu_qwen_downproj_workgroup_matches_tinygrad) {
  /* Probe parity with tinygrad_latest:
   *   Tensor.empty((25,3072)).matmul(Tensor.empty((1024,3072)).T)
   * renders @workgroup_size(16), not @workgroup_size(16,256). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_shaped_f32_buf(ctx, (int64_t[]){25, 3072}, 2);
  PolyUOp *w = make_shaped_f32_buf(ctx, (int64_t[]){1024, 3072}, 2);
  PolyUOp *wt = poly_permute(ctx, w, (int64_t[]){1, 0}, 2);
  PolyUOp *y = poly_dot(ctx, x, wt);
  PolyUOp *out = make_shaped_f32_buf(ctx, (int64_t[]){25, 1024}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, y));

  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_INT_EQ(linear_schedule->n_src, 1);

  int n_lin = 0;
  PolyUOp **lin =
      poly_linearize_webgpu(ctx, poly_test_linear_call_body(linear_schedule, 0), &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_INT_EQ(count_special_named(lin, n_lin, "gidx0"), 1);
  ASSERT_INT_EQ(count_special_named(lin, n_lin, "gidx1"), 1);
  ASSERT_INT_EQ(count_special_named(lin, n_lin, "lidx0"), 1);
  ASSERT_INT_EQ((int)special_bound_hi_named(lin, n_lin, "gidx0"), 16);
  ASSERT_INT_EQ((int)special_bound_hi_named(lin, n_lin, "gidx1"), 25);
  ASSERT_INT_EQ((int)special_bound_hi_named(lin, n_lin, "lidx0"), 16);
  char *wgsl = poly_render_wgsl(ctx, lin, n_lin, "qwen_downproj");
  ASSERT_NOT_NULL(wgsl);

  int dims[3];
  int prod = wgsl_workgroup_product(wgsl, dims);
  ASSERT_TRUE(prod <= 256);
  ASSERT_INT_EQ(dims[0], 16);
  ASSERT_INT_EQ(dims[1], 1);
  ASSERT_INT_EQ(dims[2], 1);
  ASSERT_TRUE(strstr(wgsl, "@workgroup_size(16,256") == NULL);

  free(wgsl);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, full_rewrite_post_index_lowering_narrows_i64_addressing) {
  VecKernel k = make_vec_copy_with_weak_index_expr(32);
  PolyRewriteOpts opts = {
      .optimize = false,

      .caps = {.max_vec_width = 1},
      .device = POLY_DEVICE_WEBGPU,
  };

  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(k.ctx, k.sink, opts);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(k.ctx, rewritten, &n_topo);
  ASSERT_TRUE(n_topo > 0);
  ASSERT_INT_EQ(count_indexes_with_i64_addr(topo, n_topo), 0);

  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, full_rewrite_lowers_weakint_like_address_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType weak_like = POLY_WEAKINT;
  weak_like.bitsize = 144;

  PolyUOp *p0 = program_param(ctx, POLY_FLOAT32, 32, 0);
  PolyUOp *p1 = program_param(ctx, POLY_FLOAT32, 32, 1);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(32));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, weak_like, poly_arg_int(1));
  PolyUOp *addr = poly_uop2(ctx, POLY_OP_ADD, weak_like, range, one, poly_arg_none());
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, addr, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, range, poly_arg_none());
  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, load0, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin = 0;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  for (int i = 0; i < n_lin; i++) {
    if (!poly_dtype_is_weak(lin[i]->dtype)) continue;
    ASSERT_INT_EQ(lin[i]->op, POLY_OP_CONST);
    bool casted = false;
    for (int j = 0; j < n_lin; j++)
      if (lin[j]->op == POLY_OP_CAST && lin[j]->n_src == 1 && lin[j]->src[0] == lin[i] &&
          !poly_dtype_is_weak(lin[j]->dtype))
        casted = true;
    ASSERT_TRUE(casted);
  }

  char *src = poly_render_c(ctx, lin, n_lin, "index_like_addr");
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, " weakint ") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, linearize_webgpu_narrows_i64_addressing) {
  VecKernel k = make_vec_copy_with_weak_index_expr(32);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(k.ctx, k.sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n_lin > 0);
  ASSERT_INT_EQ(count_indexes_with_i64_addr(lin, n_lin), 0);

  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

/* Unary op end-to-end */

TEST(codegen, e2e_neg) {
  /* b[i] = -a[i] for i in 0..5 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = program_param(ctx, POLY_FLOAT32, 6, 0);
  PolyUOp *p1 = program_param(ctx, POLY_FLOAT32, 6, 1);

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(6));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, range, poly_arg_none());

  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, neg, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, sink, &n);
  char *src = poly_render_c(ctx, lin, n, "vecneg");

  PolyProgram *prog = poly_compile_c(src, "vecneg");
  ASSERT_NOT_NULL(prog);

  float a[6] = {1.0f, -2.5f, 3.14f, 0.0f, -100.0f, 42.0f};
  float b[6] = {0};

  void *args[2] = {a, b};
  poly_program_call(prog, args, 2);

  for (int i = 0; i < 6; i++) {
    ASSERT_FLOAT_EQ(b[i], -a[i], 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}
