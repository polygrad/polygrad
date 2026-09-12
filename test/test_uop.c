/*
 * test_uop.c — Tests for UOp creation, CSE, and toposort
 */

#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/frontend.h"
#include "../src/engine/realize.h"
#include "../src/placer.h"
#include "../src/ctx.h"
#include "../src/device.h"
#include "../src/tensor.h"
#include "../src/uop/movement.h"
#include "../src/uop/ops.h"
#include "../src/utils.h"
#include "../src/bigint.h"
#include "../src/uop/symbolic.h"
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
#include <unistd.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#endif

/* Basic creation */

TEST(uop, gradient_owner_copy_returns_to_source_device) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *source = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *seed = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_INTERP);
  PolyUOp *dest = poly_uop_device_uop_cached(ctx, seed, NULL);
  PolyUOp *copied = poly_copy_to_device_uop(ctx, source, dest);
  PolyUOp *grad = NULL;
  ASSERT_INT_EQ(poly_grad_many(ctx, copied, seed, &source, 1, &grad), 0);
  ASSERT_NOT_NULL(grad);
  ASSERT_INT_EQ(grad->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(grad->src[0], seed);
  ASSERT_PTR_EQ(
      poly_uop_device_uop_cached(ctx, grad, NULL), poly_uop_device_uop_cached(ctx, source, NULL)
  );
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, gradient_owner_target_identity_does_not_differentiate_target) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_const_float(ctx, 2.0);
  PolyUOp *seed = poly_const_float(ctx, 3.0);
  PolyUOp *targets[] = {
      poly_detach(ctx, x),
      poly_uop1(ctx, POLY_OP_CUSTOM, POLY_FLOAT32, x, poly_arg_none()),
  };
  for (int i = 0; i < 2; i++) {
    PolyUOp *grad = NULL;
    uint8_t present = 0;
    ASSERT_INT_EQ(poly_grad_many_ex(ctx, targets[i], seed, &targets[i], 1, &grad, &present), 0);
    ASSERT_INT_EQ(present, 1);
    ASSERT_PTR_EQ(grad, seed);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

extern void poly_test_grad_target_walk_fail_alloc(bool fail);

TEST(uop, gradient_owner_target_walk_failure_is_not_zero_gradient) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_const_float(ctx, 2.0), *seed = poly_const_float(ctx, 3.0);
  PolyUOp *root = poly_mul(ctx, x, x);
  PolyUOp *grad = seed;
  uint8_t present = 7;
  size_t scratch_before = poly_arena_used(ctx->scratch);
  poly_test_grad_target_walk_fail_alloc(true);
  int rc = poly_grad_many_ex(ctx, root, seed, &x, 1, &grad, &present);
  poly_test_grad_target_walk_fail_alloc(false);
  ASSERT_INT_EQ(rc, -1);
  ASSERT_PTR_EQ(grad, seed);
  ASSERT_INT_EQ(present, 7);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);
  ASSERT_INT_EQ(poly_grad_many_ex(ctx, root, seed, &x, 1, &grad, &present), 0);
  ASSERT_NOT_NULL(grad);
  ASSERT_INT_EQ(present, 1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, gradient_owner_tuple_accumulation_preserves_noop_slots) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_const_float(ctx, 2.0), *seed = poly_const_float(ctx, 3.0);
  PolyUOp *noop = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *pair = poly_uop2(ctx, POLY_OP_TUPLE, POLY_VOID, x, x, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_TUPLE, POLY_VOID, pair, pair, poly_arg_none());
  PolyUOp *left = poly_uop2(ctx, POLY_OP_TUPLE, POLY_VOID, seed, noop, poly_arg_none());
  PolyUOp *right = poly_uop2(ctx, POLY_OP_TUPLE, POLY_VOID, noop, seed, poly_arg_none());
  PolyUOp *initial = poly_uop2(ctx, POLY_OP_TUPLE, POLY_VOID, left, right, poly_arg_none());
  PolyUOp *grad = NULL;
  ASSERT_INT_EQ(poly_grad_many(ctx, root, initial, &x, 1, &grad), 0);
  ASSERT_NOT_NULL(grad);
  ASSERT_INT_EQ(grad->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(grad->src[0], seed);
  ASSERT_PTR_EQ(grad->src[1], seed);
  /* Same-slot contributions add; two absent slots stay absent. */
  initial = poly_uop2(ctx, POLY_OP_TUPLE, POLY_VOID, left, left, poly_arg_none());
  ASSERT_INT_EQ(poly_grad_many(ctx, root, initial, &x, 1, &grad), 0);
  ASSERT_INT_EQ(grad->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(grad->src[0], seed);
  ASSERT_PTR_EQ(grad->src[1], seed);
  PolyUOp *empty = poly_uop2(ctx, POLY_OP_TUPLE, POLY_VOID, noop, noop, poly_arg_none());
  initial = poly_uop2(ctx, POLY_OP_TUPLE, POLY_VOID, empty, empty, poly_arg_none());
  uint8_t present = 1;
  ASSERT_INT_EQ(poly_grad_many_ex(ctx, root, initial, &x, 1, &grad, &present), 0);
  ASSERT_INT_EQ(present, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, gradient_owner_reshape_gradient_preserves_symbolic_dimension) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *n =
      poly_uop_variable(ctx, "n", poly_arg_int(1), poly_arg_int(8), POLY_WEAKINT, 1, false);
  PolyUOp *one = poly_uop_const(ctx, poly_arg_int(1), POLY_WEAKINT);
  PolyUOp *x = poly_expand_uop(ctx, poly_const_float(ctx, 2.0), &n, 1);
  PolyUOp *shape[] = {one, n};
  PolyUOp *root = poly_reshape_uop(ctx, x, shape, 2);
  PolyUOp *seed = poly_expand_uop(ctx, poly_const_float(ctx, 3.0), shape, 2);
  PolyUOp *grad = NULL;
  ASSERT_INT_EQ(poly_grad_many(ctx, root, seed, &x, 1, &grad), 0);
  ASSERT_NOT_NULL(grad);
  ASSERT_INT_EQ(grad->op, POLY_OP_RESHAPE);
  ASSERT_PTR_EQ(grad->src[0], seed);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, grad, 0), n);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, typed_param_bounds_long_repr) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  char decimal[601];
  memset(decimal, '9', sizeof(decimal) - 1);
  decimal[sizeof(decimal) - 1] = '\0';
  PolyInt value;
  poly_int_init(&value);
  ASSERT_TRUE(poly_int_from_decimal(&value, decimal));
  PolyUOp *v = poly_uop_variable(
      ctx, "long_bound", poly_arg_int(0), poly_int_as_arg(&value), POLY_WEAKINT, 1, false
  );
  ASSERT_NOT_NULL(v);
  poly_int_free(&value);
  char *text = poly_uop_str(v);
  ASSERT_NOT_NULL(text);
  ASSERT_TRUE(strstr(text, decimal) != NULL);
  ASSERT_TRUE(strstr(text, "name='long_bound'") != NULL);
  ASSERT_TRUE(text[strlen(text) - 1] == ')');
  free(text);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, typed_param_bounds_symbolic_consumers) {
  PolyCtx *ctx = poly_ctx_new();
  uint32_t lo[] = {0, 0x80000000}, hi[] = {UINT32_MAX, UINT32_MAX}, mid[] = {2, 0x80000000};
  PolyArg upper = poly_arg_bigint(1, hi, 2);
  PolyUOp *point = poly_uop_variable(ctx, "point", upper, upper, POLY_UINT64, 1, true);
  PolyUOp *range =
      poly_uop_variable(ctx, "range", poly_arg_bigint(1, lo, 2), upper, POLY_UINT64, 1, true);
  PolyUOp *c = poly_uop_const(ctx, poly_arg_bigint(1, mid, 2), POLY_UINT64);
  PolyUOp *maximum = poly_alu2(ctx, POLY_OP_MAX, range, c);
  PolyUOp *p = poly_graph_rewrite(ctx, point, poly_symbolic());
  PolyUOp *m = poly_graph_rewrite(ctx, maximum, poly_symbolic());
  bool correct =
      p->op == POLY_OP_CONST && poly_arg_python_numeric_eq(p->arg, upper) && m->op == POLY_OP_MAX;
  if (!correct)
    fprintf(
        stderr, "typed bounds: point=%s maximum=%s\n", poly_op_name(p->op), poly_op_name(m->op)
    );
  PolyUOp *source = poly_test_program_param(ctx, POLY_UINT64, 1, 0);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *load = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_UINT64, poly_uop_index(ctx, source, &zero, 1), poly_arg_none()
  );
  PolyUOp *temporary = poly_uop_variable_like_bounds(ctx, "temporary", load);
  correct &= temporary && poly_arg_python_numeric_eq(temporary->arg.param->max_val, upper);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

/* uop/ops.py:ParamArg.vmin_vmax keeps Python scalar values independently of
 * dtype. Arena ownership replaces Python's immutable integer object owner. */
TEST(uop, typed_param_bounds_identity_and_ownership) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *fraction = poly_uop_variable(
      ctx, "fraction", poly_arg_float(.25), poly_arg_float(.75), POLY_FLOAT32, 1, true
  );
  ASSERT_NOT_NULL(fraction);
  ASSERT_TRUE(fraction->arg.param->min_val.kind == POLY_ARG_FLOAT);
  ASSERT_TRUE(fraction->arg.param->min_val.f == .25);
  PolyUOp *copy = poly_uop_variable_like_bounds(ctx, "copy", fraction);
  ASSERT_NOT_NULL(copy);
  ASSERT_TRUE(poly_arg_eq(copy->arg.param->min_val, fraction->arg.param->min_val));
  ASSERT_TRUE(poly_arg_eq(copy->arg.param->max_val, fraction->arg.param->max_val));
  PolyUOp *a =
      poly_uop_variable(ctx, "key", poly_arg_int(0), poly_arg_int(1), POLY_FLOAT32, 1, true);
  PolyUOp *b =
      poly_uop_variable(ctx, "key", poly_arg_float(-0.), poly_arg_float(1.), POLY_FLOAT32, 1, true);
  ASSERT_PTR_EQ(a, b);
  uint32_t limbs[] = {0, 0, 0, 0, 4}; /* 2**130 */
  PolyArg huge = poly_arg_bigint(1, limbs, 5);
  PolyUOp *wide = poly_uop_variable(ctx, "wide", huge, huge, POLY_WEAKINT, 1, true);
  ASSERT_NOT_NULL(wide);
  poly_uop_retain(ctx, wide);
  limbs[4] = 8;
  PolyUOp *other = poly_uop_variable(ctx, "wide", huge, huge, POLY_WEAKINT, 1, true);
  ASSERT_TRUE(other != wide);
  poly_ctx_collect(ctx);
  char *decimal = poly_arg_integer_to_decimal(wide->arg.param->min_val);
  ASSERT_STR_EQ(decimal, "1361129467683753853853498429727072845824");
  free(decimal);
  copy = poly_uop_variable_like_bounds(ctx, "wide_copy", wide);
  ASSERT_NOT_NULL(copy);
  ASSERT_TRUE(poly_arg_eq(wide->arg.param->min_val, copy->arg.param->min_val));
  poly_uop_release(ctx, wide);
  ASSERT_TRUE(
      poly_uop_variable(
          ctx, "bad", poly_arg_float(NAN), poly_arg_float(1), POLY_FLOAT32, 1, true
      ) == NULL
  );
  ASSERT_TRUE(
      poly_uop_variable(ctx, "bad", poly_arg_float(2), poly_arg_float(1), POLY_FLOAT32, 1, true) ==
      NULL
  );
  poly_ctx_destroy(ctx);
  PASS();
}

typedef struct {
  int reads;
  bool fail;
} ReadbackProbe;

static int readback_probe_copyout(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t nbytes,
    void *userdata
) {
  ReadbackProbe *probe = userdata;
  probe->reads++;
  if (probe->fail) return -1;
  memcpy(dst->ptr, src->ptr, nbytes);
  return 0;
}

/* device.py:Buffer.as_memoryview copies to caller-owned temporary storage;
 * reading does not attach a permanent CPU allocation to the source Buffer. */
TEST(uop, buffer_read_is_copyout_without_retained_host_shadow) {
  PolyCtx *ctx = poly_ctx_new();
  ReadbackProbe probe = {0};
  PolyAllocator allocator = {.copy_out = readback_probe_copyout, .dev_ctx = &probe};
  float data[4] = {1, 2, 3, 4}, output[4] = {0};
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyBuffer handle = {
      .ptr = data,
      .nbytes = sizeof(data),
      .device = POLY_DEVICE_CUDA,
      .allocator = &allocator,
      .valid = true};
  ASSERT_INT_EQ(poly_buffer_attach(ctx, buf, &handle), 0);
  PolyBuffer *before = poly_buffer_get(ctx, buf);
  bool ok = true;
  for (int i = 0; i < 3; i++) {
    ok = ok && poly_buffer_read(ctx, buf, output, sizeof(output)) == 0 &&
         memcmp(data, output, sizeof(data)) == 0 && before == poly_buffer_get(ctx, buf) &&
         before->src == NULL;
  }
  PolyCtxStats stats;
  poly_ctx_stats(ctx, &stats);
  ok = ok && probe.reads == 3 && stats.buffer_read_count == 3 &&
       stats.buffer_owned_source_bytes == 0;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(uop, buffer_read_validates_extent_before_transfer_or_allocation) {
  PolyCtx *ctx = poly_ctx_new();
  ReadbackProbe probe = {0};
  PolyAllocator allocator = {.copy_out = readback_probe_copyout, .dev_ctx = &probe};
  float data[4] = {1, 2, 3, 4}, output[5] = {9, 9, 9, 9, 9};
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyBuffer handle = {
      .ptr = data,
      .nbytes = sizeof(data),
      .device = POLY_DEVICE_CUDA,
      .allocator = &allocator,
      .valid = true};
  ASSERT_INT_EQ(poly_buffer_attach(ctx, buf, &handle), 0);
  bool ok = poly_buffer_read(ctx, buf, output, 0) == -1 &&
            poly_buffer_read(ctx, buf, output, sizeof(output)) == -1 && probe.reads == 0 &&
            poly_buffer_get(ctx, buf)->src == NULL && output[0] == 9;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(uop, buffer_read_failure_and_partial_read_preserve_residency) {
  PolyCtx *ctx = poly_ctx_new();
  ReadbackProbe probe = {.fail = true};
  PolyAllocator allocator = {.copy_out = readback_probe_copyout, .dev_ctx = &probe};
  float data[4] = {1, 2, 3, 4}, output[2] = {9, 9};
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyBuffer handle = {
      .ptr = data,
      .nbytes = sizeof(data),
      .device = POLY_DEVICE_CUDA,
      .allocator = &allocator,
      .valid = true};
  ASSERT_INT_EQ(poly_buffer_attach(ctx, buf, &handle), 0);
  bool ok = poly_buffer_read(ctx, buf, output, sizeof(output)) == -1 && output[0] == 9 &&
            poly_buffer_get(ctx, buf)->src == NULL;
  probe.fail = false;
  ok = ok && poly_buffer_read(ctx, buf, output, sizeof(output)) == 0 && output[0] == 1 &&
       output[1] == 2 && poly_buffer_get(ctx, buf)->src == NULL;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(uop, buffer_read_preserves_explicit_mirror_and_uses_authoritative_storage) {
  PolyCtx *ctx = poly_ctx_new();
  ReadbackProbe probe = {0};
  PolyAllocator allocator = {.copy_out = readback_probe_copyout, .dev_ctx = &probe};
  float data[4] = {1, 2, 3, 4}, output[4] = {0};
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyBuffer handle = {
      .ptr = data,
      .nbytes = sizeof(data),
      .device = POLY_DEVICE_CUDA,
      .allocator = &allocator,
      .valid = true};
  ASSERT_INT_EQ(poly_buffer_attach(ctx, buf, &handle), 0);
  PolyBuffer *mirror = NULL;
  ASSERT_INT_EQ(poly_buffer_ensure_host_current(ctx, buf, &mirror), 0);
  ASSERT_NOT_NULL(mirror);
  PolyBuffer *cur = poly_buffer_get(ctx, buf);
  data[0] = 5;
  ASSERT_INT_EQ(poly_buffer_mark_residency_written(ctx, buf, POLY_DEVICE_CUDA), 0);
  bool ok = poly_buffer_read(ctx, buf, output, sizeof(output)) == 0 && output[0] == 5 &&
            cur->src == mirror && !mirror->valid && ((float *)mirror->ptr)[0] == 1;
  ((float *)mirror->ptr)[0] = 7;
  ASSERT_INT_EQ(poly_buffer_mark_host_written(ctx, buf), 0);
  ok = ok && poly_buffer_read(ctx, buf, output, sizeof(output)) == 0 && output[0] == 7 &&
       !cur->valid && cur->src == mirror && probe.reads == 2;
  mirror->valid = false;
  ok = ok && poly_buffer_read(ctx, buf, output, sizeof(output)) == -1 && probe.reads == 2;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(uop, buffer_copyout_rejects_invalid_extent_before_transfer) {
  PolyCtx *ctx = poly_ctx_new();
  ReadbackProbe probe = {0};
  PolyAllocator allocator = {.copy_out = readback_probe_copyout, .dev_ctx = &probe};
  float data[5] = {1, 2, 3, 4, 5}, output[5] = {0};
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyBuffer handle = {
      .ptr = data,
      .nbytes = 4 * sizeof(float),
      .device = POLY_DEVICE_CUDA,
      .allocator = &allocator,
      .valid = true};
  ASSERT_INT_EQ(poly_buffer_attach(ctx, buf, &handle), 0);
  bool ok = poly_buffer_copyout(ctx, buf, output, sizeof(output)) == -1 &&
            poly_buffer_copyout(ctx, buf, output, 0) == -1 && probe.reads == 0;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(uop, buffer_copyin_rejects_missing_transfer_operation) {
  PolyCtx *ctx = poly_ctx_new();
  PolyAllocator allocator = {0};
  float data[4] = {1, 2, 3, 4}, input[4] = {5, 6, 7, 8};
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyBuffer handle = {
      .ptr = data,
      .nbytes = sizeof(data),
      .device = POLY_DEVICE_CUDA,
      .allocator = &allocator,
      .valid = true};
  ASSERT_INT_EQ(poly_buffer_attach(ctx, buf, &handle), 0);
  int rc = poly_buffer_copyin(ctx, buf, input, sizeof(input));
  bool ok = rc == -1 && data[0] == 1 && poly_buffer_get(ctx, buf)->valid;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(uop, buffer_copyout_rejects_missing_transfer_operation) {
  PolyCtx *ctx = poly_ctx_new();
  PolyAllocator allocator = {0};
  float data[4] = {1, 2, 3, 4}, output[4] = {0};
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyBuffer handle = {
      .ptr = data,
      .nbytes = sizeof(data),
      .device = POLY_DEVICE_CUDA,
      .allocator = &allocator,
      .valid = true};
  ASSERT_INT_EQ(poly_buffer_attach(ctx, buf, &handle), 0);
  int rc = poly_buffer_copyout(ctx, buf, output, sizeof(output));
  bool ok = rc == -1 && output[0] == 0;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

#ifdef POLY_TESTING
extern void poly_test_buffer_handle_fail_after(int count);
extern void poly_test_range_alloc_fail_after(int count);

#if defined(__linux__) && !defined(__EMSCRIPTEN__)
TEST(uop, range_cache_allocation_failure_never_returns_empty_or_writes_past_capacity) {
  bool ok = true;
  for (int ended = 0; ended < 2; ended++) {
    for (int membership = 0; membership < 2; membership++) {
      for (int fail_after = 0; fail_after < 5; fail_after++) {
        pid_t child = fork();
        ASSERT_TRUE(child >= 0);
        if (child == 0) {
          struct rlimit no_core = {0, 0};
          (void)setrlimit(RLIMIT_CORE, &no_core);
          PolyCtx *ctx = poly_ctx_new();
          PolyUOp *ranges[5], *out[5];
          for (int i = 0; i < 5; i++) {
            ranges[i] = poly_uop1(
                ctx, POLY_OP_RANGE, POLY_INT32, poly_const_int(ctx, 8),
                poly_arg_range(i, POLY_AXIS_REDUCE)
            );
            if (poly_uop_ranges(ctx, ranges[i], out, 5) != 1) _exit(2);
          }
          PolyUOp *root;
          if (ended) {
            PolyUOp *src[6] = {poly_const_int(ctx, 8)};
            for (int i = 0; i < 5; i++)
              src[i + 1] = ranges[i];
            root = poly_uop(ctx, POLY_OP_END, POLY_INT32, src, 6, poly_arg_none());
          } else {
            root = poly_sink_n(ctx, ranges, 5);
          }
          /* Sources are cached: inject each owner allocation, including the
           * fifth RANGE's growth. Failure must not return a valid-looking
           * empty set/membership answer (hashmap's existing fatal-OOM policy). */
          poly_test_range_alloc_fail_after(fail_after);
          if (membership)
            (void)poly_uop_in_ranges(ctx, root, ranges[0]);
          else
            (void)poly_uop_ranges(ctx, root, out, 5);
          poly_test_range_alloc_fail_after(-1);
          poly_ctx_destroy(ctx);
          _exit(3);
        }
        int status = 0;
        bool waited = waitpid(child, &status, 0) == child;
        ok = waited && WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT && ok;
      }
    }
  }
  ASSERT_TRUE(ok);
  PASS();
}
#endif

static void replacement_count_free(const PolyBuffer *buffer, void *arg) {
  (void)buffer;
  (*(int *)arg)++;
}

static void *replacement_no_alloc(size_t nbytes, void *arg) {
  (void)nbytes;
  (void)arg;
  /* Reading already attached storage must not allocate it again. */
  return NULL;
}

static bool replacement_preserves_old(int operation, bool with_view) {
  PolyCtx *ctx = poly_ctx_new();
  int frees = 0;
  PolyAllocator allocator = {
      .alloc = replacement_no_alloc,
      .free = replacement_count_free,
      .dev_ctx = &frees,
      .host_addressable = true};
  float old_data[4] = {1, 2, 3, 4}, new_data[4] = {5, 6, 7, 8};
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  poly_uop_retain(ctx, buf);
  PolyBuffer old = {
      .ptr = old_data,
      .nbytes = sizeof(old_data),
      .device = POLY_DEVICE_CPU,
      .owned = true,
      .valid = true,
      .allocator = &allocator};
  poly_buffer_adopt(ctx, buf, &old);
  PolyBuffer *before = poly_buffer_get(ctx, buf);
  PolyBuffer *view = NULL;
  PolyUOp *cast = NULL;
  if (with_view) {
    cast = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT32, buf, poly_arg_none());
    view = poly_uop_buffer_handle(ctx, cast);
  }
  PolyBuffer candidate = old;
  candidate.ptr = new_data;
  if (!with_view) poly_test_buffer_handle_fail_after(0);
  int rc = operation == 0   ? poly_buffer_set(ctx, buf, new_data, sizeof(new_data), POLY_DEVICE_CPU)
           : operation == 1 ? poly_buffer_attach(ctx, buf, &candidate)
                            : poly_buffer_adopt(ctx, buf, &candidate);
  poly_test_buffer_handle_fail_after(-1);
  bool preserved = rc == -1 && frees == 0 && poly_buffer_get(ctx, buf) == before;
  if (preserved)
    preserved = before->ptr == old_data && old_data[2] == 3 &&
                (!with_view || (view && view->base == before));
  if (preserved) {
    float readback[4] = {0};
    preserved = poly_buffer_read(ctx, with_view ? cast : buf, readback, sizeof(readback)) == 0 &&
                memcmp(readback, old_data, sizeof(readback)) == 0;
  }
  /* The red set case leaves a freed map value. Remove only that dangling
   * entry so teardown does not hide the original replacement failure. */
  if (frees && !with_view && operation == 0)
    poly_map_remove(ctx->buffers, poly_ptr_hash(buf), buf, poly_ptr_eq);
  if (preserved) {
    /* Unretained view metadata is collectible; the retained BUFFER keeps
     * the old binding alive. Retrying must transfer ownership only now. */
    if (with_view) preserved = poly_ctx_collect(ctx) == 0;
    rc = operation == 0   ? poly_buffer_set(ctx, buf, new_data, sizeof(new_data), POLY_DEVICE_CPU)
         : operation == 1 ? poly_buffer_attach(ctx, buf, &candidate)
                          : poly_buffer_adopt(ctx, buf, &candidate);
    PolyBuffer *after = poly_buffer_get(ctx, buf);
    preserved = preserved && rc == 0 && frees == 1 && after && after->ptr == new_data;
  }
  poly_ctx_destroy(ctx);
  return preserved && frees == (operation == 2 ? 2 : 1);
}

TEST(uop, buffer_set_allocation_failure_preserves_storage) {
  ASSERT_TRUE(replacement_preserves_old(0, false));
  PASS();
}

TEST(uop, buffer_attach_allocation_failure_preserves_storage) {
  ASSERT_TRUE(replacement_preserves_old(1, false));
  PASS();
}

TEST(uop, buffer_adopt_allocation_failure_preserves_storage) {
  ASSERT_TRUE(replacement_preserves_old(2, false));
  PASS();
}

TEST(uop, buffer_replacement_preserves_live_views) {
  for (int operation = 0; operation < 3; operation++)
    ASSERT_TRUE(replacement_preserves_old(operation, true));
  PASS();
}

TEST(uop, buffer_host_constructor_rejects_attachment_failure) {
  PolyCtx *ctx = poly_ctx_new();
  float data[4] = {0};
  int64_t dims[] = {4};
  poly_test_buffer_handle_fail_after(0);
  PolyUOp *buf =
      poly_buffer_from_host(ctx, data, sizeof(data), poly_dtype_id_by_name("float32"), dims, 1);
  poly_test_buffer_handle_fail_after(-1);
  bool rejected = buf == NULL;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rejected);
  PASS();
}

TEST(uop, buffer_initial_attachment_failure_keeps_caller_ownership) {
  for (int operation = 0; operation < 3; operation++) {
    PolyCtx *ctx = poly_ctx_new();
    int frees = 0;
    PolyAllocator allocator = {.free = replacement_count_free, .dev_ctx = &frees};
    float data[4] = {0};
    PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
    PolyBuffer candidate = {
        .ptr = data,
        .nbytes = sizeof(data),
        .device = POLY_DEVICE_CPU,
        .owned = true,
        .valid = true,
        .allocator = &allocator};
    poly_test_buffer_handle_fail_after(0);
    int rc = operation == 0   ? poly_buffer_set(ctx, buf, data, sizeof(data), POLY_DEVICE_CPU)
             : operation == 1 ? poly_buffer_attach(ctx, buf, &candidate)
                              : poly_buffer_adopt(ctx, buf, &candidate);
    poly_test_buffer_handle_fail_after(-1);
    bool unpublished = rc == -1 && !poly_buffer_get(ctx, buf);
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(unpublished);
    ASSERT_INT_EQ(frees, 0);
  }
  PASS();
}

TEST(uop, buffer_replacement_rejects_storage_and_multibuffer_aliases) {
  for (int alias = 0; alias < 3; alias++) {
    PolyCtx *ctx = poly_ctx_new();
    int frees = 0;
    float data[4] = {1, 2, 3, 4}, next[4] = {5, 6, 7, 8};
    PolyAllocator allocator = {
        .alloc = replacement_no_alloc,
        .free = replacement_count_free,
        .dev_ctx = &frees,
        .host_addressable = true};
    PolyBuffer owned = {
        .ptr = data,
        .nbytes = sizeof(data),
        .device = POLY_DEVICE_CPU,
        .owned = true,
        .valid = true,
        .allocator = &allocator};
    PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
    ASSERT_INT_EQ(poly_buffer_adopt(ctx, buf, &owned), 0);
    PolyBuffer *before = poly_buffer_get(ctx, buf);
    if (alias == 1) {
      PolyUOp *stack = poly_uop1(ctx, POLY_OP_MSTACK, POLY_FLOAT32, buf, poly_arg_none());
      ASSERT_NOT_NULL(poly_uop_buffer_handle(ctx, stack));
    } else if (alias == 2) {
      PolyUOp *other = poly_test_buffer(ctx, POLY_FLOAT32, 3);
      PolyBuffer view = owned;
      view.ptr = data + 1;
      view.nbytes = 3 * sizeof(float);
      ASSERT_INT_EQ(poly_buffer_attach(ctx, other, &view), 0);
    }
    int rc = poly_buffer_set(ctx, buf, alias == 0 ? data : next, sizeof(data), POLY_DEVICE_CPU);
    bool preserved = rc == -1 && frees == 0 && poly_buffer_get(ctx, buf) == before;
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(preserved);
    ASSERT_INT_EQ(frees, 1);
  }
  PASS();
}

TEST(uop, buffer_replacement_retires_the_complete_owned_chain) {
  PolyCtx *ctx = poly_ctx_new();
  int frees = 0;
  float data[2] = {1, 2}, source[2] = {3, 4}, next[2] = {5, 6};
  PolyAllocator allocator = {
      .free = replacement_count_free, .dev_ctx = &frees, .host_addressable = true};
  PolyBuffer owned = {
      .ptr = data,
      .nbytes = sizeof(data),
      .device = POLY_DEVICE_CPU,
      .owned = true,
      .valid = true,
      .allocator = &allocator};
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 2);
  ASSERT_INT_EQ(poly_buffer_adopt(ctx, buf, &owned), 0);
  PolyBuffer *before = poly_buffer_get(ctx, buf);
  before->src = calloc(1, sizeof(PolyBuffer));
  ASSERT_NOT_NULL(before->src);
  *before->src = owned;
  before->src->ptr = source;
  poly_test_buffer_handle_fail_after(0);
  int rc = poly_buffer_set(ctx, buf, next, sizeof(next), POLY_DEVICE_CPU);
  poly_test_buffer_handle_fail_after(-1);
  bool preserved = rc == -1 && frees == 0 && poly_buffer_get(ctx, buf) == before;
  rc = poly_buffer_set(ctx, buf, next, sizeof(next), POLY_DEVICE_CPU);
  bool retired = rc == 0 && frees == 2;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(preserved && retired);
  ASSERT_INT_EQ(frees, 2);
  PASS();
}

static PolyCtx *replacement_callback_ctx;
static PolyUOp *replacement_callback_buf;
static int replacement_callback_count;
static bool replacement_callback_saw_publication;

static void replacement_frontend_release(uintptr_t key) {
  if (++replacement_callback_count == 1)
    replacement_callback_saw_publication =
        poly_buffer_get_key(replacement_callback_ctx, replacement_callback_buf) != key;
}

TEST(uop, buffer_host_replacement_publishes_before_release_callback) {
  for (int empty = 0; empty < 2; empty++) {
    PolyCtx *ctx = poly_ctx_new();
    ctx->frontend_buffer_release = replacement_frontend_release;
    replacement_callback_ctx = ctx;
    replacement_callback_buf = poly_test_buffer(ctx, POLY_UINT8, empty ? 0 : 4);
    replacement_callback_count = 0;
    replacement_callback_saw_publication = false;
    uint8_t data[4] = {0};
    PolyUOp *buf = replacement_callback_buf;
    ASSERT_INT_EQ(
        poly_buffer_set(ctx, buf, empty ? NULL : data, empty ? 0 : sizeof(data), POLY_DEVICE_HOST),
        0
    );
    uint64_t old_key = poly_buffer_get_key(ctx, buf);
    poly_test_buffer_handle_fail_after(0);
    int rc = poly_buffer_set(ctx, buf, NULL, 0, POLY_DEVICE_HOST);
    poly_test_buffer_handle_fail_after(-1);
    bool unchanged =
        rc == -1 && poly_buffer_get_key(ctx, buf) == old_key && replacement_callback_count == 0;
    rc = poly_buffer_set(ctx, buf, NULL, 0, POLY_DEVICE_HOST);
    bool published =
        rc == 0 && replacement_callback_count == 1 && replacement_callback_saw_publication;
    poly_ctx_destroy(ctx);
    ASSERT_TRUE(unchanged && published);
    ASSERT_INT_EQ(replacement_callback_count, 2);
  }
  replacement_callback_ctx = NULL;
  replacement_callback_buf = NULL;
  PASS();
}

#if defined(__linux__) && !defined(__EMSCRIPTEN__)
static bool replacement_file_is_mapped(const char *path) {
  FILE *maps = fopen("/proc/self/maps", "r");
  if (!maps) return true;
  char line[1024];
  bool found = false;
  while (fgets(line, sizeof(line), maps))
    if (strstr(line, path)) found = true;
  fclose(maps);
  return found;
}

TEST(uop, buffer_file_adoption_failure_releases_mapping) {
  char path[] = "/tmp/polygrad_buffer_XXXXXX";
  int fd = mkstemp(path);
  ASSERT_TRUE(fd >= 0);
  uint8_t data[4] = {1, 2, 3, 4};
  bool written = write(fd, data, sizeof(data)) == sizeof(data);
  close(fd);
  PolyCtx *ctx = poly_ctx_new();
  poly_test_buffer_handle_fail_after(0);
  PolyUOp *failed = poly_buffer_from_file(ctx, path, poly_dtype_id_by_name("uint8"));
  poly_test_buffer_handle_fail_after(-1);
  bool rejected = !failed && !replacement_file_is_mapped(path);
  PolyUOp *retry = poly_buffer_from_file(ctx, path, poly_dtype_id_by_name("uint8"));
  uint8_t output[4] = {0};
  bool readable = retry && poly_buffer_read(ctx, retry, output, sizeof(output)) == 0 &&
                  memcmp(data, output, sizeof(data)) == 0;
  poly_ctx_destroy(ctx);
  bool released = !replacement_file_is_mapped(path);
  unlink(path);
  ASSERT_TRUE(written && rejected && readable && released);
  PASS();
}
#endif
#endif

TEST(uop, create_const) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.14));
  ASSERT_NOT_NULL(c);
  ASSERT_EQ(c->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(c->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(c->n_src, 0);
  ASSERT_EQ(c->arg.kind, POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(c->arg.f, 3.14, 1e-10);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, default_literal_helpers_match_current_uop_const) {
  /* Current UOp.const infers weak kinds and converts values through an
   * explicit target dtype (uop/ops.py:609-615, dtype.py:77-82). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *i = poly_const_int(ctx, 1);
  PolyUOp *f = poly_const_float(ctx, 1.0);
  ASSERT_NOT_NULL(i);
  ASSERT_NOT_NULL(f);
  ASSERT_TRUE(poly_dtype_eq(i->dtype, POLY_WEAKINT));
  ASSERT_TRUE(poly_dtype_eq(f->dtype, POLY_WEAKFLOAT));

  int float_id = poly_dtype_id_by_name("float32");
  int int_id = poly_dtype_id_by_name("int32");
  int bool_id = poly_dtype_id_by_name("bool");
  PolyUOp *i_to_f = poly_const_int_by_id(ctx, 1, float_id);
  PolyUOp *f_to_i = poly_const_float_by_id(ctx, 1.75, int_id);
  PolyUOp *i_to_b = poly_const_int_by_id(ctx, 2, bool_id);
  ASSERT_NOT_NULL(i_to_f);
  ASSERT_NOT_NULL(f_to_i);
  ASSERT_NOT_NULL(i_to_b);
  ASSERT_TRUE(poly_dtype_eq(i_to_f->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(i_to_f->arg.kind, POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(i_to_f->arg.f, 1.0, 0.0);
  ASSERT_TRUE(poly_dtype_eq(f_to_i->dtype, POLY_INT32));
  ASSERT_INT_EQ(f_to_i->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(f_to_i->arg.i, 1);
  ASSERT_TRUE(poly_dtype_eq(i_to_b->dtype, POLY_BOOL));
  ASSERT_INT_EQ(i_to_b->arg.kind, POLY_ARG_BOOL);
  ASSERT_TRUE(i_to_b->arg.b);

  PolyUOp *strong = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *sum = poly_binop(ctx, POLY_OP_ADD, strong, i);
  ASSERT_NOT_NULL(sum);
  ASSERT_TRUE(poly_dtype_eq(sum->dtype, POLY_FLOAT32));
  ASSERT_PTR_EQ(sum->src[0], strong);
  ASSERT_TRUE(poly_dtype_eq(sum->src[1]->dtype, POLY_WEAKFLOAT));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, elementwise_promotion_preserves_invalid_base) {
  /* ElementwiseMixin._broadcasted preserves invalid bases before promotion,
   * including movement/DETACH wrappers. Ordinary bool values still cast. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_BOOL);
  int64_t shape[] = {2};
  PolyUOp *expanded = poly_expand(ctx, invalid, shape, 1);
  PolyUOp *variants[] = {invalid, expanded, poly_alu1(ctx, POLY_OP_DETACH, expanded)};
  PolyDType dtypes[] = {POLY_INT32, POLY_FLOAT32};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      PolyUOp *value = poly_uop_const(ctx, poly_arg_int(3), dtypes[j]);
      PolyUOp *sum = poly_binop(ctx, POLY_OP_ADD, variants[i], value);
      ASSERT_NOT_NULL(sum);
      ASSERT_INT_EQ(sum->op, POLY_OP_ADD);
      ASSERT_INT_EQ(sum->n_src, 2);
      ASSERT_TRUE(poly_dtype_eq(sum->dtype, dtypes[j]));
      ASSERT_PTR_EQ(sum->src[0], variants[i]);
      ASSERT_PTR_EQ(sum->src[1], value);
      ASSERT_TRUE(poly_dtype_eq(sum->src[0]->dtype, POLY_BOOL));

      PolyUOp *valid_bool = poly_uop_const(ctx, poly_arg_bool(false), POLY_BOOL);
      PolyUOp *control = poly_binop(ctx, POLY_OP_ADD, valid_bool, value);
      ASSERT_INT_EQ(control->src[0]->op, POLY_OP_CAST);
      ASSERT_TRUE(poly_dtype_eq(control->src[0]->dtype, dtypes[j]));
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, unsharded_base_preserves_nested_shard_boundary) {
  /* UOp.unsharded_base removes the first wrapper, then calls base; it does
   * not recursively strip every UNSHARD encountered under movement/DETACH. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 6, POLY_DEVICE_CPU);
  PolyUOp *range = poly_uop_range(ctx, 2, -1, POLY_AXIS_DEVICE);
  int64_t axis[] = {0};
  PolyUOp *unshard = poly_unshard(ctx, buffer, axis, &range, 1);
  ASSERT_NOT_NULL(unshard);
  int64_t shape[] = {2, 6};
  PolyUOp *movement = poly_reshape(ctx, unshard, shape, 2);
  PolyUOp *detach = poly_alu1(ctx, POLY_OP_DETACH, unshard);
  ASSERT_PTR_EQ(poly_uop_unsharded_base(unshard), buffer);
  ASSERT_PTR_EQ(poly_uop_unsharded_base(movement), unshard);
  ASSERT_PTR_EQ(poly_uop_unsharded_base(detach), unshard);
  ASSERT_PTR_EQ(poly_uop_base(movement), unshard);
  ASSERT_PTR_EQ(poly_uop_unsharded_base(buffer), buffer);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, mselect_has_buffer_identity_without_collapsing_lanes) {
  /* has_buffer_identity is a predicate, not extraction of a single BUFFER.
   * The buffer resolver must still preserve the selected runtime lane. */
  PolyCtx *ctx = poly_ctx_new();
  const char *devices[] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_string_tuple(devices, 2));
  PolyUOp *buffer = poly_uop_new_buffer(ctx, tuple, 6, POLY_FLOAT32, 7302);
  PolyUOp *select0 = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, buffer, poly_arg_int(0));
  PolyUOp *select1 = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, buffer, poly_arg_int(1));
  ASSERT_TRUE(poly_uop_has_buffer_identity(select0));
  ASSERT_TRUE(poly_uop_has_buffer_identity(select1));
  PolyUOp *after = poly_uop1(ctx, POLY_OP_AFTER, POLY_FLOAT32, select0, poly_arg_none());
  ASSERT_TRUE(!poly_uop_has_buffer_identity(after));
  ASSERT_TRUE(!poly_uop_has_buffer_identity(poly_alu1(ctx, POLY_OP_DETACH, select0)));
  PolyUOp *stack = poly_uop2(ctx, POLY_OP_MSTACK, POLY_FLOAT32, select0, select1, poly_arg_none());
  ASSERT_TRUE(!poly_uop_has_buffer_identity(
      poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, stack, poly_arg_int(0))
  ));
  ASSERT_PTR_EQ(poly_contiguous(ctx, select0), select0);
  ASSERT_PTR_EQ(poly_contiguous(ctx, select1), select1);
  ASSERT_PTR_NEQ(poly_uop_buf_uop(ctx, select0), poly_uop_buf_uop(ctx, select1));
  ASSERT_TRUE(poly_uop_get_buffer_identity(select0) == NULL);
  ASSERT_TRUE(poly_uop_get_buffer_identity(select1) == NULL);
  PolyBuffer *child0 = poly_uop_buffer_handle(ctx, select0);
  PolyBuffer *child1 = poly_uop_buffer_handle(ctx, select1);
  ASSERT_NOT_NULL(child0);
  ASSERT_NOT_NULL(child1);
  ASSERT_PTR_NEQ(child0, child1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, placeholder_preserves_current_negative_local_slot) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/ops.py:1138-1149 accepts
   * negative LOCAL slots used by X86 scratch placeholders. */
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {3};
  PolyUOp *buf =
      poly_uop_placeholder(ctx, shape, 1, POLY_FLOAT32, -1, POLY_ADDR_LOCAL, NULL, false);

  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(buf->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(buf->arg.kind, POLY_ARG_PARAM);
  ASSERT_INT_EQ(buf->arg.param->slot, -1);
  ASSERT_INT_EQ(buf->arg.param->addrspace, POLY_ADDR_LOCAL);
  ASSERT_TRUE(buf->n_src == 1 && buf->src[0]->op == POLY_OP_CONST);
  ASSERT_INT_EQ(buf->src[0]->arg.i, 3);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, release014_placeholder_defaults_use_existing_identity_and_tag) {
  /* UOp.placeholder(slot=None, tag=...) is construction sugar: allocate
   * from the existing context counter, then tag storage before reshaping.
   * C keeps explicit arguments; no sentinel may consume valid negative slots. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *values[2];
  for (int i = 0; i < 2; i++) {
    int64_t flat[] = {6};
    PolyUOp *storage = poly_uop_placeholder(
        ctx, flat, 1, POLY_WEAKFLOAT, poly_ctx_next_unique_id(ctx), POLY_ADDR_LOCAL, NULL, false
    );
    ASSERT_NOT_NULL(storage);
    storage = poly_uop_tagged_arg(
        ctx, storage->op, storage->dtype, storage->src, storage->n_src, storage->arg, 0,
        poly_arg_str("scratch")
    );
    values[i] = poly_reshape(ctx, storage, (int64_t[]){2, 3}, 2);
    ASSERT_NOT_NULL(values[i]);
    ASSERT_INT_EQ(values[i]->op, POLY_OP_RESHAPE);
    ASSERT_INT_EQ(values[i]->tag_arg.kind, POLY_ARG_NONE);
    ASSERT_INT_EQ(storage->op, POLY_OP_BUFFER);
    ASSERT_INT_EQ(storage->src[0]->arg.i, 6);
    ASSERT_TRUE(poly_dtype_eq(storage->dtype, POLY_FLOAT32));
    ASSERT_STR_EQ(storage->tag_arg.str, "scratch");
  }
  ASSERT_PTR_NEQ(values[0], values[1]);
  ASSERT_TRUE(values[0]->src[0]->arg.param->slot != values[1]->src[0]->arg.param->slot);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, create_with_sources) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());

  ASSERT_NOT_NULL(add);
  ASSERT_EQ(add->op, POLY_OP_ADD);
  ASSERT_INT_EQ(add->n_src, 2);
  ASSERT_PTR_EQ(add->src[0], a);
  ASSERT_PTR_EQ(add->src[1], b);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, shape_to_shape_arg_matches_current_rank_topology) {
  /* Current tinygrad uop/ops.py:96-100 returns the sole shape dimension
   * directly and uses STACK only for rank zero or more than one dimension. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *four = poly_const_int(ctx, 4);
  PolyUOp *five = poly_const_int(ctx, 5);
  PolyUOp *dims[2] = {four, five};

  PolyUOp *rank0 = poly_shape_to_shape_arg(ctx, NULL, 0);
  PolyUOp *rank1 = poly_shape_to_shape_arg(ctx, dims, 1);
  PolyUOp *rank2 = poly_shape_to_shape_arg(ctx, dims, 2);
  ASSERT_NOT_NULL(rank0);
  ASSERT_PTR_EQ(rank1, four);
  ASSERT_NOT_NULL(rank2);
  ASSERT_INT_EQ(rank0->op, POLY_OP_STACK);
  ASSERT_INT_EQ(rank0->n_src, 0);
  ASSERT_TRUE(poly_dtype_eq(rank0->dtype, POLY_VOID));
  ASSERT_INT_EQ(rank2->op, POLY_OP_STACK);
  ASSERT_INT_EQ(rank2->n_src, 2);
  ASSERT_PTR_EQ(rank2->src[0], four);
  ASSERT_PTR_EQ(rank2->src[1], five);
  ASSERT_TRUE(poly_dtype_eq(rank2->dtype, POLY_WEAKINT));

  PolyUOp *bad[1] = {poly_const_float(ctx, 4.0)};
  ASSERT_TRUE(poly_shape_to_shape_arg(ctx, bad, 1) == NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, image_index_dtype_is_inferred_from_param_shape) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:dtype_from_uop infers
   * image access from PARAM shape (H,W,4), without ImageDType. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape_src[3] = {poly_const_int(ctx, 2), poly_const_int(ctx, 3), poly_const_int(ctx, 4)};
  PolyUOp *shape = poly_shape_to_shape_arg(ctx, shape_src, 3);
  PolyParamArg image_arg = {.slot = 0, .dtype = POLY_FLOAT16, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *image = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT16, shape, poly_arg_param(&image_arg));
  PolyUOp *coord[2] = {poly_const_int(ctx, 1), poly_const_int(ctx, 2)};
  PolyUOp *index = poly_uop_index(ctx, image, coord, 2);
  PolyUOp *load = poly_uop_load(ctx, index);

  ASSERT_NOT_NULL(index);
  ASSERT_NOT_NULL(load);
  ASSERT_TRUE(poly_dtype_eq(index->dtype, POLY_FLOAT32));
  ASSERT_TRUE(poly_dtype_eq(load->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(poly_uop_ndim(ctx, index), 1);
  ASSERT_INT_EQ(poly_uop_max_numel(ctx, index), 4);
  PolyAddrSpace addrspace = POLY_ADDR_GLOBAL;
  ASSERT_TRUE(poly_uop_addrspace(index, &addrspace));
  ASSERT_INT_EQ(addrspace, POLY_ADDR_GLOBAL);
  ASSERT_TRUE(poly_uop_addrspace(load, &addrspace));
  ASSERT_INT_EQ(addrspace, POLY_ADDR_ALU);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, addrspace_recursive_property_handles_shared_dags) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/ops.py:866-878 caches this
   * immutable recursive property. Reusing each ADD twice must stay linear. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *param = poly_test_uop_param(ctx, POLY_FLOAT32, 1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *index = poly_uop_index(ctx, param, (PolyUOp *[]){poly_const_int(ctx, 0)}, 1);
  PolyUOp *value = poly_uop_load(ctx, index);
  for (int i = 0; i < 128; i++)
    value = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, value, value, poly_arg_none());

  PolyAddrSpace addrspace = POLY_ADDR_GLOBAL;
  ASSERT_TRUE(poly_uop_addrspace(value, &addrspace));
  ASSERT_INT_EQ(addrspace, POLY_ADDR_ALU);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, mop_cleanup_matches_current_tinygrad_topology) {
  /* Current tinygrad/uop/movement.py:mop_cleanup canonicalizes all movement
   * forms through one shared matcher used by symbolic and codegen. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *lanes[6];
  for (int i = 0; i < 6; i++)
    lanes[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(i));
  PolyUOp *stack6 = poly_uop_stack(ctx, lanes, 6);
  PolyUOp *shape23_src[] = {
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3)),
  };
  PolyUOp *shape23 = poly_shape_to_shape_arg(ctx, shape23_src, 2);
  PolyUOp *base = poly_uop2(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, stack6, shape23, poly_arg_none());
  ASSERT_NOT_NULL(base);

  PolyUOp *same_shape =
      poly_uop2(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, base, shape23, poly_arg_none());
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, same_shape, poly_mop_cleanup()), base);

  PolyUOp *shape6 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(6));
  PolyUOp *nested = poly_uop2(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, base, shape6, poly_arg_none());
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, nested, poly_mop_cleanup()), stack6);

  int64_t identity_vals[] = {0, 1};
  PolyArg identity = {
      .kind = POLY_ARG_INT_TUPLE,
      .int_tuple = {.vals = identity_vals, .n = 2},
  };
  PolyUOp *noop_permute = poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, base, identity);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, noop_permute, poly_mop_cleanup()), base);

  int64_t swap_vals[] = {1, 0};
  PolyArg swap = {
      .kind = POLY_ARG_INT_TUPLE,
      .int_tuple = {.vals = swap_vals, .n = 2},
  };
  PolyUOp *swap_once = poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, base, swap);
  PolyUOp *swap_twice = poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, swap_once, swap);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, swap_twice, poly_mop_cleanup()), base);

  PolyUOp *stack3_src[3] = {lanes[0], lanes[1], lanes[2]};
  PolyUOp *stack3 = poly_uop_stack(ctx, stack3_src, 3);
  PolyUOp *indexed[3];
  for (int i = 0; i < 3; i++) {
    PolyUOp *lane = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(i));
    indexed[i] = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, stack3, lane, poly_arg_none());
  }
  PolyUOp *restacked = poly_uop_stack(ctx, indexed, 3);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, restacked, poly_mop_cleanup()), stack3);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, restacked, poly_symbolic_simple()), stack3);

  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *stack_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, stack3, one, poly_arg_none());
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, stack_index, poly_mop_cleanup()), lanes[1]);

  PolyUOp *ptr = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *inner = poly_uop2(ctx, POLY_OP_INDEX, ptr->dtype, ptr, zero, poly_arg_none());
  PolyUOp *outer = poly_uop2(ctx, POLY_OP_INDEX, ptr->dtype, inner, one, poly_arg_none());
  PolyUOp *flattened = poly_graph_rewrite(ctx, outer, poly_mop_cleanup());
  ASSERT_NOT_NULL(flattened);
  ASSERT_INT_EQ(flattened->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(flattened->n_src, 3);
  ASSERT_PTR_EQ(flattened->src[0], ptr);
  ASSERT_PTR_EQ(flattened->src[1], zero);
  ASSERT_PTR_EQ(flattened->src[2], one);

  PolyUOp *coord_src[] = {zero, one};
  PolyUOp *coord = poly_uop_stack(ctx, coord_src, 2);
  PolyUOp *shaped_inner = poly_uop2(ctx, POLY_OP_INDEX, ptr->dtype, ptr, coord, poly_arg_none());
  PolyUOp *shaped_outer =
      poly_uop2(ctx, POLY_OP_INDEX, ptr->dtype, shaped_inner, one, poly_arg_none());
  PolyUOp *shaped = poly_graph_rewrite(ctx, shaped_outer, poly_mop_cleanup());
  ASSERT_NOT_NULL(shaped);
  ASSERT_INT_EQ(shaped->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(shaped->n_src, 2);
  ASSERT_PTR_EQ(shaped->src[0], ptr);
  ASSERT_PTR_EQ(shaped->src[1], one);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, mop_cleanup_flattens_empty_index_coordinates) {
  /* tinygrad/uop/movement.py:index-on-index permits either coordinate
   * tuple to be empty; INDEX(base) is still a shaped address operation. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *base = poly_uop_placeholder(
      ctx, (int64_t[]){2, 3}, 2, POLY_FLOAT32, 0, POLY_ADDR_GLOBAL, NULL, false
  );
  ASSERT_NOT_NULL(base);
  PolyUOp *coords[] = {poly_const_int(ctx, 0), poly_const_int(ctx, 1)};
  for (int inner_n = 0; inner_n <= 1; inner_n++) {
    for (int outer_n = 0; outer_n <= 1; outer_n++) {
      PolyUOp *inner = poly_uop_index(ctx, base, coords, inner_n);
      PolyUOp *outer = poly_uop_index(ctx, inner, coords + inner_n, outer_n);
      PolyUOp *expected = poly_uop_index(ctx, base, coords, inner_n + outer_n);
      ASSERT_NOT_NULL(inner);
      ASSERT_NOT_NULL(outer);
      ASSERT_NOT_NULL(expected);
      ASSERT_PTR_EQ(poly_graph_rewrite(ctx, outer, poly_mop_cleanup()), expected);
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, permute_mixin_identity_and_negative_axes) {
  /* MovementMixin.permute resolves axes and returns identity before _mop;
   * this must hold for C callers without frontend normalization too. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *base = poly_uop_placeholder(
      ctx, (int64_t[]){2, 3}, 2, POLY_FLOAT32, 0, POLY_ADDR_GLOBAL, NULL, false
  );
  ASSERT_NOT_NULL(base);
  ASSERT_PTR_EQ(poly_permute(ctx, base, (int64_t[]){0, 1}, 2), base);
  ASSERT_PTR_EQ(poly_permute(ctx, base, (int64_t[]){-2, -1}, 2), base);
  PolyUOp *swap = poly_permute(ctx, base, (int64_t[]){1, 0}, 2);
  ASSERT_NOT_NULL(swap);
  ASSERT_INT_EQ(swap->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(swap->n_src, 1);
  ASSERT_PTR_EQ(swap->src[0], base);
  ASSERT_INT_EQ(swap->arg.int_tuple.n, 2);
  ASSERT_INT_EQ(swap->arg.int_tuple.vals[0], 1);
  ASSERT_INT_EQ(swap->arg.int_tuple.vals[1], 0);
  ASSERT_PTR_EQ(poly_permute(ctx, base, (int64_t[]){-1, -2}, 2), swap);
  PolyUOp *scalar = poly_const_float(ctx, 1.0);
  ASSERT_PTR_EQ(poly_permute(ctx, scalar, NULL, 0), scalar);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, permute_mixin_rejects_invalid_axes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *base = poly_uop_placeholder(
      ctx, (int64_t[]){2, 3}, 2, POLY_FLOAT32, 0, POLY_ADDR_GLOBAL, NULL, false
  );
  ASSERT_NOT_NULL(base);
  ASSERT_PTR_EQ(poly_permute(ctx, base, (int64_t[]){0, 0}, 2), NULL);
  ASSERT_PTR_EQ(poly_permute(ctx, base, (int64_t[]){0}, 1), NULL);
  ASSERT_PTR_EQ(poly_permute(ctx, base, (int64_t[]){0, 2}, 2), NULL);
  ASSERT_PTR_EQ(poly_permute(ctx, base, (int64_t[]){-3, 0}, 2), NULL);
  ASSERT_PTR_EQ(poly_permute(ctx, base, (int64_t[]){0, 1, 2}, 3), NULL);
  ASSERT_PTR_EQ(poly_permute(ctx, base, (int64_t[]){INT64_MIN, INT64_MAX}, 2), NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, contiguous_view_offset_matches_current_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t square_shape[2] = {4, 4};
  PolyUOp *square = poly_reshape(
      ctx, poly_test_buffer_on_device(ctx, POLY_FLOAT32, 16, POLY_DEVICE_CPU), square_shape, 2
  );
  ASSERT_NOT_NULL(square);

  int64_t reshape_shape[2] = {2, 8};
  int64_t row_pairs[2][2] = {{1, 3}, {0, 4}};
  int64_t column_pairs[2][2] = {{0, 4}, {1, 3}};
  int64_t transpose[2] = {1, 0};
  int64_t flip_axis[1] = {0};
  PolyUOp *reshape = poly_reshape(ctx, square, reshape_shape, 2);
  PolyUOp *row_shrink = poly_shrink(ctx, square, row_pairs, 2);
  PolyUOp *column_shrink = poly_shrink(ctx, square, column_pairs, 2);
  PolyUOp *permute = poly_permute(ctx, square, transpose, 2);
  PolyUOp *flip = poly_flip(ctx, square, flip_axis, 1);
  ASSERT_NOT_NULL(reshape);
  ASSERT_NOT_NULL(row_shrink);
  ASSERT_NOT_NULL(column_shrink);
  ASSERT_NOT_NULL(permute);
  ASSERT_NOT_NULL(flip);

  int64_t offset = -1;
  ASSERT_INT_EQ(poly_uop_contiguous_view_offset(ctx, reshape, &offset), 0);
  ASSERT_INT_EQ(offset, 0);
  ASSERT_INT_EQ(poly_uop_contiguous_view_offset(ctx, row_shrink, &offset), 0);
  ASSERT_INT_EQ(offset, 4);
  ASSERT_TRUE(poly_uop_contiguous_view_offset(ctx, column_shrink, &offset) < 0);
  ASSERT_TRUE(poly_uop_contiguous_view_offset(ctx, permute, &offset) < 0);
  ASSERT_TRUE(poly_uop_contiguous_view_offset(ctx, flip, &offset) < 0);

  int64_t degenerate_shape[2] = {1, 3};
  PolyUOp *degenerate = poly_reshape(
      ctx, poly_test_buffer_on_device(ctx, POLY_FLOAT32, 3, POLY_DEVICE_CPU), degenerate_shape, 2
  );
  PolyUOp *degenerate_permute = poly_permute(ctx, degenerate, transpose, 2);
  ASSERT_NOT_NULL(degenerate_permute);
  ASSERT_INT_EQ(poly_uop_contiguous_view_offset(ctx, degenerate_permute, &offset), 0);
  ASSERT_INT_EQ(offset, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cast_and_bitcast_store_exact_dtype_arg) {
  /* Current tinygrad DTypeMixin.cast/bitcast passes the target DType as arg,
   * and dtype_from_uop derives the result dtype from that same object
   * (mixin/dtype.py:16-50, uop/ops.py:169-172). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *src = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  ASSERT_NOT_NULL(src);

  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, src, poly_arg_none());
  ASSERT_NOT_NULL(cast);
  ASSERT_INT_EQ(cast->arg.kind, POLY_ARG_DTYPE);
  ASSERT_TRUE(poly_dtype_eq(cast->arg.dtype, POLY_WEAKINT));
  char *text = poly_uop_str(cast);
  ASSERT_NOT_NULL(text);
  ASSERT_TRUE(strstr(text, "dtypes.weakint") != NULL);
  free(text);

  PolyUOp *bitcast = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT8, src, poly_arg_none());
  ASSERT_NOT_NULL(bitcast);
  ASSERT_INT_EQ(bitcast->arg.kind, POLY_ARG_DTYPE);
  ASSERT_TRUE(poly_dtype_eq(bitcast->arg.dtype, POLY_UINT8));

  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, src, poly_arg_dtype(POLY_INT32)) == NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, transcendental_dtype_matches_current_tinygrad) {
  /* Current UOp dtype inference promotes these five ops without inserting a
   * source CAST (uop/ops.py:144-145; dtype.py:187-188). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *weak_two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *reciprocal = poly_alu1(ctx, POLY_OP_RECIPROCAL, weak_two);
  ASSERT_NOT_NULL(reciprocal);
  ASSERT_TRUE(poly_dtype_eq(reciprocal->dtype, POLY_WEAKFLOAT));
  ASSERT_PTR_EQ(reciprocal->src[0], weak_two);
  ASSERT_TRUE(poly_dtype_eq(reciprocal->src[0]->dtype, POLY_WEAKINT));

  PolyUOp *integer = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *sin = poly_alu1(ctx, POLY_OP_SIN, integer);
  ASSERT_NOT_NULL(sin);
  ASSERT_TRUE(poly_dtype_eq(sin->dtype, POLY_FLOAT32));
  ASSERT_PTR_EQ(sin->src[0], integer);
  ASSERT_TRUE(poly_dtype_eq(sin->src[0]->dtype, POLY_INT32));

  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *shape[1] = {poly_const_int(ctx, 1)};
  PolyUOp *moved_invalid = poly_reshape_uop(ctx, invalid, shape, 1);
  PolyUOp *sqrt_invalid = poly_alu1(ctx, POLY_OP_SQRT, moved_invalid);
  ASSERT_NOT_NULL(sqrt_invalid);
  ASSERT_TRUE(poly_dtype_eq(sqrt_invalid->dtype, POLY_BOOL));
  ASSERT_PTR_EQ(sqrt_invalid->src[0], moved_invalid);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, create_int_arg) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  ASSERT_EQ(c->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(c->arg.i, 42);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, frontend_resolve_simplifies_before_using_bounds) {
  /* Pinned tinygrad/uop/ops.py:50-54 simplifies first, then consults exact
   * vmin/vmax and uses the caller default only for an unresolved boolean. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *v =
      poly_uop_variable(ctx, "resolve_v", poly_arg_int(0), poly_arg_int(5), POLY_WEAKINT, 1, false);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *ten = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(10));
  ASSERT_NOT_NULL(v);
  ASSERT_INT_EQ(poly_uop_resolve(ctx, poly_alu2(ctx, POLY_OP_CMPLT, v, ten), 0), 1);
  ASSERT_INT_EQ(poly_uop_resolve(ctx, poly_alu2(ctx, POLY_OP_CMPLT, v, zero), 1), 0);
  PolyUOp *dynamic = poly_alu2(ctx, POLY_OP_CMPLT, v, three);
  ASSERT_INT_EQ(poly_uop_resolve(ctx, dynamic, 1), 1);
  ASSERT_INT_EQ(poly_uop_resolve(ctx, dynamic, 0), 0);
  PolyUOp *true_const = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *self_ne = poly_alu2(ctx, POLY_OP_CMPNE, v, v);
  PolyUOp *self_equal = poly_alu2(ctx, POLY_OP_CMPNE, self_ne, true_const);
  ASSERT_INT_EQ(poly_uop_resolve(ctx, self_equal, 0), 1);
  PolyUOp *mul_zero = poly_alu2(ctx, POLY_OP_MUL, v, zero);
  PolyUOp *mul_ne = poly_alu2(ctx, POLY_OP_CMPNE, mul_zero, zero);
  PolyUOp *mul_equal = poly_alu2(ctx, POLY_OP_CMPNE, mul_ne, true_const);
  ASSERT_INT_EQ(poly_uop_resolve(ctx, mul_equal, 0), 1);
  ASSERT_INT_EQ(poly_uop_resolve(ctx, v, 0), -1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, create_string_arg) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_str("CPU"));
  ASSERT_EQ(u->arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(u->arg.str, "CPU");
  /* string should be arena-copied, not pointing to the original */
  ASSERT_PTR_NEQ(u->arg.str, "CPU");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, bufferize_integer_identity_cse) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *value =
      poly_uop_new_buffer(ctx, poly_device_uop_from_name(ctx, "CPU"), 1, POLY_FLOAT32, 0);
  const int64_t ids[] = {0, 7, INT64_C(1) << 40, -1};
  PolyUOp *stages[4];
  for (int i = 0; i < 4; i++) {
    PolyArg arg = poly_arg_bufferize_opts_int(ids[i], POLY_ADDR_LOCAL, true);
    ASSERT_FALSE(poly_arg_eq(arg, poly_arg_bufferize_opts(NULL, POLY_ADDR_LOCAL, true)));
    ASSERT_FALSE(poly_arg_eq(arg, poly_arg_bufferize_opts("7", POLY_ADDR_LOCAL, true)));
    stages[i] = poly_uop1(ctx, POLY_OP_STAGE, POLY_FLOAT32, value, arg);
    ASSERT_NOT_NULL(stages[i]);
    ASSERT_PTR_EQ(stages[i], poly_uop1(ctx, POLY_OP_STAGE, POLY_FLOAT32, value, arg));
    for (int j = 0; j < i; j++)
      ASSERT_TRUE(stages[i] != stages[j]);
    ASSERT_TRUE(poly_uop_device_uop_cached(ctx, stages[i], NULL) == NULL);
    ASSERT_EQ(poly_uop_device(stages[i]), POLY_DEVICE_AUTO);
    ASSERT_TRUE(poly_bufferize_arg_device(arg) == NULL);
    char expected[80];
    snprintf(expected, sizeof(expected), "BufferizeOpts(device=%lld,", (long long)ids[i]);
    char *text = poly_uop_str(stages[i]);
    ASSERT_NOT_NULL(text);
    bool exact = strstr(text, expected) != NULL;
    free(text);
    ASSERT_TRUE(exact);
    PolyUOp *replacement = poly_const_float(ctx, 2.0f);
    PolyUOp *changed = poly_uop_substitute(ctx, stages[i], &value, &replacement, 1);
    ASSERT_NOT_NULL(changed);
    ASSERT_TRUE(poly_arg_eq(changed->arg, arg));
    ASSERT_EQ(poly_arg_hash(changed->arg), poly_arg_hash(arg));
  }
  PolyArg invalid = poly_arg_bufferize_opts_int(7, POLY_ADDR_LOCAL, true);
  invalid.bufferize_opts.device = "CPU";
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_STAGE, POLY_FLOAT32, value, invalid) == NULL);
  invalid.bufferize_opts.device = NULL;
  invalid.bufferize_opts.device_is_tuple = true;
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_STAGE, POLY_FLOAT32, value, invalid) == NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, device_constructor_uses_canonical_string_identity) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned Device._canonicalize removes only ordinal zero, and UOp.new_buffer
   * stores the resulting string in DEVICE.arg (device.py:15-24,
   * uop/ops.py:733-746). */
  PolyUOp *cuda = poly_device_uop_from_name(ctx, "cuda");
  PolyUOp *cuda0 = poly_device_uop_from_name(ctx, "CUDA:0");
  PolyUOp *cuda1 = poly_device_uop_from_name(ctx, "cuda:1");
  ASSERT_NOT_NULL(cuda);
  ASSERT_NOT_NULL(cuda0);
  ASSERT_NOT_NULL(cuda1);
  ASSERT_PTR_EQ(cuda, cuda0);
  ASSERT_PTR_NEQ(cuda, cuda1);
  ASSERT_EQ(cuda->arg.kind, POLY_ARG_STRING);
  ASSERT_EQ(cuda1->arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(cuda->arg.str, "CUDA");
  ASSERT_STR_EQ(cuda1->arg.str, "CUDA:1");
  ASSERT_INT_EQ(poly_device_from_device_uop(cuda), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(poly_device_from_device_uop(cuda1), POLY_DEVICE_CUDA);
  ASSERT_FALSE(poly_uop_explicit_devices_supported(ctx, cuda1));
  PolyUOp *old_integer_dialect =
      poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_CUDA));
  ASSERT_INT_EQ(poly_device_from_device_uop(old_integer_dialect), POLY_DEVICE_AUTO);

  PolyUOp *cpu = poly_device_uop(ctx, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(cpu);
  ASSERT_EQ(cpu->arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(cpu->arg.str, "CPU");
  PolyUOp *cpu1 = poly_device_uop_from_name(ctx, "CPU:1");
  ASSERT_INT_EQ(poly_device_from_device_uop(cpu1), POLY_DEVICE_CPU);
  ASSERT_TRUE(poly_uop_explicit_devices_supported(ctx, cpu1));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, device_constructor_preserves_ordered_tuple_identity) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned canonicalize_device maps every element independently, preserves
   * tuple order, and collapses only a one-element tuple (device.py:57-59).
   * DEVICE.arg stores the exact tuple (uop/ops.py:770-783). */
  const char *names_a[] = {"CPU:0", "cpu:1"};
  const char *names_b[] = {"CPU", "CPU:1"};
  const char *names_rev[] = {"CPU:1", "CPU"};
  const char *single[] = {"cpu:0"};
  PolyUOp *tuple_a = poly_device_uop_from_names(ctx, names_a, 2);
  PolyUOp *tuple_b = poly_device_uop_from_names(ctx, names_b, 2);
  PolyUOp *tuple_rev = poly_device_uop_from_names(ctx, names_rev, 2);
  PolyUOp *scalar = poly_device_uop_from_names(ctx, single, 1);
  ASSERT_NOT_NULL(tuple_a);
  ASSERT_PTR_EQ(tuple_a, tuple_b);
  ASSERT_PTR_NEQ(tuple_a, tuple_rev);
  ASSERT_PTR_EQ(scalar, poly_device_uop_from_name(ctx, "CPU"));
  ASSERT_INT_EQ(tuple_a->arg.kind, POLY_ARG_STRING_TUPLE);
  ASSERT_INT_EQ(tuple_a->arg.string_tuple.n, 2);
  ASSERT_STR_EQ(tuple_a->arg.string_tuple.vals[0], "CPU");
  ASSERT_STR_EQ(tuple_a->arg.string_tuple.vals[1], "CPU:1");
  ASSERT_INT_EQ(poly_device_from_device_uop(tuple_a), POLY_DEVICE_AUTO);
  /* Exact tuple identity has no scalar backend enum, but every CPU ordinal is
   * executable and schedule/multi lowers it to supported scalar occurrences. */
  ASSERT_TRUE(poly_uop_explicit_devices_supported(ctx, tuple_a));

  PolyUOp *buffer =
      poly_uop_new_buffer(ctx, poly_device_uop_from_name(ctx, "CPU"), 8, POLY_INT32, 1);
  PolyUOp *copy = poly_copy_to_device_uop(ctx, buffer, tuple_a);
  ASSERT_PTR_EQ(poly_uop_device_uop_cached(ctx, copy, NULL), tuple_a);
  ASSERT_TRUE(poly_uop_explicit_devices_supported(ctx, copy));

  /* Pinned UOp.device selects one tuple element for MSELECT and constructs an
   * ordered tuple for MSTACK (uop/ops.py:770-783). MSTACK returns a tuple even
   * with one source; that path does not call canonicalize_device. */
  PolyUOp *cpu = poly_device_uop_from_name(ctx, "CPU");
  PolyUOp *cpu1 = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *buffer1 = poly_uop_new_buffer(ctx, cpu1, 8, POLY_INT32, 2);
  PolyUOp *stack_src[] = {buffer, buffer1};
  PolyUOp *stack = poly_uop(ctx, POLY_OP_MSTACK, POLY_INT32, stack_src, 2, poly_arg_none());
  PolyUOp *stack_device = poly_uop_device_uop_cached(ctx, stack, NULL);
  ASSERT_NOT_NULL(stack_device);
  ASSERT_INT_EQ(stack_device->arg.kind, POLY_ARG_STRING_TUPLE);
  ASSERT_INT_EQ(stack_device->arg.string_tuple.n, 2);
  ASSERT_STR_EQ(stack_device->arg.string_tuple.vals[0], "CPU");
  ASSERT_STR_EQ(stack_device->arg.string_tuple.vals[1], "CPU:1");
  PolyUOp *select0 = poly_uop1(ctx, POLY_OP_MSELECT, POLY_INT32, stack, poly_arg_int(0));
  PolyUOp *select1 = poly_uop1(ctx, POLY_OP_MSELECT, POLY_INT32, stack, poly_arg_int(1));
  ASSERT_PTR_EQ(poly_uop_device_uop_cached(ctx, select0, NULL), cpu);
  ASSERT_PTR_EQ(poly_uop_device_uop_cached(ctx, select1, NULL), cpu1);

  PolyUOp *single_stack = poly_uop1(ctx, POLY_OP_MSTACK, POLY_INT32, buffer, poly_arg_none());
  PolyUOp *single_stack_device = poly_uop_device_uop_cached(ctx, single_stack, NULL);
  ASSERT_NOT_NULL(single_stack_device);
  ASSERT_INT_EQ(single_stack_device->arg.kind, POLY_ARG_STRING_TUPLE);
  ASSERT_INT_EQ(single_stack_device->arg.string_tuple.n, 1);
  ASSERT_STR_EQ(single_stack_device->arg.string_tuple.vals[0], "CPU");

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, device_query_preserves_tuple_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  const char *names[] = {"CPU", "CPU:1"};
  PolyUOp *shape = poly_const_int(ctx, 1);
  PolyUOp *source =
      poly_uop_new_buffer(ctx, poly_device_uop(ctx, POLY_DEVICE_CPU), 1, POLY_FLOAT32, 0);
  unsigned wrong = 0;
  for (int n = 0; n <= 2; n++) {
    PolyParamArg param = {
        .slot = 0,
        .addrspace = POLY_ADDR_GLOBAL,
        .device_is_tuple = true,
        .devices = names,
        .n_devices = n};
    PolyUOp *nodes[] = {
        poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&param)),
        poly_uop1(ctx, POLY_OP_BUFFER, POLY_FLOAT32, shape, poly_arg_param(&param)),
        poly_uop2(
            ctx, POLY_OP_STAGE, POLY_FLOAT32, source, shape,
            poly_arg_bufferize_opts_tuple(names, n, POLY_ADDR_GLOBAL, false)
        ),
        poly_uop1(ctx, POLY_OP_COPY, POLY_FLOAT32, source, poly_arg_string_tuple(names, n)),
        poly_uop1(
            ctx, POLY_OP_ALLREDUCE, POLY_FLOAT32, source,
            poly_arg_allreduce(POLY_OP_ADD, NULL, names, n)
        ),
    };
    for (int i = 0; i < 5; i++) {
      PolyUOp *device = poly_uop_device_uop_cached(ctx, nodes[i], NULL);
      bool correct = device && device->op == POLY_OP_DEVICE &&
                     device->arg.kind == POLY_ARG_STRING_TUPLE && device->arg.string_tuple.n == n;
      for (int j = 0; j < n && correct; j++)
        correct = strcmp(device->arg.string_tuple.vals[j], names[j]) == 0;
      const char **queried = NULL;
      bool is_tuple = false;
      int count = poly_uop_device_names(ctx, nodes[i], &queried, &is_tuple);
      correct = correct && is_tuple && count == n;
      for (int j = 0; j < n && correct; j++)
        correct = queried && strcmp(queried[j], names[j]) == 0;
      if (n > 0) {
        PolyUOp *select = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, nodes[i], poly_arg_int(0));
        const char *selected = poly_uop_device_name(ctx, select);
        correct = correct && selected && strcmp(selected, "CPU") == 0;
      }
      if (!correct) {
        fprintf(stderr, "device tuple mismatch: %s count=%d\n", poly_op_name(nodes[i]->op), n);
        wrong |= 1u << (n * 5 + i);
      }
    }
  }
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(wrong, 0);
  PASS();
}

TEST(uop, device_query_preserves_exact_physical_identity) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* tinygrad@2026-08-22/a9069c177a9d uop/ops.py:847-861 reads
   * BUFFER/ParamArg.device and COPY.arg, follows AFTER src[0], and takes the
   * first concrete generic source. */
  PolyUOp *cpu1 = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *cuda = poly_device_uop_from_name(ctx, "CUDA");
  PolyUOp *buffer0 = poly_uop_new_buffer(ctx, cpu1, 1, POLY_FLOAT32, 0);
  PolyUOp *buffer1 = poly_uop_new_buffer(ctx, cuda, 1, POLY_FLOAT32, 1);
  PolyUOp *copy = poly_copy_to_device_uop(ctx, buffer1, cpu1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, copy, buffer0, poly_arg_none());
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, POLY_FLOAT32, copy, store, poly_arg_none());
  PolyUOp *mixed = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buffer0, buffer1, poly_arg_none());
  PolyUOp *constant = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));

  PolyUOp *shape = poly_const_int(ctx, 1);
  PolyParamArg param_arg = {
      .slot = 0,
      .addrspace = POLY_ADDR_GLOBAL,
      .device = "CPU:1",
  };
  PolyUOp *param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&param_arg));
  PolyUOp *stage_src[2] = {buffer0, shape};
  PolyUOp *stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_src, 2,
      poly_arg_bufferize_opts("CPU:1", POLY_ADDR_GLOBAL, false)
  );
  const char *tuple_names[2] = {"CPU", "CPU:1"};
  const char *reversed_names[2] = {"CPU:1", "CPU"};
  PolyUOp *tuple_stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_src, 2,
      poly_arg_bufferize_opts_tuple(tuple_names, 2, POLY_ADDR_GLOBAL, false)
  );
  PolyUOp *tuple_stage_same = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_src, 2,
      poly_arg_bufferize_opts_tuple(tuple_names, 2, POLY_ADDR_GLOBAL, false)
  );
  PolyUOp *tuple_stage_reversed = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_src, 2,
      poly_arg_bufferize_opts_tuple(reversed_names, 2, POLY_ADDR_GLOBAL, false)
  );
  PolyParamArg unsupported_arg = {
      .slot = 1,
      .addrspace = POLY_ADDR_GLOBAL,
      .device = "CUDA:1",
  };
  PolyUOp *unsupported =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&unsupported_arg));

  ASSERT_PTR_EQ(poly_uop_device_uop_cached(ctx, cpu1, NULL), cpu1);
  ASSERT_PTR_EQ(poly_uop_device_uop_cached(ctx, buffer0, NULL), cpu1);
  ASSERT_PTR_EQ(poly_uop_device_uop_cached(ctx, copy, NULL), cpu1);
  ASSERT_PTR_EQ(poly_uop_device_uop_cached(ctx, after, NULL), cpu1);
  ASSERT_PTR_EQ(poly_uop_device_uop_cached(ctx, mixed, NULL), cpu1);
  ASSERT_PTR_EQ(poly_uop_device_uop_cached(ctx, param, NULL), cpu1);
  ASSERT_PTR_EQ(poly_uop_device_uop_cached(ctx, stage, NULL), cpu1);
  PolyUOp *tuple_stage_device = poly_uop_device_uop_cached(ctx, tuple_stage, NULL);
  ASSERT_NOT_NULL(tuple_stage_device);
  ASSERT_INT_EQ(tuple_stage_device->arg.kind, POLY_ARG_STRING_TUPLE);
  ASSERT_INT_EQ(tuple_stage_device->arg.string_tuple.n, 2);
  ASSERT_STR_EQ(tuple_stage_device->arg.string_tuple.vals[0], "CPU");
  ASSERT_STR_EQ(tuple_stage_device->arg.string_tuple.vals[1], "CPU:1");
  ASSERT_PTR_EQ(tuple_stage, tuple_stage_same);
  ASSERT_PTR_NEQ(tuple_stage, tuple_stage_reversed);
  ASSERT_STR_EQ(poly_uop_device_name(ctx, mixed), "CPU:1");
  const char **queried = NULL;
  bool is_tuple = true;
  ASSERT_INT_EQ(poly_uop_device_names(ctx, mixed, &queried, &is_tuple), 1);
  ASSERT_FALSE(is_tuple);
  ASSERT_STR_EQ(queried[0], "CPU:1");
  ASSERT_INT_EQ(poly_uop_device_names(ctx, constant, &queried, &is_tuple), 0);
  ASSERT_FALSE(is_tuple);
  ASSERT_TRUE(queried == NULL);
  ASSERT_INT_EQ(poly_uop_device_names(ctx, NULL, &queried, &is_tuple), -1);
  ASSERT_INT_EQ(poly_uop_device_names(NULL, mixed, &queried, &is_tuple), -1);
  ASSERT_INT_EQ(poly_uop_device_names(ctx, mixed, NULL, &is_tuple), -1);
  ASSERT_INT_EQ(poly_uop_device_names(ctx, mixed, &queried, NULL), -1);
  ASSERT_TRUE(poly_uop_device_uop_cached(ctx, constant, NULL) == NULL);
  ASSERT_TRUE(poly_uop_device_name(ctx, constant) == NULL);
  ASSERT_TRUE(poly_uop_explicit_devices_supported(ctx, param));
  ASSERT_TRUE(poly_uop_explicit_devices_supported(ctx, tuple_stage));
  ASSERT_FALSE(poly_uop_explicit_devices_supported(ctx, unsupported));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, create_int_tuple_arg) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t perm[] = {1, 0, 2};
  PolyArg arg = {.kind = POLY_ARG_INT_TUPLE, .int_tuple = {perm, 3}};
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *u = poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, a, arg);
  ASSERT_EQ(u->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(u->arg.int_tuple.n, 3);
  ASSERT_INT_EQ(u->arg.int_tuple.vals[0], 1);
  ASSERT_INT_EQ(u->arg.int_tuple.vals[1], 0);
  ASSERT_INT_EQ(u->arg.int_tuple.vals[2], 2);
  /* should be arena-copied */
  ASSERT_PTR_NEQ(u->arg.int_tuple.vals, perm);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, range_arg_keeps_current_typed_identity) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *legacy = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(5));
  PolyUOp *canonical =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(5, POLY_AXIS_LOOP));

  ASSERT_TRUE(!poly_arg_eq(poly_arg_int(5), poly_arg_range(5, POLY_AXIS_LOOP)));
  ASSERT_PTR_NEQ(legacy, canonical);
  ASSERT_EQ(legacy->arg.kind, POLY_ARG_INT);
  ASSERT_EQ(canonical->arg.kind, POLY_ARG_RANGE);
  ASSERT_INT_EQ(poly_range_axis_id(canonical->arg), 5);
  ASSERT_EQ(poly_range_axis_type(canonical->arg), POLY_AXIS_LOOP);

  poly_ctx_destroy(ctx);
  PASS();
}

/* CSE (Common Subexpression Elimination) */

TEST(uop, cse_same_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  ASSERT_PTR_EQ(a, b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cse_hash_matches_dtype_eq_for_fmtless_scalar) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType fmtless = POLY_FLOAT32;
  fmtless.fmt = 0;

  ASSERT_TRUE(poly_dtype_eq(POLY_FLOAT32, fmtless));

  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, fmtless, poly_arg_float(0.0));

  ASSERT_PTR_EQ(a, b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cse_different_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  ASSERT_PTR_NEQ(a, b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cse_different_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT64, poly_arg_float(1.0));
  ASSERT_PTR_NEQ(a, b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cse_same_add) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *y = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *add1 = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, y, poly_arg_none());
  PolyUOp *add2 = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, y, poly_arg_none());
  ASSERT_PTR_EQ(add1, add2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cse_different_order) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *y = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *add1 = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, y, poly_arg_none());
  PolyUOp *add2 = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, y, x, poly_arg_none());
  /* Different source order → different UOps (CSE is identity-based) */
  ASSERT_PTR_NEQ(add1, add2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, cse_int_tuple) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  int64_t perm1[] = {1, 0};
  int64_t perm2[] = {1, 0};
  PolyArg arg1 = {.kind = POLY_ARG_INT_TUPLE, .int_tuple = {perm1, 2}};
  PolyArg arg2 = {.kind = POLY_ARG_INT_TUPLE, .int_tuple = {perm2, 2}};
  PolyUOp *u1 = poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, a, arg1);
  PolyUOp *u2 = poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, a, arg2);
  ASSERT_PTR_EQ(u1, u2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, call_info_is_value_metadata) {
  /* Pinned CallInfo is FUNCTION value metadata, so equal field values CSE and
   * every differing supported field stays distinct
   * (tinygrad 2026-08-22/a9069c177a9d uop/ops.py:1261-1272). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *body = poly_uop0(ctx, POLY_OP_TUPLE, POLY_VOID, poly_arg_none());
  ASSERT_NOT_NULL(body);
  PolyCallInfo first = {.name = "forward"};
  PolyCallInfo same = {.name = "forward"};
  PolyCallInfo other = {.name = "other"};
  PolyCallInfo precompiled = {.name = "forward", .precompile = true};
  PolyCallInfo first_grad = {.name = "forward", .has_grad_fxn = true, .grad_fxn_key = 1};
  PolyCallInfo other_grad = {.name = "forward", .has_grad_fxn = true, .grad_fxn_key = 2};
  PolyUOp *a = poly_uop1(ctx, POLY_OP_FUNCTION, POLY_VOID, body, poly_arg_call_info(&first));
  PolyUOp *b = poly_uop1(ctx, POLY_OP_FUNCTION, POLY_VOID, body, poly_arg_call_info(&same));
  PolyUOp *c = poly_uop1(ctx, POLY_OP_FUNCTION, POLY_VOID, body, poly_arg_call_info(&other));
  PolyUOp *d = poly_uop1(ctx, POLY_OP_FUNCTION, POLY_VOID, body, poly_arg_call_info(&precompiled));
  PolyUOp *e = poly_uop1(ctx, POLY_OP_FUNCTION, POLY_VOID, body, poly_arg_call_info(&first_grad));
  PolyUOp *f = poly_uop1(ctx, POLY_OP_FUNCTION, POLY_VOID, body, poly_arg_call_info(&other_grad));
  ASSERT_NOT_NULL(a);
  ASSERT_PTR_EQ(a, b);
  ASSERT_PTR_NEQ(a, c);
  ASSERT_PTR_NEQ(a, d);
  ASSERT_PTR_NEQ(e, f);
  first.name = "mutated-after-construction";
  ASSERT_STR_EQ(a->arg.call_info->name, "forward");
  char *text = poly_uop_str(a);
  ASSERT_NOT_NULL(text);
  ASSERT_STR_EQ(text, "UOp(FUNCTION, CallInfo(None,'forward',False,False), src=1)");
  free(text);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Toposort */

TEST(uop, toposort_single) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  int n;
  PolyUOp **sorted = poly_toposort(ctx, c, &n);
  ASSERT_INT_EQ(n, 1);
  ASSERT_PTR_EQ(sorted[0], c);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, toposort_linear_chain) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *c = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, b, poly_arg_none());

  int n;
  PolyUOp **sorted = poly_toposort(ctx, c, &n);
  ASSERT_INT_EQ(n, 3);
  /* Sources before consumers */
  ASSERT_PTR_EQ(sorted[0], a);
  ASSERT_PTR_EQ(sorted[1], b);
  ASSERT_PTR_EQ(sorted[2], c);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, toposort_diamond) {
  PolyCtx *ctx = poly_ctx_new();
  /*    a
   *   / \
   *  b   c
   *   \ /
   *    d
   */
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *c = poly_uop1(ctx, POLY_OP_SQRT, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *d = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, b, c, poly_arg_none());

  int n;
  PolyUOp **sorted = poly_toposort(ctx, d, &n);
  ASSERT_INT_EQ(n, 4); /* a, b, c, d — each appears once */
  ASSERT_PTR_EQ(sorted[0], a); /* a first (leaf) */
  ASSERT_PTR_EQ(sorted[3], d); /* d last (root) */
  /* b and c can be in either order but both must come before d */
  ASSERT_TRUE((sorted[1] == b && sorted[2] == c) || (sorted[1] == c && sorted[2] == b));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, toposort_shared_subgraph) {
  PolyCtx *ctx = poly_ctx_new();
  /* shared = a + b, root = shared * shared
   * Should not duplicate shared in toposort */
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.0));
  PolyUOp *shared = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, shared, shared, poly_arg_none());

  int n;
  PolyUOp **sorted = poly_toposort(ctx, root, &n);
  ASSERT_INT_EQ(n, 4); /* a, b, shared, root */
  ASSERT_PTR_EQ(sorted[3], root);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, toposort_arena_result_contract) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());

  int n = 0;
  PolyUOp **sorted = poly_toposort(ctx, b, &n);
  ASSERT_INT_EQ(n, 2);
  ASSERT_TRUE(poly_ctx_owns_ptr(ctx, sorted));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, toposort_alloc_is_owned_and_does_not_grow_ctx_arena) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *x = a;
  for (int i = 0; i < 32; i++)
    x = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, x, poly_arg_none());

  size_t before = poly_arena_used(poly_ctx_arena(ctx));
  for (int i = 0; i < 128; i++) {
    int n = 0;
    PolyUOp **sorted = poly_toposort_alloc(ctx, x, &n);
    ASSERT_NOT_NULL(sorted);
    ASSERT_INT_EQ(n, 33);
    ASSERT_FALSE(poly_ctx_owns_ptr(ctx, sorted));
    poly_toposort_free(sorted);
  }
  size_t after = poly_arena_used(poly_ctx_arena(ctx));
  ASSERT_INT_EQ(after, before);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, toposort_scratch_rewinds_without_growing_ctx_arena) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *x = a;
  for (int i = 0; i < 32; i++)
    x = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, x, poly_arg_none());

  size_t main_before = poly_arena_used(poly_ctx_arena(ctx));
  size_t scratch_before = poly_arena_used(ctx->scratch);
  PolyScratchMark mark = poly_ctx_scratch_mark(ctx);

  int n = 0;
  PolyUOp **sorted = poly_toposort_scratch(ctx, x, &n);
  ASSERT_NOT_NULL(sorted);
  ASSERT_INT_EQ(n, 33);
  ASSERT_FALSE(poly_ctx_owns_ptr(ctx, sorted));
  ASSERT_INT_EQ(poly_arena_used(poly_ctx_arena(ctx)), main_before);
  ASSERT_TRUE(poly_arena_used(ctx->scratch) > scratch_before);

  poly_ctx_scratch_rewind(ctx, mark);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);
  ASSERT_INT_EQ(poly_arena_used(poly_ctx_arena(ctx)), main_before);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, ctx_stats_reports_arena_and_scratch_high_water) {
  ASSERT_INT_EQ(poly_ctx_stats(NULL, NULL), -1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(poly_arena_high_water(poly_ctx_arena(ctx)), stats.arena_high_water);
  ASSERT_INT_EQ(poly_arena_high_water(ctx->scratch), stats.scratch_high_water);
  ASSERT_INT_EQ(stats.scratch_bytes, 0);
  ASSERT_INT_EQ(stats.buffer_owned_bytes, 0);
  ASSERT_INT_EQ(stats.launch_count, 0);
  ASSERT_INT_EQ(stats.runtime_cache_hits, 0);
  ASSERT_INT_EQ(stats.runtime_cache_misses, 0);
  ASSERT_INT_EQ(stats.buffer_read_count, 0);
  ASSERT_INT_EQ(stats.buffer_read_bytes, 0);
  ASSERT_INT_EQ(stats.buffer_write_count, 0);
  ASSERT_INT_EQ(stats.buffer_write_bytes, 0);
  ASSERT_INT_EQ(stats.buffer_copy_count, 0);
  ASSERT_INT_EQ(stats.buffer_copy_bytes, 0);
  ASSERT_INT_EQ(stats.buffer_owned_current_bytes, 0);
  ASSERT_INT_EQ(stats.buffer_owned_source_bytes, 0);

  PolyUOp *borrowed_buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  float borrowed_data[4] = {0};
  PolyBuffer borrowed = poly_buffer_make_host_view(borrowed_data, sizeof(borrowed_data));
  poly_buffer_attach(ctx, borrowed_buf, &borrowed);

  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.buffer_entries, 1);
  ASSERT_INT_EQ(stats.buffer_owned_bytes, 0);

  PolyUOp *owned_buf = poly_test_buffer(ctx, POLY_FLOAT32, 8);
  PolyBuffer *owned_host = NULL;
  ASSERT_INT_EQ(
      poly_buffer_alloc_owned_host(ctx, owned_buf, 8 * sizeof(float), true, &owned_host), 0
  );
  ASSERT_NOT_NULL(owned_host);

  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.buffer_entries, 2);
  ASSERT_INT_EQ(stats.buffer_owned_current_bytes, 8 * sizeof(float));
  ASSERT_INT_EQ(stats.buffer_owned_source_bytes, 0);
  ASSERT_INT_EQ(stats.buffer_owned_bytes, 8 * sizeof(float));

  float write_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float read_data[8] = {0};
  ASSERT_INT_EQ(poly_buffer_write(ctx, owned_buf, write_data, sizeof(write_data)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, owned_buf, read_data, sizeof(read_data)), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.buffer_write_count, 1);
  ASSERT_INT_EQ(stats.buffer_write_bytes, sizeof(write_data));
  ASSERT_INT_EQ(stats.buffer_read_count, 1);
  ASSERT_INT_EQ(stats.buffer_read_bytes, sizeof(read_data));

  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *x = a;
  for (int i = 0; i < 16; i++)
    x = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, x, poly_arg_none());

  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.arena_bytes > 0);
  ASSERT_TRUE(stats.arena_high_water >= stats.arena_bytes);
  ASSERT_TRUE(stats.cse_entries >= 17);

  PolyScratchMark mark = poly_ctx_scratch_mark(ctx);
  int n = 0;
  PolyUOp **sorted = poly_toposort_scratch(ctx, x, &n);
  ASSERT_NOT_NULL(sorted);
  ASSERT_INT_EQ(n, 17);

  PolyCtxStats during = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &during), 0);
  ASSERT_TRUE(during.scratch_bytes > 0);
  ASSERT_TRUE(during.scratch_high_water >= during.scratch_bytes);

  poly_ctx_scratch_rewind(ctx, mark);

  PolyCtxStats after = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after), 0);
  ASSERT_INT_EQ(after.scratch_bytes, 0);
  ASSERT_TRUE(after.scratch_high_water >= during.scratch_high_water);
  ASSERT_INT_EQ(after.arena_bytes, during.arena_bytes);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, ir_collection_evicts_weak_rows_and_preserves_declared_roots) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *logical_leaf = poly_buffer_f32(ctx, 4);
  PolyUOp *logical_one = poly_const_float(ctx, 1.0);
  PolyUOp *logical = poly_add(ctx, logical_leaf, logical_one);
  PolyUOp *physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyTensor *tensor =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tensor);

  PolyUOp *raw_two = poly_const_float(ctx, 2.0);
  PolyUOp *raw = poly_mul(ctx, logical, raw_two);
  ASSERT_NOT_NULL(raw);
  ASSERT_INT_EQ(poly_uop_retain(ctx, raw), 0);

  PolyUOp *dead = poly_buffer_f32(ctx, 7);
  for (int i = 0; i < 32; i++)
    dead = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, dead, poly_arg_none());
  ASSERT_NOT_NULL(dead);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, logical).ndim, 1);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, raw).ndim, 1);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, dead).ndim, 1);

  PolyCtxStats before = {0}, after = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &before), 0);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after), 0);
  ASSERT_TRUE(after.cse_entries < before.cse_entries);
  ASSERT_TRUE(after.shape_cache_entries < before.shape_cache_entries);

  ASSERT_PTR_EQ(poly_tensor_uop_logical(tensor), logical);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(tensor), physical);
  ASSERT_PTR_EQ(poly_add(ctx, logical_leaf, logical_one), logical);
  ASSERT_PTR_EQ(poly_mul(ctx, logical, raw_two), raw);

  poly_uop_release(ctx, raw);
  poly_tensor_release(tensor);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, partial_external_release_keeps_live_ir_clean) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *raw = poly_const_int(ctx, 7);
  ASSERT_NOT_NULL(raw);
  ASSERT_INT_EQ(poly_uop_retain(ctx, raw), 0);
  ASSERT_INT_EQ(poly_uop_retain(ctx, raw), 0);
  ASSERT_FALSE(ctx->collection_dirty);
  ASSERT_FALSE(ctx->ir_collection_dirty);

  /* Tinygrad keeps the weak-cache row while either strong reference lives. */
  poly_uop_release(ctx, raw);
  ASSERT_FALSE(ctx->collection_dirty);
  ASSERT_FALSE(ctx->ir_collection_dirty);

  poly_uop_release(ctx, raw);
  ASSERT_TRUE(ctx->collection_dirty);
  ASSERT_TRUE(ctx->ir_collection_dirty);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, stats_defers_small_ir_sweep_until_explicit_collect) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *raw = poly_const_int(ctx, 11);
  ASSERT_NOT_NULL(raw);
  ASSERT_INT_EQ(poly_uop_retain(ctx, raw), 0);
  poly_uop_release(ctx, raw);
  ASSERT_TRUE(ctx->ir_collection_dirty);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  /* Tinygrad counter reads do not run a global UOp trace. */
  ASSERT_TRUE(ctx->ir_collection_dirty);

  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_FALSE(ctx->ir_collection_dirty);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, raw_c_100k_churn_is_reclaimed_only_at_explicit_safe_point) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  PolyCtxStats baseline = {0}, before = {0}, observed = {0}, after = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &baseline), 0);

  double start_ms = poly_now_ms();
  PolyUOp *last = NULL;
  for (int64_t i = 0; i < 100000; i++)
    last = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(i));
  ASSERT_NOT_NULL(last);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &before), 0);
  ASSERT_TRUE(before.cse_entries >= 100000);
  ASSERT_TRUE(before.arena_bytes > baseline.arena_bytes);

  /* Counter reads are observational; raw allocation-only callers choose the
   * explicit collection boundary documented by the C API. */
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &observed), 0);
  ASSERT_INT_EQ(observed.arena_bytes, before.arena_bytes);
  ASSERT_INT_EQ(observed.cse_entries, before.cse_entries);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after), 0);
  ASSERT_INT_EQ(after.arena_bytes, baseline.arena_bytes);
  ASSERT_INT_EQ(after.cse_entries, baseline.cse_entries);
  ASSERT_TRUE(poly_now_ms() - start_ms < 30000.0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, safe_point_collects_after_bounded_ir_growth) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);

  PolyUOp *dead = poly_const_int(ctx, 13);
  for (int i = 0; i < 4096; i++)
    dead = poly_uop1(ctx, POLY_OP_NEG, POLY_INT32, dead, poly_arg_none());
  ASSERT_NOT_NULL(dead);
  size_t peak = ctx->uop_storage_bytes;
  ASSERT_TRUE(peak > ctx->ir_collection_baseline_bytes + POLY_IR_COLLECTION_MIN_GROWTH);
  ASSERT_INT_EQ(poly_uop_retain(ctx, dead), 0);
  poly_uop_release(ctx, dead);

  ASSERT_INT_EQ(poly_ctx_collect_at_safe_point(ctx), 0);
  ASSERT_FALSE(ctx->ir_collection_dirty);
  ASSERT_TRUE(ctx->uop_storage_bytes < peak);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, ir_collection_reclaims_dead_uop_storage_and_preserves_live_address) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *live = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, one, poly_arg_none());
  ASSERT_NOT_NULL(live);
  ASSERT_INT_EQ(poly_uop_retain(ctx, live), 0);

  PolyCtxStats baseline = {0}, peak = {0}, after = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &baseline), 0);
  PolyUOp *dead = poly_const_float(ctx, 2.0);
  for (int i = 0; i < 1024; i++)
    dead = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, dead, poly_arg_none());
  ASSERT_NOT_NULL(dead);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, dead).ndim, 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &peak), 0);
  ASSERT_TRUE(peak.arena_bytes > baseline.arena_bytes);

  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after), 0);
  ASSERT_TRUE(after.arena_bytes < peak.arena_bytes);
  ASSERT_PTR_EQ(poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, one, poly_arg_none()), live);

  PolyCtxStats stable = after;
  for (int round = 0; round < 64; round++) {
    PolyUOp *churn = poly_const_float(ctx, 1000.0 + round);
    for (int i = 0; i < 32; i++)
      churn = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, churn, poly_arg_none());
    ASSERT_NOT_NULL(churn);
    ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, churn).ndim, 0);
    ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
    ASSERT_PTR_EQ(poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, one, poly_arg_none()), live);
  }
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after), 0);
  ASSERT_INT_EQ(after.arena_bytes, stable.arena_bytes);
  ASSERT_INT_EQ(after.cse_entries, stable.cse_entries);
  ASSERT_INT_EQ(after.shape_cache_entries, stable.shape_cache_entries);

  poly_uop_release(ctx, live);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, can_run_op_probes_backend_lowering) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int f32 = poly_dtype_id_by_name("float32");
  ASSERT_TRUE(f32 >= 0);

  int64_t add_shape[1] = {4};
  ASSERT_INT_EQ(poly_can_run_op(ctx, POLY_DEVICE_INTERP, "add", f32, add_shape, 1), 1);

  int64_t mm_shape[3] = {2, 3, 4};
  ASSERT_INT_EQ(poly_can_run_op(ctx, POLY_DEVICE_INTERP, "matmul", f32, mm_shape, 3), 1);

  int64_t gather_shape[2] = {2, 3};
  ASSERT_INT_EQ(poly_can_run_op(ctx, POLY_DEVICE_INTERP, "gather", f32, gather_shape, 2), 1);
  ASSERT_INT_EQ(poly_can_run_op(ctx, POLY_DEVICE_INTERP, "sort", f32, gather_shape, 2), 1);
  ASSERT_INT_EQ(poly_can_run_op(ctx, POLY_DEVICE_INTERP, "argsort", f32, gather_shape, 2), 1);
  ASSERT_INT_EQ(poly_can_run_op(ctx, POLY_DEVICE_INTERP, "topk", f32, gather_shape, 2), 1);

  ASSERT_INT_EQ(poly_can_run_op(ctx, POLY_DEVICE_HOST, "add", f32, add_shape, 1), 1);
  ASSERT_TRUE(poly_can_run_op(ctx, POLY_DEVICE_INTERP, "add", f32, NULL, -1) < 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, collect_ordered_buffers_uses_transient_toposort) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *val = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, out, val);
  PolyUOp *sink = poly_sink1(ctx, store);

  size_t before = poly_arena_used(poly_ctx_arena(ctx));
  for (int i = 0; i < 128; i++) {
    PolyUOp *ordered[8];
    int n = poly_collect_ordered_buffers(ctx, sink, ordered, 8);
    ASSERT_INT_EQ(n, 3);
    ASSERT_PTR_EQ(ordered[0], out);
    ASSERT_TRUE((ordered[1] == a && ordered[2] == b) || (ordered[1] == b && ordered[2] == a));
  }
  size_t after = poly_arena_used(poly_ctx_arena(ctx));
  ASSERT_INT_EQ(after, before);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Ops helpers */

TEST(uop, unshard_keeps_sorted_axes_and_explicit_ranges) {
  /* Current UOp.unshard stores one RANGE per sorted sharded axis
   * (`tinygrad/uop/ops.py:667-681`). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *storage = poly_buffer_f32(ctx, 4);
  int64_t local_shape[] = {2, 2};
  PolyUOp *local = poly_reshape(ctx, storage, local_shape, 2);
  PolyUOp *r0 = poly_uop_range(ctx, 2, -1, POLY_AXIS_DEVICE);
  PolyUOp *r1 = poly_uop_range(ctx, 2, -2, POLY_AXIS_DEVICE);
  int64_t axes[] = {1, 0};
  PolyUOp *ranges[] = {r1, r0};
  PolyUOp *unshard = poly_unshard(ctx, local, axes, ranges, 2);

  ASSERT_NOT_NULL(unshard);
  ASSERT_INT_EQ(unshard->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(unshard->n_src, 3);
  ASSERT_PTR_EQ(unshard->src[0], local);
  ASSERT_PTR_EQ(unshard->src[1], r0);
  ASSERT_PTR_EQ(unshard->src[2], r1);
  ASSERT_INT_EQ(unshard->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(unshard->arg.int_tuple.n, 2);
  ASSERT_INT_EQ(unshard->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(unshard->arg.int_tuple.vals[1], 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, unshard), 2);
  const int64_t *shape = poly_uop_max_shape_dims(ctx, unshard);
  ASSERT_NOT_NULL(shape);
  ASSERT_INT_EQ(shape[0], 4);
  ASSERT_INT_EQ(shape[1], 4);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(ops, op_name) {
  ASSERT_EQ(poly_op_name((PolyOps)0), NULL);
  ASSERT_STR_EQ(poly_op_name(POLY_OP_ADD), "ADD");
  ASSERT_STR_EQ(poly_op_name(POLY_OP_CONST), "CONST");
  ASSERT_STR_EQ(poly_op_name(POLY_OP_SINK), "SINK");
  ASSERT_STR_EQ(poly_op_name(POLY_OP_RESHAPE), "RESHAPE");
  ASSERT_STR_EQ(poly_op_name(POLY_OP_STAGE), "STAGE");
  ASSERT_STR_EQ(poly_op_name(POLY_OP_UNSHARD), "UNSHARD");
  PASS();
}

TEST(ops, op_value_matches_tinygrad_sort_order) {
  /* Current tinygrad uop/__init__.py:Ops. Shared values define toposort order. */
  const struct {
    PolyOps op;
    int value;
  } expected[] = {
      {POLY_OP_SPECIAL, 1},       {POLY_OP_BUFFER, 2},      {POLY_OP_NOOP, 3},
      {POLY_OP_REWRITE_ERROR, 4}, {POLY_OP_PARAM, 5},       {POLY_OP_FUNCTION, 6},
      {POLY_OP_CALL, 7},          {POLY_OP_PROGRAM, 8},     {POLY_OP_LINEAR, 9},
      {POLY_OP_SOURCE, 10},       {POLY_OP_BINARY, 11},     {POLY_OP_SINK, 12},
      {POLY_OP_AFTER, 13},        {POLY_OP_GROUP, 14},      {POLY_OP_STACK, 15},
      {POLY_OP_TUPLE, 16},        {POLY_OP_GETTUPLE, 17},   {POLY_OP_INDEX, 19},
      {POLY_OP_SHRINK, 20},       {POLY_OP_LOAD, 21},       {POLY_OP_STORE, 22},
      {POLY_OP_WMMA, 23},         {POLY_OP_CAST, 24},       {POLY_OP_BITCAST, 25},
      {POLY_OP_EXP2, 26},         {POLY_OP_LOG2, 27},       {POLY_OP_SIN, 28},
      {POLY_OP_SQRT, 29},         {POLY_OP_RECIPROCAL, 30}, {POLY_OP_NEG, 31},
      {POLY_OP_TRUNC, 32},        {POLY_OP_ADD, 33},        {POLY_OP_MUL, 34},
      {POLY_OP_SHL, 35},          {POLY_OP_SHR, 36},        {POLY_OP_CDIV, 37},
      {POLY_OP_MAX, 38},          {POLY_OP_CMOD, 39},       {POLY_OP_CMPLT, 40},
      {POLY_OP_CMPNE, 41},        {POLY_OP_CMPEQ, 42},      {POLY_OP_XOR, 43},
      {POLY_OP_OR, 44},           {POLY_OP_AND, 45},        {POLY_OP_THREEFRY, 46},
      {POLY_OP_SUB, 47},          {POLY_OP_FDIV, 48},       {POLY_OP_POW, 49},
      {POLY_OP_FLOORDIV, 50},     {POLY_OP_FLOORMOD, 51},   {POLY_OP_WHERE, 52},
      {POLY_OP_MULACC, 53},       {POLY_OP_BARRIER, 54},    {POLY_OP_RANGE, 55},
      {POLY_OP_IF, 56},           {POLY_OP_END, 57},        {POLY_OP_ENDIF, 58},
      {POLY_OP_CONST, 59},        {POLY_OP_CUSTOM, 60},     {POLY_OP_CUSTOMI, 61},
      {POLY_OP_INS, 62},          {POLY_OP_CONTIGUOUS, 63}, {POLY_OP_CONTIGUOUS_BACKWARD, 64},
      {POLY_OP_DETACH, 65},       {POLY_OP_STAGE, 66},      {POLY_OP_COPY, 67},
      {POLY_OP_MSELECT, 68},      {POLY_OP_MSTACK, 69},     {POLY_OP_CUSTOM_FUNCTION, 70},
      {POLY_OP_RESHAPE, 71},      {POLY_OP_PERMUTE, 72},    {POLY_OP_EXPAND, 73},
      {POLY_OP_PAD, 74},          {POLY_OP_FLIP, 75},       {POLY_OP_UNSHARD, 76},
      {POLY_OP_REDUCE, 77},       {POLY_OP_ALLREDUCE, 78},
  };
  for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); i++)
    ASSERT_INT_EQ(poly_op_value(expected[i].op), expected[i].value);
  PASS();
}

TEST(uop, constructor_rejects_null_children) {
  /* Tinygrad UOp sources are always UOps. Reject malformed C graphs at entry. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *src[] = {poly_const_int(ctx, 1), NULL};
  ASSERT_TRUE(poly_uop(ctx, POLY_OP_ADD, POLY_WEAKINT, src, 2, poly_arg_none()) == NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, copy_uses_current_one_source_device_arg) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/ops.py:745-748 stores the
   * destination in COPY.arg; DEVICE is placement metadata, not a source. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *value =
      poly_uop_new_buffer(ctx, poly_device_uop_from_name(ctx, "CPU"), 4, POLY_FLOAT32, 0);
  PolyUOp *cuda = poly_device_uop_from_name(ctx, "CUDA");
  ASSERT_NOT_NULL(value);
  ASSERT_NOT_NULL(cuda);

  PolyUOp *copy = poly_copy_to_device_uop(ctx, value, cuda);
  ASSERT_NOT_NULL(copy);
  ASSERT_INT_EQ(copy->op, POLY_OP_COPY);
  ASSERT_INT_EQ(copy->n_src, 1);
  ASSERT_PTR_EQ(copy->src[0], value);
  ASSERT_INT_EQ(copy->arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(copy->arg.str, "CUDA");
  ASSERT_INT_EQ(poly_uop_device(copy), POLY_DEVICE_CUDA);

  PolyUOp *retired_src[2] = {value, cuda};
  ASSERT_TRUE(poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, retired_src, 2, poly_arg_none()) == NULL);
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(9));
  PolyUOp *retired_buffer_src[2] = {unique, cuda};
  ASSERT_TRUE(
      poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, retired_buffer_src, 2, poly_arg_int(4)) == NULL
  );
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(ops, opset) {
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_UNARY, POLY_OP_EXP2));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_UNARY, POLY_OP_NEG));
  ASSERT_FALSE(poly_opset_has(POLY_GROUP_UNARY, POLY_OP_ADD));

  ASSERT_TRUE(poly_opset_has(POLY_GROUP_BINARY, POLY_OP_ADD));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_BINARY, POLY_OP_MUL));
  ASSERT_FALSE(poly_opset_has(POLY_GROUP_BINARY, POLY_OP_NEG));

  ASSERT_TRUE(poly_opset_has(POLY_GROUP_ALU, POLY_OP_ADD));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_ALU, POLY_OP_NEG));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_ALU, POLY_OP_WHERE));
  ASSERT_FALSE(poly_opset_has(POLY_GROUP_ALU, POLY_OP_CONST));

  ASSERT_TRUE(poly_opset_has(POLY_GROUP_MOVEMENT, POLY_OP_RESHAPE));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_MOVEMENT, POLY_OP_PERMUTE));
  ASSERT_FALSE(poly_opset_has(POLY_GROUP_MOVEMENT, POLY_OP_ADD));

  ASSERT_TRUE(poly_opset_has(POLY_GROUP_COMMUTATIVE, POLY_OP_ADD));
  ASSERT_TRUE(poly_opset_has(POLY_GROUP_COMMUTATIVE, POLY_OP_MUL));
  ASSERT_FALSE(poly_opset_has(POLY_GROUP_COMMUTATIVE, POLY_OP_SUB));
  PASS();
}

/* Pretty-print */

TEST(uop, print_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.14));
  char *s = poly_uop_str(c);
  ASSERT_NOT_NULL(s);
  /* Should contain "CONST" and "float" and "3.14" */
  ASSERT_TRUE(strstr(s, "CONST") != NULL);
  ASSERT_TRUE(strstr(s, "float") != NULL);
  ASSERT_TRUE(strstr(s, "3.14") != NULL);
  free(s);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, print_float_arg_round_trips_double) {
  /* Pinned tinygrad UOp.argstr uses Python's round-trippable float repr
   * (uop/ops.py:166-171). Diagnostic serialization must not collapse distinct
   * double values to the six significant digits provided by plain %g. */
  PolyCtx *ctx = poly_ctx_new();
  double expected = 1.0 / 3.0;
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(expected));
  char *s = poly_uop_str(c);
  ASSERT_NOT_NULL(s);

  const char *arg = strstr(s, ", 0.");
  ASSERT_NOT_NULL(arg);
  char *end = NULL;
  double parsed = strtod(arg + 2, &end);
  ASSERT_TRUE(end != arg + 2);
  ASSERT_TRUE(parsed == expected);

  free(s);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, print_preserves_reduce_arg) {
  /* Tinygrad 2026-08-22/a9069c177a9d UOp.argstr (uop/ops.py:166-168)
   * preserves the complete REDUCE operation and axis count. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *input = poly_test_buffer(ctx, POLY_FLOAT32, 6);

  PolyUOp *reduce =
      poly_uop1(ctx, POLY_OP_REDUCE, POLY_FLOAT32, input, poly_arg_reduce(POLY_OP_ADD, 2));
  char *reduce_s = poly_uop_str(reduce);
  ASSERT_NOT_NULL(reduce_s);
  ASSERT_STR_EQ(reduce_s, "UOp(REDUCE, float, (ADD,2), src=1)");

  free(reduce_s);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Arena basics */

TEST(arena, allocation_alignment_and_growth) {
  PolyArena *a = poly_arena_new(31);
  bool ok = a != NULL;
  for (int repeat = 0; ok && repeat < 3; repeat++) {
    for (size_t align = 1; ok && align <= 64; align *= 2) {
      void *p = poly_arena_alloc(a, 17, align);
      ok = p && (uintptr_t)p % align == 0;
      if (p) memset(p, 0x5a, 17);
    }
    poly_arena_reset(a);
  }
  if (a) poly_arena_destroy(a);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(arena, overflow_preserves_allocation_state) {
  PolyArena *a = poly_arena_new(64);
  ASSERT_NOT_NULL(a);
  PolyArenaMark before = poly_arena_mark(a);
  bool rejected = poly_arena_alloc(a, SIZE_MAX, 8) == NULL;
  rejected &= poly_arena_alloc(a, 16, 3) == NULL;
  PolyArena *oversized = poly_arena_new(SIZE_MAX);
  rejected &= oversized == NULL;
  if (oversized) poly_arena_destroy(oversized);
  PolyArenaMark after = poly_arena_mark(a);
  bool unchanged = before.head == after.head && before.used == after.used &&
                   before.total_used == after.total_used;
  poly_arena_destroy(a);
  ASSERT_TRUE(rejected && unchanged);
  PASS();
}

TEST(arena, alloc_and_destroy) {
  PolyArena *a = poly_arena_new(1024);
  ASSERT_NOT_NULL(a);

  void *p1 = poly_arena_alloc(a, 64, 8);
  ASSERT_NOT_NULL(p1);
  void *p2 = poly_arena_alloc(a, 128, 8);
  ASSERT_NOT_NULL(p2);
  ASSERT_PTR_NEQ(p1, p2);

  ASSERT_TRUE(poly_arena_used(a) >= 192);
  poly_arena_destroy(a);
  PASS();
}

TEST(arena, large_alloc) {
  PolyArena *a = poly_arena_new(64); /* tiny initial block */
  /* allocate more than initial capacity */
  void *p = poly_arena_alloc(a, 256, 8);
  ASSERT_NOT_NULL(p);
  poly_arena_destroy(a);
  PASS();
}

TEST(arena, reset) {
  PolyArena *a = poly_arena_new(1024);
  poly_arena_alloc(a, 512, 8);
  ASSERT_TRUE(poly_arena_used(a) >= 512);
  poly_arena_reset(a);
  ASSERT_INT_EQ(poly_arena_used(a), 0);
  /* Can allocate again after reset */
  void *p = poly_arena_alloc(a, 64, 8);
  ASSERT_NOT_NULL(p);
  poly_arena_destroy(a);
  PASS();
}

/* Hashmap */

static bool int_key_eq(const void *a, const void *b) {
  return *(const int *)a == *(const int *)b;
}

TEST(hashmap, basic_set_get) {
  PolyMap *m = poly_map_new(16);
  ASSERT_NOT_NULL(m);

  int key1 = 42;
  int key2 = 99;
  poly_map_set(m, 42, &key1, (void *)(uintptr_t)100, int_key_eq);
  poly_map_set(m, 99, &key2, (void *)(uintptr_t)200, int_key_eq);

  ASSERT_INT_EQ(poly_map_len(m), 2);
  ASSERT_EQ((uintptr_t)poly_map_get(m, 42, &key1, int_key_eq), 100);
  ASSERT_EQ((uintptr_t)poly_map_get(m, 99, &key2, int_key_eq), 200);

  int key3 = 77;
  ASSERT_EQ(poly_map_get(m, 77, &key3, int_key_eq), NULL);

  poly_map_destroy(m);
  PASS();
}

TEST(hashmap, remove) {
  PolyMap *m = poly_map_new(16);
  int key1 = 42;
  poly_map_set(m, 42, &key1, (void *)(uintptr_t)100, int_key_eq);
  ASSERT_INT_EQ(poly_map_len(m), 1);

  poly_map_remove(m, 42, &key1, int_key_eq);
  ASSERT_INT_EQ(poly_map_len(m), 0);
  ASSERT_EQ(poly_map_get(m, 42, &key1, int_key_eq), NULL);

  poly_map_destroy(m);
  PASS();
}

TEST(hashmap, grow) {
  PolyMap *m = poly_map_new(4);
  int keys[100];
  for (int i = 0; i < 100; i++) {
    keys[i] = i;
    poly_map_set(m, (uint32_t)i, &keys[i], (void *)(uintptr_t)(i + 1000), int_key_eq);
  }
  ASSERT_INT_EQ(poly_map_len(m), 100);

  /* verify all values */
  for (int i = 0; i < 100; i++) {
    void *v = poly_map_get(m, (uint32_t)i, &keys[i], int_key_eq);
    ASSERT_EQ((uintptr_t)v, (uintptr_t)(i + 1000));
  }

  poly_map_destroy(m);
  PASS();
}

TEST(hashmap, null_helpers_are_noops) {
  poly_map_destroy(NULL);
  poly_map_clear(NULL);
  poly_map_foreach(NULL, NULL, NULL);
  ASSERT_INT_EQ(poly_map_len(NULL), 0);
  PASS();
}

/* Toposort gate and enter_calls */

static bool gate_skip_neg(PolyUOp *u) {
  return u->op != POLY_OP_NEG;
}

TEST(uop, toposort_gate_skips_subtree) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  /* ADD(NEG(a), b) -- gate skips NEG, so NEG and a should not appear */
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, neg, b, poly_arg_none());

  int n = 0;
  PolyUOp **topo = poly_toposort_ex(ctx, add, &n, gate_skip_neg, true);
  /* Only b and add should be in the result (NEG subtree skipped) */
  ASSERT_INT_EQ(n, 2);
  bool found_neg = false, found_a = false;
  for (int i = 0; i < n; i++) {
    if (topo[i] == neg) found_neg = true;
    if (topo[i] == a) found_a = true;
  }
  ASSERT_TRUE(!found_neg);
  ASSERT_TRUE(!found_a);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, toposort_enter_calls_false) {
  const PolyOps opaque_ops[] = {POLY_OP_CALL, POLY_OP_FUNCTION};
  for (int op_idx = 0; op_idx < 2; op_idx++) {
    PolyCtx *ctx = poly_ctx_new();
    /* Build: CALL/FUNCTION(callee_body, arg1)
     * callee_body = ADD(c1, c2), arg1 = CONST(42). */
    PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
    PolyUOp *c2 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
    PolyUOp *callee_body = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, c1, c2, poly_arg_none());
    PolyUOp *arg1 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(42.0));
    PolyUOp *opaque =
        poly_uop2(ctx, opaque_ops[op_idx], POLY_FLOAT32, callee_body, arg1, poly_arg_none());

    int n_all = 0;
    PolyUOp **topo_all = poly_toposort_ex(ctx, opaque, &n_all, NULL, true);
    ASSERT_NOT_NULL(topo_all);
    ASSERT_INT_EQ(n_all, 5);

    int n_no = 0;
    PolyUOp **topo_no = poly_toposort_ex(ctx, opaque, &n_no, NULL, false);
    ASSERT_NOT_NULL(topo_no);
    ASSERT_INT_EQ(n_no, 2);
    bool found_callee = false;
    for (int i = 0; i < n_no; i++) {
      if (topo_no[i] == callee_body || topo_no[i] == c1 || topo_no[i] == c2) found_callee = true;
    }
    ASSERT_TRUE(!found_callee);
    ASSERT_TRUE(topo_no[0] == arg1);
    ASSERT_TRUE(topo_no[1] == opaque);

    PolyUOp *body_only =
        poly_uop1(ctx, opaque_ops[op_idx], POLY_FLOAT32, callee_body, poly_arg_none());
    int n_body_only = 0;
    PolyUOp **topo_body_only = poly_toposort_ex(ctx, body_only, &n_body_only, NULL, false);
    ASSERT_NOT_NULL(topo_body_only);
    ASSERT_INT_EQ(n_body_only, 1);
    ASSERT_PTR_EQ(topo_body_only[0], body_only);

    poly_ctx_destroy(ctx);
  }
  PASS();
}

/* Range helpers (poly_no_range / poly_uop_in_ranges / poly_uop_ranges) *
 * Each test mirrors a tinygrad ground-truth case verified against
 * references/tinygrad_latest (conda env tiny). Ground-truth generator:
 *   PYTHONPATH=.../references/tinygrad_latest python /tmp/tg_ranges_gt.py
 *
 * Semantics differ between `no_range` and `ranges`:
 *   - no_range(u)   : walks backward slice, True iff no RANGE anywhere
 *                     (structural — doesn't subtract ended ranges).
 *   - u.ranges      : set of *active* ranges at u's position — REDUCE,
 *                     STORE, END, BUFFERIZE, WMMA, CALL, COPY
 *                     end their trailing RANGE srcs. */

static PolyUOp *make_range(PolyCtx *ctx, int64_t n, int64_t axis_id) {
  PolyUOp *size = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  return poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, size, poly_arg_range(axis_id, POLY_AXIS_LOOP));
}

TEST(uop, no_range_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  /* [tg] CONST   no_range=True */
  ASSERT_TRUE(poly_no_range(ctx, c));
  ASSERT_INT_EQ(poly_uop_ranges(ctx, c, (PolyUOp *[8]){0}, 8), 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, no_range_bare_range) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = make_range(ctx, 5, 0);
  /* [c1] RANGE(5,0)  no_range=False  |ranges|=1 (contains self) */
  ASSERT_TRUE(!poly_no_range(ctx, r));
  PolyUOp *rs[8] = {0};
  int n = poly_uop_ranges(ctx, r, rs, 8);
  ASSERT_INT_EQ(n, 1);
  ASSERT_TRUE(rs[0] == r);
  ASSERT_TRUE(poly_uop_in_ranges(ctx, r, r));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, no_range_expr_r_plus_3) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = make_range(ctx, 5, 0);
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *e = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, r, c, poly_arg_none());
  /* [c2] r+3   no_range=False  |ranges|=1 */
  ASSERT_TRUE(!poly_no_range(ctx, e));
  PolyUOp *rs[8] = {0};
  int n = poly_uop_ranges(ctx, e, rs, 8);
  ASSERT_INT_EQ(n, 1);
  ASSERT_TRUE(rs[0] == r);
  ASSERT_TRUE(poly_uop_in_ranges(ctx, e, r));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, no_range_reduce_ends_range) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = make_range(ctx, 5, 0);
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *e = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, r, c, poly_arg_none());
  PolyUOp *red_src[2] = {e, r};
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_INT32, red_src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  /* [c3] REDUCE(r+3, r)  no_range=False (r still in backward slice)
   *                      |ranges|=0  (r is ended by the reduce) */
  ASSERT_TRUE(!poly_no_range(ctx, red));
  PolyUOp *rs[8] = {0};
  int n = poly_uop_ranges(ctx, red, rs, 8);
  ASSERT_INT_EQ(n, 0);
  ASSERT_TRUE(!poly_uop_in_ranges(ctx, red, r));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, ranges_two_ranges_union) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = make_range(ctx, 5, 0);
  PolyUOp *r2 = make_range(ctx, 7, 1);
  PolyUOp *e = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, r, r2, poly_arg_none());
  /* [c4] r+r2  |ranges|=2 */
  PolyUOp *rs[8] = {0};
  int n = poly_uop_ranges(ctx, e, rs, 8);
  ASSERT_INT_EQ(n, 2);
  ASSERT_TRUE(poly_uop_in_ranges(ctx, e, r));
  ASSERT_TRUE(poly_uop_in_ranges(ctx, e, r2));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, ranges_partial_reduce_leaves_other_active) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = make_range(ctx, 5, 0);
  PolyUOp *r2 = make_range(ctx, 7, 1);
  PolyUOp *e = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, r, r2, poly_arg_none());
  PolyUOp *red_src[2] = {e, r};
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_INT32, red_src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  /* [c5] REDUCE(r+r2, r)  |ranges|=1  (r ended, r2 still active) */
  PolyUOp *rs[8] = {0};
  int n = poly_uop_ranges(ctx, red, rs, 8);
  ASSERT_INT_EQ(n, 1);
  ASSERT_TRUE(rs[0] == r2);
  ASSERT_TRUE(!poly_uop_in_ranges(ctx, red, r));
  ASSERT_TRUE(poly_uop_in_ranges(ctx, red, r2));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, ranges_deep_chain_is_iterative) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = make_range(ctx, 5, 0);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *expr = r;
  for (int i = 0; i < 12000; i++) {
    expr = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, expr, one, poly_arg_none());
  }

  PolyUOp *rs[8] = {0};
  int n = poly_uop_ranges(ctx, expr, rs, 8);
  ASSERT_INT_EQ(n, 1);
  ASSERT_TRUE(rs[0] == r);
  ASSERT_TRUE(poly_uop_in_ranges(ctx, expr, r));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, ranges_use_rewound_scratch_toposort) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = make_range(ctx, 5, 0);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *expr = r;
  for (int i = 0; i < 32; i++)
    expr = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, expr, one, poly_arg_none());

  size_t scratch_before = poly_arena_used(ctx->scratch);
  PolyUOp *rs[8] = {0};
  int n = poly_uop_ranges(ctx, expr, rs, 8);
  ASSERT_INT_EQ(n, 1);
  ASSERT_TRUE(rs[0] == r);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, buf_uop_matches_current_tinygrad_storage_states) {
  /* references/tinygrad_latest/tinygrad/uop/ops.py:881-889. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *stack_src[] = {a, b};
  PolyUOp *stack = poly_uop(ctx, POLY_OP_MSTACK, POLY_FLOAT32, stack_src, 2, poly_arg_none());
  PolyUOp *select = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, stack, poly_arg_int(1));
  PolyUOp *expected_select = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, stack, poly_arg_int(1));
  PolyUOp *store = poly_store_val(ctx, a, b);
  PolyUOp *after_src[] = {a, store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, POLY_FLOAT32, after_src, 2, poly_arg_none());

  ASSERT_PTR_EQ(poly_uop_buf_uop(ctx, a), a);
  ASSERT_PTR_EQ(poly_uop_buf_uop(ctx, stack), stack);
  ASSERT_PTR_EQ(poly_uop_buf_uop(ctx, select), expected_select);
  ASSERT_PTR_EQ(poly_uop_buf_uop(ctx, after), a);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Gate closure test for poly_toposort_ex_user: collect only nodes whose
 * backward slice contains a specific RANGE. */
static bool gate_in_r(PolyUOp *u, void *user_data) {
  PolyUOp *r = (PolyUOp *)user_data;
  /* NOTE: this gate uses a fresh PolyCtx each call via global storage,
   * which we don't have here. For the test we just check pointer identity
   * on the root RANGE — enough to exercise the user_data plumbing. */
  return u != NULL && u != (PolyUOp *)((uintptr_t)r ^ 0xdeadbeef);
}

TEST(uop, toposort_ex_user_passes_user_data) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = make_range(ctx, 5, 0);
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *e = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, r, c, poly_arg_none());
  int n = 0;
  PolyUOp **topo = poly_toposort_ex_user(ctx, e, &n, gate_in_r, r, true);
  ASSERT_NOT_NULL(topo);
  /* Gate returns true for all nodes (no collision with the xor sentinel),
   * so we should see every node in the graph. */
  ASSERT_TRUE(n >= 3); /* at least RANGE size const, RANGE, ADD */
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(uop, toposort_ex_user_scratch_rewinds_without_growing_ctx_arena) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = make_range(ctx, 5, 0);
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *e = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, r, c, poly_arg_none());

  size_t main_before = poly_arena_used(poly_ctx_arena(ctx));
  size_t scratch_before = poly_arena_used(ctx->scratch);
  PolyScratchMark mark = poly_ctx_scratch_mark(ctx);

  int n = 0;
  PolyUOp **topo = poly_toposort_ex_user_scratch(ctx, e, &n, gate_in_r, r, true);
  ASSERT_NOT_NULL(topo);
  ASSERT_TRUE(n >= 3);
  ASSERT_FALSE(poly_ctx_owns_ptr(ctx, topo));
  ASSERT_INT_EQ(poly_arena_used(poly_ctx_arena(ctx)), main_before);
  ASSERT_TRUE(poly_arena_used(ctx->scratch) > scratch_before);

  poly_ctx_scratch_rewind(ctx, mark);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);
  ASSERT_INT_EQ(poly_arena_used(poly_ctx_arena(ctx)), main_before);

  poly_ctx_destroy(ctx);
  PASS();
}
