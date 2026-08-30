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

#include "../src/ctx.h"
#include "../src/codegen/codegen.h"
#include "../src/engine/realize.h"
#include "../src/schedule/rangeify.h"
#include "../src/schedule/schedule.h"
#include "../src/tensor.h"

typedef void (*TestFn)(int *passed, int *failed);

typedef struct {
  const char *suite;
  const char *name;
  TestFn fn;
  unsigned flags;
} TestEntry;

/* Test-only spelling of Tinygrad's full_rewrite_to_sink -> do_linearize boundary. */
static inline PolyUOp **poly_test_full_rewrite_and_linearize_ex(
    PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts, int *n_out
) {
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  if (!rewritten) {
    if (n_out) *n_out = 0;
    return NULL;
  }
  return poly_do_linearize(ctx, rewritten, n_out);
}

static inline PolyUOp **poly_test_full_rewrite_and_linearize(
    PolyCtx *ctx, PolyUOp *sink, int *n_out
) {
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  if (!rewritten) {
    if (n_out) *n_out = 0;
    return NULL;
  }
  return poly_do_linearize(ctx, rewritten, n_out);
}

/* Current Tinygrad UOp.new_buffer(canonicalize_device(device), ...). */
static inline PolyUOp *poly_test_buffer_on_device(
    PolyCtx *ctx, PolyDType dtype, int64_t size, PolyDevice device
) {
  if (device == POLY_DEVICE_AUTO) {
    device = poly_ctx_get_preferred_device(ctx);
    if (!poly_device_can_execute(device)) device = poly_device_default();
  }
  PolyUOp *device_uop = poly_device_uop(ctx, device);
  return device_uop
             ? poly_uop_new_buffer(
                   ctx, device_uop, size, dtype, poly_ctx_next_unique_id(ctx)
               )
             : NULL;
}

static inline PolyUOp *poly_test_buffer(
    PolyCtx *ctx, PolyDType dtype, int64_t size
) {
  return poly_test_buffer_on_device(ctx, dtype, size, POLY_DEVICE_AUTO);
}

/* Approved Polygrad portable logical-storage fixture. */
static inline PolyUOp *poly_test_logical_buffer(
    PolyCtx *ctx, PolyDType dtype, int64_t size
) {
  return poly_uop_new_logical_buffer(ctx, dtype, size);
}

/* Test-only adapter for tinygrad CreationMixin.empty with UOp dimensions. */
static inline PolyTensor *poly_test_tensor_empty_var_on_device(
    PolyCtx *ctx,
    PolyDType dtype,
    PolyUOp *batch_var,
    const int64_t *inner_dims,
    int n_inner,
    PolyDevice device
) {
  if (!ctx || !batch_var || n_inner < 0 || n_inner >= POLY_MAX_DIMS ||
      (n_inner > 0 && !inner_dims))
    return NULL;
  if (device == POLY_DEVICE_AUTO) {
    device = poly_ctx_get_preferred_device(ctx);
    if (!poly_device_can_execute(device)) device = poly_device_default();
  }
  PolyUOp *shape[POLY_MAX_DIMS];
  shape[0] = batch_var;
  for (int i = 0; i < n_inner; i++)
    shape[i + 1] = poly_uop0(
        ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(inner_dims[i])
    );
  return poly_tensor_empty_uop(ctx, dtype, shape, n_inner + 1, device);
}

static inline PolyUOp *poly_test_buffer_var(
    PolyCtx *ctx,
    PolyDType dtype,
    PolyUOp *batch_var,
    const int64_t *inner_dims,
    int n_inner
) {
  PolyTensor *tensor = poly_test_tensor_empty_var_on_device(
      ctx, dtype, batch_var, inner_dims, n_inner, POLY_DEVICE_AUTO
  );
  return tensor ? poly_tensor_uop_physical(tensor) : NULL;
}

static inline PolyUOp *poly_test_buffer_var_by_id(
    PolyCtx *ctx,
    int dtype_id,
    PolyUOp *batch_var,
    const int64_t *inner_dims,
    int n_inner,
    PolyDevice device
) {
  PolyDType dtype;
  if (!poly_dtype_by_id(dtype_id, &dtype)) return NULL;
  PolyTensor *tensor = poly_test_tensor_empty_var_on_device(
      ctx, dtype, batch_var, inner_dims, n_inner, device
  );
  return tensor ? poly_tensor_uop_physical(tensor) : NULL;
}

/* Current Tinygrad transform_to_call -> create_linear_with_vars boundary for
 * explicit C effect sinks. */
static inline PolyUOp *poly_test_create_linear(PolyCtx *ctx, PolyUOp *sink) {
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_effect_sink(ctx, sink, &vars, &n_vars);
  free(vars);
  return linear;
}

/* tinygrad@2026-08-22/a9069c177a9d schedule/rangeify.py:83-89 stores
 * through a value-shaped view while the allocator owns the flat BUFFER. */
static inline PolyUOp *poly_test_store_to_buffer(
    PolyCtx *ctx, PolyUOp *buffer, PolyUOp *value
) {
  if (!ctx || !buffer || !value) return NULL;
  int ndim = poly_uop_ndim(ctx, value);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  PolyUOp *shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    shape[i] = poly_uop_shape_dim(ctx, value, i);
    if (!shape[i]) return NULL;
  }
  PolyUOp *target = poly_reshape_uop(ctx, buffer, shape, ndim);
  return target ? poly_store_val(ctx, target, value) : NULL;
}

/* Current Tinygrad compiler kernels require SINK(arg=KernelInfo). */
static inline PolyUOp *poly_test_kernel_sink(
    PolyCtx *ctx, PolyUOp **src, int n_src, const char *name
) {
  PolyKernelInfo info = {.name = name};
  return poly_uop(
      ctx, POLY_OP_SINK, POLY_VOID, src, n_src, poly_arg_kernel_info(&info)
  );
}

/* tinygrad@2026-08-22/a9069c177a9d UOp.placeholder: final-program storage
 * parameters are scalar value UOps with their flat extent in src[0]. */
static inline PolyUOp *poly_test_program_param(
    PolyCtx *ctx, PolyDType dtype, int64_t numel, int slot
) {
  int64_t shape[] = {numel};
  return poly_uop_placeholder(
      ctx, shape, 1, dtype, slot, POLY_ADDR_GLOBAL, NULL, false
  );
}

/* Current Tinygrad UOp.param: scalar dtype plus UOp shape and ParamArg storage metadata. */
static inline PolyUOp *poly_test_uop_param(
    PolyCtx *ctx,
    PolyDType dtype,
    int64_t numel,
    int slot,
    PolyAddrSpace addrspace
) {
  PolyUOp *shape = numel < 0
                       ? poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none())
                       : poly_const_int(ctx, numel);
  PolyParamArg arg = {.slot = slot, .dtype = dtype, .addrspace = addrspace};
  PolyOps op = addrspace == POLY_ADDR_GLOBAL ? POLY_OP_PARAM : POLY_OP_BUFFER;
  return shape ? poly_uop1(ctx, op, dtype, shape, poly_arg_param(&arg)) : NULL;
}

static inline PolyUOp *poly_test_linear_call_body(PolyUOp *linear, int index) {
  if (!linear || linear->op != POLY_OP_LINEAR || index < 0 || index >= linear->n_src)
    return NULL;
  PolyUOp *call = linear->src[index];
  return call && call->op == POLY_OP_CALL && call->n_src > 0 ? call->src[0] : NULL;
}

static inline PolyUOp *poly_test_linear_call(PolyUOp *linear, int index) {
  return linear && linear->op == POLY_OP_LINEAR && index >= 0 && index < linear->n_src &&
                 linear->src[index] && linear->src[index]->op == POLY_OP_CALL
             ? linear->src[index]
             : NULL;
}

static inline bool poly_test_linear_call_is_copy(PolyUOp *linear, int index) {
  PolyUOp *body = poly_test_linear_call_body(linear, index);
  return body && body->op == POLY_OP_COPY;
}

static inline int poly_test_linear_call_n_buffers(PolyUOp *linear, int index) {
  PolyUOp *call = poly_test_linear_call(linear, index);
  return call ? call->n_src - 1 : -1;
}

static inline PolyUOp *poly_test_linear_call_buffer(
    PolyUOp *linear, int call_index, int buffer_index
) {
  PolyUOp *call = poly_test_linear_call(linear, call_index);
  return call && buffer_index >= 0 && buffer_index + 1 < call->n_src
             ? call->src[buffer_index + 1]
             : NULL;
}

static inline PolyUOp *poly_test_linear_values(
    PolyCtx *ctx, PolyUOp **values, int n_values, PolyUOp **realized
) {
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear =
      poly_linear_with_vars(ctx, values, n_values, realized, &vars, &n_vars);
  free(vars);
  return linear;
}

enum {
  POLY_TEST_COMMON = 1u << 0,
};

#define MAX_TESTS 2048
extern TestEntry g_tests[MAX_TESTS];
extern int g_n_tests;

#define POLY_TEST_REGISTER(suite, name, test_flags) \
  static void test_##suite##_##name(int *_passed, int *_failed); \
  __attribute__((constructor)) \
  static void register_##suite##_##name(void) { \
    g_tests[g_n_tests++] = (TestEntry){ #suite, #name, test_##suite##_##name, (test_flags) }; \
  } \
  static void test_##suite##_##name(int *_passed, int *_failed)

/* Backend portability is the default. Only tests that exercise one backend's
 * private renderer/runtime use TEST_BACKEND and remain additive through the
 * test-specific-* Make targets. */
#define TEST(suite, name) POLY_TEST_REGISTER(suite, name, POLY_TEST_COMMON)
#define TEST_COMMON(suite, name) TEST(suite, name)
#define TEST_BACKEND(suite, name) POLY_TEST_REGISTER(suite, name, 0)

#define PASS() do { (*_passed)++; return; } while(0)

#define FAIL(fmt, ...) do { \
  fprintf(stderr, "    FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
  (*_failed)++; return; \
} while(0)

/* SKIP: marks a test as intentionally not run yet (e.g. waiting on a
 * dependent feature). Counts as a pass but logs the reason so the
 * skipped test is visible in the runner output. */
#define SKIP(reason) do { \
  fprintf(stderr, "    SKIP %s:%d: " reason "\n", __FILE__, __LINE__); \
  (*_passed)++; return; \
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

/* Double-precision ULP distance. */
static inline int64_t poly_double_ulp_index(double f) {
  int64_t i;
  memcpy(&i, &f, sizeof(i));
  if (i < 0) i = (int64_t)((uint64_t)0x8000000000000000ULL - (uint64_t)i);
  return i;
}

#define ASSERT_DOUBLE_NAN(a) do { \
  double _a = (double)(a); \
  if (!isnan(_a)) FAIL("%s = %.17g, expected NaN", #a, _a); \
} while(0)

#define ASSERT_DOUBLE_INF(a, sign) do { \
  double _a = (double)(a); \
  if (!isinf(_a) || ((sign) > 0 && _a < 0) || ((sign) < 0 && _a > 0)) \
    FAIL("%s = %.17g, expected %sinf", #a, _a, (sign) < 0 ? "-" : "+"); \
} while(0)

#define ASSERT_DOUBLE_ABS(a, b, tol) do { \
  double _da = (double)(a), _db = (double)(b); \
  if (isnan(_da) && isnan(_db)) { /* ok */ } \
  else if (isnan(_da) || isnan(_db)) \
    FAIL("%s=%.17g, expected %.17g (NaN mismatch)", #a, _da, _db); \
  else if (fabs(_da - _db) > (double)(tol)) \
    FAIL("%s=%.17g, expected %.17g (abs err %.17g, tol %.17g)", \
      #a, _da, _db, fabs(_da - _db), (double)(tol)); \
} while(0)

#define ASSERT_DOUBLE_ULP(a, b, max_ulps) do { \
  double _da = (double)(a), _db = (double)(b); \
  if (isnan(_da) && isnan(_db)) { /* ok */ } \
  else if (isnan(_da) || isnan(_db)) \
    FAIL("%s=%.17g, expected %.17g (NaN mismatch)", #a, _da, _db); \
  else if (isinf(_da) || isinf(_db)) { \
    if (!(isinf(_da) && isinf(_db) && ((_da > 0) == (_db > 0)))) \
      FAIL("%s=%.17g, expected %.17g (inf mismatch)", #a, _da, _db); \
  } else { \
    int64_t _ia = poly_double_ulp_index(_da), _ib = poly_double_ulp_index(_db); \
    int64_t _d = llabs(_ia - _ib); \
    if (_d > (int64_t)(max_ulps)) \
      FAIL("%s=%.17g, expected %.17g (%lld ulps, max %lld)", \
        #a, _da, _db, (long long)_d, (long long)(max_ulps)); \
  } \
} while(0)

#define ASSERT_DOUBLE_NEAR(a, b, max_ulps, abs_tol) do { \
  double _da = (double)(a), _db = (double)(b); \
  if (isnan(_da) && isnan(_db)) { /* ok */ } \
  else if (isnan(_da) || isnan(_db)) \
    FAIL("%s=%.17g, expected %.17g (NaN mismatch)", #a, _da, _db); \
  else if (isinf(_da) && isinf(_db) && ((_da > 0) == (_db > 0))) { /* ok */ } \
  else if (isinf(_da) || isinf(_db)) \
    FAIL("%s=%.17g, expected %.17g (inf mismatch)", #a, _da, _db); \
  else if (fabs(_da - _db) <= (double)(abs_tol)) { /* within abs tol */ } \
  else { \
    int64_t _ia = poly_double_ulp_index(_da), _ib = poly_double_ulp_index(_db); \
    int64_t _d = llabs(_ia - _ib); \
    if (_d > (int64_t)(max_ulps)) \
      FAIL("%s=%.17g, expected %.17g (%lld ulps, max %lld; abs %.17g, tol %.17g)", \
        #a, _da, _db, (long long)_d, (long long)(max_ulps), \
        fabs(_da - _db), (double)(abs_tol)); \
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

typedef struct {
  PolyUOp *buffer;
  PolyBuffer handle;
} PolyTestBufferView;

#define POLY_TEST_HOST_VIEW(buffer_uop, data_ptr) \
  ((PolyTestBufferView){ \
      .buffer = (buffer_uop), \
      .handle = { \
          .ptr = (void *)(data_ptr), \
          .nbytes = 0, \
          .device = POLY_DEVICE_CPU, \
          .owned = false, \
          .allocator = NULL, \
          .src = NULL, \
          .valid = true, \
      }, \
  })

static inline size_t poly_test_buffer_nbytes(PolyCtx *ctx, PolyUOp *buf) {
  if (!buf) return 0;
  int64_t numel = -1;
  PolyShape shape = poly_uop_max_shape(ctx, buf);
  if (shape.ndim >= 0) numel = poly_shape_numel(shape);
  if (shape.ndim > 0 && shape.dims) free(shape.dims);
  if (numel < 0 && buf->arg.kind == POLY_ARG_INT) numel = buf->arg.i;
  if (numel < 0) numel = 0;
  return (size_t)numel * (size_t)poly_dtype_itemsize(buf->dtype);
}

static inline void poly_test_attach_buffer_views(
    PolyCtx *ctx, PolyTestBufferView *views, int n_views
) {
  for (int i = 0; i < n_views; i++) {
    PolyBuffer h = views[i].handle;
    if (h.nbytes == 0) h.nbytes = poly_test_buffer_nbytes(ctx, views[i].buffer);
    h.valid = true;
    poly_buffer_attach(ctx, views[i].buffer, &h);
  }
}

static inline int poly_test_readback_buffer_views(
    PolyCtx *ctx, PolyTestBufferView *views, int n_views
) {
  for (int i = 0; i < n_views; i++) {
    /* Explicit CUDA/HIP test bindings already carry device pointers and
     * perform their own backend readback. Only mirror host-addressable views. */
    if (!poly_device_is_host_addressable(views[i].handle.device)) continue;
    PolyBuffer *current = poly_buffer_get(ctx, views[i].buffer);
    if (!current || !views[i].handle.ptr) return -1;
    size_t nbytes =
        views[i].handle.nbytes ? views[i].handle.nbytes
                               : poly_test_buffer_nbytes(ctx, views[i].buffer);
    if (nbytes > 0 &&
        poly_buffer_read(ctx, views[i].buffer, views[i].handle.ptr, nbytes) != 0)
      return -1;
  }
  return 0;
}

static inline int poly_test_run_linear_buffer_views(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyTestBufferView *views,
    int n_views,
    PolyVarBinding *vars,
    int n_vars
) {
  poly_test_attach_buffer_views(ctx, views, n_views);
  /* Current Tinygrad executes LINEAR directly through run_linear
   * (tinygrad/engine/realize.py:315-323). */
  int ret = poly_run_linear(
      ctx, linear, vars, n_vars, NULL, 0, true, false, false);
  if (ret != 0) return ret;
  return poly_test_readback_buffer_views(ctx, views, n_views);
}

static inline int poly_test_realize_buffer_views(
    PolyCtx *ctx, PolyUOp *sink, PolyTestBufferView *views, int n_views
) {
  poly_test_attach_buffer_views(ctx, views, n_views);
  /* Test helpers pass schedule-ready STORE/ASSIGN sinks. Keep them on the
   * effect-sink layer instead of treating the sink itself as a tensor value. */
  int ret = poly_realize_sink(ctx, sink);
  if (ret != 0) return ret;
  /* CPU execution aliases these host views directly. Device backends own a
   * separate current residency, so mirror the observable post-run state back
   * through the public ctx buffer API before tests inspect their arrays. */
  return poly_test_readback_buffer_views(ctx, views, n_views);
}

static inline int poly_test_realize_buffer_views_vars(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyTestBufferView *views,
    int n_views,
    PolyVarBinding *vars,
    int n_vars
) {
  poly_test_attach_buffer_views(ctx, views, n_views);
  PolyVarBinding *default_vars = NULL;
  int n_default_vars = 0;
  PolyUOp *linear = poly_linear_effect_sink(
      ctx, sink, &default_vars, &n_default_vars);
  if (!linear) return -1;
  int total_vars = n_default_vars + n_vars;
  PolyVarBinding *merged = total_vars > 0
                               ? malloc((size_t)total_vars * sizeof(*merged))
                               : NULL;
  if (total_vars > 0 && !merged) {
    free(default_vars);
    return -1;
  }
  if (n_default_vars > 0)
    memcpy(merged, default_vars, (size_t)n_default_vars * sizeof(*merged));
  int n_merged = n_default_vars;
  for (int i = 0; i < n_vars; i++) {
    int found = -1;
    for (int j = 0; j < n_merged; j++)
      if (merged[j].var == vars[i].var) found = j;
    if (found >= 0)
      merged[found].value = vars[i].value;
    else
      merged[n_merged++] = vars[i];
  }
  int ret = poly_test_run_linear_buffer_views(
      ctx, linear, views, n_views, merged, n_merged);
  free(merged);
  free(default_vars);
  return ret;
}

static inline int poly_test_run_all(void) {
  int total_passed = 0, total_failed = 0;
  const char *current_suite = "";

  for (int i = 0; i < g_n_tests; i++) {
    if (strcmp(current_suite, g_tests[i].suite) != 0) {
      current_suite = g_tests[i].suite;
      printf("\n  %s:\n", current_suite);
    }

    int passed = 0, failed = 0;
    /* Emit the active test before entering it. Sanitizer crashes can happen
     * inside generated/JIT paths before PASS/FAIL is printed, and Apptainer
     * ASan sometimes reports only recursive DEADLYSIGNAL lines. */
    printf("    [RUN ] %s\n", g_tests[i].name);
    fflush(stdout);
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
