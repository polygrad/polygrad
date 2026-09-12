/*
 * Current Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py ClangRenderer.
 */

#define _POSIX_C_SOURCE 200809L

#include "renderer/cstyle.h"
#include "codegen/codegen.h"
#include "bigint.h"
#include "ctx.h"
#include "utils.h"
#include "uop/upat.h"
#include "uop/ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include <unistd.h>

/* Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py:_wmma_name. */
bool poly_wmma_name(const PolyUOp *uop, char *out, size_t out_size) {
  if (!uop || !out || out_size == 0 || uop->op != POLY_OP_WMMA ||
      uop->arg.kind != POLY_ARG_TENSOR_CORE)
    return false;
  const char *dtype_in = poly_dtype_eq(uop->arg.tensor_core.dtype_in, POLY_FLOAT16)
                             ? "half"
                             : poly_dtype_name(uop->arg.tensor_core.dtype_in);
  const char *dtype_out =
      poly_dtype_eq(uop->dtype, POLY_FLOAT16) ? "half" : poly_dtype_name(uop->dtype);
  int written = snprintf(
      out, out_size, "WMMA_%d_%d_%d_%s_%s", uop->arg.tensor_core.dims[0],
      uop->arg.tensor_core.dims[1], uop->arg.tensor_core.dims[2], dtype_in, dtype_out
  );
  if (written < 0 || (size_t)written >= out_size) return false;
  for (char *p = out; *p; p++)
    if (*p == ' ') *p = '_';
  return true;
}

/* String builder */

typedef struct {
  char *buf;
  int len;
  int cap;
} StrBuf;

static void sb_init(StrBuf *sb) {
  sb->cap = 512;
  sb->buf = malloc(sb->cap);
  sb->buf[0] = '\0';
  sb->len = 0;
}

static void sb_printf(StrBuf *sb, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int need = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);

  while (sb->len + need + 1 > sb->cap) {
    sb->cap *= 2;
    sb->buf = realloc(sb->buf, sb->cap);
  }

  va_start(ap, fmt);
  sb->len += vsnprintf(sb->buf + sb->len, sb->cap - sb->len, fmt, ap);
  va_end(ap);
}

static void sb_puts(StrBuf *sb, const char *s) {
  sb_printf(sb, "%s", s);
}

/* Pointer → string hash map (for renderer) */

typedef struct {
  PolyUOp **keys;
  char **vals;
  int cap;
} StrMap;

static void smap_init(StrMap *m, int n) {
  m->cap = (n < 4) ? 16 : n * 3;
  m->keys = calloc(m->cap, sizeof(PolyUOp *));
  m->vals = calloc(m->cap, sizeof(char *));
}

static void smap_set(StrMap *m, PolyUOp *key, char *val) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
  while (m->keys[h] && m->keys[h] != key)
    h = (h + 1) % m->cap;
  if (m->keys[h] == key) free(m->vals[h]); /* replace existing */
  m->keys[h] = key;
  m->vals[h] = val;
}

static char *smap_get(StrMap *m, PolyUOp *key) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
  while (m->keys[h]) {
    if (m->keys[h] == key) return m->vals[h];
    h = (h + 1) % m->cap;
  }
  return NULL;
}

static void smap_destroy(StrMap *m) {
  for (int i = 0; i < m->cap; i++)
    if (m->vals[i]) free(m->vals[i]);
  free(m->keys);
  free(m->vals);
}

static int cpu_thread_count(void) {
  /* tinygrad@2026-08-22/a9069c177a9d renderer/cstyle.py:261-263. */
  int n_env = poly_getenv_int("NUM_CPU_THREADS", 0);
  if (n_env > 0) return n_env;
#ifdef _SC_NPROCESSORS_ONLN
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  if (n > 0 && n < INT32_MAX) return (int)n;
#endif
  return 1;
}

PolyRendererCaps poly_c_renderer_caps(void) {
  bool has_threads = poly_getenv_flag_default("THREADS", true);
  return (PolyRendererCaps){
      .device = "CPU",
      .has_mulacc = false,
      .has_threefry = false,
      /* Pinned ClangRenderer removes EXP2, LOG2, and SIN from code_for_op;
       * the shared codegen decomposition handles them before C rendering
       * (tinygrad/renderer/cstyle.py:246-269). */
      .has_exp2 = false,
      .has_log2 = false,
      .has_sin = false,
      .has_fdiv = true,
      .supports_float16 = true,
      /* The runtime C compiler does not provide Tinygrad Clang's portable
       * x86/arm __bf16 surface; pm_dtype_decomps lowers BF16 to float32. */
      .supports_bfloat16 = false,
      .has_int64 = true,
      .has_local = false,
      .has_threads = has_threads,
      /* Clang/C rendering follows tinygrad's CStyle pm_render path, which
       * inserts masked-load alt values and scalarizes vector comparisons.
       * Direct ISA backends that keep packed integer masks set this capability
       * themselves. */
      .has_simd_int = false,
      .max_vec_width = 4,
      .max_threads = has_threads ? cpu_thread_count() : 0,
      .global_max = {has_threads ? cpu_thread_count() : 0, 0, 0},
  };
}

static PolyUOp *replace_dtype_src(PolyCtx *ctx, PolyUOp *u, PolyDType dtype, PolyUOp **src) {
  /* C construction mechanics for Tinygrad UOp.replace(dtype=..., src=...). */
  PolyArg arg =
      (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) ? poly_arg_dtype(dtype) : u->arg;
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, u->op, dtype, src, u->n_src, arg, u->tag, u->tag_arg)
             : poly_uop(ctx, u->op, dtype, src, u->n_src, arg);
}

/* Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py:76-89. */
static PolyUOp *non_native_float_where(PolyCtx *ctx, PolyUOp *w, const PolyBindings *bindings) {
  (void)bindings;
  PolyUOp *src[] = {
      w->src[0],
      poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, w->src[1], poly_arg_none()),
      poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, w->src[2], poly_arg_none()),
  };
  PolyUOp *where = replace_dtype_src(ctx, w, POLY_FLOAT32, src);
  return where ? poly_uop1(ctx, POLY_OP_CAST, w->dtype, where, poly_arg_none()) : NULL;
}

static PolyUOp *non_native_float_alu(PolyCtx *ctx, PolyUOp *u, const PolyBindings *bindings) {
  (void)bindings;
  PolyUOp **src = malloc((size_t)u->n_src * sizeof(*src));
  if (u->n_src > 0 && !src) return NULL;
  for (int i = 0; i < u->n_src; i++)
    src[i] = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, u->src[i], poly_arg_none());
  PolyUOp *alu = replace_dtype_src(ctx, u, POLY_FLOAT32, src);
  free(src);
  return alu ? poly_uop1(ctx, POLY_OP_CAST, u->dtype, alu, poly_arg_none()) : NULL;
}

static PolyUOp *non_native_float_comparison(
    PolyCtx *ctx,
    PolyUOp *alu,
    const PolyBindings *bindings
) {
  (void)bindings;
  PolyUOp *src[] = {
      poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, alu->src[0], poly_arg_none()),
      poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, alu->src[1], poly_arg_none()),
  };
  return replace_dtype_src(ctx, alu, alu->dtype, src);
}

static PolyUOp *non_native_float_cast_to(PolyCtx *ctx, PolyUOp *y, const PolyBindings *bindings) {
  (void)bindings;
  PolyUOp *x = y->src[0];
  if (poly_dtype_eq(x->dtype, POLY_FLOAT32)) return NULL;
  PolyUOp *f32 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, x, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, y->dtype, f32, poly_arg_none());
}

static PolyUOp *non_native_float_cast_from(PolyCtx *ctx, PolyUOp *x, const PolyBindings *bindings) {
  (void)bindings;
  if (poly_dtype_eq(x->dtype, POLY_FLOAT32)) return NULL;
  PolyUOp *f32 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, x->src[0], poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, x->dtype, f32, poly_arg_none());
}

static PolyPatternMatcher *create_non_native_float_pats(
    const PolyDType *dtypes,
    int n_dtypes,
    bool casting
) {
  if (!dtypes || n_dtypes <= 0) return NULL;
  PolyOpSet alu_without_where = POLY_GROUP_ALU;
  alu_without_where.bits[POLY_OP_WHERE / 64] &= ~(UINT64_C(1) << (POLY_OP_WHERE % 64));

  PolyUPat *where_src[] = {
      poly_upat_any("b"),
      poly_upat_dtype("x", (PolyDType *)dtypes, n_dtypes),
      poly_upat_dtype("y", (PolyDType *)dtypes, n_dtypes),
  };
  PolyUPat *where =
      poly_upat_set_dtype(poly_upat_op(POLY_OP_WHERE, where_src, 3, "w"), dtypes, n_dtypes);
  PolyUPat *alu =
      poly_upat_set_dtype(poly_upat_ops(alu_without_where, NULL, 0, "x"), dtypes, n_dtypes);
  PolyDType bool_dtype = POLY_BOOL;
  PolyUPat *comparison = poly_upat_set_dtype(
      poly_upat_ops2(
          POLY_GROUP_ALU, poly_upat_dtype("x", (PolyDType *)dtypes, n_dtypes),
          poly_upat_dtype("y", (PolyDType *)dtypes, n_dtypes), "alu"
      ),
      &bool_dtype, 1
  );

  PolyNamedRule rules[5] = {
      POLY_RULE(where, non_native_float_where),
      POLY_RULE(alu, non_native_float_alu),
      POLY_RULE(comparison, non_native_float_comparison),
  };
  int n_rules = 3;
  if (casting) {
    rules[n_rules++] = POLY_RULE(
        poly_upat_set_dtype(poly_upat_op1(POLY_OP_CAST, poly_upat_any("x"), "y"), dtypes, n_dtypes),
        non_native_float_cast_to
    );
    rules[n_rules++] = POLY_RULE(
        poly_upat_op1(POLY_OP_CAST, poly_upat_dtype("y", (PolyDType *)dtypes, n_dtypes), "x"),
        non_native_float_cast_from
    );
  }
  return poly_pm_new_named(rules, n_rules);
}

/* Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py:91-102. */
static PolyUOp *cast_float_to_bf16(PolyCtx *ctx, PolyUOp *x, PolyDType bf16_dtype) {
  PolyUOp *bits = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT32, x, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *minus_one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(-1));
  PolyUOp *c16 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(16));
  PolyUOp *c7fff = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0x7fff));
  PolyUOp *cffff = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0xffff));
  PolyUOp *c10000 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0x10000));
  PolyUOp *c7f800000 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0x7f800000));
  PolyUOp *neg_bits = poly_uop2(ctx, POLY_OP_MUL, POLY_UINT32, bits, minus_one, poly_arg_none());
  PolyUOp *is_finite = poly_uop2(
      ctx, POLY_OP_CMPNE, POLY_BOOL,
      poly_uop2(ctx, POLY_OP_AND, POLY_UINT32, neg_bits, c7f800000, poly_arg_none()), zero,
      poly_arg_none()
  );
  PolyUOp *rounded = poly_uop2(
      ctx, POLY_OP_ADD, POLY_UINT32,
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_UINT32, bits,
          poly_uop2(
              ctx, POLY_OP_AND, POLY_UINT32,
              poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, bits, c16, poly_arg_none()), one,
              poly_arg_none()
          ),
          poly_arg_none()
      ),
      c7fff, poly_arg_none()
  );
  PolyUOp *low_nonzero = poly_uop2(
      ctx, POLY_OP_CMPNE, POLY_BOOL,
      poly_uop2(ctx, POLY_OP_AND, POLY_UINT32, bits, cffff, poly_arg_none()), zero, poly_arg_none()
  );
  PolyUOp *nan_adjusted = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_UINT32, low_nonzero,
      poly_uop2(ctx, POLY_OP_OR, POLY_UINT32, bits, c10000, poly_arg_none()), bits, poly_arg_none()
  );
  PolyUOp *selected =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_UINT32, is_finite, rounded, nan_adjusted, poly_arg_none());
  PolyUOp *raw = poly_uop1(
      ctx, POLY_OP_CAST, POLY_UINT16,
      poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, selected, c16, poly_arg_none()), poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_BITCAST, bf16_dtype, raw, poly_arg_none());
}

static PolyUOp *manual_bf16_to_float(PolyCtx *ctx, PolyUOp *root, const PolyBindings *bindings) {
  (void)bindings;
  PolyUOp *raw = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT16, root->src[0], poly_arg_none());
  PolyUOp *wide = poly_uop1(ctx, POLY_OP_CAST, POLY_UINT32, raw, poly_arg_none());
  PolyUOp *shift = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(16));
  return poly_uop1(
      ctx, POLY_OP_BITCAST, POLY_FLOAT32,
      poly_uop2(ctx, POLY_OP_SHL, POLY_UINT32, wide, shift, poly_arg_none()), poly_arg_none()
  );
}

static PolyUOp *manual_float_to_bf16(PolyCtx *ctx, PolyUOp *root, const PolyBindings *bindings) {
  (void)bindings;
  return cast_float_to_bf16(ctx, root->src[0], root->dtype);
}

static _Thread_local PolyPatternMatcher *g_pm_manual_bf16_cast = NULL;

/* Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py:103-109. */
PolyPatternMatcher *poly_pm_manual_bf16_cast(void) {
  if (g_pm_manual_bf16_cast) return g_pm_manual_bf16_cast;
  PolyDType bf16 = POLY_BFLOAT16, f32 = POLY_FLOAT32;
  PolyNamedRule rules[] = {
      POLY_RULE(
          poly_upat_set_dtype(
              poly_upat_op1(POLY_OP_CAST, poly_upat_dtype("x", &bf16, 1), NULL), &f32, 1
          ),
          manual_bf16_to_float
      ),
      POLY_RULE(
          poly_upat_set_dtype(
              poly_upat_op1(POLY_OP_CAST, poly_upat_dtype("x", &f32, 1), NULL), &bf16, 1
          ),
          manual_float_to_bf16
      ),
  };
  g_pm_manual_bf16_cast =
      poly_pm_thread_cache(poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_manual_bf16_cast;
}

static PolyUOp *cast_via_float(PolyCtx *ctx, PolyUOp *root, const PolyBindings *bindings) {
  (void)bindings;
  PolyUOp *f32 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[0], poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, root->dtype, f32, poly_arg_none());
}

static _Thread_local PolyPatternMatcher *g_clang_renderer_extra = NULL;

PolyPatternMatcher *poly_clang_renderer_extra_matcher(void) {
  if (g_clang_renderer_extra) return g_clang_renderer_extra;
  PolyDType f64 = POLY_FLOAT64, f16 = POLY_FLOAT16, bf16 = POLY_BFLOAT16;
  PolyNamedRule cast_rules[] = {
      POLY_RULE(
          poly_upat_set_dtype(
              poly_upat_op1(POLY_OP_CAST, poly_upat_dtype("x", &f64, 1), NULL), &f16, 1
          ),
          cast_via_float
      ),
      POLY_RULE(
          poly_upat_set_dtype(
              poly_upat_op1(POLY_OP_CAST, poly_upat_dtype("x", &f64, 1), NULL), &bf16, 1
          ),
          cast_via_float
      ),
      POLY_RULE(
          poly_upat_set_dtype(
              poly_upat_op1(POLY_OP_CAST, poly_upat_dtype("x", &bf16, 1), NULL), &f16, 1
          ),
          cast_via_float
      ),
  };
  PolyPatternMatcher *casts =
      poly_pm_new_named(cast_rules, (int)(sizeof(cast_rules) / sizeof(cast_rules[0])));
  PolyPatternMatcher *non_native = create_non_native_float_pats(&bf16, 1, true);
  PolyPatternMatcher *with_non_native = poly_pm_concat(casts, non_native);
  PolyPatternMatcher *complete =
      with_non_native ? poly_pm_concat(with_non_native, poly_pm_manual_bf16_cast()) : NULL;
  poly_pm_destroy(casts);
  poly_pm_destroy(non_native);
  poly_pm_destroy(with_non_native);
  g_clang_renderer_extra = poly_pm_thread_cache(complete);
  return g_clang_renderer_extra;
}

static PolyUOp *hip_bf16_const(PolyCtx *ctx, PolyUOp *root, const PolyBindings *bindings) {
  (void)bindings;
  PolyUOp *f32 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, root->arg);
  return cast_float_to_bf16(ctx, f32, root->dtype);
}

static PolyUOp *fp8_cast_via_float(PolyCtx *ctx, PolyUOp *root, const PolyBindings *bindings) {
  (void)bindings;
  if (!root || root->n_src != 1 || !poly_dtype_is_fp8(root->dtype) ||
      !poly_dtype_is_fp8(root->src[0]->dtype) || poly_dtype_eq(root->dtype, root->src[0]->dtype))
    return NULL;
  PolyUOp *f32 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[0], poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, root->dtype, f32, poly_arg_none());
}

static PolyUOp *hip_fp8_wmma_bitcast(PolyCtx *ctx, PolyUOp *root, const PolyBindings *bindings) {
  (void)bindings;
  bool ocp_input = root && root->n_src > 0 &&
                   (poly_dtype_eq(root->src[0]->dtype, POLY_FP8E4M3) ||
                    poly_dtype_eq(root->src[0]->dtype, POLY_FP8E5M2));
  if (!root || root->op != POLY_OP_WMMA || root->n_src != 3 ||
      !poly_dtype_eq(root->dtype, POLY_FLOAT32) || !ocp_input ||
      poly_uop_max_numel(ctx, root->src[0]) != 8)
    return NULL;
  PolyUOp *src[] = {
      poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT64, root->src[0], poly_arg_none()),
      poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT64, root->src[1], poly_arg_none()),
      root->src[2],
  };
  return poly_uop(ctx, POLY_OP_WMMA, root->dtype, src, 3, root->arg);
}

static _Thread_local PolyPatternMatcher *g_cuda_renderer_extra = NULL;

/* Tinygrad 2026-08-22/a9069c177a9d CUDARenderer.extra_matcher. */
PolyPatternMatcher *poly_cuda_renderer_extra_matcher(void) {
  if (g_cuda_renderer_extra) return g_cuda_renderer_extra;
  PolyDType fp8s[] = {
      POLY_FP8E4M3,
      POLY_FP8E5M2,
      POLY_FP8E4M3FNUZ,
      POLY_FP8E5M2FNUZ,
  };
  PolyPatternMatcher *non_native = create_non_native_float_pats(fp8s, 4, false);
  PolyUPat *cross_src = poly_upat_dtype("x", fp8s, 4);
  PolyUPat *cross = poly_upat_set_dtype(poly_upat_op1(POLY_OP_CAST, cross_src, "y"), fp8s, 4);
  PolyNamedRule cast_rules[] = {POLY_RULE(cross, fp8_cast_via_float)};
  PolyPatternMatcher *casts = poly_pm_new_named(cast_rules, 1);
  PolyPatternMatcher *complete = poly_pm_concat(non_native, casts);
  poly_pm_destroy(non_native);
  poly_pm_destroy(casts);
  g_cuda_renderer_extra = poly_pm_thread_cache(complete);
  return g_cuda_renderer_extra;
}

static _Thread_local PolyPatternMatcher *g_hip_renderer_extra = NULL;

/* Tinygrad 2026-08-22/a9069c177a9d HIPRenderer.extra_matcher class rows. */
PolyPatternMatcher *poly_hip_renderer_extra_matcher(void) {
  if (g_hip_renderer_extra) return g_hip_renderer_extra;
  PolyDType non_native_dtypes[] = {
      POLY_BFLOAT16, POLY_FP8E4M3, POLY_FP8E5M2, POLY_FP8E4M3FNUZ, POLY_FP8E5M2FNUZ,
  };
  PolyDType bf16 = POLY_BFLOAT16, f32 = POLY_FLOAT32;
  PolyPatternMatcher *non_native = create_non_native_float_pats(non_native_dtypes, 5, true);
  PolyNamedRule hip_rules[] = {
      POLY_RULE(
          poly_upat_set_dtype(poly_upat_op(POLY_OP_WMMA, NULL, 0, "x"), &f32, 1),
          hip_fp8_wmma_bitcast
      ),
      POLY_RULE(poly_upat_set_dtype(poly_upat_cvar("x"), &bf16, 1), hip_bf16_const),
  };
  PolyPatternMatcher *hip =
      poly_pm_new_named(hip_rules, (int)(sizeof(hip_rules) / sizeof(hip_rules[0])));
  PolyPatternMatcher *complete = poly_pm_concat(non_native, hip);
  poly_pm_destroy(non_native);
  poly_pm_destroy(hip);
  g_hip_renderer_extra = poly_pm_thread_cache(complete);
  return g_hip_renderer_extra;
}

/* Render helpers */

/* Render INT64_MIN without spelling an out-of-range positive literal followed
 * by unary minus. */
static char *render_int64_const(int64_t v, char *buf, int cap) {
  if (v == INT64_MIN)
    snprintf(buf, cap, "(-9223372036854775807ll - 1ll)");
  else
    snprintf(buf, cap, "%lldll", (long long)v);
  return buf;
}

/* Render a float constant, dtype-aware. Pinned cstyle.py:40-43 renders half
 * constants through a larger float literal and explicitly casts to half. */
static char *render_float_const(double v, PolyDType dt, char *buf, int cap) {
  PolyDType scalar = dt;
  bool is_f64 = poly_dtype_eq(scalar, POLY_FLOAT64);
  bool is_f16 = poly_dtype_eq(scalar, POLY_FLOAT16);
  char literal[96];
  if (isinf(v)) {
    if (is_f64)
      snprintf(buf, cap, v > 0 ? "__builtin_inf()" : "(-__builtin_inf())");
    else if (is_f16)
      snprintf(buf, cap, v > 0 ? "((__fp16)(__builtin_inff()))" : "((__fp16)(-__builtin_inff()))");
    else
      snprintf(buf, cap, v > 0 ? "__builtin_inff()" : "(-__builtin_inff())");
    return buf;
  }
  if (isnan(v)) {
    if (is_f64)
      snprintf(buf, cap, "__builtin_nan(\"\")");
    else if (is_f16)
      snprintf(buf, cap, "((__fp16)(__builtin_nanf(\"\")))");
    else
      snprintf(buf, cap, "__builtin_nanf(\"\")");
    return buf;
  }
  if (is_f64) {
    /* Full precision double literal: enough digits to round-trip, no suffix. */
    snprintf(buf, cap, "%.17g", v);
    if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E')) {
      int len = (int)strlen(buf);
      if (len + 2 < cap) {
        buf[len] = '.';
        buf[len + 1] = '0';
        buf[len + 2] = '\0';
      }
    }
    return buf;
  }
  if (is_f16) {
    /* A double round-trip literal is harmlessly rounded to f32 by the suffix,
     * then to f16 by the explicit cast, matching tinygrad's emitted C type
     * boundary without pre-rounding the argument in the renderer. */
    snprintf(literal, sizeof(literal), "%.17g", v);
    if (!strchr(literal, '.') && !strchr(literal, 'e') && !strchr(literal, 'E')) {
      int len = (int)strlen(literal);
      if (len + 2 < (int)sizeof(literal)) {
        literal[len] = '.';
        literal[len + 1] = '0';
        literal[len + 2] = '\0';
      }
    }
    snprintf(buf, cap, "((__fp16)(%sf))", literal);
    return buf;
  }
  /* Float32: truncate to float32, add the 'f' suffix. */
  snprintf(literal, sizeof(literal), "%.9g", (double)(float)v);
  if (!strchr(literal, '.') && !strchr(literal, 'e') && !strchr(literal, 'E')) {
    int len = (int)strlen(literal);
    if (len + 2 < (int)sizeof(literal)) {
      literal[len] = '.';
      literal[len + 1] = '0';
      literal[len + 2] = '\0';
    }
  }
  int len = (int)strlen(literal);
  if (len + 1 < cap) {
    snprintf(buf, cap, "%sf", literal);
  }
  return buf;
}

static void render_ctype_nonptr(PolyDType dt, int lanes, char *buf, int cap);

/* Current tinygrad renderer/cstyle.py:26-47 renders both a direct CONST and
 * CAST(strong, CONST(weak/bool)) from the value using the destination dtype.
 * Return one owned scalar literal so the CONST and CAST paths cannot drift. */
static char *render_const_literal(PolyUOp *c, PolyDType dtype, bool vector_bool) {
  if (!c || c->op != POLY_OP_CONST) return NULL;
  PolyDType scalar = dtype;
  char val[192];
  if (poly_dtype_is_float(scalar)) {
    render_float_const(c->arg.f, scalar, val, sizeof(val));
  } else if (poly_dtype_is_bool(scalar)) {
    snprintf(val, sizeof(val), "%d", c->arg.b ? (vector_bool ? -1 : 1) : 0);
  } else if (poly_dtype_eq(scalar, POLY_INT64)) {
    if (c->arg.kind == POLY_ARG_BIGINT) {
      char *decimal = poly_arg_integer_to_decimal(c->arg);
      if (!decimal) return NULL;
      size_t n = strlen(decimal) + 3;
      char *ret = malloc(n);
      if (ret) snprintf(ret, n, "%sll", decimal);
      free(decimal);
      return ret;
    }
    render_int64_const(c->arg.i, val, sizeof(val));
  } else if (poly_dtype_eq(scalar, POLY_UINT64)) {
    snprintf(val, sizeof(val), "%lluull", (unsigned long long)poly_arg_integer_to_u64_mod(c->arg));
  } else if (poly_dtype_eq(scalar, POLY_UINT32)) {
    snprintf(val, sizeof(val), "%uu", (unsigned)(uint32_t)poly_arg_integer_to_u64_mod(c->arg));
  } else if (poly_dtype_eq(scalar, POLY_UINT8) || poly_dtype_eq(scalar, POLY_UINT16)) {
    char type[64];
    render_ctype_nonptr(scalar, 1, type, sizeof(type));
    snprintf(
        val, sizeof(val), "((%s)(%uu))", type,
        (unsigned)(uint32_t)poly_arg_integer_to_u64_mod(c->arg)
    );
  } else if (poly_dtype_eq(scalar, POLY_INT8) || poly_dtype_eq(scalar, POLY_INT16)) {
    char type[64];
    render_ctype_nonptr(scalar, 1, type, sizeof(type));
    if (c->arg.kind == POLY_ARG_BIGINT) {
      char *decimal = poly_arg_integer_to_decimal(c->arg);
      if (!decimal) return NULL;
      size_t n = strlen(type) + strlen(decimal) + 7;
      char *ret = malloc(n);
      if (ret) snprintf(ret, n, "((%s)(%s))", type, decimal);
      free(decimal);
      return ret;
    }
    snprintf(val, sizeof(val), "((%s)(%lld))", type, (long long)c->arg.i);
  } else if (c->arg.kind == POLY_ARG_BIGINT) {
    return poly_arg_integer_to_decimal(c->arg);
  } else {
    snprintf(val, sizeof(val), "%lld", (long long)c->arg.i);
  }
  return strdup(val);
}

/* Exact C equivalent of helpers.strip_parens used by pinned
 * cstyle.py:62-63: remove one balanced outer pair, not merely the first and
 * last characters. */
static char *render_strip_parens(const char *expr) {
  if (!expr) return strdup("");
  size_t len = strlen(expr);
  if (len < 2 || expr[0] != '(' || expr[len - 1] != ')') return strdup(expr);
  int depth = 0;
  for (size_t i = 1; i + 1 < len; i++) {
    if (expr[i] == '(')
      depth++;
    else if (expr[i] == ')' && --depth < 0)
      return strdup(expr);
  }
  if (depth != 0) return strdup(expr);
  char *stripped = malloc(len - 1);
  if (!stripped) return strdup(expr);
  memcpy(stripped, expr + 1, len - 2);
  stripped[len - 2] = '\0';
  return stripped;
}

/* Tinygrad 2026-08-22/a9069c177a9d CStyleLanguage._render_dtype receives
 * scalar DType and UOp.max_numel() as separate arguments. */
static bool render_is_bf16(PolyDType dt) {
  PolyDType s = dt;
  return s.priority == POLY_BFLOAT16.priority && s.bitsize == 16;
}

static void render_ctype_nonptr(PolyDType dt, int lanes, char *buf, int cap) {
  if (render_is_bf16(dt)) {
    /* CPU C follows tinygrad's non-native bf16 lowering: arithmetic is
     * promoted to f32 and bf16 storage is addressed as raw 16-bit lanes. */
    if (lanes > 1)
      snprintf(
          buf, cap, "unsigned short __attribute__((vector_size(%d)))",
          poly_dtype_itemsize(dt) * lanes
      );
    else
      snprintf(buf, cap, "unsigned short");
    return;
  }
  if (lanes <= 1) {
    snprintf(buf, cap, "%s", dt.name);
    return;
  }
  PolyDType s = dt;
  /* Clang/GCC vector bool is problematic; model vector masks as int vectors. */
  if (poly_dtype_is_bool(s)) {
    int vec_sz = (int)sizeof(int) * lanes;
    snprintf(buf, cap, "int __attribute__((vector_size(%d)))", vec_sz);
  } else {
    int vec_sz = poly_dtype_itemsize(s) * lanes;
    snprintf(buf, cap, "%s __attribute__((vector_size(%d)))", s.name, vec_sz);
  }
}

static void render_ctype(PolyDType dt, int lanes, char *buf, int cap) {
  render_ctype_nonptr(dt, lanes, buf, cap);
}

/* Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py:187 render_type uses
 * UOp.max_numel(); DType never carries lanes. */
static int render_uop_lanes(PolyCtx *ctx, const PolyUOp *u) {
  if (!u) return 1;
  int64_t max_numel = poly_uop_max_numel(ctx, u);
  if (max_numel > 1 && max_numel <= INT_MAX) return (int)max_numel;
  return 1;
}

static bool render_index_lane_ptr(StrMap *names, PolyUOp *ptr_uop, int lane, char *buf, int cap) {
  PolyUOp *idx = poly_as_memory_slice(ptr_uop);
  if (!idx || idx->n_src < 2) return false;
  char *base = smap_get(names, idx->src[0]);
  char *idx_s = smap_get(names, idx->src[1]);
  if (lane == 0)
    snprintf(buf, cap, "(%s+%s)", base ? base : "0", idx_s ? idx_s : "0");
  else
    snprintf(buf, cap, "(%s+(%s)+%d)", base ? base : "0", idx_s ? idx_s : "0", lane);
  return true;
}

static void render_vector_load_expr(
    StrBuf *body,
    StrMap *names,
    PolyUOp *ptr_uop,
    int lanes,
    const char *dtype_s
) {
  sb_printf(body, "((%s){", dtype_s);
  for (int lane = 0; lane < lanes; lane++) {
    char lane_ptr[512];
    if (lane) sb_puts(body, ",");
    if (render_index_lane_ptr(names, ptr_uop, lane, lane_ptr, sizeof(lane_ptr)))
      sb_printf(body, "(*%s)", lane_ptr);
    else
      sb_puts(body, "0");
  }
  sb_puts(body, "})");
}

/* Render an ALU expression.
 * For vec4 types, GCC vector extensions handle +, -, *, /, <<, >>, &, |, ^,
 * <, !=, == natively. WHERE/MAX need special handling. */
static void render_alu(
    char *buf,
    int cap,
    PolyOps op,
    PolyDType dtype,
    int lanes,
    const char *s0,
    const char *s1,
    const char *s2
) {
  bool is_vec = lanes > 1;
  PolyDType sdt = dtype;
  switch (op) {
  /* unary */
  case POLY_OP_NEG:
    /* Pinned CStyleLanguage.code_for_op renders every NEG as arithmetic -x
     * (renderer/cstyle.py:128-130). Assignment to bool normalizes the result. */
    snprintf(buf, cap, "(-%s)", s0); /* works on vectors */
    break;
  case POLY_OP_SQRT:
    if (is_vec) {
      /* Vec SQRT: per-element via initializer (no vec builtin) */
      char vt[128];
      render_ctype(dtype, lanes, vt, sizeof(vt));
      const char *fn = poly_dtype_eq(sdt, POLY_FLOAT64) ? "__builtin_sqrt" : "__builtin_sqrtf";
      snprintf(
          buf, cap, "((%s){%s(%s[0]),%s(%s[1]),%s(%s[2]),%s(%s[3])})", vt, fn, s0, fn, s0, fn, s0,
          fn, s0
      );
    } else {
      snprintf(
          buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "__builtin_sqrt(%s)" : "__builtin_sqrtf(%s)",
          s0
      );
    }
    break;
  case POLY_OP_TRUNC:
    snprintf(
        buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "__builtin_trunc(%s)" : "__builtin_truncf(%s)",
        s0
    );
    break;
  case POLY_OP_EXP2:
    snprintf(buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "exp2(%s)" : "exp2f(%s)", s0);
    break;
  case POLY_OP_LOG2:
    snprintf(buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "log2(%s)" : "log2f(%s)", s0);
    break;
  case POLY_OP_SIN:
    snprintf(buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "sin(%s)" : "sinf(%s)", s0);
    break;
  case POLY_OP_RECIPROCAL:
    if (is_vec) {
      /* Vec RECIPROCAL: (type){1.0f,...} / x */
      char vt[128];
      render_ctype(dtype, lanes, vt, sizeof(vt));
      char one[128];
      render_ctype(dtype, lanes, one, sizeof(one));
      snprintf(buf, cap, "((%s){1.0f,1.0f,1.0f,1.0f}/%s)", vt, s0);
    } else {
      snprintf(buf, cap, "(1/%s)", s0);
    }
    break;
  /* binary — +, -, *, /, <<, >>, &, |, ^, <, !=, == all work on GCC vectors */
  case POLY_OP_ADD:
    snprintf(buf, cap, "(%s+%s)", s0, s1);
    break;
  case POLY_OP_SUB:
    snprintf(buf, cap, "(%s-%s)", s0, s1);
    break;
  case POLY_OP_MUL:
    snprintf(buf, cap, "(%s*%s)", s0, s1);
    break;
  case POLY_OP_FDIV:
    snprintf(buf, cap, "(%s/%s)", s0, s1);
    break;
  case POLY_OP_IDIV:
    snprintf(buf, cap, "(%s/%s)", s0, s1);
    break;
  case POLY_OP_MOD:
    snprintf(buf, cap, "(%s%%%s)", s0, s1);
    break;
  case POLY_OP_SHL:
    snprintf(buf, cap, "(%s<<%s)", s0, s1);
    break;
  case POLY_OP_SHR:
    snprintf(buf, cap, "(%s>>%s)", s0, s1);
    break;
  case POLY_OP_AND:
    snprintf(buf, cap, "(%s&%s)", s0, s1);
    break;
  case POLY_OP_OR:
    snprintf(buf, cap, "(%s|%s)", s0, s1);
    break;
  case POLY_OP_XOR:
    snprintf(buf, cap, "(%s^%s)", s0, s1);
    break;
  case POLY_OP_CMPLT:
    snprintf(buf, cap, "(%s<%s)", s0, s1);
    break;
  case POLY_OP_CMPNE:
    snprintf(buf, cap, "(%s!=%s)", s0, s1);
    break;
  case POLY_OP_CMPEQ:
    snprintf(buf, cap, "(%s==%s)", s0, s1);
    break;
  case POLY_OP_MAX:
    if (is_vec) {
      /* vec MAX: bitwise select using comparison mask.
       * GCC vec comparison returns -1 (all bits set) or 0 per lane. */
      char int_type[128];
      PolyDType idt = poly_dtype_is_float(sdt) ? POLY_INT32 : sdt;
      render_ctype(idt, lanes, int_type, sizeof(int_type));
      char dst_type[128];
      render_ctype(dtype, lanes, dst_type, sizeof(dst_type));
      snprintf(
          buf, cap, "((%s)(((%s)(%s>%s) & (%s)%s) | (~(%s)(%s>%s) & (%s)%s)))", dst_type, int_type,
          s0, s1, int_type, s0, int_type, s0, s1, int_type, s1
      );
    } else {
      snprintf(buf, cap, "((%s>%s)?%s:%s)", s0, s1, s0, s1);
    }
    break;
  case POLY_OP_POW:
    snprintf(buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "pow(%s, %s)" : "powf(%s, %s)", s0, s1);
    break;
  /* ternary */
  case POLY_OP_WHERE:
    if (is_vec) {
      /* vec WHERE(mask, a, b): bitwise select.
       * mask is int-typed (from CMPLT: -1 or 0 per lane). */
      char int_type[128];
      PolyDType idt = poly_dtype_is_float(sdt) ? POLY_INT32 : sdt;
      render_ctype(idt, lanes, int_type, sizeof(int_type));
      char dst_type[128];
      render_ctype(dtype, lanes, dst_type, sizeof(dst_type));
      snprintf(
          buf, cap, "((%s)((%s & (%s)%s) | (~%s & (%s)%s)))", dst_type, s0, int_type, s1, s0,
          int_type, s2
      );
    } else {
      snprintf(buf, cap, "(%s?%s:%s)", s0, s1, s2);
    }
    break;
  case POLY_OP_MULACC:
    snprintf(buf, cap, "((%s*%s)+%s)", s0, s1, s2);
    break;
  default:
    snprintf(buf, cap, "/* unknown op %d */0", op);
    break;
  }
}

static int range_slot(PolyUOp **ranges, int *n_ranges, PolyUOp *r, bool create) {
  if (!r) return -1;
  for (int i = 0; i < *n_ranges; i++) {
    if (ranges[i] == r) return i;
  }
  if (!create || *n_ranges >= 128) return -1;
  ranges[*n_ranges] = r;
  (*n_ranges)++;
  return *n_ranges - 1;
}

/* C Renderer */

typedef struct {
  char type[256];
  char name[256];
  int order; /* compact C runtime args[] position */
  int sort_key; /* numbered PARAM slot */
  bool is_alu;
  bool runtime_core_id; /* tinygrad CPU THREAD runtime variable */
} RenderParam;

char *poly_render_c(PolyCtx *ctx, PolyUOp **uops, int n, const char *fn_name) {
  if (!ctx || n < 0 || (n && !uops)) return NULL;
  StrBuf decls; /* variable declarations at function scope */
  StrBuf body; /* function body with assignments */
  sb_init(&decls);
  sb_init(&body);

  StrMap names;
  smap_init(&names, n);

  /* Tinygrad 2026-08-22/a9069c177a9d cstyle.py:194,232-237 counts direct
   * consumers, then inlines single-consumer expressions unless EXPAND_SSA.
   */
  PolyMap *uop_indices = poly_map_new((size_t)(n > 0 ? n * 2 : 16));
  int *indices = malloc((size_t)n * sizeof(*indices));
  int *child_count = calloc((size_t)n, sizeof(*child_count));
  RenderParam *params = NULL;
  int param_capacity = 0;
  if (!indices || !child_count) goto fail;
  for (int i = 0; i < n; i++) {
    if (uops[i]->op == POLY_OP_PARAM) param_capacity++;
    indices[i] = i;
    poly_map_set(uop_indices, poly_ptr_hash(uops[i]), uops[i], &indices[i], poly_ptr_eq);
  }
  /* Match CStyleLanguage's parameter dict without allocating by kernel size. */
  params = calloc((size_t)param_capacity, sizeof(*params));
  if (param_capacity && !params) goto fail;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < uops[i]->n_src; j++) {
      int *source_index =
          poly_map_get(uop_indices, poly_ptr_hash(uops[i]->src[j]), uops[i]->src[j], poly_ptr_eq);
      if (source_index) child_count[*source_index]++;
    }
  }
  bool expand_ssa = poly_getenv_flag("EXPAND_SSA") || poly_getenv_flag("POLY_EXPAND_SSA");

  /* function parameter entries: (type_str, name_str, sort_key) */
  int n_params = 0;
  int n_buffer_params = 0;

  /* prefix counters */
  int c_val = 0, c_alu = 0, c_cast = 0, c_acc = 0;
  int depth = 1;

  /* Range liveness tracking: emit END only after last non-END use. */
  PolyUOp *live_ranges[128];
  int live_remaining[128];
  int n_live_ranges = 0;
  memset(live_ranges, 0, sizeof(live_ranges));
  memset(live_remaining, 0, sizeof(live_remaining));

  PolyUOp *open_ranges[128];
  int n_open_ranges = 0;
  memset(open_ranges, 0, sizeof(open_ranges));

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_RANGE) (void)range_slot(live_ranges, &n_live_ranges, u, true);
    if (u->op == POLY_OP_END) continue;
    for (int j = 0; j < u->n_src; j++) {
      if (u->src[j] && u->src[j]->op == POLY_OP_RANGE) {
        int ri = range_slot(live_ranges, &n_live_ranges, u->src[j], true);
        if (ri >= 0) live_remaining[ri]++;
      }
    }
  }

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];

    /* --- SINK: skip ------------------------------------------------- */
    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP) continue;

    if (u->op != POLY_OP_END) {
      for (int j = 0; j < u->n_src; j++) {
        if (u->src[j] && u->src[j]->op == POLY_OP_RANGE) {
          int ri = range_slot(live_ranges, &n_live_ranges, u->src[j], false);
          if (ri >= 0 && live_remaining[ri] > 0) live_remaining[ri]--;
        }
      }
    }

    /* ParamArg.addrspace selects pointer storage or an ALU scalar, as in
     * current tinygrad cstyle.py:151-155,207-211. */
    if (u->op == POLY_OP_PARAM) {
      int64_t slot = poly_program_buffer_slot(u);
      if (slot < 0) goto fail;
      char name[32];
      snprintf(name, sizeof(name), "data%lld", (long long)slot);
      smap_set(&names, u, strdup(name));

      PolyDType base = u->dtype;
      char base_type[128];
      render_ctype_nonptr(base, 1, base_type, sizeof(base_type));
      bool is_alu = poly_uop_is_alu_param(u);
      snprintf(
          params[n_params].type, sizeof(params[n_params].type),
          is_alu ? "const %s" : "%s* restrict", base_type
      );
      snprintf(params[n_params].name, sizeof(params[n_params].name), "%s", name);
      params[n_params].sort_key = (int)slot;
      params[n_params].is_alu = is_alu;
      const char *expr = poly_uop_expr(u);
      params[n_params].runtime_core_id = is_alu && expr && strcmp(expr, "core_id") == 0;
      params[n_params].order = -1;
      if (!is_alu) n_buffer_params++;
      n_params++;
      continue;
    }

    /* --- CONST: inline literal -------------------------------------- */
    if (u->op == POLY_OP_CONST) {
      PolyDType sdt = u->dtype;
      /* Current cstyle render_type uses max_numel, so a shaped scalar CONST
       * broadcasts using its shape-derived register width. */
      int lanes = render_uop_lanes(ctx, u);
      char *literal = render_const_literal(u, sdt, lanes > 1);
      if (!literal) goto fail;
      if (lanes > 1) {
        char dtype_s[128];
        render_ctype(u->dtype, lanes, dtype_s, sizeof(dtype_s));
        StrBuf vexpr;
        sb_init(&vexpr);
        sb_printf(&vexpr, "((%s){", dtype_s);
        for (int j = 0; j < lanes; j++)
          sb_printf(&vexpr, "%s%s", j > 0 ? "," : "", literal);
        sb_puts(&vexpr, "})");
        smap_set(&names, u, vexpr.buf);
      } else {
        smap_set(&names, u, strdup(literal));
      }
      free(literal);
      continue;
    }

    /* --- STACK: current shaped register literal -------------------- */
    if (u->op == POLY_OP_STACK) {
      if (u->n_src == 1) {
        char *s = smap_get(&names, u->src[0]);
        smap_set(&names, u, strdup(s ? s : "0"));
        continue;
      }
      char dtype_s[128];
      render_ctype(u->dtype, render_uop_lanes(ctx, u), dtype_s, sizeof(dtype_s));
      StrBuf vexpr;
      sb_init(&vexpr);
      sb_printf(&vexpr, "((%s){", dtype_s);
      if (u->n_src > 0) {
        for (int j = 0; j < u->n_src; j++) {
          if (j) sb_puts(&vexpr, ",");
          char *s = smap_get(&names, u->src[j]);
          sb_puts(&vexpr, s ? s : "0");
        }
      } else if (u->arg.kind == POLY_ARG_INT_TUPLE) {
        for (int j = 0; j < u->arg.int_tuple.n; j++) {
          if (j) sb_puts(&vexpr, ",");
          sb_printf(&vexpr, "%lld", (long long)u->arg.int_tuple.vals[j]);
        }
      }
      sb_puts(&vexpr, "})");
      smap_set(&names, u, vexpr.buf); /* takes ownership */
      continue;
    }

    /* --- INDEX: pointer arithmetic or vector lane extract ------------ */
    if (u->op == POLY_OP_INDEX) {
      char *buf_s = smap_get(&names, u->src[0]);
      char *idx_s = smap_get(&names, u->src[1]);
      StrBuf expr;
      sb_init(&expr);
      if (poly_is_program_memory_base(u->src[0]))
        sb_printf(&expr, "(%s+%s)", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      else
        sb_printf(&expr, "(%s[%s])", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      smap_set(&names, u, expr.buf);
      continue;
    }

    /* --- SHRINK: late codegen memory slice -------------------------- */
    if (u->op == POLY_OP_SHRINK) {
      char *buf_s = smap_get(&names, u->src[0]);
      char *idx_s = smap_get(&names, u->src[1]);
      StrBuf expr;
      sb_init(&expr);
      sb_printf(&expr, "(%s+%s)", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      smap_set(&names, u, expr.buf);
      continue;
    }

    /* --- RANGE: for loop -------------------------------------------- */
    if (u->op == POLY_OP_RANGE) {
      char *range = poly_range_str(u->arg);
      char *name = malloc(strlen(range) + 6);
      snprintf(name, strlen(range) + 6, "%cidx%s", poly_axis_letter(u->arg), range);
      free(range);
      smap_set(&names, u, name);

      char *bound = smap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
      sb_printf(&body, "for (int %s = 0; %s < %s; %s++) {\n", name, name, bound, name);
      depth++;
      if (n_open_ranges < 128) open_ranges[n_open_ranges++] = u;
      continue;
    }

    /* --- END / ENDIF: close brace ----------------------------------- */
    if (u->op == POLY_OP_END || u->op == POLY_OP_ENDIF) {
      if (u->op == POLY_OP_END && u->n_src > 1 && u->src[1]->op == POLY_OP_RANGE) {
        PolyUOp *want = u->src[1];
        int wi = range_slot(live_ranges, &n_live_ranges, want, false);
        if (wi >= 0 && live_remaining[wi] > 0) continue; /* too early */

        int pos = -1;
        for (int p = n_open_ranges - 1; p >= 0; p--) {
          if (open_ranges[p] == want) {
            pos = p;
            break;
          }
        }
        if (pos < 0) continue; /* duplicate/stale END */

        bool can_close = true;
        for (int p = n_open_ranges - 1; p >= pos; p--) {
          int oi = range_slot(live_ranges, &n_live_ranges, open_ranges[p], false);
          if (oi >= 0 && live_remaining[oi] > 0 && open_ranges[p] != want) {
            can_close = false;
            break;
          }
        }
        if (!can_close) continue;

        while (n_open_ranges > pos) {
          depth--;
          for (int d = 0; d < depth; d++)
            sb_puts(&body, "  ");
          sb_puts(&body, "}\n");
          n_open_ranges--;
        }
        continue;
      }

      /* Defensive: END with non-RANGE source is a structural violation.
       * Upstream invariant (rangeify/codegen) should prevent this.
       * Debug: abort loudly.  Release: skip silently as safety belt. */
      if (u->op == POLY_OP_END && u->n_src > 1 && u->src[1]->op != POLY_OP_RANGE) {
#ifndef NDEBUG
        fprintf(
            stderr,
            "polygrad: render_c: END node references non-RANGE source "
            "(op=%s) -- structural invariant violation\n",
            poly_op_name(u->src[1]->op)
        );
        assert(0 && "END source must be RANGE");
#endif
        continue;
      }

      depth--;
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
      sb_puts(&body, "}\n");
      if (u->op == POLY_OP_END && n_open_ranges > 0) n_open_ranges--;
      continue;
    }

    /* Pinned tinygrad renderer/cstyle.py:13,161-164 renders final new-style
     * LOCAL BUFFER(dtype, CONST<int>(size), ParamArg(slot, LOCAL)) storage. */
    if (u->op == POLY_OP_BUFFER && poly_program_memory_is(u, POLY_ADDR_LOCAL)) {
      char name[32];
      snprintf(name, sizeof(name), "smem%lld", (long long)poly_program_buffer_slot(u));
      smap_set(&names, u, strdup(name));
      char dtype_s[128];
      render_ctype(poly_program_buffer_dtype(u), 1, dtype_s, sizeof(dtype_s));
      sb_printf(&decls, "  %s %s[%lld];\n", dtype_s, name, (long long)poly_program_buffer_size(u));
      continue;
    }

    /* --- register accumulator (float r0[1];) ------------------------ */
    if (u->op == POLY_OP_BUFFER && poly_program_memory_is(u, POLY_ADDR_REG)) {
      char name[32];
      snprintf(name, sizeof(name), "r%lld", (long long)poly_program_buffer_slot(u));
      smap_set(&names, u, strdup(name));

      /* Tinygrad CStyleLanguage.render_buffer emits scalar storage arrays;
       * UOp shape determines their length (renderer/cstyle.py:169-173). */
      PolyDType base = poly_program_buffer_dtype(u);
      int64_t reg_size = poly_program_buffer_size(u);
      char base_type[128];
      render_ctype(base, 1, base_type, sizeof(base_type));
      sb_printf(&decls, "  %s %s[%lld];\n", base_type, name, (long long)reg_size);
      continue;
    }

    /* --- AFTER: pass-through (use src[0]'s name) -------------------- */
    if (u->op == POLY_OP_AFTER) {
      char *src_name = smap_get(&names, u->src[0]);
      if (src_name) smap_set(&names, u, strdup(src_name));
      continue;
    }

    /* --- LOAD: dereference indexed pointer -------------------------- */
    if (u->op == POLY_OP_LOAD) {
      char name[32];
      snprintf(name, sizeof(name), "val%d", c_val++);
      smap_set(&names, u, strdup(name));

      char *bidx = smap_get(&names, u->src[0]);
      int lanes = render_uop_lanes(ctx, u);
      char dtype_s[128];
      render_ctype(u->dtype, lanes, dtype_s, sizeof(dtype_s));
      sb_printf(&decls, "  %s %s;\n", dtype_s, name);
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");

      /* Pinned tinygrad final IR: LOAD(INDEX(buf, idx), alt, gate). */
      PolyUOp *idx_uop = poly_as_index(u->src[0]);
      bool is_lane_load =
          idx_uop && idx_uop->n_src >= 1 && !poly_is_program_memory_base(idx_uop->src[0]);
      PolyUOp *gate_uop =
          (u->n_src >= 3 && poly_dtype_is_bool(u->src[2]->dtype)) ? u->src[2] : NULL;
      if (gate_uop && u->n_src >= 2) {
        char *gate_s = smap_get(&names, gate_uop);
        char *alt_s = smap_get(&names, u->src[1]);
        if (is_lane_load) {
          sb_printf(&body, "%s = (%s?%s:%s);\n", name, gate_s, bidx, alt_s);
        } else if (lanes > 1) {
          sb_printf(&body, "if (%s) %s = ", gate_s, name);
          if (poly_as_memory_slice(u->src[0]))
            sb_printf(&body, "(*((%s*)(%s)))", dtype_s, bidx);
          else
            render_vector_load_expr(&body, &names, u->src[0], lanes, dtype_s);
          sb_printf(&body, "; else %s = %s;\n", name, alt_s);
        } else {
          sb_printf(&body, "%s = (%s?(*%s):%s);\n", name, gate_s, bidx, alt_s);
        }
      } else if (gate_uop) {
        char *gate_s = smap_get(&names, gate_uop);
        if (is_lane_load) {
          sb_printf(&body, "%s = (%s?%s:(%s)0);\n", name, gate_s, bidx, dtype_s);
        } else if (lanes > 1) {
          sb_printf(&body, "if (%s) %s = ", gate_s, name);
          if (poly_as_memory_slice(u->src[0]))
            sb_printf(&body, "(*((%s*)(%s)))", dtype_s, bidx);
          else
            render_vector_load_expr(&body, &names, u->src[0], lanes, dtype_s);
          sb_printf(&body, "; else memset(&%s, 0, sizeof(%s));\n", name, name);
        } else {
          sb_printf(&body, "%s = (%s?(*%s):(%s)0);\n", name, gate_s, bidx, dtype_s);
        }
      } else {
        if (is_lane_load) {
          sb_printf(&body, "%s = %s;\n", name, bidx);
        } else if (lanes > 1 && poly_as_memory_slice(u->src[0])) {
          sb_printf(&body, "%s = (*((%s*)(%s)));\n", name, dtype_s, bidx);
        } else {
          sb_printf(&body, "%s = (*%s);\n", name, bidx);
        }
      }
      continue;
    }

    /* --- STORE: write to indexed pointer or accumulator ------------- */
    if (u->op == POLY_OP_STORE) {
      char *target = smap_get(&names, u->src[0]);
      char *val = smap_get(&names, u->src[1]);
      int lanes = u->n_src >= 2 ? render_uop_lanes(ctx, u->src[1]) : 1;
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
      /* Guard: STORE src[0] is always set by construction, but null-check
       * satisfies the analyzer's path-sensitive null-deref tracking. */
      if (u->src[0] && poly_program_memory_is(u->src[0], POLY_ADDR_LOCAL))
        sb_printf(&body, "%s = %s;\n", target, val);
      else if (u->src[1] && lanes > 1 && poly_as_memory_slice(u->src[0])) {
        char dtype_s[128];
        render_ctype(u->src[1]->dtype, lanes, dtype_s, sizeof(dtype_s));
        sb_printf(&body, "*((%s*)(%s)) = %s;\n", dtype_s, target, val);
      } else
        sb_printf(&body, "*%s = %s;\n", target, val);
      continue;
    }

    /* --- CAST: type conversion -------------------------------------- */
    if (u->op == POLY_OP_CAST) {
      /* Current CStyleLanguage._render always inlines CAST(CONST), and
       * base_rewrite formats the value using the strong destination dtype. */
      if (u->n_src == 1 && u->src[0] && u->src[0]->op == POLY_OP_CONST &&
          (poly_dtype_is_weak(u->src[0]->dtype) || poly_dtype_is_bool(u->src[0]->dtype)) &&
          render_uop_lanes(ctx, u) == 1) {
        char *literal = render_const_literal(u->src[0], u->dtype, false);
        if (!literal) goto fail;
        smap_set(&names, u, literal);
        continue;
      }
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      smap_set(&names, u, strdup(name));

      char *src_s = smap_get(&names, u->src[0]);
      int lanes = render_uop_lanes(ctx, u);
      int src_lanes = render_uop_lanes(ctx, u->src[0]);
      char dtype_s[128];
      render_ctype(u->dtype, lanes, dtype_s, sizeof(dtype_s));
      sb_printf(&decls, "  %s %s;\n", dtype_s, name);
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");

      /* Vector → vector CAST: __builtin_convertvector (tinygrad cstyle.py:24) */
      if (lanes > 1 && src_lanes > 1) {
        sb_printf(&body, "%s = __builtin_convertvector(%s, %s);\n", name, src_s, dtype_s);
      }
      /* Scalar → vector CAST (non-pointer): broadcast via initializer */
      else if (lanes > 1 && src_lanes <= 1) {
        char scalar_type[128];
        render_ctype(u->dtype, 1, scalar_type, sizeof(scalar_type));
        sb_printf(&body, "{ %s _sc = (%s)(%s); ", scalar_type, scalar_type, src_s);
        sb_printf(&body, "%s = (%s){", name, dtype_s);
        for (int vi = 0; vi < lanes; vi++)
          sb_printf(&body, "%s_sc", vi > 0 ? "," : "");
        sb_printf(&body, "}; }\n");
      }
      /* Vector → scalar: extract element 0, then cast */
      else if (lanes <= 1 && src_lanes > 1) {
        sb_printf(&body, "%s = (%s)((%s)[0]);\n", name, dtype_s, src_s);
      } else {
        sb_printf(&body, "%s = (%s)(%s);\n", name, dtype_s, src_s);
      }
      continue;
    }

    /* --- BITCAST: reinterpret bits (union punning, C11-legal) ------- */
    if (u->op == POLY_OP_BITCAST) {
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      smap_set(&names, u, strdup(name));

      char *src_s = smap_get(&names, u->src[0]);
      int lanes = render_uop_lanes(ctx, u);
      int src_lanes = render_uop_lanes(ctx, u->src[0]);
      char src_type[128], dst_type[128];
      render_ctype(u->src[0]->dtype, src_lanes, src_type, sizeof(src_type));
      render_ctype(u->dtype, lanes, dst_type, sizeof(dst_type));
      sb_printf(&decls, "  %s %s;\n", dst_type, name);
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
      /* Vector → scalar bitcast: extract element 0, then reinterpret */
      if (lanes <= 1 && src_lanes > 1) {
        char elem_type[128];
        render_ctype(u->src[0]->dtype, 1, elem_type, sizeof(elem_type));
        sb_printf(
            &body, "{ %s _bc = (%s)[0]; memcpy(&%s, &_bc, sizeof(%s)); }\n", elem_type, src_s, name,
            name
        );
      } else {
        sb_printf(
            &body, "{ %s _bc = %s; memcpy(&%s, &_bc, sizeof(%s)); }\n", src_type, src_s, name, name
        );
      }
      continue;
    }

    /* --- ALU ops: arithmetic expressions ---------------------------- */
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      const bool associative = u->op == POLY_OP_ADD || u->op == POLY_OP_MUL ||
                               u->op == POLY_OP_XOR || u->op == POLY_OP_OR || u->op == POLY_OP_AND;
      char *stripped[3] = {NULL, NULL, NULL};
      const char *sources[3] = {"", "", ""};
      for (int j = 0; j < u->n_src && j < 3; j++) {
        const char *source = smap_get(&names, u->src[j]);
        if (associative && u->src[j]->op == u->op) {
          stripped[j] = render_strip_parens(source);
          sources[j] = stripped[j];
        } else {
          sources[j] = source ? source : "";
        }
      }
      const char *s0 = sources[0], *s1 = sources[1], *s2 = sources[2];
      size_t expr_cap = strlen(s0 ? s0 : "") + strlen(s1 ? s1 : "") + strlen(s2 ? s2 : "") + 1024;
      char *expr = malloc(expr_cap);
      if (!expr)
        expr = strdup("0");
      else
        render_alu(
            expr, (int)expr_cap, u->op, u->dtype, render_uop_lanes(ctx, u), s0 ? s0 : "",
            s1 ? s1 : "", s2 ? s2 : ""
        );
      for (int j = 0; j < 3; j++)
        free(stripped[j]);

      /* Pinned cstyle.py:232-237 keeps WHERE materialized but directly embeds
       * a one-use ALU expression in its consumer by default. */
      if (u->op != POLY_OP_WHERE && child_count[i] == 1 && !expand_ssa) {
        smap_set(&names, u, expr);
        continue;
      }

      char name[32];
      snprintf(name, sizeof(name), "alu%d", c_alu++);
      smap_set(&names, u, strdup(name));

      char dtype_s[128];
      render_ctype(u->dtype, render_uop_lanes(ctx, u), dtype_s, sizeof(dtype_s));
      sb_printf(&decls, "  %s %s;\n", dtype_s, name);
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
      sb_printf(&body, "%s = %s;\n", name, expr);
      free(expr);
      continue;
    }

    /* --- IF: conditional -------------------------------------------- */
    if (u->op == POLY_OP_IF) {
      char *cond_s = smap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
      sb_printf(&body, "if (%s) {\n", cond_s);
      depth++;
      continue;
    }
  }

  /* Debug: check depth balance */
  if (depth != 1) {
    int n_ranges = 0, n_ends = 0, n_ends_norange = 0;
    PolyUOp *range_ptrs[64];
    int64_t range_sizes[64];
    int range_end_count[64];
    for (int i = 0; i < n; i++) {
      if (uops[i]->op == POLY_OP_RANGE && n_ranges < 64) {
        range_ptrs[n_ranges] = uops[i];
        range_sizes[n_ranges] = (uops[i]->n_src > 0 && uops[i]->src[0]->op == POLY_OP_CONST)
                                    ? uops[i]->src[0]->arg.i
                                    : -1;
        range_end_count[n_ranges] = 0;
        n_ranges++;
      }
      if (uops[i]->op == POLY_OP_END) {
        n_ends++;
        if (uops[i]->n_src > 1 && uops[i]->src[1]->op == POLY_OP_RANGE) {
          for (int r = 0; r < n_ranges; r++) {
            if (range_ptrs[r] == uops[i]->src[1]) range_end_count[r]++;
          }
        } else {
          n_ends_norange++;
        }
      }
    }
    fprintf(
        stderr,
        "polygrad render_c: DEPTH MISMATCH: depth=%d (expected 1) "
        "%d RANGEs, %d ENDs (%d without RANGE ref)\n",
        depth, n_ranges, n_ends, n_ends_norange
    );
    for (int r = 0; r < n_ranges; r++) {
      fprintf(
          stderr, "  RANGE[%d] %p size=%lld %s\n", r, (void *)range_ptrs[r],
          (long long)range_sizes[r], (range_end_count[r] > 0) ? "HAS_END" : "ORPHAN"
      );
      fprintf(stderr, "    END count: %d\n", range_end_count[r]);
    }
  }

  /* Sort params by arg index (PARAM 0, 1, 2, ...) */
  for (int i = 1; i < n_params; i++) {
    RenderParam kp = params[i];
    int j = i - 1;
    while (j >= 0 && params[j].sort_key > kp.sort_key) {
      params[j + 1] = params[j];
      j--;
    }
    params[j + 1] = kp;
  }

  /* tinygrad@2026-08-22/a9069c177a9d ProgramInfo.globals keeps sparse
   * PARAM identities, while CPUProgram passes selected buffers positionally.
   * This wrapper is the C-only adapter from those names to compact args[]. */
  int buffer_arg = 0, var_arg = 0;
  for (int i = 0; i < n_params; i++) {
    if (params[i].runtime_core_id) continue;
    params[i].order = params[i].is_alu ? n_buffer_params + var_arg++ : buffer_arg++;
  }

  /* Build complete source */
  StrBuf out;
  sb_init(&out);
  sb_puts(&out, "#include <math.h>\n");
  sb_puts(&out, "#include <stdint.h>\n");
  sb_puts(&out, "#include <stdbool.h>\n");
  sb_puts(&out, "#include <string.h>\n");

  /* function signature */
  sb_printf(&out, "void %s(", fn_name);
  for (int i = 0; i < n_params; i++) {
    if (i > 0) sb_puts(&out, ", ");
    sb_printf(&out, "%s %s", params[i].type, params[i].name);
  }
  sb_puts(&out, ") {\n");
  if (decls.len > 0) sb_puts(&out, decls.buf);
  sb_puts(&out, body.buf);
  sb_puts(&out, "}\n");

  /* _call wrapper: takes void** and dispatches to the typed function */
  sb_printf(&out, "void %s_call(void **args) {\n", fn_name);
  sb_printf(&out, "  %s(", fn_name);
  for (int i = 0; i < n_params; i++) {
    if (i > 0) sb_puts(&out, ", ");
    int arg_idx = params[i].order;
    if (params[i].runtime_core_id) {
      sb_puts(&out, "0");
      continue;
    }
    /* CPUProgram numeric vals use the signed64 C transport before the
     * ordinary typed function conversion; buffers remain addresses. */
    if (strchr(params[i].type, '*')) {
      /* extract base type (before '* restrict') */
      char base[128];
      const char *star = strchr(params[i].type, '*');
      int blen = (int)(star - params[i].type);
      if (blen >= (int)sizeof(base)) blen = (int)sizeof(base) - 1;
      memcpy(base, params[i].type, blen);
      base[blen] = '\0';
      sb_printf(&out, "(%s*)args[%d]", base, arg_idx);
    } else {
      sb_printf(&out, "*(int64_t*)args[%d]", arg_idx);
    }
  }
  sb_puts(&out, ");\n}\n");

  /* _call_core wrapper: same ABI as tinygrad's CPU runtimevars path. The
   * runtime calls the same compiled kernel once per worker with a different
   * core_id, while non-threaded kernels ignore the second argument. */
  sb_printf(&out, "void %s_call_core(void **args, int core_id) {\n", fn_name);
  sb_printf(&out, "  %s(", fn_name);
  for (int i = 0; i < n_params; i++) {
    if (i > 0) sb_puts(&out, ", ");
    int arg_idx = params[i].order;
    if (params[i].runtime_core_id) {
      sb_puts(&out, "core_id");
      continue;
    }
    if (strchr(params[i].type, '*')) {
      char base[128];
      const char *star = strchr(params[i].type, '*');
      int blen = (int)(star - params[i].type);
      if (blen >= (int)sizeof(base)) blen = (int)sizeof(base) - 1;
      memcpy(base, params[i].type, blen);
      base[blen] = '\0';
      sb_printf(&out, "(%s*)args[%d]", base, arg_idx);
    } else {
      sb_printf(&out, "*(int64_t*)args[%d]", arg_idx);
    }
  }
  sb_puts(&out, ");\n}\n");

  /* cleanup */
  free(params);
  free(decls.buf);
  free(body.buf);
  free(child_count);
  free(indices);
  poly_map_destroy(uop_indices);
  smap_destroy(&names);

  return out.buf;

fail:
  free(params);
  free(decls.buf);
  free(body.buf);
  free(child_count);
  free(indices);
  poly_map_destroy(uop_indices);
  smap_destroy(&names);
  return NULL;
}
