/*
 * renderer/hip.c — current Tinygrad renderer/cstyle.py:HIPRenderer port
 *
 * Walks linearized UOps and emits HIP C++ source code for AMD GPUs.
 * Uses AMD-specific syntax from Tinygrad's
 * AMDHIPRenderer (cstyle.py:466-563).
 *
 * Key differences from CUDA renderer:
 *   - Kernel attribute: __attribute__((global)) + amdgpu_flat_work_group_size
 *   - Thread indices: __ockl_get_group_id / __ockl_get_local_id
 *   - Math intrinsics: __ocml_exp2_f32, __ocml_log2_f32, etc.
 *   - FMA: __builtin_fmaf (not __fmaf_rn)
 *   - Shared memory: __attribute__((shared, aligned(16)))
 *   - Barrier: __syncthreads() (portable, works in HIP)
 */

#ifdef POLY_HAS_HIP

#define _POSIX_C_SOURCE 200809L

#include "codegen/codegen.h"
#include "codegen/opt/tc.h"
#include "renderer/cstyle.h"
#include "bigint.h"
#include "engine/schedule.h"
#include "uop/upat.h"
#include "uop/ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <limits.h>
#include "utils.h"

/* String builder */

typedef struct {
  char *buf;
  int len;
  int cap;
} HipStrBuf;

static void hsb_init(HipStrBuf *sb) {
  sb->cap = 512;
  sb->buf = malloc(sb->cap);
  sb->buf[0] = '\0';
  sb->len = 0;
}

static void hsb_printf(HipStrBuf *sb, const char *fmt, ...) {
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

static void hsb_puts(HipStrBuf *sb, const char *s) {
  hsb_printf(sb, "%s", s);
}

/* Pointer -> string hash map */

typedef struct {
  PolyUOp **keys;
  char **vals;
  int cap;
} HipStrMap;

static void hsmap_init(HipStrMap *m, int n) {
  m->cap = (n < 4) ? 16 : n * 3;
  m->keys = calloc(m->cap, sizeof(PolyUOp *));
  m->vals = calloc(m->cap, sizeof(char *));
}

static void hsmap_set(HipStrMap *m, PolyUOp *key, char *val) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
  while (m->keys[h] && m->keys[h] != key)
    h = (h + 1) % m->cap;
  if (m->keys[h] == key) free(m->vals[h]);
  m->keys[h] = key;
  m->vals[h] = val;
}

static char *hsmap_get(HipStrMap *m, PolyUOp *key) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
  while (m->keys[h]) {
    if (m->keys[h] == key) return m->vals[h];
    h = (h + 1) % m->cap;
  }
  return NULL;
}

static void hsmap_destroy(HipStrMap *m) {
  for (int i = 0; i < m->cap; i++)
    if (m->vals[i]) free(m->vals[i]);
  free(m->keys);
  free(m->vals);
}

/* Type rendering */

/* Map scalar PolyDType to a short identifier-safe name for vector typedefs.
 * E.g. POLY_FLOAT16 -> "half", POLY_FLOAT32 -> "float", POLY_INT32 -> "int".
 * Returns the raw C type for scalars and the typedef alias for vectors. */
/* Match by priority (not full poly_dtype_eq) so pointer-derived dtypes work. */
static const char *hip_scalar_alias(PolyDType s) {
  if (poly_dtype_eq(s, POLY_FP8E4M3)) return "hip_fp8";
  if (poly_dtype_eq(s, POLY_FP8E5M2)) return "hip_bf8";
  if (poly_dtype_eq(s, POLY_BOOL)) return "bool";
  if (poly_dtype_eq(s, POLY_INT8)) return "signed_char";
  if (poly_dtype_eq(s, POLY_UINT8)) return "unsigned_char";
  if (poly_dtype_eq(s, POLY_INT16)) return "short";
  if (poly_dtype_eq(s, POLY_UINT16)) return "unsigned_short";
  if (poly_dtype_eq(s, POLY_INT32)) return "int";
  if (poly_dtype_eq(s, POLY_UINT32)) return "unsigned_int";
  if (poly_dtype_eq(s, POLY_INT64)) return "long";
  if (poly_dtype_eq(s, POLY_UINT64)) return "unsigned_long";
  if (poly_dtype_eq(s, POLY_FLOAT16)) return "half";
  if (poly_dtype_eq(s, POLY_BFLOAT16)) return "hip_bfloat16";
  if (poly_dtype_eq(s, POLY_FLOAT32)) return "float";
  if (poly_dtype_eq(s, POLY_FLOAT64)) return "double";
  return s.name;
}

/* Map scalar PolyDType to the actual C type used in HIP code.
 * Matches by priority+bitsize (not full poly_dtype_eq) so pointer-derived
 * dtypes also resolve correctly. */
static const char *hip_scalar_ctype(PolyDType s) {
  if (poly_dtype_eq(s, POLY_FP8E4M3)) return "hip_fp8";
  if (poly_dtype_eq(s, POLY_FP8E5M2)) return "hip_bf8";
  if (s.priority == POLY_FLOAT16.priority && s.bitsize == 16) return "half";
  if (s.priority == POLY_BFLOAT16.priority && s.bitsize == 16) return "hip_bfloat16";
  return s.name;
}

/* Render a HIP C++ type for a PolyDType.
 * Scalars: raw C type (_Float16, float, int, etc.)
 * Vectors: typedef alias (half4, float4, int4, etc.)
 * Vector typedefs must be emitted in the kernel prefix. */
static void hip_render_ctype(PolyDType dtype, int lanes, char *buf, int cap) {
  if (lanes <= 1) {
    snprintf(buf, cap, "%s", hip_scalar_ctype(dtype));
    return;
  }
  snprintf(buf, cap, "%s%d", hip_scalar_alias(dtype), lanes);
}

/* Pinned CStyleLanguage.render_access (renderer/cstyle.py:179-184).  The
 * scalar base type stays on BUFFER; a width-changing INDEX/SHRINK is accessed
 * by casting its address to the vector pointer type and dereferencing it. */
static void hip_render_access_expr(
    char *buf, int cap, const char *address, PolyDType value_dtype, int lanes
) {
  if (lanes > 1) {
    char value_type[128];
    hip_render_ctype(value_dtype, lanes, value_type, sizeof(value_type));
    snprintf(buf, cap, "*((%s*)(%s))", value_type, address ? address : "0");
  } else {
    snprintf(buf, cap, "*%s", address ? address : "0");
  }
}

/* Render helpers */

static char *hip_render_int64_const(int64_t v, char *buf, int cap) {
  if (v == INT64_MIN)
    snprintf(buf, cap, "(-9223372036854775807ll - 1ll)");
  else
    snprintf(buf, cap, "%lldll", (long long)v);
  return buf;
}

static char *hip_render_float_const(double v, PolyDType dt, char *buf, int cap) {
  bool is_f64 = poly_dtype_eq(dt, POLY_FLOAT64);
  if (isinf(v)) {
    if (is_f64)
      snprintf(buf, cap, v > 0 ? "(__builtin_inf())" : "(-__builtin_inf())");
    else
      snprintf(buf, cap, v > 0 ? "(__builtin_inff())" : "(-__builtin_inff())");
    return buf;
  }
  if (isnan(v)) {
    snprintf(buf, cap, is_f64 ? "(__builtin_nan(\"\"))" : "(__builtin_nanf(\"\"))");
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
  } else {
    /* f32: round-trip float32 through text with 'f' suffix. */
    snprintf(buf, cap, "%.9g", (double)(float)v);
    if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E')) {
      int len = (int)strlen(buf);
      if (len + 2 < cap) {
        buf[len] = '.';
        buf[len + 1] = '0';
        buf[len + 2] = '\0';
      }
    }
    int len = (int)strlen(buf);
    if (len + 1 < cap) {
      buf[len] = 'f';
      buf[len + 1] = '\0';
    }
  }
  return buf;
}

/* Current renderer/cstyle.py:26-47 formats CAST(strong, CONST(weak/bool))
 * as a literal of the strong destination dtype. */
static char *hip_render_const_literal(PolyUOp *c, PolyDType dtype) {
  if (!c || c->op != POLY_OP_CONST) return NULL;
  PolyDType scalar = dtype;
  char val[192];
  if (poly_dtype_eq(scalar, POLY_FLOAT16) || poly_dtype_eq(scalar, POLY_BFLOAT16)) {
    char tmp[64];
    hip_render_float_const(c->arg.f, POLY_FLOAT32, tmp, sizeof(tmp));
    snprintf(val, sizeof(val), "((%s)(%s))", hip_scalar_ctype(scalar), tmp);
  } else if (poly_dtype_is_float(scalar)) {
    hip_render_float_const(c->arg.f, scalar, val, sizeof(val));
  } else if (poly_dtype_is_bool(scalar)) {
    snprintf(val, sizeof(val), "%d", c->arg.b ? 1 : 0);
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
    hip_render_int64_const(c->arg.i, val, sizeof(val));
  } else if (poly_dtype_eq(scalar, POLY_UINT64)) {
    snprintf(
        val, sizeof(val), "%lluull",
        (unsigned long long)poly_arg_integer_to_u64_mod(c->arg)
    );
  } else if (poly_dtype_eq(scalar, POLY_UINT32)) {
    snprintf(
        val, sizeof(val), "%uu",
        (unsigned)(uint32_t)poly_arg_integer_to_u64_mod(c->arg)
    );
  } else if (poly_dtype_eq(scalar, POLY_UINT8) || poly_dtype_eq(scalar, POLY_UINT16) ||
             poly_dtype_eq(scalar, POLY_INT8) || poly_dtype_eq(scalar, POLY_INT16)) {
    if (poly_dtype_is_unsigned(scalar))
      snprintf(
          val, sizeof(val), "((%s)(%uu))", hip_scalar_ctype(scalar),
          (unsigned)(uint32_t)poly_arg_integer_to_u64_mod(c->arg)
      );
    else
      snprintf(
          val, sizeof(val), "((%s)(%lld))", hip_scalar_ctype(scalar),
          (long long)c->arg.i
      );
  } else if (c->arg.kind == POLY_ARG_BIGINT) {
    return poly_arg_integer_to_decimal(c->arg);
  } else {
    snprintf(val, sizeof(val), "%lld", (long long)c->arg.i);
  }
  return strdup(val);
}

static void hip_render_alu(
    char *buf,
    int cap,
    PolyOps op,
    PolyDType dtype,
    const char *s0,
    const char *s1,
    const char *s2
) {
  switch (op) {
  case POLY_OP_NEG:
    /* Pinned CStyleLanguage/HIPRenderer inherits arithmetic NEG. */
    snprintf(buf, cap, "(-%s)", s0);
    break;
  case POLY_OP_SQRT:
    snprintf(
        buf, cap,
        poly_dtype_eq(dtype, POLY_FLOAT64) ? "__ocml_sqrt_f64(%s)" : "__ocml_sqrt_f32(%s)", s0
    );
    break;
  case POLY_OP_TRUNC:
    snprintf(
        buf, cap,
        poly_dtype_eq(dtype, POLY_FLOAT64) ? "__ocml_trunc_f64(%s)" : "__ocml_trunc_f32(%s)", s0
    );
    break;
  case POLY_OP_EXP2:
    snprintf(
        buf, cap,
        poly_dtype_eq(dtype, POLY_FLOAT64) ? "__ocml_exp2_f64(%s)" : "__ocml_exp2_f32(%s)", s0
    );
    break;
  case POLY_OP_LOG2:
    snprintf(
        buf, cap,
        poly_dtype_eq(dtype, POLY_FLOAT64) ? "__ocml_log2_f64(%s)" : "__ocml_log2_f32(%s)", s0
    );
    break;
  case POLY_OP_SIN:
    snprintf(
        buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64) ? "__ocml_sin_f64(%s)" : "__ocml_sin_f32(%s)",
        s0
    );
    break;
  case POLY_OP_RECIPROCAL:
    snprintf(buf, cap, "(1/%s)", s0);
    break;
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
    snprintf(buf, cap, "((%s>%s)?%s:%s)", s0, s1, s0, s1);
    break;
  case POLY_OP_POW:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64) ? "pow(%s, %s)" : "powf(%s, %s)", s0, s1);
    break;
  case POLY_OP_WHERE:
    snprintf(buf, cap, "(%s?%s:%s)", s0, s1, s2);
    break;
  case POLY_OP_MULACC:
    snprintf(
        buf, cap,
        poly_dtype_eq(dtype, POLY_FLOAT64) ? "__builtin_fma(%s,%s,%s)" : "__builtin_fmaf(%s,%s,%s)",
        s0, s1, s2
    );
    break;
  default:
    snprintf(buf, cap, "/* unknown op %d */0", op);
    break;
  }
}

static int hip_range_slot(PolyUOp **ranges, int *n_ranges, PolyUOp *r, bool create) {
  if (!r) return -1;
  for (int i = 0; i < *n_ranges; i++) {
    if (ranges[i] == r) return i;
  }
  if (!create || *n_ranges >= 128) return -1;
  ranges[*n_ranges] = r;
  (*n_ranges)++;
  return *n_ranges - 1;
}

static bool hip_arch_is(const char *arch, const char *base) {
  if (!arch || !base) return false;
  size_t n = strlen(base);
  return strncmp(arch, base, n) == 0 && (arch[n] == '\0' || arch[n] == ':');
}

static bool hip_is_cdna(const char *arch) {
  return hip_arch_is(arch, "gfx942") || hip_arch_is(arch, "gfx950");
}

static PolyRendererCaps poly_hip_renderer_caps(const char *arch) {
  int n_tensor_cores = 0;
  const PolyTensorCore *tensor_cores = poly_tc_get_amd(arch, &n_tensor_cores);
  bool supports_fp8 = hip_arch_is(arch, "gfx950");
  return (PolyRendererCaps){
      .device = "AMD",
      /* Pinned HIPRenderer inherits CStyleLanguage.code_for_op and advertises
       * neither MULACC nor FDIV (renderer/cstyle.py:128-136,472-508). */
      .has_mulacc = false,
      .has_threefry = false,
      .has_exp2 = true,
      .has_log2 = true,
      .has_sin = true,
      .has_fdiv = false,
      .supports_float16 = true,
      .supports_bfloat16 = true,
      .supports_fp8e4m3 = supports_fp8,
      .supports_fp8e5m2 = supports_fp8,
      .has_int64 = true,
      .has_local = true,
      /* Pinned Renderer.supports_float4=True and split_load_store keeps
       * aligned float32/float16 memory operations at width four. BF16 remains
       * scalar through the shared dtype allowlist (devectorizer.py:155-177). */
      .max_vec_width = 4,
      .global_max = {2147483647, 65535, 65535},
      .tensor_cores = tensor_cores,
      .n_tensor_cores = n_tensor_cores,
  };
}

/* HIP Linearizer */

PolyUOp *poly_rewrite_hip(PolyCtx *ctx, PolyUOp *sink) {
  const char *arch = poly_hip_arch();
  PolyPatternMatcher *extra = poly_hip_renderer_extra_matcher();
  PolyPatternMatcher *extra_with_manual = NULL;
  if (!hip_arch_is(arch, "gfx950")) {
    /* Tinygrad 2026-08-22/a9069c177a9d HIPRenderer.__init__ appends the
     * shared manual BF16 casts on every architecture before CDNA4. */
    extra_with_manual = poly_pm_concat(extra, poly_pm_manual_bf16_cast());
    if (!extra_with_manual) return NULL;
    extra = extra_with_manual;
  }
  PolyRewriteOpts opts = {
      .optimize =
          poly_kernel_optimize_enabled(sink), /* shared optimized pipeline (tinygrad parity) */
      .beam_width = poly_kernel_beam(sink),
      .caps = poly_hip_renderer_caps(arch),
      .device = POLY_DEVICE_HIP,
      .opt_policy = POLY_OPT_TC_ONLY,
      /* Pinned HIPRenderer keeps BF16 in supported_dtypes, so BF16 storage
       * and WMMA fragments bypass pm_dtype_decomps. Ordinary BF16 ALU/casts
       * are handled only by the renderer-final matcher
       * (renderer/cstyle.py:472-486,515,574-575). */
      .extra_matcher = extra,
  };
  PolyUOp *ret = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  poly_pm_destroy(extra_with_manual);
  return ret;
}

PolyUOp **poly_linearize_hip(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  sink = poly_rewrite_hip(ctx, sink);
  return poly_do_linearize(ctx, sink, n_out);
}

/* Track which OCML/OCKL functions are used */

typedef struct {
  bool uses_special; /* needs ockl workitem functions */
  bool uses_exp2_f32;
  bool uses_log2_f32;
  bool uses_sin_f32;
  bool uses_sqrt_f32;
  bool uses_trunc_f32;
  bool uses_exp2_f64;
  bool uses_log2_f64;
  bool uses_sin_f64;
  bool uses_sqrt_f64;
  bool uses_trunc_f64;
  bool uses_wmma;
  bool uses_f16;
  bool uses_bf16;
  bool uses_fp8;
  bool uses_f32_to_fp8;
  /* Collect unique vector dtypes that need typedefs */
  PolyDType vec_dtypes[32];
  int vec_lanes[32];
  int n_vec_dtypes;
} HipUsedFuncs;

/* C collection mechanics for current uops_to_dtypes and HIP render prefix selection. */
static void hip_scan_used_funcs(PolyCtx *ctx, PolyUOp **uops, int n, HipUsedFuncs *used) {
  memset(used, 0, sizeof(*used));
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_SPECIAL) used->uses_special = true;
    if (poly_dtype_eq(u->dtype, POLY_FLOAT16)) used->uses_f16 = true;
    if (poly_dtype_eq(u->dtype, POLY_BFLOAT16)) used->uses_bf16 = true;
    if (poly_dtype_is_fp8(u->dtype)) used->uses_fp8 = true;
    if (u->op == POLY_OP_CAST && u->n_src == 1 && poly_dtype_is_fp8(u->dtype) &&
        (poly_dtype_eq(u->src[0]->dtype, POLY_FLOAT32) ||
         u->src[0]->op == POLY_OP_CONST))
      used->uses_f32_to_fp8 = true;
    bool is_f64 = poly_dtype_eq(u->dtype, POLY_FLOAT64);
    switch (u->op) {
    case POLY_OP_EXP2:
      if (is_f64)
        used->uses_exp2_f64 = true;
      else
        used->uses_exp2_f32 = true;
      break;
    case POLY_OP_LOG2:
      if (is_f64)
        used->uses_log2_f64 = true;
      else
        used->uses_log2_f32 = true;
      break;
    case POLY_OP_SIN:
      if (is_f64)
        used->uses_sin_f64 = true;
      else
        used->uses_sin_f32 = true;
      break;
    case POLY_OP_SQRT:
      if (is_f64)
        used->uses_sqrt_f64 = true;
      else
        used->uses_sqrt_f32 = true;
      break;
    case POLY_OP_TRUNC:
      if (is_f64)
        used->uses_trunc_f64 = true;
      else
        used->uses_trunc_f32 = true;
      break;
    case POLY_OP_WMMA: {
      used->uses_wmma = true;
      if (u->arg.kind == POLY_ARG_TENSOR_CORE) {
        used->uses_f16 |= poly_dtype_eq(u->arg.tensor_core.dtype_in, POLY_FLOAT16);
        used->uses_bf16 |= poly_dtype_eq(u->arg.tensor_core.dtype_in, POLY_BFLOAT16);
        used->uses_fp8 |= poly_dtype_is_fp8(u->arg.tensor_core.dtype_in);
      }
      break;
    }
    default:
      break;
    }
    /* Current CStyle render_type uses the scalar dtype plus max_numel. */
    int64_t lanes = poly_uop_max_numel(ctx, u);
    if (lanes > 1 && lanes <= INT_MAX && used->n_vec_dtypes < 32) {
      bool dup = false;
      for (int j = 0; j < used->n_vec_dtypes; j++) {
        if (poly_dtype_eq(used->vec_dtypes[j], u->dtype) && used->vec_lanes[j] == lanes) {
          dup = true;
          break;
        }
      }
      if (!dup) {
        used->vec_dtypes[used->n_vec_dtypes] = u->dtype;
        used->vec_lanes[used->n_vec_dtypes++] = (int)lanes;
      }
    }
  }
}

static const char *hip_wmma_type_name(PolyDType dtype) {
  if (poly_dtype_eq(dtype, POLY_FLOAT32)) return "f32";
  if (poly_dtype_eq(dtype, POLY_FLOAT16)) return "f16";
  if (poly_dtype_eq(dtype, POLY_BFLOAT16)) return "bf16";
  return NULL;
}

/* Tinygrad 2026-08-22/a9069c177a9d HIPRenderer.render_kernel emits the
 * architecture-specific builtin binding for each wmma_args signature. */
static bool hip_render_wmma_prefix(
    HipStrBuf *out, PolyUOp *wmma, const char *arch
) {
  if (!wmma || wmma->n_src != 3 || wmma->arg.kind != POLY_ARG_TENSOR_CORE)
    return false;
  char name[128];
  if (!poly_wmma_name(wmma, name, sizeof(name))) return false;
  int n = wmma->arg.tensor_core.dims[0];
  int m = wmma->arg.tensor_core.dims[1];
  int k = wmma->arg.tensor_core.dims[2];
  PolyDType dtype_in = wmma->arg.tensor_core.dtype_in;

  if (hip_is_cdna(arch)) {
    if (!poly_dtype_eq(wmma->dtype, POLY_FLOAT32)) return false;
    const char *suffix = NULL;
    if (k == 16) {
      if (poly_dtype_eq(dtype_in, POLY_FLOAT16)) suffix = "f16";
      if (poly_dtype_eq(dtype_in, POLY_BFLOAT16)) suffix = "bf16_1k";
    } else if (k == 32) {
      if (poly_dtype_eq(dtype_in, POLY_FLOAT16)) suffix = "_f16";
      if (poly_dtype_eq(dtype_in, POLY_BFLOAT16)) suffix = "_bf16";
      if (poly_dtype_eq(dtype_in, POLY_FP8E4M3)) suffix = "_fp8_fp8";
      if (poly_dtype_eq(dtype_in, POLY_FP8E5M2)) suffix = "_bf8_bf8";
    } else if (k == 128 && poly_dtype_is_fp8(dtype_in)) {
      suffix = "_f8f6f4";
    }
    if (!suffix) return false;
    hsb_printf(
        out, "#define __%s __builtin_amdgcn_mfma_%sf32_%dx%dx%d%s\n", name,
        k == 128 ? "scale_" : "", n, m, k, suffix
    );
    return true;
  }

  if (hip_arch_is(arch, "gfx1200") || hip_arch_is(arch, "gfx1201")) {
    const char *dtype_out = hip_wmma_type_name(wmma->dtype);
    const char *input = hip_wmma_type_name(dtype_in);
    if (!dtype_out || !input) return false;
    hsb_printf(
        out, "#define __%s __builtin_amdgcn_wmma_%s_16x16x16_%s_w32_gfx12\n",
        name, dtype_out, input
    );
    return true;
  }

  if (poly_dtype_eq(wmma->dtype, POLY_INT32)) {
    hsb_puts(out, "typedef int wmma_int4 __attribute__((ext_vector_type(4)));\n");
    hsb_printf(
        out,
        "static inline __attribute__((device)) int8 __%s(signed_char16 a, signed_char16 b, int8 c) {\n"
        "  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, __builtin_bit_cast(wmma_int4, a),\n"
        "    true, __builtin_bit_cast(wmma_int4, b), c, false);\n}\n",
        name
    );
    return true;
  }
  if (poly_dtype_eq(wmma->dtype, POLY_FLOAT32)) {
    const char *input = poly_dtype_eq(dtype_in, POLY_FLOAT16) ? "f16" : "bf16";
    hsb_printf(
        out, "#define __%s __builtin_amdgcn_wmma_f32_16x16x16_%s_w32\n", name,
        input
    );
    return true;
  }
  if (poly_dtype_eq(wmma->dtype, POLY_FLOAT16) &&
      poly_dtype_eq(dtype_in, POLY_FLOAT16)) {
    hsb_printf(
        out,
        "static inline __attribute__((device)) half8 __%s(half16 a, half16 b, half8 c) {\n"
        "  half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }\n"
        "  c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);\n"
        "  for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;\n}\n",
        name
    );
    return true;
  }
  return false;
}

/* HIP Renderer */

char *poly_render_hip(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n,
    const char *fn_name,
    int launch_bounds,
    const char *arch
) {
  HipStrBuf decls, body;
  hsb_init(&decls);
  hsb_init(&body);

  HipStrMap names;
  hsmap_init(&names, n);

  char *param_types[64];
  char *param_names[64];
  int param_order[64];
  int n_params = 0;

  int c_val = 0, c_alu = 0, c_cast = 0, c_acc = 0;
  int depth = 1;

  /* Range liveness tracking */
  PolyUOp *live_ranges[128];
  int live_remaining[128];
  int n_live_ranges = 0;
  memset(live_ranges, 0, sizeof(live_ranges));
  memset(live_remaining, 0, sizeof(live_remaining));

  PolyUOp *open_ranges[128];
  int n_open_ranges = 0;
  memset(open_ranges, 0, sizeof(open_ranges));

  /* Pre-scan: count range references for liveness */
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_RANGE) (void)hip_range_slot(live_ranges, &n_live_ranges, u, true);
    if (u->op == POLY_OP_END) continue;
    for (int j = 0; j < u->n_src; j++) {
      if (u->src[j] && u->src[j]->op == POLY_OP_RANGE) {
        int ri = hip_range_slot(live_ranges, &n_live_ranges, u->src[j], true);
        if (ri >= 0) live_remaining[ri]++;
      }
    }
  }

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];

    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP) continue;

    if (u->op != POLY_OP_END) {
      for (int j = 0; j < u->n_src; j++) {
        if (u->src[j] && u->src[j]->op == POLY_OP_RANGE) {
          int ri = hip_range_slot(live_ranges, &n_live_ranges, u->src[j], false);
          if (ri >= 0 && live_remaining[ri] > 0) live_remaining[ri]--;
        }
      }
    }

    /* Current cstyle renders GLOBAL PARAMs as pointers and ALU PARAMs as
     * scalar kernel arguments from the same numbered ParamArg sequence. */
    if (u->op == POLY_OP_PARAM) {
      int64_t slot = poly_program_buffer_slot(u);
      if (slot < 0) {
        for (int j = 0; j < n_params; j++) {
          free(param_types[j]);
          free(param_names[j]);
        }
        free(decls.buf);
        free(body.buf);
        hsmap_destroy(&names);
        return NULL;
      }
      char name[32];
      snprintf(name, sizeof(name), "data%lld", (long long)slot);
      hsmap_set(&names, u, strdup(name));

      PolyDType base = u->dtype;
      char type[64];
      snprintf(
          type, sizeof(type), poly_uop_is_alu_param(u) ? "const %s" : "%s*",
          hip_scalar_ctype(base)
      );
      param_types[n_params] = strdup(type);
      param_names[n_params] = strdup(name);
      param_order[n_params] = (int)slot;
      n_params++;
      continue;
    }

    /* --- CONST -------------------------------------------------------- */
    if (u->op == POLY_OP_CONST) {
      char *literal = hip_render_const_literal(u, u->dtype);
      if (!literal) return NULL;
      hsmap_set(&names, u, literal);
      continue;
    }

    /* --- INDEX: pointer arithmetic or vector lane extract ------------- */
    if (u->op == POLY_OP_INDEX) {
      char *buf_s = hsmap_get(&names, u->src[0]);
      char *idx_s = hsmap_get(&names, u->src[1]);
      char expr[256];
      if (poly_is_program_memory_base(u->src[0]))
        snprintf(expr, sizeof(expr), "(%s+%s)", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      else
        snprintf(expr, sizeof(expr), "%s[%s]", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      hsmap_set(&names, u, strdup(expr));
      continue;
    }

    /* --- SHRINK: late codegen memory slice --------------------------- */
    if (u->op == POLY_OP_SHRINK) {
      char *buf_s = hsmap_get(&names, u->src[0]);
      char *idx_s = hsmap_get(&names, u->src[1]);
      char expr[256];
      snprintf(expr, sizeof(expr), "(%s+%s)", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      hsmap_set(&names, u, strdup(expr));
      continue;
    }

    /* --- SPECIAL: GPU thread index (AMD OCKL intrinsics) -------------- */
    if (u->op == POLY_OP_SPECIAL) {
      const char *sname = u->arg.str ? u->arg.str : "gidx0";
      hsmap_set(&names, u, strdup(sname));

      /* Determine which dimension: last char of name */
      int dim_idx = 0;
      int slen = (int)strlen(sname);
      if (slen > 0) dim_idx = sname[slen - 1] - '0';

      /* HIP uses OCKL workitem intrinsics */
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");
      if (sname[0] == 'l') {
        /* Local index: __ockl_get_local_id */
        hsb_printf(&body, "int %s = __ockl_get_local_id(%d);\n", sname, dim_idx);
      } else {
        /* Global index: __ockl_get_group_id * __ockl_get_local_size + __ockl_get_local_id */
        hsb_printf(
            &body,
            "int %s = "
            "(__ockl_get_group_id(%d)*__ockl_get_local_size(%d)+__ockl_get_local_id(%d));\n",
            sname, dim_idx, dim_idx, dim_idx
        );
      }

      /* Bounds check for global indices only */
      if (sname[0] != 'l') {
        char *bound = hsmap_get(&names, u->src[0]);
        if (bound) {
          for (int d = 0; d < depth; d++)
            hsb_puts(&body, "  ");
          hsb_printf(&body, "if (%s >= %s) return;\n", sname, bound);
        }
      }
      continue;
    }

    /* --- BARRIER ------------------------------------------------------ */
    if (u->op == POLY_OP_BARRIER) {
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");
      hsb_puts(&body, "__syncthreads();\n");
      continue;
    }

    /* --- RANGE: for loop ---------------------------------------------- */
    if (u->op == POLY_OP_RANGE) {
      char *range = poly_range_str(u->arg);
      char *name = malloc(strlen(range) + 6);
      snprintf(name, strlen(range) + 6, "%cidx%s", poly_axis_letter(u->arg), range);
      free(range);
      hsmap_set(&names, u, name);

      char *bound = hsmap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");
      hsb_printf(&body, "for (int %s = 0; %s < %s; %s++) {\n", name, name, bound, name);
      depth++;
      if (n_open_ranges < 128) open_ranges[n_open_ranges++] = u;
      continue;
    }

    /* --- END / ENDIF -------------------------------------------------- */
    if (u->op == POLY_OP_END || u->op == POLY_OP_ENDIF) {
      if (u->op == POLY_OP_END && u->n_src > 1 && u->src[1]->op == POLY_OP_RANGE) {
        PolyUOp *want = u->src[1];
        int wi = hip_range_slot(live_ranges, &n_live_ranges, want, false);
        if (wi >= 0 && live_remaining[wi] > 0) continue;

        int pos = -1;
        for (int p = n_open_ranges - 1; p >= 0; p--) {
          if (open_ranges[p] == want) {
            pos = p;
            break;
          }
        }
        if (pos < 0) continue;

        bool can_close = true;
        for (int p = n_open_ranges - 1; p >= pos; p--) {
          int oi = hip_range_slot(live_ranges, &n_live_ranges, open_ranges[p], false);
          if (oi >= 0 && live_remaining[oi] > 0 && open_ranges[p] != want) {
            can_close = false;
            break;
          }
        }
        if (!can_close) continue;

        while (n_open_ranges > pos) {
          depth--;
          for (int d = 0; d < depth; d++)
            hsb_puts(&body, "  ");
          hsb_puts(&body, "}\n");
          n_open_ranges--;
        }
        continue;
      }

      depth--;
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");
      hsb_puts(&body, "}\n");
      if (u->op == POLY_OP_END && n_open_ranges > 0) n_open_ranges--;
      continue;
    }

    /* Pinned tinygrad's C-style renderers consume the LOCAL BUFFER emitted by
     * pm_add_buffers_local directly. */
    if (u->op == POLY_OP_BUFFER && poly_program_memory_is(u, POLY_ADDR_LOCAL)) {
      char name[32];
      snprintf(
          name, sizeof(name), "smem%lld",
          (long long)poly_program_buffer_slot(u)
      );
      hsmap_set(&names, u, strdup(name));

      /* HIP shared memory: __attribute__((shared, aligned(16))) */
      int64_t smem_size = poly_program_buffer_size(u);
      PolyDType base = poly_program_buffer_dtype(u);
      hsb_printf(
          &decls, "  __attribute__((shared, aligned(16))) %s %s[%lld];\n",
          hip_scalar_ctype(base), name, (long long)smem_size
      );
      continue;
    }

    /* --- register buffer --------------------------------------------- */
    if (u->op == POLY_OP_BUFFER && poly_program_memory_is(u, POLY_ADDR_REG)) {
      char name[32];
      snprintf(name, sizeof(name), "r%lld", (long long)poly_program_buffer_slot(u));
      hsmap_set(&names, u, strdup(name));

      PolyDType base = poly_program_buffer_dtype(u);
      hsb_printf(
          &decls, "  %s %s[%lld];\n", hip_scalar_ctype(base), name,
          (long long)poly_program_buffer_size(u)
      );
      continue;
    }

    /* --- AFTER -------------------------------------------------------- */
    if (u->op == POLY_OP_AFTER) {
      char *src_name = hsmap_get(&names, u->src[0]);
      if (src_name) hsmap_set(&names, u, strdup(src_name));
      continue;
    }

    /* --- LOAD --------------------------------------------------------- */
    if (u->op == POLY_OP_LOAD) {
      char name[32];
      snprintf(name, sizeof(name), "val%d", c_val++);
      hsmap_set(&names, u, strdup(name));

      char *bidx = hsmap_get(&names, u->src[0]);
      char access[512];
      int lanes = (int)poly_uop_max_numel(ctx, u);
      hip_render_access_expr(access, sizeof(access), bidx, u->dtype, lanes);
      {
        char ctype[128];
        hip_render_ctype(u->dtype, lanes, ctype, sizeof(ctype));
        hsb_printf(&decls, "  %s %s;\n", ctype, name);
      }
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");

      /* Pinned tinygrad final IR: LOAD(INDEX(buf, idx), alt, gate). */
      PolyUOp *gate_uop = (u->n_src >= 3 && poly_dtype_is_bool(u->src[2]->dtype))
                              ? u->src[2]
                              : NULL;
      if (gate_uop && u->n_src >= 2) {
        char *gate_s = hsmap_get(&names, gate_uop);
        char *alt_s = hsmap_get(&names, u->src[1]);
        hsb_printf(&body, "%s = (%s?%s:%s);\n", name, gate_s, access, alt_s);
      } else if (gate_uop) {
        char *gate_s = hsmap_get(&names, gate_uop);
        char ctype[128];
        hip_render_ctype(u->dtype, lanes, ctype, sizeof(ctype));
        hsb_printf(&body, "%s = (%s?%s:(%s)0);\n", name, gate_s, access, ctype);
      } else {
        hsb_printf(&body, "%s = %s;\n", name, access);
      }
      continue;
    }

    /* --- STORE -------------------------------------------------------- */
    if (u->op == POLY_OP_STORE) {
      char *target = hsmap_get(&names, u->src[0]);
      char *val = hsmap_get(&names, u->src[1]);
      char access[512];
      int lanes = u->src[1] ? (int)poly_uop_max_numel(ctx, u->src[1]) : 1;
      hip_render_access_expr(
          access, sizeof(access), target, u->src[1] ? u->src[1]->dtype : POLY_VOID, lanes
      );
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");
      hsb_printf(&body, "%s = %s;\n", access, val);
      continue;
    }

    /* --- CAST --------------------------------------------------------- */
    if (u->op == POLY_OP_CAST) {
      if (u->n_src == 1 && u->src[0] && u->src[0]->op == POLY_OP_CONST &&
          poly_dtype_is_fp8(u->dtype) && poly_uop_max_numel(ctx, u) == 1) {
        char value[96];
        if (isnan(u->src[0]->arg.f))
          snprintf(value, sizeof(value), "NAN");
        else if (isinf(u->src[0]->arg.f))
          snprintf(value, sizeof(value), u->src[0]->arg.f > 0 ? "INFINITY" : "-INFINITY");
        else
          hip_render_float_const(
              u->src[0]->arg.f, POLY_FLOAT32, value, sizeof(value)
          );
        char expr[160];
        snprintf(
            expr, sizeof(expr), "f32_to_fp8(%s, %d)", value,
            poly_dtype_eq(u->dtype, POLY_FP8E5M2) ? 1 : 0
        );
        hsmap_set(&names, u, strdup(expr));
        continue;
      }
      if (u->n_src == 1 && u->src[0] && u->src[0]->op == POLY_OP_CONST &&
          (poly_dtype_is_weak(u->src[0]->dtype) || poly_dtype_is_bool(u->src[0]->dtype)) &&
          poly_uop_max_numel(ctx, u) == 1) {
        char *literal = hip_render_const_literal(u->src[0], u->dtype);
        if (!literal) return NULL;
        hsmap_set(&names, u, literal);
        continue;
      }
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      hsmap_set(&names, u, strdup(name));

      char *src_s = hsmap_get(&names, u->src[0]);
      {
        char ctype[128];
        hip_render_ctype(u->dtype, (int)poly_uop_max_numel(ctx, u), ctype, sizeof(ctype));
        hsb_printf(&decls, "  %s %s;\n", ctype, name);
      }
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");
      if (poly_dtype_is_fp8(u->dtype) &&
          poly_dtype_eq(u->src[0]->dtype, POLY_FLOAT32)) {
        hsb_printf(
            &body, "%s = f32_to_fp8(%s, %d);\n", name, src_s,
            poly_dtype_eq(u->dtype, POLY_FP8E5M2) ? 1 : 0
        );
      } else if (poly_dtype_eq(u->dtype, POLY_FLOAT32) &&
                 poly_dtype_is_fp8(u->src[0]->dtype)) {
        hsb_printf(
            &body, "%s = __builtin_amdgcn_cvt_f32_%s((unsigned int)%s, 0);\n", name,
            poly_dtype_eq(u->src[0]->dtype, POLY_FP8E5M2) ? "bf8" : "fp8", src_s
        );
      } else {
        char ctype2[128];
        hip_render_ctype(u->dtype, (int)poly_uop_max_numel(ctx, u), ctype2, sizeof(ctype2));
        hsb_printf(&body, "%s = (%s)(%s);\n", name, ctype2, src_s);
      }
      continue;
    }

    /* --- BITCAST: use template helper --------------------------------- */
    if (u->op == POLY_OP_BITCAST) {
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      hsmap_set(&names, u, strdup(name));

      char *src_s = hsmap_get(&names, u->src[0]);
      char dst_type[128], src_type[128];
      hip_render_ctype(u->dtype, (int)poly_uop_max_numel(ctx, u), dst_type, sizeof(dst_type));
      hip_render_ctype(
          u->src[0]->dtype, (int)poly_uop_max_numel(ctx, u->src[0]), src_type,
          sizeof(src_type)
      );
      hsb_printf(&decls, "  %s %s;\n", dst_type, name);
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");
      hsb_printf(&body, "%s = tg_bitcast<%s>((%s)(%s));\n", name, dst_type, src_type, src_s);
      continue;
    }

    /* --- ALU ---------------------------------------------------------- */
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      char expr[512];
      const char *s0 = (u->n_src > 0) ? hsmap_get(&names, u->src[0]) : "";
      const char *s1 = (u->n_src > 1) ? hsmap_get(&names, u->src[1]) : "";
      const char *s2 = (u->n_src > 2) ? hsmap_get(&names, u->src[2]) : "";
      hip_render_alu(expr, sizeof(expr), u->op, u->dtype, s0, s1, s2);

      char name[32];
      snprintf(name, sizeof(name), "alu%d", c_alu++);
      hsmap_set(&names, u, strdup(name));

      {
        char ctype[128];
        hip_render_ctype(u->dtype, (int)poly_uop_max_numel(ctx, u), ctype, sizeof(ctype));
        hsb_printf(&decls, "  %s %s;\n", ctype, name);
      }
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");
      hsb_printf(&body, "%s = %s;\n", name, expr);
      continue;
    }

    /* --- IF ----------------------------------------------------------- */
    if (u->op == POLY_OP_IF) {
      char *cond_s = hsmap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");
      hsb_printf(&body, "if (%s) {\n", cond_s);
      depth++;
      continue;
    }

    /* --- VECTORIZE: vector literal ----------------------------------- */
    if (u->op == POLY_OP_STACK) {
      char name[32];
      snprintf(name, sizeof(name), "vec%d", c_alu++);
      hsmap_set(&names, u, strdup(name));

      char ctype[128];
      hip_render_ctype(u->dtype, (int)poly_uop_max_numel(ctx, u), ctype, sizeof(ctype));
      hsb_printf(&decls, "  %s %s;\n", ctype, name);
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");

      if (u->n_src == 1) {
        char *s = hsmap_get(&names, u->src[0]);
        hsb_printf(&body, "%s = (%s)(%s);\n", name, ctype, s ? s : "0");
      } else {
        hsb_printf(&body, "%s = (%s){", name, ctype);
        for (int j = 0; j < u->n_src; j++) {
          if (j) hsb_puts(&body, ", ");
          char *s = hsmap_get(&names, u->src[j]);
          hsb_puts(&body, s ? s : "0");
        }
        hsb_puts(&body, "};\n");
      }
      continue;
    }

    /* --- WMMA: matrix multiply-accumulate ----------------------------- */
    if (u->op == POLY_OP_WMMA) {
      char name[32];
      snprintf(name, sizeof(name), "wmma%d", c_alu++);
      hsmap_set(&names, u, strdup(name));

      char ctype[128];
      int64_t lanes = poly_uop_max_numel(ctx, u);
      hip_render_ctype(u->dtype, (int)lanes, ctype, sizeof(ctype));
      hsb_printf(&decls, "  %s %s;\n", ctype, name);
      for (int d = 0; d < depth; d++)
        hsb_puts(&body, "  ");

      char *a_s = (u->n_src > 0) ? hsmap_get(&names, u->src[0]) : "0";
      char *b_s = (u->n_src > 1) ? hsmap_get(&names, u->src[1]) : "0";
      char *c_s = (u->n_src > 2) ? hsmap_get(&names, u->src[2]) : "0";
      char helper[128];
      if (!poly_wmma_name(u, helper, sizeof(helper))) return NULL;
      hsb_printf(
          &body, "%s = __%s(%s, %s, %s", name, helper, a_s ? a_s : "0",
          b_s ? b_s : "0", c_s ? c_s : "0"
      );
      if (hip_is_cdna(arch)) {
        if (u->arg.tensor_core.dims[2] == 128) {
          int fp8 = poly_dtype_eq(u->arg.tensor_core.dtype_in, POLY_FP8E5M2) ? 1 : 0;
          hsb_printf(&body, ", %d, %d, 0, 0, 0, 0", fp8, fp8);
        } else {
          hsb_puts(&body, ", 0, 0, 0");
        }
      }
      hsb_puts(&body, ");\n");
      continue;
    }
  }

  /* Sort params */
  for (int i = 1; i < n_params; i++) {
    int ko = param_order[i];
    char *kt = param_types[i], *kn = param_names[i];
    int j = i - 1;
    while (j >= 0 && param_order[j] > ko) {
      param_order[j + 1] = param_order[j];
      param_types[j + 1] = param_types[j];
      param_names[j + 1] = param_names[j];
      j--;
    }
    param_order[j + 1] = ko;
    param_types[j + 1] = kt;
    param_names[j + 1] = kn;
  }

  /* Scan which OCML/OCKL functions are used */
  HipUsedFuncs used;
  hip_scan_used_funcs(ctx, uops, n, &used);

  /* Build complete HIP source */
  HipStrBuf out;
  hsb_init(&out);

  /* Prefix: compiled via comgr with -nogpuinc (no built-in headers).
   * All device function declarations must be provided explicitly,
   * matching tinygrad's AMDHIPRenderer (cstyle.py:466-563). */
  hsb_puts(&out, "typedef long unsigned int size_t;\n");
  hsb_puts(&out, "#define INFINITY (__builtin_inff())\n");
  hsb_puts(&out, "#define NAN (__builtin_nanf(\"\"))\n");

  if (used.uses_bf16)
    hsb_printf(
        &out, "typedef %s hip_bfloat16;\n",
        hip_arch_is(arch, "gfx950") ? "__bf16" : "unsigned short"
    );
  if (used.uses_f16) hsb_puts(&out, "#define half _Float16\n");
  if (used.uses_fp8) {
    hsb_puts(&out, "typedef unsigned char hip_bf8;\n");
    hsb_puts(&out, "typedef unsigned char hip_fp8;\n");
  }
  if (used.uses_f32_to_fp8) {
    hsb_puts(
        &out,
        "static inline __attribute__((device)) unsigned char f32_to_fp8(float v, int is_bf8) {\n"
        "  v = (((*(unsigned*)&v)&0x7F800000)!=0x7F800000)?"
        "__builtin_amdgcn_fmed3f(v,is_bf8?57344.0f:448.0f,is_bf8?-57344.0f:-448.0f) : v;\n"
        "  return (unsigned char)(is_bf8?__builtin_amdgcn_cvt_pk_bf8_f32(v,v,0,false):"
        "__builtin_amdgcn_cvt_pk_fp8_f32(v,v,0,false));\n"
        "}\n"
    );
  }

  /* Bitcast template helper. With -nogpuinc, __device__/__forceinline__
   * are unavailable; use __attribute__ equivalents. */
  hsb_puts(
      &out,
      "template <class T, class F> __attribute__((device, always_inline)) T tg_bitcast(F v) {\n"
  );
  hsb_puts(&out, "  union U { F f; T t; }; U u; u.f = v; return u.t;\n");
  hsb_puts(&out, "}\n");

  /* OCKL workitem function declarations (only when SPECIAL ops are used) */
  if (used.uses_special) {
    hsb_puts(
        &out,
        "extern \"C\" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);\n"
    );
    hsb_puts(
        &out,
        "extern \"C\" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);\n"
    );
    hsb_puts(
        &out,
        "extern \"C\" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);\n"
    );
  }

  /* OCML math function declarations (only what's used) */
#define EMIT_OCML(flag, name, attr)                                                                \
  if (used.flag)                                                                                   \
  hsb_printf(                                                                                      \
      &out, "extern \"C\" __attribute__((device%s)) float __ocml_%s_f32(float);\n", attr, name     \
  )
#define EMIT_OCML_F64(flag, name, attr)                                                            \
  if (used.flag)                                                                                   \
  hsb_printf(                                                                                      \
      &out, "extern \"C\" __attribute__((device%s)) double __ocml_%s_f64(double);\n", attr, name   \
  )

  EMIT_OCML(uses_exp2_f32, "exp2", ", pure");
  EMIT_OCML(uses_log2_f32, "log2", ", pure");
  EMIT_OCML(uses_sqrt_f32, "sqrt", ", const");
  EMIT_OCML(uses_sin_f32, "sin", "");
  EMIT_OCML(uses_trunc_f32, "trunc", "");
  EMIT_OCML_F64(uses_exp2_f64, "exp2", ", pure");
  EMIT_OCML_F64(uses_log2_f64, "log2", ", pure");
  EMIT_OCML_F64(uses_sqrt_f64, "sqrt", ", const");
  EMIT_OCML_F64(uses_sin_f64, "sin", "");
  EMIT_OCML_F64(uses_trunc_f64, "trunc", "");

#undef EMIT_OCML
#undef EMIT_OCML_F64

  /* Vector type typedefs (ext_vector_type requires typedef in HIP C++) */
  for (int vi = 0; vi < used.n_vec_dtypes; vi++) {
    PolyDType vdt = used.vec_dtypes[vi];
    int lanes = used.vec_lanes[vi];
    char tname[64];
    hip_render_ctype(vdt, lanes, tname, sizeof(tname));
    const char *sctype = hip_scalar_ctype(vdt);
    hsb_printf(
        &out, "typedef %s %s __attribute__((ext_vector_type(%d)));\n", sctype, tname, lanes
    );
  }

  char wmma_names[32][128];
  int n_wmma_names = 0;
  for (int i = 0; i < n; i++) {
    if (uops[i]->op != POLY_OP_WMMA) continue;
    char name[128];
    if (!poly_wmma_name(uops[i], name, sizeof(name))) continue;
    bool seen = false;
    for (int j = 0; j < n_wmma_names; j++)
      if (strcmp(wmma_names[j], name) == 0) seen = true;
    if (seen) continue;
    if (!hip_render_wmma_prefix(&out, uops[i], arch)) return NULL;
    if (n_wmma_names < 32)
      snprintf(wmma_names[n_wmma_names++], sizeof(wmma_names[0]), "%s", name);
  }

  hsb_puts(&out, "\n");

  /* Kernel signature: AMD-specific attributes */
  hsb_printf(
      &out,
      "extern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, "
      "%d))) %s(",
      launch_bounds, fn_name
  );
  for (int i = 0; i < n_params; i++) {
    if (i > 0) hsb_puts(&out, ", ");
    hsb_printf(&out, "%s %s", param_types[i], param_names[i]);
  }
  hsb_puts(&out, ") {\n");
  if (decls.len > 0) hsb_puts(&out, decls.buf);
  hsb_puts(&out, body.buf);
  hsb_puts(&out, "}\n");

  /* No _call wrapper needed -- HIP uses hipModuleLaunchKernel */

  for (int i = 0; i < n_params; i++) {
    free(param_types[i]);
    free(param_names[i]);
  }
  free(decls.buf);
  free(body.buf);
  hsmap_destroy(&names);

  return out.buf;
}

#endif /* POLY_HAS_HIP */
