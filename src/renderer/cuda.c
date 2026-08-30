/*
 * renderer/cuda.c — current Tinygrad renderer/cstyle.py:CUDARenderer port
 *
 * Walks linearized UOps and emits CUDA C++ source code matching
 * tinygrad's CUDARenderer output. Also provides poly_linearize_cuda()
 * which chains: full_rewrite_to_sink → add_gpudims → linearize.
 */

#ifdef POLY_HAS_CUDA

#define _POSIX_C_SOURCE 200809L

#include "codegen/codegen.h"
#include "codegen/opt/tc.h"
#include "renderer/cstyle.h"
#include "bigint.h"
#include "utils.h"
#include "engine/schedule.h"
#include "uop/upat.h"
#include "uop/ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <limits.h>

/* C storage for Tinygrad's string-renderer output. */

typedef struct {
  char *buf;
  int len;
  int cap;
} CudaStrBuf;

static void csb_init(CudaStrBuf *sb) {
  sb->cap = 512;
  sb->buf = malloc(sb->cap);
  sb->buf[0] = '\0';
  sb->len = 0;
}

static void csb_printf(CudaStrBuf *sb, const char *fmt, ...) {
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

static void csb_puts(CudaStrBuf *sb, const char *s) {
  csb_printf(sb, "%s", s);
}

/* Pointer → string hash map */

typedef struct {
  PolyUOp **keys;
  char **vals;
  int cap;
} CudaStrMap;

static void csmap_init(CudaStrMap *m, int n) {
  m->cap = (n < 4) ? 16 : n * 3;
  m->keys = calloc(m->cap, sizeof(PolyUOp *));
  m->vals = calloc(m->cap, sizeof(char *));
}

static void csmap_set(CudaStrMap *m, PolyUOp *key, char *val) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
  while (m->keys[h] && m->keys[h] != key)
    h = (h + 1) % m->cap;
  if (m->keys[h] == key) free(m->vals[h]);
  m->keys[h] = key;
  m->vals[h] = val;
}

static char *csmap_get(CudaStrMap *m, PolyUOp *key) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
  while (m->keys[h]) {
    if (m->keys[h] == key) return m->vals[h];
    h = (h + 1) % m->cap;
  }
  return NULL;
}

static void csmap_destroy(CudaStrMap *m) {
  for (int i = 0; i < m->cap; i++)
    if (m->vals[i]) free(m->vals[i]);
  free(m->keys);
  free(m->vals);
}

/* Render helpers */

static char *cuda_render_int64_const(int64_t v, char *buf, int cap) {
  if (v == INT64_MIN)
    snprintf(buf, cap, "(-9223372036854775807ll - 1ll)");
  else
    snprintf(buf, cap, "%lldll", (long long)v);
  return buf;
}

static char *cuda_render_float_const(double v, PolyDType dt, char *buf, int cap) {
  bool is_f64 = poly_dtype_eq(dt, POLY_FLOAT64);
  if (isinf(v)) {
    snprintf(buf, cap, v > 0 ? "INFINITY" : "(-INFINITY)");
    return buf;
  }
  if (isnan(v)) {
    snprintf(buf, cap, "NAN");
    return buf;
  }
  if (is_f64) {
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

/* Tinygrad 2026-08-22/a9069c177a9d CUDARenderer.type_map. */
static const char *cuda_ctype(PolyDType dt) {
  PolyDType s = dt;
  /* tinygrad@2026-08-22/a9069c177a9d CUDARenderer.type_map uses `uint`,
   * allowing _render_dtype to form CUDA's native uint2/uint4 spellings. */
  if (s.priority == POLY_UINT32.priority) return "uint";
  if (s.priority == POLY_FLOAT16.priority) return "half";
  if (s.priority == POLY_BFLOAT16.priority) return "nv_bfloat16";
  if (poly_dtype_eq(s, POLY_FP8E4M3)) return "__nv_fp8_e4m3";
  if (poly_dtype_eq(s, POLY_FP8E5M2)) return "__nv_fp8_e5m2";
  return s.name;
}

static void cuda_render_ctype_nonptr(PolyDType dtype, int lanes, char *buf, int cap) {
  if (lanes <= 1) {
    snprintf(buf, cap, "%s", cuda_ctype(dtype));
    return;
  }
  if (poly_dtype_is_bool(dtype)) {
    snprintf(buf, cap, "int%d", lanes);
  } else {
    snprintf(buf, cap, "%s%d", cuda_ctype(dtype), lanes);
  }
}

/* tinygrad@2026-08-22/a9069c177a9d renderer/cstyle.py:175-191 derives
 * register width from UOp.max_numel(); storage PARAMs remain scalar pointers. */
static int cuda_uop_lanes(PolyCtx *ctx, const PolyUOp *u) {
  if (!u) return 1;
  int64_t max_numel = poly_uop_max_numel(ctx, u);
  if (max_numel > 1 && max_numel <= INT_MAX) return (int)max_numel;
  return 1;
}

static void cuda_render_uop_ctype(
    PolyCtx *ctx, const PolyUOp *u, char *buf, int cap
) {
  cuda_render_ctype_nonptr(u->dtype, cuda_uop_lanes(ctx, u), buf, cap);
}

/* Current renderer/cstyle.py:26-47 uses the CAST destination dtype when a
 * strong program literal wraps a weak/bool CONST. */
static char *cuda_render_const_literal(PolyUOp *c, PolyDType dtype) {
  if (!c || c->op != POLY_OP_CONST) return NULL;
  PolyDType scalar = dtype;
  char val[192];
  if (poly_dtype_eq(scalar, POLY_FLOAT16)) {
    char tmp[64];
    cuda_render_float_const(c->arg.f, POLY_FLOAT32, tmp, sizeof(tmp));
    snprintf(val, sizeof(val), "__float2half(%s)", tmp);
  } else if (poly_dtype_eq(scalar, POLY_BFLOAT16)) {
    char tmp[64];
    cuda_render_float_const(c->arg.f, POLY_FLOAT32, tmp, sizeof(tmp));
    snprintf(val, sizeof(val), "__float2bfloat16(%s)", tmp);
  } else if (poly_dtype_is_float(scalar)) {
    cuda_render_float_const(c->arg.f, scalar, val, sizeof(val));
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
    cuda_render_int64_const(c->arg.i, val, sizeof(val));
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
    char type[64];
    cuda_render_ctype_nonptr(scalar, 1, type, sizeof(type));
    if (poly_dtype_is_unsigned(scalar))
      snprintf(
          val, sizeof(val), "((%s)(%uu))", type,
          (unsigned)(uint32_t)poly_arg_integer_to_u64_mod(c->arg)
      );
    else
      snprintf(val, sizeof(val), "((%s)(%lld))", type, (long long)c->arg.i);
  } else if (c->arg.kind == POLY_ARG_BIGINT) {
    return poly_arg_integer_to_decimal(c->arg);
  } else {
    snprintf(val, sizeof(val), "%lld", (long long)c->arg.i);
  }
  return strdup(val);
}

/* Pinned tinygrad CStyleLanguage.render_access
 * (tinygrad/renderer/cstyle.py:179-184) dereferences vector memory through a
 * pointer to the accessed vector dtype.  The devectorizer proves alignment
 * before producing this vector LOAD/STORE; the PARAM itself remains scalar. */
static void cuda_render_access_expr(
    PolyCtx *ctx, char *buf, int cap, const char *address, const PolyUOp *index
) {
  int lanes = cuda_uop_lanes(ctx, index);
  if (lanes > 1) {
    char value_type[128];
    cuda_render_ctype_nonptr(index->dtype, lanes, value_type, sizeof(value_type));
    snprintf(buf, cap, "*((%s*)(%s))", value_type, address ? address : "0");
  } else {
    snprintf(buf, cap, "*%s", address ? address : "0");
  }
}

/* Tinygrad 2026-08-22/a9069c177a9d CUDARenderer.render_kernel FP8 vectors. */
static bool cuda_vector_needs_prefix(PolyDType dtype, int count) {
  if (poly_dtype_eq(dtype, POLY_FP8E4M3) || poly_dtype_eq(dtype, POLY_FP8E5M2))
    return count == 2 || count == 4 || count == 8 || count == 16;
  return (count == 4 || count == 8) &&
         (dtype.priority == POLY_FLOAT16.priority ||
          dtype.priority == POLY_BFLOAT16.priority);
}

static const char *cuda_lane_name(int idx) {
  static const char *lanes[] = {
      "x", "y", "z", "w", "a", "b", "c", "d", "e", "f", "g",  "h",  "i",  "j",  "k",  "l",
      "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w0", "x0", "y0", "z0", "a0", "b0",
  };
  if (idx < 0 || idx >= (int)(sizeof(lanes) / sizeof(lanes[0]))) return "x";
  return lanes[idx];
}

static void cuda_render_vector_prefix(CudaStrBuf *out, PolyDType scalar, int count) {
  char vec_t[64], scalar_t[64];
  cuda_render_ctype_nonptr(scalar, count, vec_t, sizeof(vec_t));
  cuda_render_ctype_nonptr(scalar, 1, scalar_t, sizeof(scalar_t));
  int align = poly_dtype_itemsize(scalar) * count;

  csb_printf(out, "struct __align__(%d) %s { ", align, vec_t);
  for (int i = 0; i < count; i++) {
    if (i) csb_puts(out, " ");
    csb_printf(out, "%s %s;", scalar_t, cuda_lane_name(i));
  }
  csb_puts(out, " }; ");
  csb_printf(out, "__device__ %s make_%s(", vec_t, vec_t);
  for (int i = 0; i < count; i++) {
    if (i) csb_puts(out, ", ");
    csb_printf(out, "%s %s", scalar_t, cuda_lane_name(i));
  }
  csb_printf(out, ") { %s r = {", vec_t);
  for (int i = 0; i < count; i++) {
    if (i) csb_puts(out, ", ");
    csb_puts(out, cuda_lane_name(i));
  }
  csb_puts(out, "}; return r; }\n");
}

static const char *cuda_wmma_dtype_in(PolyDType dtype) {
  if (poly_dtype_eq(dtype, POLY_FLOAT32)) return "tf32";
  if (poly_dtype_eq(dtype, POLY_FLOAT16)) return "f16";
  if (poly_dtype_eq(dtype, POLY_BFLOAT16)) return "bf16";
  if (poly_dtype_eq(dtype, POLY_FP8E4M3)) return "e4m3";
  if (poly_dtype_eq(dtype, POLY_FP8E5M2)) return "e5m2";
  return NULL;
}

static const char *cuda_wmma_dtype_out(PolyDType dtype) {
  if (poly_dtype_eq(dtype, POLY_FLOAT32)) return "f32";
  if (poly_dtype_eq(dtype, POLY_FLOAT16)) return "f16";
  return NULL;
}

/* Tinygrad 2026-08-22/a9069c177a9d CUDARenderer.render_kernel emits one
 * inline-PTX helper per signature returned by wmma_args. */
static bool cuda_render_wmma_helper(CudaStrBuf *out, PolyCtx *ctx, PolyUOp *wmma) {
  if (!wmma || wmma->n_src != 3 || wmma->arg.kind != POLY_ARG_TENSOR_CORE)
    return false;
  char name[128], types[3][64];
  if (!poly_wmma_name(wmma, name, sizeof(name))) return false;
  int sizes[3], n_operands[3];
  PolyDType dtypes[3] = {
      wmma->arg.tensor_core.dtype_in, wmma->arg.tensor_core.dtype_in, wmma->dtype
  };
  for (int i = 0; i < 3; i++) {
    int64_t lanes = poly_uop_max_numel(ctx, wmma->src[i]);
    if (lanes <= 0 || lanes > INT_MAX) return false;
    sizes[i] = (int)lanes;
    n_operands[i] = sizes[i] * poly_dtype_itemsize(dtypes[i]) / 4;
    if (n_operands[i] <= 0) return false;
    cuda_render_ctype_nonptr(dtypes[i], sizes[i], types[i], sizeof(types[i]));
  }
  const char *dtype_in = cuda_wmma_dtype_in(wmma->arg.tensor_core.dtype_in);
  const char *dtype_out = cuda_wmma_dtype_out(wmma->dtype);
  if (!dtype_in || !dtype_out) return false;

  csb_printf(
      out, "__device__ %s __%s(%s a, %s b, %s c){\n", types[2], name,
      types[0], types[1], types[2]
  );
  csb_puts(out, "  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);\n");
  csb_printf(
      out, "  asm(\"mma.sync.aligned.m%dn%dk%d.row.col.%s.%s.%s.%s\"\n",
      wmma->arg.tensor_core.dims[1], wmma->arg.tensor_core.dims[0],
      wmma->arg.tensor_core.dims[2], dtype_out, dtype_in, dtype_in, dtype_out
  );
  csb_puts(out, "      \"{");
  for (int i = 0; i < n_operands[2]; i++) {
    if (i) csb_puts(out, ", ");
    csb_printf(out, "%%%d", i);
  }
  csb_puts(out, "}, {");
  for (int i = 0; i < n_operands[0]; i++) {
    if (i) csb_puts(out, ", ");
    csb_printf(out, "%%%d", n_operands[2] + i);
  }
  csb_puts(out, "},\"\n      \"{");
  for (int i = 0; i < n_operands[1]; i++) {
    if (i) csb_puts(out, ", ");
    csb_printf(out, "%%%d", n_operands[2] + n_operands[0] + i);
  }
  csb_puts(out, "}, {");
  for (int i = 0; i < n_operands[2]; i++) {
    if (i) csb_puts(out, ", ");
    csb_printf(out, "%%%d", i);
  }
  csb_puts(out, "};\"\n    : ");
  for (int i = 0; i < n_operands[2]; i++) {
    if (i) csb_puts(out, ", ");
    csb_printf(out, "\"+r\"(c_pk[%d])", i);
  }
  csb_puts(out, "\n    : ");
  for (int i = 0; i < n_operands[0]; i++) {
    if (i) csb_puts(out, ", ");
    csb_printf(out, "\"r\"(a_pk[%d])", i);
  }
  for (int i = 0; i < n_operands[1]; i++) {
    if (i || n_operands[0]) csb_puts(out, ", ");
    csb_printf(out, "\"r\"(b_pk[%d])", i);
  }
  csb_puts(out, ");\n  return c;\n}\n");
  return true;
}

static int cuda_child_count(PolyUOp **uops, int n, PolyUOp *needle) {
  int count = 0;
  if (!needle) return 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (!u) continue;
    for (int j = 0; j < u->n_src; j++)
      if (u->src[j] == needle) count++;
  }
  return count;
}

static char *cuda_render_vector_expr(PolyCtx *ctx, CudaStrMap *names, PolyUOp *u) {
  char ctype[128];
  cuda_render_uop_ctype(ctx, u, ctype, sizeof(ctype));

  CudaStrBuf expr;
  csb_init(&expr);
  if (u->n_src == 1) {
    char *s = csmap_get(names, u->src[0]);
    csb_printf(&expr, "(%s)(%s)", ctype, s ? s : "0");
  } else if (u->n_src > 1) {
    csb_printf(&expr, "make_%s(", ctype);
    for (int j = 0; j < u->n_src; j++) {
      if (j) csb_puts(&expr, ", ");
      char *s = csmap_get(names, u->src[j]);
      csb_puts(&expr, s ? s : "0");
    }
    csb_puts(&expr, ")");
  } else if (u->arg.kind == POLY_ARG_INT_TUPLE) {
    csb_printf(&expr, "make_%s(", ctype);
    for (int j = 0; j < u->arg.int_tuple.n; j++) {
      if (j) csb_puts(&expr, ", ");
      csb_printf(&expr, "%lld", (long long)u->arg.int_tuple.vals[j]);
    }
    csb_puts(&expr, ")");
  } else {
    csb_printf(&expr, "(%s)0", ctype);
  }
  return expr.buf;
}


static bool cuda_is_half(PolyDType dt) {
  PolyDType s = dt;
  return s.priority == POLY_FLOAT16.priority || s.priority == POLY_BFLOAT16.priority;
}

/* Exact C equivalent of helpers.strip_parens for the same-op associative
 * rendering rule in pinned CStyleLanguage.base_rewrite. */
static char *cuda_strip_parens(const char *expr) {
  if (!expr) return strdup("");
  size_t n = strlen(expr);
  if (n < 2 || expr[0] != '(' || expr[n - 1] != ')') return strdup(expr);
  int depth = 0;
  for (size_t i = 0; i < n; i++) {
    if (expr[i] == '(')
      depth++;
    else if (expr[i] == ')')
      depth--;
    if (depth == 0 && i != n - 1) return strdup(expr);
    if (depth < 0) return strdup(expr);
  }
  if (depth != 0) return strdup(expr);
  char *ret = malloc(n - 1);
  if (!ret) return NULL;
  memcpy(ret, expr + 1, n - 2);
  ret[n - 2] = '\0';
  return ret;
}

static char *cuda_render_alu(
    PolyOps op,
    PolyDType dtype,
    const char *s0,
    const char *s1,
    const char *s2
) {
  bool is_half = cuda_is_half(dtype);
  CudaStrBuf out;
  csb_init(&out);
  switch (op) {
  case POLY_OP_NEG:
    /* Pinned CStyleLanguage/CUDARenderer inherits arithmetic NEG. */
    csb_printf(&out, "(-%s)", s0);
    break;
  case POLY_OP_SQRT:
    csb_printf(
        &out,
        is_half                              ? "hsqrt(%s)"
        : poly_dtype_eq(dtype, POLY_FLOAT64) ? "sqrt(%s)"
                                             : "sqrtf(%s)",
        s0
    );
    break;
  case POLY_OP_TRUNC:
    csb_printf(
        &out,
        is_half                              ? "htrunc(%s)"
        : poly_dtype_eq(dtype, POLY_FLOAT64) ? "trunc(%s)"
                                             : "truncf(%s)",
        s0
    );
    break;
  case POLY_OP_EXP2:
    csb_printf(
        &out,
        is_half                              ? "hexp2(%s)"
        : poly_dtype_eq(dtype, POLY_FLOAT64) ? "exp2(%s)"
                                             : "exp2f(%s)",
        s0
    );
    break;
  case POLY_OP_LOG2:
    csb_printf(
        &out,
        is_half                              ? "hlog2(%s)"
        : poly_dtype_eq(dtype, POLY_FLOAT64) ? "log2(%s)"
                                             : "log2f(%s)",
        s0
    );
    break;
  case POLY_OP_SIN:
    csb_printf(
        &out,
        is_half                              ? "hsin(%s)"
        : poly_dtype_eq(dtype, POLY_FLOAT64) ? "sin(%s)"
                                             : "sinf(%s)",
        s0
    );
    break;
  case POLY_OP_RECIPROCAL:
    csb_printf(&out, is_half ? "hrcp(%s)" : "(1/%s)", s0);
    break;
  case POLY_OP_ADD:
    csb_printf(&out, "(%s+%s)", s0, s1);
    break;
  case POLY_OP_SUB:
    csb_printf(&out, "(%s-%s)", s0, s1);
    break;
  case POLY_OP_MUL:
    csb_printf(&out, "(%s*%s)", s0, s1);
    break;
  case POLY_OP_FDIV:
    csb_printf(&out, "(%s/%s)", s0, s1);
    break;
  case POLY_OP_IDIV:
    csb_printf(&out, "(%s/%s)", s0, s1);
    break;
  case POLY_OP_MOD:
    csb_printf(&out, "(%s%%%s)", s0, s1);
    break;
  case POLY_OP_SHL:
    csb_printf(&out, "(%s<<%s)", s0, s1);
    break;
  case POLY_OP_SHR:
    csb_printf(&out, "(%s>>%s)", s0, s1);
    break;
  case POLY_OP_AND:
    csb_printf(&out, "(%s&%s)", s0, s1);
    break;
  case POLY_OP_OR:
    csb_printf(&out, "(%s|%s)", s0, s1);
    break;
  case POLY_OP_XOR:
    csb_printf(&out, "(%s^%s)", s0, s1);
    break;
  case POLY_OP_CMPLT:
    csb_printf(&out, "(%s<%s)", s0, s1);
    break;
  case POLY_OP_CMPNE:
    csb_printf(&out, "(%s!=%s)", s0, s1);
    break;
  case POLY_OP_CMPEQ:
    csb_printf(&out, "(%s==%s)", s0, s1);
    break;
  case POLY_OP_MAX:
    csb_printf(&out, "((%s>%s)?%s:%s)", s0, s1, s0, s1);
    break;
  case POLY_OP_POW:
    csb_printf(
        &out, poly_dtype_eq(dtype, POLY_FLOAT64) ? "pow(%s, %s)" : "powf(%s, %s)", s0, s1
    );
    break;
  case POLY_OP_WHERE:
    csb_printf(&out, "(%s?%s:%s)", s0, s1, s2);
    break;
  case POLY_OP_MULACC:
    csb_printf(
        &out,
        is_half                              ? "__hfma(%s,%s,%s)"
        : poly_dtype_eq(dtype, POLY_FLOAT64) ? "fma(%s,%s,%s)"
        : poly_dtype_is_float(dtype)         ? "__fmaf_rn(%s,%s,%s)"
                                             : "(%s*%s+%s)",
        s0, s1, s2
    );
    break;
  default:
    csb_printf(&out, "/* unknown op %d */0", op);
    break;
  }
  return out.buf;
}

static int cuda_range_slot(PolyUOp **ranges, int *n_ranges, PolyUOp *r, bool create) {
  if (!r) return -1;
  for (int i = 0; i < *n_ranges; i++) {
    if (ranges[i] == r) return i;
  }
  if (!create || *n_ranges >= 128) return -1;
  ranges[*n_ranges] = r;
  (*n_ranges)++;
  return *n_ranges - 1;
}

/* CUDA Linearizer */

PolyUOp *poly_rewrite_cuda(PolyCtx *ctx, PolyUOp *sink) {
  /* tinygrad CUDA still uses the normal postrange apply_opts path for
   * non-TC kernels. Tensor core matching is only the first branch inside
   * that heuristic. Keep CUDA on the shared heuristic policy so LOCAL/UNROLL
  * scheduling still happens on ordinary kernels such as broadcast matmul. */
  int arch = poly_cuda_arch_major() * 10 + poly_cuda_arch_minor();
  int n_tensor_cores = 0;
  const PolyTensorCore *tensor_cores = poly_tc_get_cuda(arch, &n_tensor_cores);
  PolyRewriteOpts opts = {
      .optimize =
          poly_kernel_optimize_enabled(sink), /* shared optimized pipeline (tinygrad parity) */
      .beam_width = poly_kernel_beam(sink),
      .caps =
          {
              .device = "CUDA",
              /* Pinned CUDARenderer inherits RECIPROCAL from CStyleLanguage
               * but does not advertise MULACC in code_for_op. */
              .has_mulacc = false,
              .has_exp2 = true,
              .has_log2 = true,
              .has_sin = true,
              /* Tinygrad CUDARenderer.supported_dtypes(). */
              .supports_float16 = arch >= 53,
              .supports_bfloat16 = arch >= 80,
              .supports_fp8e4m3 = arch >= 89,
              .supports_fp8e5m2 = arch >= 89,
              .has_int64 = true,
              .has_local = true,
              /* Pinned Renderer.supports_float4 defaults true
               * (renderer/__init__.py:62); CUDARenderer does not override it. */
              .max_vec_width = 4,
              .global_max = {2147483647, 65535, 65535},
              .local_max = {1024, 1024, 64},
              .tensor_cores = tensor_cores,
              .n_tensor_cores = n_tensor_cores,
      },
      .device = POLY_DEVICE_CUDA,
      .opt_policy = POLY_OPT_HEURISTIC,
      .extra_matcher = poly_cuda_renderer_extra_matcher(),
  };
  return poly_full_rewrite_to_sink_ex(ctx, sink, opts);
}

PolyUOp **poly_linearize_cuda(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  sink = poly_rewrite_cuda(ctx, sink);
  return poly_do_linearize(ctx, sink, n_out);
}

/* CUDA Renderer */

char *poly_render_cuda(
    PolyCtx *ctx, PolyUOp **uops, int n, const char *fn_name, int launch_bounds
) {
  if (!ctx) return NULL;
  CudaStrBuf decls, body;
  csb_init(&decls);
  csb_init(&body);

  CudaStrMap names;
  csmap_init(&names, n);

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
    if (u->op == POLY_OP_RANGE) (void)cuda_range_slot(live_ranges, &n_live_ranges, u, true);
    if (u->op == POLY_OP_END) continue;
    for (int j = 0; j < u->n_src; j++) {
      if (u->src[j] && u->src[j]->op == POLY_OP_RANGE) {
        int ri = cuda_range_slot(live_ranges, &n_live_ranges, u->src[j], true);
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
          int ri = cuda_range_slot(live_ranges, &n_live_ranges, u->src[j], false);
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
        csmap_destroy(&names);
        return NULL;
      }
      char name[32];
      snprintf(name, sizeof(name), "data%lld", (long long)slot);
      csmap_set(&names, u, strdup(name));

      char type[64];
      if (poly_uop_is_alu_param(u)) {
        char base_type[64];
        cuda_render_ctype_nonptr(u->dtype, 1, base_type, sizeof(base_type));
        snprintf(type, sizeof(type), "const %s", base_type);
      } else {
        char base_type[64];
        cuda_render_ctype_nonptr(u->dtype, 1, base_type, sizeof(base_type));
        snprintf(type, sizeof(type), "%s* __restrict__", base_type);
      }
      param_types[n_params] = strdup(type);
      param_names[n_params] = strdup(name);
      param_order[n_params] = (int)slot;
      n_params++;
      continue;
    }

    /* --- CONST -------------------------------------------------------- */
    if (u->op == POLY_OP_CONST) {
      char *literal = cuda_render_const_literal(u, u->dtype);
      if (!literal) return NULL;
      csmap_set(&names, u, literal);
      continue;
    }

    /* --- INDEX: pointer arithmetic or vector lane extract ------------- */
    if (u->op == POLY_OP_INDEX) {
      char *buf_s = csmap_get(&names, u->src[0]);
      char *idx_s = csmap_get(&names, u->src[1]);
      char expr[256];
      if (poly_is_program_memory_base(u->src[0]))
        snprintf(expr, sizeof(expr), "(%s+%s)", buf_s, idx_s);
      else if (u->src[1] && u->src[1]->op == POLY_OP_CONST && u->src[1]->arg.kind == POLY_ARG_INT)
        snprintf(expr, sizeof(expr), "%s.%s", buf_s ? buf_s : "0", cuda_lane_name((int)u->src[1]->arg.i));
      else
        snprintf(expr, sizeof(expr), "((&(%s).x)[%s])", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      csmap_set(&names, u, strdup(expr));
      continue;
    }

    /* --- SHRINK: late codegen memory slice --------------------------- */
    if (u->op == POLY_OP_SHRINK) {
      char *buf_s = csmap_get(&names, u->src[0]);
      char *idx_s = csmap_get(&names, u->src[1]);
      char expr[256];
      snprintf(expr, sizeof(expr), "(%s+%s)", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      csmap_set(&names, u, strdup(expr));
      continue;
    }

    /* --- SPECIAL: GPU workitem index ---------------------------------- */
    if (u->op == POLY_OP_SPECIAL) {
      const char *sname = u->arg.str ? u->arg.str : "gidx0";
      csmap_set(&names, u, strdup(sname));

      /* Determine which dimension: last char of name */
      int dim_idx = 0;
      int slen = (int)strlen(sname);
      if (slen > 0) dim_idx = sname[slen - 1] - '0';
      char dim_char = 'x';
      if (dim_idx == 1)
        dim_char = 'y';
      else if (dim_idx == 2)
        dim_char = 'z';

      /* tinygrad CUDA maps:
       *   gidxN -> blockIdx.{x,y,z}
       *   lidxN -> threadIdx.{x,y,z}
       *   idxN  -> blockIdx*blockDim+threadIdx
       * Keep that contract here so launch dimensions and rendered indexing
       * stay aligned with the reference runtime. */
      for (int d = 0; d < depth; d++)
        csb_puts(&body, "  ");
      if (sname[0] == 'l') {
        csb_printf(&body, "int %s = threadIdx.%c;\n", sname, dim_char);
      } else if (sname[0] == 'i') {
        csb_printf(
            &body, "int %s = (blockIdx.%c*blockDim.%c+threadIdx.%c);\n", sname, dim_char, dim_char,
            dim_char
        );
      } else {
        csb_printf(&body, "int %s = blockIdx.%c;\n", sname, dim_char);
      }

      /* Bounds check for non-local indices only.
       * Skip for lidx: all threads in the block must execute barriers. */
      if (sname[0] != 'l') {
        char *bound = csmap_get(&names, u->src[0]);
        if (bound) {
          for (int d = 0; d < depth; d++)
            csb_puts(&body, "  ");
          csb_printf(&body, "if (%s >= %s) return;\n", sname, bound);
        }
      }
      continue;
    }

    /* --- BARRIER ------------------------------------------------------ */
    if (u->op == POLY_OP_BARRIER) {
      for (int d = 0; d < depth; d++)
        csb_puts(&body, "  ");
      csb_puts(&body, "__syncthreads();\n");
      continue;
    }

    /* --- RANGE: for loop (reduce loops stay as loops) ----------------- */
    if (u->op == POLY_OP_RANGE) {
      char *range = poly_range_str(u->arg);
      char *name = malloc(strlen(range) + 6);
      snprintf(name, strlen(range) + 6, "%cidx%s", poly_axis_letter(u->arg), range);
      free(range);
      csmap_set(&names, u, name);

      char *bound = csmap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++)
        csb_puts(&body, "  ");
      csb_printf(&body, "for (int %s = 0; %s < %s; %s++) {\n", name, name, bound, name);
      depth++;
      if (n_open_ranges < 128) open_ranges[n_open_ranges++] = u;
      continue;
    }

    /* --- END / ENDIF -------------------------------------------------- */
    if (u->op == POLY_OP_END || u->op == POLY_OP_ENDIF) {
      if (u->op == POLY_OP_END && u->n_src > 1 && u->src[1]->op == POLY_OP_RANGE) {
        PolyUOp *want = u->src[1];
        int wi = cuda_range_slot(live_ranges, &n_live_ranges, want, false);
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
          int oi = cuda_range_slot(live_ranges, &n_live_ranges, open_ranges[p], false);
          if (oi >= 0 && live_remaining[oi] > 0 && open_ranges[p] != want) {
            can_close = false;
            break;
          }
        }
        if (!can_close) continue;

        while (n_open_ranges > pos) {
          depth--;
          for (int d = 0; d < depth; d++)
            csb_puts(&body, "  ");
          csb_puts(&body, "}\n");
          n_open_ranges--;
        }
        continue;
      }

      depth--;
      for (int d = 0; d < depth; d++)
        csb_puts(&body, "  ");
      csb_puts(&body, "}\n");
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
      csmap_set(&names, u, strdup(name));

      int64_t smem_size = poly_program_buffer_size(u);
      {
        char ctype[128];
        cuda_render_ctype_nonptr(poly_program_buffer_dtype(u), 1, ctype, sizeof(ctype));
        csb_printf(&decls, "  __shared__ %s %s[%lld];\n", ctype, name, (long long)smem_size);
      }
      continue;
    }

    /* --- register buffer --------------------------------------------- */
    if (u->op == POLY_OP_BUFFER && poly_program_memory_is(u, POLY_ADDR_REG)) {
      char name[32];
      snprintf(name, sizeof(name), "r%lld", (long long)poly_program_buffer_slot(u));
      csmap_set(&names, u, strdup(name));

      PolyDType base = poly_program_buffer_dtype(u);
      int64_t reg_size = poly_program_buffer_size(u);
      char ctype[128];
      cuda_render_ctype_nonptr(base, 1, ctype, sizeof(ctype));
      csb_printf(&decls, "  %s %s[%lld];\n", ctype, name, (long long)reg_size);
      continue;
    }

    /* --- AFTER -------------------------------------------------------- */
    if (u->op == POLY_OP_AFTER) {
      char *src_name = csmap_get(&names, u->src[0]);
      if (src_name) csmap_set(&names, u, strdup(src_name));
      continue;
    }

    /* --- COPY: transparent placement wrapper ------------------------- */
    if (u->op == POLY_OP_COPY && u->n_src > 0) {
      char *src_name = csmap_get(&names, u->src[0]);
      if (src_name) csmap_set(&names, u, strdup(src_name));
      continue;
    }

    /* --- LOAD --------------------------------------------------------- */
    if (u->op == POLY_OP_LOAD) {
      char name[32];
      snprintf(name, sizeof(name), "val%d", c_val++);
      csmap_set(&names, u, strdup(name));

      char *bidx = csmap_get(&names, u->src[0]);
      char ctype[128];
      cuda_render_uop_ctype(ctx, u, ctype, sizeof(ctype));
      char access[512];
      cuda_render_access_expr(ctx, access, sizeof(access), bidx, u->src[0]);
      csb_printf(&decls, "  %s %s;\n", ctype, name);
      for (int d = 0; d < depth; d++)
        csb_puts(&body, "  ");

      /* Pinned tinygrad final IR: LOAD(INDEX(buf, idx), alt, gate). */
      PolyUOp *gate_uop =
          (u->n_src >= 3 && poly_dtype_is_bool(u->src[2]->dtype))
              ? u->src[2]
              : NULL;
      if (gate_uop && u->n_src >= 2) {
        char *gate_s = csmap_get(&names, gate_uop);
        char *alt_s = csmap_get(&names, u->src[1]);
        csb_printf(&body, "%s = (%s?%s:%s);\n", name, gate_s, access, alt_s);
      } else if (gate_uop) {
        char *gate_s = csmap_get(&names, gate_uop);
        csb_printf(&body, "%s = (%s?%s:(%s)0);\n", name, gate_s, access, ctype);
      } else {
        csb_printf(&body, "%s = %s;\n", name, access);
      }
      continue;
    }

    /* --- STORE -------------------------------------------------------- */
    if (u->op == POLY_OP_STORE) {
      char *target = csmap_get(&names, u->src[0]);
      char *val = csmap_get(&names, u->src[1]);

      for (int d = 0; d < depth; d++)
        csb_puts(&body, "  ");
      /* Pinned CStyleLanguage.render_access is shared by GLOBAL and LOCAL
       * stores (renderer/cstyle.py:58,179-184).  In particular a vector
       * SHRINK over scalar shared memory is a typed vector-pointer lvalue. */
      char access[512];
      cuda_render_access_expr(
          ctx, access, sizeof(access), target, u->src[0]
      );
      csb_printf(&body, "%s = %s;\n", access, val ? val : "null");

      continue;
    }

    /* --- CAST --------------------------------------------------------- */
    if (u->op == POLY_OP_CAST) {
      if (u->n_src == 1 && u->src[0] && u->src[0]->op == POLY_OP_CONST &&
          (poly_dtype_is_weak(u->src[0]->dtype) || poly_dtype_is_bool(u->src[0]->dtype)) &&
          poly_uop_max_numel(ctx, u) == 1) {
        char *literal = cuda_render_const_literal(
            u->src[0], poly_dtype_is_fp8(u->dtype) ? POLY_FLOAT32 : u->dtype
        );
        if (!literal) return NULL;
        if (poly_dtype_is_fp8(u->dtype)) {
          const char *ctype = cuda_ctype(u->dtype);
          size_t len = strlen(ctype) + strlen(literal) + 7;
          char *typed = malloc(len);
          if (!typed) {
            free(literal);
            return NULL;
          }
          snprintf(typed, len, "((%s)(%s))", ctype, literal);
          free(literal);
          literal = typed;
        }
        csmap_set(&names, u, literal);
        continue;
      }
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      csmap_set(&names, u, strdup(name));

      char *src_s = csmap_get(&names, u->src[0]);
      char ctype[128];
      cuda_render_uop_ctype(ctx, u, ctype, sizeof(ctype));
      csb_printf(&decls, "  %s %s;\n", ctype, name);
      for (int d = 0; d < depth; d++)
        csb_puts(&body, "  ");
      csb_printf(&body, "%s = (%s)(%s);\n", name, ctype, src_s);
      continue;
    }

    /* --- BITCAST: use template helper --------------------------------- */
    if (u->op == POLY_OP_BITCAST) {
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      csmap_set(&names, u, strdup(name));

      char *src_s = csmap_get(&names, u->src[0]);
      char dst_type[128], src_type[128];
      cuda_render_uop_ctype(ctx, u, dst_type, sizeof(dst_type));
      cuda_render_uop_ctype(ctx, u->src[0], src_type, sizeof(src_type));
      csb_printf(&decls, "  %s %s;\n", dst_type, name);
      for (int d = 0; d < depth; d++)
        csb_puts(&body, "  ");
      csb_printf(&body, "%s = tg_bitcast<%s>((%s)(%s));\n", name, dst_type, src_type, src_s);
      continue;
    }

    /* --- WMMA -------------------------------------------------------- */
    if (u->op == POLY_OP_WMMA) {
      char helper[128];
      if (!poly_wmma_name(u, helper, sizeof(helper))) return NULL;
      char name[32], ctype[128];
      snprintf(name, sizeof(name), "wmma%d", c_alu++);
      csmap_set(&names, u, strdup(name));
      cuda_render_uop_ctype(ctx, u, ctype, sizeof(ctype));
      csb_printf(&decls, "  %s %s;\n", ctype, name);
      for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
      csb_printf(
          &body, "%s = __%s(%s, %s, %s);\n", name, helper,
          csmap_get(&names, u->src[0]), csmap_get(&names, u->src[1]),
          csmap_get(&names, u->src[2])
      );
      continue;
    }

    /* --- ALU ---------------------------------------------------------- */
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      const char *rendered[3] = {"", "", ""};
      char *stripped[3] = {NULL, NULL, NULL};
      bool associative =
          u->op == POLY_OP_ADD || u->op == POLY_OP_MUL || u->op == POLY_OP_XOR ||
          u->op == POLY_OP_OR || u->op == POLY_OP_AND;
      for (int j = 0; j < u->n_src && j < 3; j++) {
        const char *source = csmap_get(&names, u->src[j]);
        if (!source) source = "";
        if (associative && u->src[j]->op == u->op) {
          stripped[j] = cuda_strip_parens(source);
          rendered[j] = stripped[j] ? stripped[j] : source;
        } else {
          rendered[j] = source;
        }
      }
      char *expr =
          cuda_render_alu(u->op, u->dtype, rendered[0], rendered[1], rendered[2]);
      for (int j = 0; j < 3; j++)
        free(stripped[j]);
      if (!expr) {
        free(decls.buf);
        free(body.buf);
        csmap_destroy(&names);
        return NULL;
      }

      /* Pinned CStyleLanguage._render: one-consumer non-WHERE ALU remains an
       * expression unless EXPAND_SSA requests explicit statements. */
      bool expand_ssa =
          poly_getenv_flag("EXPAND_SSA") || poly_getenv_flag("POLY_EXPAND_SSA");
      if (u->op != POLY_OP_WHERE && cuda_child_count(uops, n, u) == 1 && !expand_ssa) {
        csmap_set(&names, u, expr);
        continue;
      }

      char name[32];
      snprintf(name, sizeof(name), "alu%d", c_alu++);
      csmap_set(&names, u, strdup(name));

      {
        char ctype[128];
        cuda_render_uop_ctype(ctx, u, ctype, sizeof(ctype));
        csb_printf(&decls, "  %s %s;\n", ctype, name);
      }
      for (int d = 0; d < depth; d++)
        csb_puts(&body, "  ");
      csb_printf(&body, "%s = %s;\n", name, expr);
      free(expr);
      continue;
    }

    /* --- VECTORIZE --------------------------------------------------- */
    if (u->op == POLY_OP_STACK) {
      char *expr = cuda_render_vector_expr(ctx, &names, u);
      /* Match tinygrad CStyleLanguage._render: one-use STACK nodes are kept as
       * expressions instead of materialized as locals. This prevents dead
       * renderer-facing STACKs that only feed structural helper ops (for
       * example UNROLL) from emitting illegal CUDA locals like float512. */
      if (cuda_child_count(uops, n, u) <= 1) {
        csmap_set(&names, u, expr);
        continue;
      }

      char name[32];
      snprintf(name, sizeof(name), "vec%d", c_alu++);
      csmap_set(&names, u, strdup(name));

      char ctype[128];
      cuda_render_uop_ctype(ctx, u, ctype, sizeof(ctype));
      csb_printf(&decls, "  %s %s;\n", ctype, name);
      for (int d = 0; d < depth; d++)
        csb_puts(&body, "  ");
      csb_printf(&body, "%s = %s;\n", name, expr);
      free(expr);
      continue;
    }

    /* --- IF ----------------------------------------------------------- */
    if (u->op == POLY_OP_IF) {
      char *cond_s = csmap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++)
        csb_puts(&body, "  ");
      csb_printf(&body, "if (%s) {\n", cond_s);
      depth++;
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

  /* Build complete CUDA source */
  CudaStrBuf out;
  csb_init(&out);

  /* Prefix: CUDA-specific defines and helpers */
  csb_puts(&out, "typedef unsigned int uint;\n");
  csb_puts(&out, "#define INFINITY (__int_as_float(0x7f800000))\n");
  csb_puts(&out, "#define NAN (__int_as_float(0x7fffffff))\n");
  csb_puts(&out, "template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) {\n");
  csb_puts(&out, "  union U { F f; T t; }; U u; u.f = v; return u.t;\n");
  csb_puts(&out, "}\n");

  /* Conditional includes for half-precision types */
  {
    bool uses_f16 = false, uses_bf16 = false, uses_fp8 = false;
    PolyDType vec_prefix_dtypes[32];
    int vec_prefix_counts[32];
    int n_vec_prefixes = 0;
    for (int i = 0; i < n; i++) {
      if (poly_dtype_eq(uops[i]->dtype, POLY_VOID)) continue;
      PolyDType s = uops[i]->dtype;
      int count = cuda_uop_lanes(ctx, uops[i]);
      if (s.priority == POLY_FLOAT16.priority) uses_f16 = true;
      if (s.priority == POLY_BFLOAT16.priority) uses_bf16 = true;
      if (poly_dtype_is_fp8(s)) uses_fp8 = true;
      if (cuda_vector_needs_prefix(s, count)) {
        bool seen = false;
        for (int j = 0; j < n_vec_prefixes; j++) {
          if (poly_dtype_eq(vec_prefix_dtypes[j], s) && vec_prefix_counts[j] == count) {
            seen = true;
            break;
          }
        }
        if (!seen && n_vec_prefixes < 32) {
          vec_prefix_dtypes[n_vec_prefixes] = s;
          vec_prefix_counts[n_vec_prefixes++] = count;
        }
      }
    }
    if (uses_fp8) csb_puts(&out, "#include <cuda_fp8.h>\n");
    if (uses_f16) csb_puts(&out, "#include <cuda_fp16.h>\n");
    if (uses_bf16) csb_puts(&out, "#include <cuda_bf16.h>\n");
    for (int i = 0; i < n_vec_prefixes; i++)
      cuda_render_vector_prefix(&out, vec_prefix_dtypes[i], vec_prefix_counts[i]);
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
      if (!cuda_render_wmma_helper(&out, ctx, uops[i])) return NULL;
      if (n_wmma_names < 32)
        snprintf(wmma_names[n_wmma_names++], sizeof(wmma_names[0]), "%s", name);
    }
  }
  csb_puts(&out, "\n");

  /* Kernel signature */
  csb_printf(
      &out, "extern \"C\" __global__ void __launch_bounds__(%d) %s(", launch_bounds, fn_name
  );
  for (int i = 0; i < n_params; i++) {
    if (i > 0) csb_puts(&out, ", ");
    csb_printf(&out, "%s %s", param_types[i], param_names[i]);
  }
  csb_puts(&out, ") {\n");
  if (decls.len > 0) csb_puts(&out, decls.buf);
  csb_puts(&out, body.buf);
  csb_puts(&out, "}\n");

  /* No _call wrapper needed — CUDA uses cuLaunchKernel */

  for (int i = 0; i < n_params; i++) {
    free(param_types[i]);
    free(param_names[i]);
  }
  free(decls.buf);
  free(body.buf);
  csmap_destroy(&names);

  return out.buf;
}

#endif /* POLY_HAS_CUDA */
