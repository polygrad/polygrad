/*
 * render_hip.c -- HIP C++ renderer + HIP linearizer
 *
 * Walks linearized UOps and emits HIP C++ source code for AMD GPUs.
 * Mirrors render_cuda.c with AMD-specific syntax from tinygrad's
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

#include "codegen.h"
#include "pat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

/* ── String builder ───────────────────────────────────────────────── */

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

/* ── Pointer -> string hash map ───────────────────────────────────── */

typedef struct {
  PolyUOp **keys;
  char **vals;
  int cap;
} HipStrMap;

static uint32_t hip_ptr_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  v = ((v >> 16) ^ v) * 0x45d9f3bU;
  v = ((v >> 16) ^ v) * 0x45d9f3bU;
  return (uint32_t)((v >> 16) ^ v);
}

static void hsmap_init(HipStrMap *m, int n) {
  m->cap = (n < 4) ? 16 : n * 3;
  m->keys = calloc(m->cap, sizeof(PolyUOp *));
  m->vals = calloc(m->cap, sizeof(char *));
}

static void hsmap_set(HipStrMap *m, PolyUOp *key, char *val) {
  uint32_t h = hip_ptr_hash(key) % m->cap;
  while (m->keys[h] && m->keys[h] != key) h = (h + 1) % m->cap;
  if (m->keys[h] == key) free(m->vals[h]);
  m->keys[h] = key;
  m->vals[h] = val;
}

static char *hsmap_get(HipStrMap *m, PolyUOp *key) {
  uint32_t h = hip_ptr_hash(key) % m->cap;
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

/* ── Type rendering ──────────────────────────────────────────────── */

/* Map scalar PolyDType to a short identifier-safe name for vector typedefs.
 * E.g. POLY_FLOAT16 -> "half", POLY_FLOAT32 -> "float", POLY_INT32 -> "int".
 * Returns the raw C type for scalars and the typedef alias for vectors. */
static const char *hip_scalar_alias(PolyDType s) {
  if (poly_dtype_eq(s, POLY_FLOAT16))  return "half";
  if (poly_dtype_eq(s, POLY_BFLOAT16)) return "bfloat16";
  if (poly_dtype_eq(s, POLY_FLOAT32))  return "float";
  if (poly_dtype_eq(s, POLY_FLOAT64))  return "double";
  if (poly_dtype_eq(s, POLY_INT8))     return "char";
  if (poly_dtype_eq(s, POLY_UINT8))    return "uchar";
  if (poly_dtype_eq(s, POLY_INT16))    return "short";
  if (poly_dtype_eq(s, POLY_UINT16))   return "ushort";
  if (poly_dtype_eq(s, POLY_INT32))    return "int";
  if (poly_dtype_eq(s, POLY_UINT32))   return "uint";
  if (poly_dtype_eq(s, POLY_INT64))    return "long";
  if (poly_dtype_eq(s, POLY_UINT64))   return "ulong";
  if (poly_dtype_eq(s, POLY_BOOL))     return "bool";
  return s.name;
}

/* Map scalar PolyDType to the actual C type used in HIP code. */
static const char *hip_scalar_ctype(PolyDType s) {
  if (poly_dtype_eq(s, POLY_FLOAT16))  return "_Float16";
  if (poly_dtype_eq(s, POLY_BFLOAT16)) return "unsigned short";
  return s.name;
}

/* Render a HIP C++ type for a PolyDType.
 * Scalars: raw C type (_Float16, float, int, etc.)
 * Vectors: typedef alias (half4, float4, int4, etc.)
 * Vector typedefs must be emitted in the kernel prefix. */
static void hip_render_ctype(PolyDType dt, char *buf, int cap) {
  PolyDType s = poly_dtype_scalar(dt);
  if (dt.count <= 1) {
    snprintf(buf, cap, "%s", hip_scalar_ctype(s));
    return;
  }
  snprintf(buf, cap, "%s%d", hip_scalar_alias(s), (int)dt.count);
}

/* ── Render helpers ───────────────────────────────────────────────── */

static char *hip_render_float_const(double v, PolyDType dt, char *buf, int cap) {
  bool is_f64 = poly_dtype_eq(poly_dtype_scalar(dt), POLY_FLOAT64);
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
      if (len + 2 < cap) { buf[len] = '.'; buf[len+1] = '0'; buf[len+2] = '\0'; }
    }
  } else {
    /* f32: round-trip float32 through text with 'f' suffix. */
    snprintf(buf, cap, "%.9g", (double)(float)v);
    if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E')) {
      int len = (int)strlen(buf);
      if (len + 2 < cap) { buf[len] = '.'; buf[len+1] = '0'; buf[len+2] = '\0'; }
    }
    int len = (int)strlen(buf);
    if (len + 1 < cap) { buf[len] = 'f'; buf[len+1] = '\0'; }
  }
  return buf;
}

static void hip_render_alu(char *buf, int cap, PolyOps op, PolyDType dtype,
                            const char *s0, const char *s1, const char *s2) {
  switch (op) {
  case POLY_OP_NEG:
    snprintf(buf, cap, poly_dtype_is_bool(dtype) ? "(!%s)" : "(-%s)", s0); break;
  case POLY_OP_SQRT:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "__ocml_sqrt_f64(%s)" : "__ocml_sqrt_f32(%s)", s0); break;
  case POLY_OP_TRUNC:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "__ocml_trunc_f64(%s)" : "__ocml_trunc_f32(%s)", s0); break;
  case POLY_OP_EXP2:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "__ocml_exp2_f64(%s)" : "__ocml_exp2_f32(%s)", s0); break;
  case POLY_OP_LOG2:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "__ocml_log2_f64(%s)" : "__ocml_log2_f32(%s)", s0); break;
  case POLY_OP_SIN:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "__ocml_sin_f64(%s)" : "__ocml_sin_f32(%s)", s0); break;
  case POLY_OP_RECIPROCAL: snprintf(buf, cap, "(1/%s)", s0); break;
  case POLY_OP_ADD:   snprintf(buf, cap, "(%s+%s)", s0, s1); break;
  case POLY_OP_SUB:   snprintf(buf, cap, "(%s-%s)", s0, s1); break;
  case POLY_OP_MUL:   snprintf(buf, cap, "(%s*%s)", s0, s1); break;
  case POLY_OP_FDIV:  snprintf(buf, cap, "(%s/%s)", s0, s1); break;
  case POLY_OP_IDIV:  snprintf(buf, cap, "(%s/%s)", s0, s1); break;
  case POLY_OP_MOD:   snprintf(buf, cap, "(%s%%%s)", s0, s1); break;
  case POLY_OP_SHL:   snprintf(buf, cap, "(%s<<%s)", s0, s1); break;
  case POLY_OP_SHR:   snprintf(buf, cap, "(%s>>%s)", s0, s1); break;
  case POLY_OP_AND:   snprintf(buf, cap, "(%s&%s)", s0, s1); break;
  case POLY_OP_OR:    snprintf(buf, cap, "(%s|%s)", s0, s1); break;
  case POLY_OP_XOR:   snprintf(buf, cap, "(%s^%s)", s0, s1); break;
  case POLY_OP_CMPLT: snprintf(buf, cap, "(%s<%s)", s0, s1); break;
  case POLY_OP_CMPNE: snprintf(buf, cap, "(%s!=%s)", s0, s1); break;
  case POLY_OP_CMPEQ: snprintf(buf, cap, "(%s==%s)", s0, s1); break;
  case POLY_OP_MAX:   snprintf(buf, cap, "((%s>%s)?%s:%s)", s0, s1, s0, s1); break;
  case POLY_OP_POW:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "pow(%s, %s)" : "powf(%s, %s)", s0, s1); break;
  case POLY_OP_WHERE:  snprintf(buf, cap, "(%s?%s:%s)", s0, s1, s2); break;
  case POLY_OP_MULACC:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "__builtin_fma(%s,%s,%s)" : "__builtin_fmaf(%s,%s,%s)", s0, s1, s2);
    break;
  default: snprintf(buf, cap, "/* unknown op %d */0", op); break;
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

/* ── HIP Linearizer ──────────────────────────────────────────────── */

PolyUOp **poly_linearize_hip(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  /* Same pipeline as CUDA: HIP GPUs also have FMA.
   * Pipeline: sym -> group_for_reduce -> pm_reduce(with sink END merge)
   *        -> sym -> decomp -> transcendental -> decomp -> gpudims */
  PolyRendererCaps hip_caps = { .has_mulacc = true, .has_threefry = false };
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_group_for_reduce(ctx, sink, 256);
  sink = poly_apply_pm_reduce(ctx, sink);
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass_caps(hip_caps));
  sink = poly_graph_rewrite(ctx, sink, poly_pm_transcendental_pass());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass_caps(hip_caps));
  sink = poly_add_gpudims(ctx, sink);
  sink = poly_apply_control_flow(ctx, sink);
  return poly_linearize_rewritten(ctx, sink, n_out);
}

/* ── Track which OCML/OCKL functions are used ─────────────────────── */

typedef struct {
  bool uses_special;    /* needs ockl workitem functions */
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
  /* Collect unique WMMA intrinsic names */
  const char *wmma_names[16];
  int n_wmma_names;
  /* Collect unique vector dtypes that need typedefs */
  PolyDType vec_dtypes[32];
  int n_vec_dtypes;
} HipUsedFuncs;

static void hip_scan_used_funcs(PolyUOp **uops, int n, HipUsedFuncs *used) {
  memset(used, 0, sizeof(*used));
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_SPECIAL) used->uses_special = true;
    bool is_f64 = poly_dtype_eq(u->dtype, POLY_FLOAT64);
    switch (u->op) {
    case POLY_OP_EXP2:  if (is_f64) used->uses_exp2_f64  = true; else used->uses_exp2_f32  = true; break;
    case POLY_OP_LOG2:  if (is_f64) used->uses_log2_f64  = true; else used->uses_log2_f32  = true; break;
    case POLY_OP_SIN:   if (is_f64) used->uses_sin_f64   = true; else used->uses_sin_f32   = true; break;
    case POLY_OP_SQRT:  if (is_f64) used->uses_sqrt_f64  = true; else used->uses_sqrt_f32  = true; break;
    case POLY_OP_TRUNC: if (is_f64) used->uses_trunc_f64 = true; else used->uses_trunc_f32 = true; break;
    case POLY_OP_WMMA: {
      used->uses_wmma = true;
      const char *wn = (u->arg.kind == POLY_ARG_STRING && u->arg.str) ? u->arg.str : NULL;
      if (wn && used->n_wmma_names < 16) {
        bool dup = false;
        for (int j = 0; j < used->n_wmma_names; j++)
          if (strcmp(used->wmma_names[j], wn) == 0) { dup = true; break; }
        if (!dup) used->wmma_names[used->n_wmma_names++] = wn;
      }
      break;
    }
    default: break;
    }
    /* Collect vector dtypes for typedef emission */
    if (u->dtype.count > 1 && !u->dtype.is_ptr && used->n_vec_dtypes < 32) {
      bool dup = false;
      for (int j = 0; j < used->n_vec_dtypes; j++) {
        if (poly_dtype_eq(poly_dtype_scalar(used->vec_dtypes[j]),
                          poly_dtype_scalar(u->dtype)) &&
            used->vec_dtypes[j].count == u->dtype.count) {
          dup = true; break;
        }
      }
      if (!dup) used->vec_dtypes[used->n_vec_dtypes++] = u->dtype;
    }
  }
}

/* ── HIP Renderer ─────────────────────────────────────────────────── */

char *poly_render_hip(PolyUOp **uops, int n, const char *fn_name, int launch_bounds) {
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
    if (u->op == POLY_OP_RANGE)
      (void)hip_range_slot(live_ranges, &n_live_ranges, u, true);
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

    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP)
      continue;

    if (u->op != POLY_OP_END) {
      for (int j = 0; j < u->n_src; j++) {
        if (u->src[j] && u->src[j]->op == POLY_OP_RANGE) {
          int ri = hip_range_slot(live_ranges, &n_live_ranges, u->src[j], false);
          if (ri >= 0 && live_remaining[ri] > 0) live_remaining[ri]--;
        }
      }
    }

    /* --- PARAM -------------------------------------------------------- */
    if (u->op == POLY_OP_PARAM) {
      char name[32];
      snprintf(name, sizeof(name), "data%lld", (long long)u->arg.i);
      hsmap_set(&names, u, strdup(name));

      PolyDType base = poly_dtype_scalar(u->dtype);
      char type[64];
      snprintf(type, sizeof(type), "%s*", base.name);
      param_types[n_params] = strdup(type);
      param_names[n_params] = strdup(name);
      param_order[n_params] = (int)u->arg.i;
      n_params++;
      continue;
    }

    /* --- DEFINE_VAR --------------------------------------------------- */
    if (u->op == POLY_OP_DEFINE_VAR) {
      const char *vname = u->arg.kind == POLY_ARG_DEFINE_VAR ? u->arg.define_var.name
                        : (u->arg.str ? u->arg.str : "var");
      hsmap_set(&names, u, strdup(vname));
      param_types[n_params] = strdup("const int");
      param_names[n_params] = strdup(vname);
      param_order[n_params] = 10000 + n_params;
      n_params++;
      continue;
    }

    /* --- CONST -------------------------------------------------------- */
    if (u->op == POLY_OP_CONST) {
      char val[64];
      if (poly_dtype_is_float(u->dtype)) {
        hip_render_float_const(u->arg.f, u->dtype, val, sizeof(val));
      } else if (poly_dtype_is_bool(u->dtype)) {
        snprintf(val, sizeof(val), "%d", u->arg.b ? 1 : 0);
      } else if (poly_dtype_eq(u->dtype, POLY_INT64)) {
        snprintf(val, sizeof(val), "%lldll", (long long)u->arg.i);
      } else if (poly_dtype_eq(u->dtype, POLY_UINT64)) {
        snprintf(val, sizeof(val), "%lluull", (unsigned long long)(uint64_t)u->arg.i);
      } else if (poly_dtype_eq(u->dtype, POLY_UINT32)) {
        snprintf(val, sizeof(val), "%uu", (unsigned)(uint32_t)u->arg.i);
      } else {
        snprintf(val, sizeof(val), "%lld", (long long)u->arg.i);
      }
      hsmap_set(&names, u, strdup(val));
      continue;
    }

    /* --- INDEX -------------------------------------------------------- */
    if (u->op == POLY_OP_INDEX) {
      char *buf_s = hsmap_get(&names, u->src[0]);
      char *idx_s = hsmap_get(&names, u->src[1]);
      char expr[256];
      snprintf(expr, sizeof(expr), "(%s+%s)", buf_s, idx_s);
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
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
      if (sname[0] == 'l') {
        /* Local index: __ockl_get_local_id */
        hsb_printf(&body, "int %s = __ockl_get_local_id(%d);\n", sname, dim_idx);
      } else {
        /* Global index: __ockl_get_group_id * __ockl_get_local_size + __ockl_get_local_id */
        hsb_printf(&body, "int %s = (__ockl_get_group_id(%d)*__ockl_get_local_size(%d)+__ockl_get_local_id(%d));\n",
                   sname, dim_idx, dim_idx, dim_idx);
      }

      /* Bounds check for global indices only */
      if (sname[0] != 'l') {
        char *bound = hsmap_get(&names, u->src[0]);
        if (bound) {
          for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
          hsb_printf(&body, "if (%s >= %s) return;\n", sname, bound);
        }
      }
      continue;
    }

    /* --- BARRIER ------------------------------------------------------ */
    if (u->op == POLY_OP_BARRIER) {
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
      hsb_puts(&body, "__syncthreads();\n");
      continue;
    }

    /* --- RANGE: for loop ---------------------------------------------- */
    if (u->op == POLY_OP_RANGE) {
      char name[32];
      snprintf(name, sizeof(name), "ridx%lld", (long long)poly_range_axis_id(u->arg));
      hsmap_set(&names, u, strdup(name));

      char *bound = hsmap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
      hsb_printf(&body, "for (int %s = 0; %s < %s; %s++) {\n",
                 name, name, bound, name);
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
          if (open_ranges[p] == want) { pos = p; break; }
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
          for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
          hsb_puts(&body, "}\n");
          n_open_ranges--;
        }
        continue;
      }

      depth--;
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
      hsb_puts(&body, "}\n");
      if (u->op == POLY_OP_END && n_open_ranges > 0) n_open_ranges--;
      continue;
    }

    /* --- DEFINE_LOCAL: shared memory array ----------------------------- */
    if (u->op == POLY_OP_DEFINE_LOCAL) {
      char name[32];
      snprintf(name, sizeof(name), "smem%d", c_acc++);
      hsmap_set(&names, u, strdup(name));

      /* HIP shared memory: __attribute__((shared, aligned(16))) */
      int smem_size = u->dtype.ptr_size > 0 ? u->dtype.ptr_size : 1;
      PolyDType base = poly_dtype_scalar(u->dtype);
      hsb_printf(&decls, "  __attribute__((shared, aligned(16))) %s %s[%d];\n",
                 base.name, name, smem_size);
      continue;
    }

    /* --- DEFINE_REG --------------------------------------------------- */
    if (u->op == POLY_OP_DEFINE_REG) {
      char name[32];
      snprintf(name, sizeof(name), "r%lld", (long long)u->arg.i);
      hsmap_set(&names, u, strdup(name));

      PolyDType base = poly_dtype_scalar(u->dtype);
      hsb_printf(&decls, "  %s %s[1];\n", base.name, name);
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
      { char ctype[128]; hip_render_ctype(u->dtype, ctype, sizeof(ctype));
      hsb_printf(&decls, "  %s %s;\n", ctype, name); }
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
      hsb_printf(&body, "%s = (*%s);\n", name, bidx);
      continue;
    }

    /* --- STORE -------------------------------------------------------- */
    if (u->op == POLY_OP_STORE) {
      char *target = hsmap_get(&names, u->src[0]);
      char *val    = hsmap_get(&names, u->src[1]);
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
      if (u->src[0]->op == POLY_OP_DEFINE_LOCAL)
        hsb_printf(&body, "%s = %s;\n", target, val);
      else
        hsb_printf(&body, "*%s = %s;\n", target, val);
      continue;
    }

    /* --- CAST --------------------------------------------------------- */
    if (u->op == POLY_OP_CAST) {
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      hsmap_set(&names, u, strdup(name));

      char *src_s = hsmap_get(&names, u->src[0]);
      { char ctype[128]; hip_render_ctype(u->dtype, ctype, sizeof(ctype));
      hsb_printf(&decls, "  %s %s;\n", ctype, name); }
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
      { char ctype2[128]; hip_render_ctype(u->dtype, ctype2, sizeof(ctype2));
      hsb_printf(&body, "%s = (%s)(%s);\n", name, ctype2, src_s); }
      continue;
    }

    /* --- BITCAST: use template helper --------------------------------- */
    if (u->op == POLY_OP_BITCAST) {
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      hsmap_set(&names, u, strdup(name));

      char *src_s = hsmap_get(&names, u->src[0]);
      char dst_type[128], src_type[128];
      hip_render_ctype(u->dtype, dst_type, sizeof(dst_type));
      hip_render_ctype(u->src[0]->dtype, src_type, sizeof(src_type));
      hsb_printf(&decls, "  %s %s;\n", dst_type, name);
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
      hsb_printf(&body, "%s = tg_bitcast<%s>((%s)(%s));\n",
                 name, dst_type, src_type, src_s);
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

      { char ctype[128]; hip_render_ctype(u->dtype, ctype, sizeof(ctype));
      hsb_printf(&decls, "  %s %s;\n", ctype, name); }
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
      hsb_printf(&body, "%s = %s;\n", name, expr);
      continue;
    }

    /* --- IF ----------------------------------------------------------- */
    if (u->op == POLY_OP_IF) {
      char *cond_s = hsmap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");
      hsb_printf(&body, "if (%s) {\n", cond_s);
      depth++;
      continue;
    }

    /* --- VECTORIZE / VCONST: vector literal -------------------------- */
    if (u->op == POLY_OP_VECTORIZE || u->op == POLY_OP_VCONST) {
      char name[32];
      snprintf(name, sizeof(name), "vec%d", c_alu++);
      hsmap_set(&names, u, strdup(name));

      char ctype[128];
      hip_render_ctype(u->dtype, ctype, sizeof(ctype));
      hsb_printf(&decls, "  %s %s;\n", ctype, name);
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");

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

    /* --- GEP: vector lane extract ------------------------------------ */
    if (u->op == POLY_OP_GEP) {
      char name[32];
      snprintf(name, sizeof(name), "gep%d", c_alu++);
      hsmap_set(&names, u, strdup(name));

      char ctype[128];
      hip_render_ctype(u->dtype, ctype, sizeof(ctype));
      hsb_printf(&decls, "  %s %s;\n", ctype, name);
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");

      char *src_s = hsmap_get(&names, u->src[0]);
      if (u->arg.kind == POLY_ARG_INT) {
        int idx = (int)u->arg.i;
        hsb_printf(&body, "%s = %s[%d];\n", name, src_s ? src_s : "0", idx);
      } else if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n == 1) {
        int idx = (int)u->arg.int_tuple.vals[0];
        /* Use array-style indexing for ext_vector_type */
        hsb_printf(&body, "%s = %s[%d];\n", name, src_s ? src_s : "0", idx);
      } else if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 1) {
        /* Multi-lane GEP: construct vector from selected lanes */
        hsb_printf(&body, "%s = (%s){", name, ctype);
        for (int j = 0; j < u->arg.int_tuple.n; j++) {
          if (j) hsb_puts(&body, ", ");
          hsb_printf(&body, "%s[%d]", src_s ? src_s : "0", (int)u->arg.int_tuple.vals[j]);
        }
        hsb_puts(&body, "};\n");
      } else {
        hsb_printf(&body, "%s = %s;\n", name, src_s ? src_s : "0");
      }
      continue;
    }

    /* --- WMMA: matrix multiply-accumulate ----------------------------- */
    if (u->op == POLY_OP_WMMA) {
      char name[32];
      snprintf(name, sizeof(name), "wmma%d", c_alu++);
      hsmap_set(&names, u, strdup(name));

      char ctype[128];
      hip_render_ctype(u->dtype, ctype, sizeof(ctype));
      hsb_printf(&decls, "  %s %s;\n", ctype, name);
      for (int d = 0; d < depth; d++) hsb_puts(&body, "  ");

      /* WMMA has 3 sources: A, B, C(accumulator). Arg is the intrinsic name. */
      char *a_s = (u->n_src > 0) ? hsmap_get(&names, u->src[0]) : "0";
      char *b_s = (u->n_src > 1) ? hsmap_get(&names, u->src[1]) : "0";
      char *c_s = (u->n_src > 2) ? hsmap_get(&names, u->src[2]) : "0";

      /* Emit: wmma0 = __builtin_amdgcn_<name>(A, B, C, 0, 0, 0);
       * MFMA builtins take 6 args: A, B, C, cbsz, abid, blgp. */
      const char *wmma_name = (u->arg.kind == POLY_ARG_STRING && u->arg.str)
                              ? u->arg.str : "WMMA_UNKNOWN";
      hsb_printf(&body, "%s = __builtin_amdgcn_%s(%s, %s, %s, 0, 0, 0);\n",
                 name, wmma_name, a_s ? a_s : "0", b_s ? b_s : "0", c_s ? c_s : "0");
      continue;
    }
  }

  /* Sort params */
  for (int i = 1; i < n_params; i++) {
    int ko = param_order[i];
    char *kt = param_types[i], *kn = param_names[i];
    int j = i - 1;
    while (j >= 0 && param_order[j] > ko) {
      param_order[j+1] = param_order[j];
      param_types[j+1] = param_types[j];
      param_names[j+1] = param_names[j];
      j--;
    }
    param_order[j+1] = ko;
    param_types[j+1] = kt;
    param_names[j+1] = kn;
  }

  /* Scan which OCML/OCKL functions are used */
  HipUsedFuncs used;
  hip_scan_used_funcs(uops, n, &used);

  /* Build complete HIP source */
  HipStrBuf out;
  hsb_init(&out);

  /* Prefix: compiled via comgr with -nogpuinc (no built-in headers).
   * All device function declarations must be provided explicitly,
   * matching tinygrad's AMDHIPRenderer (cstyle.py:466-563). */
  hsb_puts(&out, "typedef long unsigned int size_t;\n");
  hsb_puts(&out, "#define INFINITY (__builtin_inff())\n");
  hsb_puts(&out, "#define NAN (__builtin_nanf(\"\"))\n");

  /* Bitcast template helper. With -nogpuinc, __device__/__forceinline__
   * are unavailable; use __attribute__ equivalents. */
  hsb_puts(&out, "template <class T, class F> __attribute__((device, always_inline)) T tg_bitcast(F v) {\n");
  hsb_puts(&out, "  union U { F f; T t; }; U u; u.f = v; return u.t;\n");
  hsb_puts(&out, "}\n");

  /* OCKL workitem function declarations (only when SPECIAL ops are used) */
  if (used.uses_special) {
    hsb_puts(&out, "extern \"C\" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);\n");
    hsb_puts(&out, "extern \"C\" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);\n");
    hsb_puts(&out, "extern \"C\" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);\n");
  }

  /* OCML math function declarations (only what's used) */
#define EMIT_OCML(flag, name, attr) \
  if (used.flag) hsb_printf(&out, "extern \"C\" __attribute__((device%s)) float __ocml_%s_f32(float);\n", attr, name)
#define EMIT_OCML_F64(flag, name, attr) \
  if (used.flag) hsb_printf(&out, "extern \"C\" __attribute__((device%s)) double __ocml_%s_f64(double);\n", attr, name)

  EMIT_OCML(uses_exp2_f32,  "exp2",  ", pure");
  EMIT_OCML(uses_log2_f32,  "log2",  ", pure");
  EMIT_OCML(uses_sqrt_f32,  "sqrt",  ", const");
  EMIT_OCML(uses_sin_f32,   "sin",   "");
  EMIT_OCML(uses_trunc_f32, "trunc", "");
  EMIT_OCML_F64(uses_exp2_f64,  "exp2",  ", pure");
  EMIT_OCML_F64(uses_log2_f64,  "log2",  ", pure");
  EMIT_OCML_F64(uses_sqrt_f64,  "sqrt",  ", const");
  EMIT_OCML_F64(uses_sin_f64,   "sin",   "");
  EMIT_OCML_F64(uses_trunc_f64, "trunc", "");

  /* WMMA: builtins emitted directly in body, no #define needed. */

#undef EMIT_OCML
#undef EMIT_OCML_F64

  /* Vector type typedefs (ext_vector_type requires typedef in HIP C++) */
  for (int vi = 0; vi < used.n_vec_dtypes; vi++) {
    PolyDType vdt = used.vec_dtypes[vi];
    char tname[64];
    hip_render_ctype(vdt, tname, sizeof(tname));
    const char *sctype = hip_scalar_ctype(poly_dtype_scalar(vdt));
    hsb_printf(&out, "typedef %s %s __attribute__((ext_vector_type(%d)));\n",
               sctype, tname, (int)vdt.count);
  }

  hsb_puts(&out, "\n");

  /* Kernel signature: AMD-specific attributes */
  hsb_printf(&out,
    "extern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, %d))) %s(",
    launch_bounds, fn_name);
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
