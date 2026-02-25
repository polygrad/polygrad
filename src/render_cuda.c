/*
 * render_cuda.c — CUDA C renderer + CUDA linearizer
 *
 * Walks linearized UOps and emits CUDA C++ source code matching
 * tinygrad's CUDARenderer output. Also provides poly_linearize_cuda()
 * which chains: full_rewrite_to_sink → add_gpudims → linearize.
 */

#ifdef POLY_HAS_CUDA

#define _POSIX_C_SOURCE 200809L

#include "codegen.h"
#include "pat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

/* ── String builder (local copy from render_c.c) ─────────────────────── */

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

/* ── Pointer → string hash map ───────────────────────────────────────── */

typedef struct {
  PolyUOp **keys;
  char **vals;
  int cap;
} CudaStrMap;

static uint32_t cuda_ptr_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  v = ((v >> 16) ^ v) * 0x45d9f3bU;
  v = ((v >> 16) ^ v) * 0x45d9f3bU;
  return (uint32_t)((v >> 16) ^ v);
}

static void csmap_init(CudaStrMap *m, int n) {
  m->cap = (n < 4) ? 16 : n * 3;
  m->keys = calloc(m->cap, sizeof(PolyUOp *));
  m->vals = calloc(m->cap, sizeof(char *));
}

static void csmap_set(CudaStrMap *m, PolyUOp *key, char *val) {
  uint32_t h = cuda_ptr_hash(key) % m->cap;
  while (m->keys[h] && m->keys[h] != key) h = (h + 1) % m->cap;
  if (m->keys[h] == key) free(m->vals[h]);
  m->keys[h] = key;
  m->vals[h] = val;
}

static char *csmap_get(CudaStrMap *m, PolyUOp *key) {
  uint32_t h = cuda_ptr_hash(key) % m->cap;
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

/* ── Render helpers ──────────────────────────────────────────────────── */

static char *cuda_render_float_const(double v, char *buf, int cap) {
  if (isinf(v)) {
    snprintf(buf, cap, v > 0 ? "INFINITY" : "(-INFINITY)");
    return buf;
  }
  if (isnan(v)) {
    snprintf(buf, cap, "NAN");
    return buf;
  }
  /* Use enough digits to round-trip float32 constants through text. */
  snprintf(buf, cap, "%.9g", (double)(float)v);
  if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E')) {
    int len = (int)strlen(buf);
    if (len + 2 < cap) { buf[len] = '.'; buf[len+1] = '0'; buf[len+2] = '\0'; }
  }
  int len = (int)strlen(buf);
  if (len + 1 < cap) { buf[len] = 'f'; buf[len+1] = '\0'; }
  return buf;
}

static void cuda_render_alu(char *buf, int cap, PolyOps op, PolyDType dtype,
                             const char *s0, const char *s1, const char *s2) {
  switch (op) {
  case POLY_OP_NEG:
    snprintf(buf, cap, poly_dtype_is_bool(dtype) ? "(!%s)" : "(-%s)", s0); break;
  case POLY_OP_SQRT:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "sqrt(%s)" : "sqrtf(%s)", s0); break;
  case POLY_OP_TRUNC:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "trunc(%s)" : "truncf(%s)", s0); break;
  case POLY_OP_EXP2:       snprintf(buf, cap, "exp2f(%s)", s0); break;
  case POLY_OP_LOG2:       snprintf(buf, cap, "log2f(%s)", s0); break;
  case POLY_OP_SIN:        snprintf(buf, cap, "sinf(%s)", s0); break;
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
  case POLY_OP_MULACC: snprintf(buf, cap, "((%s*%s)+%s)", s0, s1, s2); break;
  default: snprintf(buf, cap, "/* unknown op %d */0", op); break;
  }
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

/* ── CUDA Linearizer ─────────────────────────────────────────────────── */

PolyUOp **poly_linearize_cuda(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  /* Inline codegen passes to insert group_for_reduce between sym and pm_reduce.
   * Pipeline: sym → group_for_reduce → pm_reduce(with sink END merge)
   *        → sym → decomp → transcendental → decomp → gpudims */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_group_for_reduce(ctx, sink, 256);
  sink = poly_apply_pm_reduce(ctx, sink);
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_transcendental_pass());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass());
  sink = poly_add_gpudims(ctx, sink);
  sink = poly_apply_control_flow(ctx, sink);
  return poly_linearize_rewritten(ctx, sink, n_out);
}

/* ── CUDA Renderer ───────────────────────────────────────────────────── */

char *poly_render_cuda(PolyUOp **uops, int n, const char *fn_name, int launch_bounds) {
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
    if (u->op == POLY_OP_RANGE)
      (void)cuda_range_slot(live_ranges, &n_live_ranges, u, true);
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

    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP)
      continue;

    if (u->op != POLY_OP_END) {
      for (int j = 0; j < u->n_src; j++) {
        if (u->src[j] && u->src[j]->op == POLY_OP_RANGE) {
          int ri = cuda_range_slot(live_ranges, &n_live_ranges, u->src[j], false);
          if (ri >= 0 && live_remaining[ri] > 0) live_remaining[ri]--;
        }
      }
    }

    /* --- PARAM -------------------------------------------------------- */
    if (u->op == POLY_OP_PARAM) {
      char name[32];
      snprintf(name, sizeof(name), "data%lld", (long long)u->arg.i);
      csmap_set(&names, u, strdup(name));

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
      csmap_set(&names, u, strdup(vname));
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
        cuda_render_float_const(u->arg.f, val, sizeof(val));
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
      csmap_set(&names, u, strdup(val));
      continue;
    }

    /* --- INDEX -------------------------------------------------------- */
    if (u->op == POLY_OP_INDEX) {
      char *buf_s = csmap_get(&names, u->src[0]);
      char *idx_s = csmap_get(&names, u->src[1]);
      char expr[256];
      snprintf(expr, sizeof(expr), "(%s+%s)", buf_s, idx_s);
      csmap_set(&names, u, strdup(expr));
      continue;
    }

    /* --- SPECIAL: GPU thread index ------------------------------------ */
    if (u->op == POLY_OP_SPECIAL) {
      const char *sname = u->arg.str ? u->arg.str : "gidx0";
      csmap_set(&names, u, strdup(sname));

      /* Determine which dimension: last char of name */
      int dim_idx = 0;
      int slen = (int)strlen(sname);
      if (slen > 0) dim_idx = sname[slen - 1] - '0';
      char dim_char = 'x';
      if (dim_idx == 1) dim_char = 'y';
      else if (dim_idx == 2) dim_char = 'z';

      /* Determine prefix: "gidx" → blockIdx*blockDim+threadIdx */
      for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
      if (sname[0] == 'l') {
        /* Local index: threadIdx */
        csb_printf(&body, "int %s = threadIdx.%c;\n", sname, dim_char);
      } else {
        /* Global index: blockIdx * blockDim + threadIdx */
        csb_printf(&body, "int %s = (blockIdx.%c*blockDim.%c+threadIdx.%c);\n",
                   sname, dim_char, dim_char, dim_char);
      }

      /* Bounds check for global indices only.
       * Skip for lidx: all block_size threads must execute (barrier requires it). */
      if (sname[0] != 'l') {
        char *bound = csmap_get(&names, u->src[0]);
        if (bound) {
          for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
          csb_printf(&body, "if (%s >= %s) return;\n", sname, bound);
        }
      }
      continue;
    }

    /* --- BARRIER ------------------------------------------------------ */
    if (u->op == POLY_OP_BARRIER) {
      for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
      csb_puts(&body, "__syncthreads();\n");
      continue;
    }

    /* --- RANGE: for loop (reduce loops stay as loops) ----------------- */
    if (u->op == POLY_OP_RANGE) {
      char name[32];
      snprintf(name, sizeof(name), "ridx%lld", (long long)poly_range_axis_id(u->arg));
      csmap_set(&names, u, strdup(name));

      char *bound = csmap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
      csb_printf(&body, "for (int %s = 0; %s < %s; %s++) {\n",
                 name, name, bound, name);
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
          if (open_ranges[p] == want) { pos = p; break; }
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
          for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
          csb_puts(&body, "}\n");
          n_open_ranges--;
        }
        continue;
      }

      depth--;
      for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
      csb_puts(&body, "}\n");
      if (u->op == POLY_OP_END && n_open_ranges > 0) n_open_ranges--;
      continue;
    }

    /* --- DEFINE_LOCAL: shared memory array ------------------------------ */
    if (u->op == POLY_OP_DEFINE_LOCAL) {
      char name[32];
      snprintf(name, sizeof(name), "smem%d", c_acc++);
      csmap_set(&names, u, strdup(name));

      /* Render as __shared__ array. Size from pointer dtype's ptr_size. */
      int smem_size = u->dtype.ptr_size > 0 ? u->dtype.ptr_size : 1;
      PolyDType base = poly_dtype_scalar(u->dtype);
      csb_printf(&decls, "  __shared__ %s %s[%d];\n", base.name, name, smem_size);
      continue;
    }

    /* --- DEFINE_REG --------------------------------------------------- */
    if (u->op == POLY_OP_DEFINE_REG) {
      char name[32];
      snprintf(name, sizeof(name), "r%lld", (long long)u->arg.i);
      csmap_set(&names, u, strdup(name));

      PolyDType base = poly_dtype_scalar(u->dtype);
      csb_printf(&decls, "  %s %s[1];\n", base.name, name);
      continue;
    }

    /* --- AFTER -------------------------------------------------------- */
    if (u->op == POLY_OP_AFTER) {
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
      csb_printf(&decls, "  %s %s;\n", u->dtype.name, name);
      for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
      csb_printf(&body, "%s = (*%s);\n", name, bidx);
      continue;
    }

    /* --- STORE -------------------------------------------------------- */
    if (u->op == POLY_OP_STORE) {
      char *target = csmap_get(&names, u->src[0]);
      char *val    = csmap_get(&names, u->src[1]);
      for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
      if (u->src[0]->op == POLY_OP_DEFINE_LOCAL)
        csb_printf(&body, "%s = %s;\n", target, val);
      else
        csb_printf(&body, "*%s = %s;\n", target, val);
      continue;
    }

    /* --- CAST --------------------------------------------------------- */
    if (u->op == POLY_OP_CAST) {
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      csmap_set(&names, u, strdup(name));

      char *src_s = csmap_get(&names, u->src[0]);
      csb_printf(&decls, "  %s %s;\n", u->dtype.name, name);
      for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
      csb_printf(&body, "%s = (%s)(%s);\n", name, u->dtype.name, src_s);
      continue;
    }

    /* --- BITCAST: use template helper --------------------------------- */
    if (u->op == POLY_OP_BITCAST) {
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      csmap_set(&names, u, strdup(name));

      char *src_s = csmap_get(&names, u->src[0]);
      const char *dst_type = u->dtype.name;
      const char *src_type = u->src[0]->dtype.name;
      csb_printf(&decls, "  %s %s;\n", dst_type, name);
      for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
      csb_printf(&body, "%s = tg_bitcast<%s>((%s)(%s));\n",
                 name, dst_type, src_type, src_s);
      continue;
    }

    /* --- ALU ---------------------------------------------------------- */
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      char expr[512];
      const char *s0 = (u->n_src > 0) ? csmap_get(&names, u->src[0]) : "";
      const char *s1 = (u->n_src > 1) ? csmap_get(&names, u->src[1]) : "";
      const char *s2 = (u->n_src > 2) ? csmap_get(&names, u->src[2]) : "";
      cuda_render_alu(expr, sizeof(expr), u->op, u->dtype, s0, s1, s2);

      char name[32];
      snprintf(name, sizeof(name), "alu%d", c_alu++);
      csmap_set(&names, u, strdup(name));

      csb_printf(&decls, "  %s %s;\n", u->dtype.name, name);
      for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
      csb_printf(&body, "%s = %s;\n", name, expr);
      continue;
    }

    /* --- IF ----------------------------------------------------------- */
    if (u->op == POLY_OP_IF) {
      char *cond_s = csmap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++) csb_puts(&body, "  ");
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
      param_order[j+1] = param_order[j];
      param_types[j+1] = param_types[j];
      param_names[j+1] = param_names[j];
      j--;
    }
    param_order[j+1] = ko;
    param_types[j+1] = kt;
    param_names[j+1] = kn;
  }

  /* Build complete CUDA source */
  CudaStrBuf out;
  csb_init(&out);

  /* Prefix: CUDA-specific defines and helpers */
  csb_puts(&out, "#define INFINITY (__int_as_float(0x7f800000))\n");
  csb_puts(&out, "#define NAN (__int_as_float(0x7fffffff))\n");
  csb_puts(&out, "template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) {\n");
  csb_puts(&out, "  union U { F f; T t; }; U u; u.f = v; return u.t;\n");
  csb_puts(&out, "}\n\n");

  /* Kernel signature */
  csb_printf(&out, "extern \"C\" __global__ void __launch_bounds__(%d) %s(",
             launch_bounds, fn_name);
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
