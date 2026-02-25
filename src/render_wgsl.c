/*
 * render_wgsl.c — WGSL compute shader renderer
 *
 * Walks linearized UOps (same input as render_c.c) and emits a WGSL
 * compute shader string. Buffers become storage bindings, loops use
 * WGSL syntax, WHERE maps to select().
 *
 * Reference: tinygrad renderer/wgsl.py
 */

#define _POSIX_C_SOURCE 200809L

#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

/* ── String builder (same as render_c.c) ─────────────────────────────── */

typedef struct {
  char *buf;
  int len;
  int cap;
} WgslStrBuf;

static void wsb_init(WgslStrBuf *sb) {
  sb->cap = 512;
  sb->buf = malloc(sb->cap);
  sb->buf[0] = '\0';
  sb->len = 0;
}

static void wsb_printf(WgslStrBuf *sb, const char *fmt, ...) {
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

static void wsb_puts(WgslStrBuf *sb, const char *s) {
  wsb_printf(sb, "%s", s);
}

/* ── Pointer → string hash map (same as render_c.c) ──────────────────── */

typedef struct {
  PolyUOp **keys;
  char **vals;
  int cap;
} WgslStrMap;

static uint32_t wgsl_ptr_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  v = ((v >> 16) ^ v) * 0x45d9f3bU;
  v = ((v >> 16) ^ v) * 0x45d9f3bU;
  return (uint32_t)((v >> 16) ^ v);
}

static void wsm_init(WgslStrMap *m, int n) {
  m->cap = (n < 4) ? 16 : n * 3;
  m->keys = calloc(m->cap, sizeof(PolyUOp *));
  m->vals = calloc(m->cap, sizeof(char *));
}

static void wsm_set(WgslStrMap *m, PolyUOp *key, char *val) {
  uint32_t h = wgsl_ptr_hash(key) % m->cap;
  while (m->keys[h] && m->keys[h] != key) h = (h + 1) % m->cap;
  if (m->keys[h] == key) free(m->vals[h]);
  m->keys[h] = key;
  m->vals[h] = val;
}

static char *wsm_get(WgslStrMap *m, PolyUOp *key) {
  uint32_t h = wgsl_ptr_hash(key) % m->cap;
  while (m->keys[h]) {
    if (m->keys[h] == key) return m->vals[h];
    h = (h + 1) % m->cap;
  }
  return NULL;
}

static void wsm_destroy(WgslStrMap *m) {
  for (int i = 0; i < m->cap; i++)
    if (m->vals[i]) free(m->vals[i]);
  free(m->keys);
  free(m->vals);
}

/* ── WGSL type name ──────────────────────────────────────────────────── */

static const char *wgsl_type_name(PolyDType dt) {
  if (poly_dtype_is_float(dt)) return "f32";
  if (poly_dtype_is_bool(dt)) return "bool";
  return "i32";
}

/* ── WGSL float constant ────────────────────────────────────────────── */

static char *render_float_const_wgsl(double v, char *buf, int cap) {
  if (isinf(v)) {
    snprintf(buf, cap, v > 0
      ? "bitcast<f32>(0x7F800000u)"
      : "bitcast<f32>(0xFF800000u)");
    return buf;
  }
  if (isnan(v)) {
    snprintf(buf, cap, "bitcast<f32>(0xFFFFFFFFu)");
    return buf;
  }
  /* Use enough digits to round-trip float32 constants through text. */
  snprintf(buf, cap, "%.9g", (double)(float)v);
  /* ensure decimal point (WGSL requires it for f32 literals) */
  if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E')) {
    int len = (int)strlen(buf);
    if (len + 2 < cap) { buf[len] = '.'; buf[len+1] = '0'; buf[len+2] = '\0'; }
  }
  /* no 'f' suffix in WGSL */
  return buf;
}

/* ── WGSL ALU expression ────────────────────────────────────────────── */

static void render_alu_wgsl(char *buf, int cap, PolyOps op, PolyDType dtype,
                            const char *s0, const char *s1, const char *s2) {
  switch (op) {
  /* unary */
  case POLY_OP_NEG:
    snprintf(buf, cap, poly_dtype_is_bool(dtype) ? "(!%s)" : "(-%s)", s0); break;
  case POLY_OP_SQRT:       snprintf(buf, cap, "sqrt(%s)", s0); break;
  case POLY_OP_TRUNC:      snprintf(buf, cap, "trunc(%s)", s0); break;
  case POLY_OP_EXP2:       snprintf(buf, cap, "exp2(%s)", s0); break;
  case POLY_OP_LOG2:       snprintf(buf, cap, "log2(%s)", s0); break;
  case POLY_OP_SIN:        snprintf(buf, cap, "sin(%s)", s0); break;
  case POLY_OP_RECIPROCAL: snprintf(buf, cap, "(1.0/%s)", s0); break;
  /* binary */
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
  case POLY_OP_MAX:   snprintf(buf, cap, "max(%s,%s)", s0, s1); break;
  case POLY_OP_POW:   snprintf(buf, cap, "pow(%s,%s)", s0, s1); break;
  /* ternary */
  case POLY_OP_WHERE:  snprintf(buf, cap, "select(%s,%s,%s)", s2, s1, s0); break;
  case POLY_OP_MULACC: snprintf(buf, cap, "((%s*%s)+%s)", s0, s1, s2); break;
  default: snprintf(buf, cap, "/* unknown op %d */0", op); break;
  }
}

/* ── WGSL Renderer ───────────────────────────────────────────────────── */

char *poly_render_wgsl(PolyUOp **uops, int n, const char *fn_name) {
  WgslStrBuf body;
  wsb_init(&body);

  WgslStrMap names;
  wsm_init(&names, n);

  /* Collect buffer binding info: (name, binding_index, dtype) */
  char *binding_names[64];
  int binding_indices[64];
  PolyDType binding_dtypes[64];
  int n_bindings = 0;

  /* prefix counters */
  int c_val = 0, c_alu = 0, c_cast = 0, c_acc = 0;
  int depth = 1;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];

    /* --- SINK: skip ------------------------------------------------- */
    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP)
      continue;

    /* --- PARAM: storage buffer binding ------------------------------ */
    if (u->op == POLY_OP_PARAM) {
      char name[32];
      snprintf(name, sizeof(name), "data%lld", (long long)u->arg.i);
      wsm_set(&names, u, strdup(name));

      binding_names[n_bindings] = strdup(name);
      binding_indices[n_bindings] = (int)u->arg.i;
      binding_dtypes[n_bindings] = poly_dtype_scalar(u->dtype);
      n_bindings++;
      continue;
    }

    /* --- DEFINE_VAR: integer parameter (as var) --------------------- */
    if (u->op == POLY_OP_DEFINE_VAR) {
      const char *vname = u->arg.kind == POLY_ARG_DEFINE_VAR ? u->arg.define_var.name
                        : (u->arg.str ? u->arg.str : "var");
      wsm_set(&names, u, strdup(vname));
      continue;
    }

    /* --- CONST: inline literal -------------------------------------- */
    if (u->op == POLY_OP_CONST) {
      char val[128];
      if (poly_dtype_is_float(u->dtype)) {
        render_float_const_wgsl(u->arg.f, val, sizeof(val));
      } else if (poly_dtype_is_bool(u->dtype)) {
        snprintf(val, sizeof(val), "%s", u->arg.b ? "true" : "false");
      } else {
        snprintf(val, sizeof(val), "%lld", (long long)u->arg.i);
      }
      wsm_set(&names, u, strdup(val));
      continue;
    }

    /* --- INDEX: array indexing (buf[idx]) --------------------------- */
    if (u->op == POLY_OP_INDEX) {
      char *buf_s = wsm_get(&names, u->src[0]);
      char *idx_s = wsm_get(&names, u->src[1]);
      char expr[256];
      snprintf(expr, sizeof(expr), "%s[%s]", buf_s, idx_s);
      wsm_set(&names, u, strdup(expr));
      continue;
    }

    /* --- RANGE: for loop -------------------------------------------- */
    if (u->op == POLY_OP_RANGE) {
      char name[32];
      snprintf(name, sizeof(name), "ridx%lld", (long long)poly_range_axis_id(u->arg));
      wsm_set(&names, u, strdup(name));

      char *bound = wsm_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++) wsb_puts(&body, "  ");
      wsb_printf(&body, "for (var %s: i32 = 0; %s < %s; %s++) {\n",
                 name, name, bound, name);
      depth++;
      continue;
    }

    /* --- END / ENDIF: close brace ----------------------------------- */
    if (u->op == POLY_OP_END || u->op == POLY_OP_ENDIF) {
      depth--;
      for (int d = 0; d < depth; d++) wsb_puts(&body, "  ");
      wsb_puts(&body, "}\n");
      continue;
    }

    /* --- DEFINE_LOCAL: accumulator variable -------------------------- */
    if (u->op == POLY_OP_DEFINE_LOCAL) {
      char name[32];
      snprintf(name, sizeof(name), "acc%d", c_acc++);
      wsm_set(&names, u, strdup(name));

      char initval[64];
      if (u->arg.kind == POLY_ARG_FLOAT)
        render_float_const_wgsl(u->arg.f, initval, sizeof(initval));
      else
        snprintf(initval, sizeof(initval), "0.0");

      const char *tn = wgsl_type_name(u->dtype);
      for (int d = 0; d < depth; d++) wsb_puts(&body, "  ");
      wsb_printf(&body, "var %s: %s = %s;\n", name, tn, initval);
      continue;
    }

    /* --- LOAD: read from array -------------------------------------- */
    if (u->op == POLY_OP_LOAD) {
      char name[32];
      snprintf(name, sizeof(name), "val%d", c_val++);
      wsm_set(&names, u, strdup(name));

      char *bidx = wsm_get(&names, u->src[0]);
      const char *tn = wgsl_type_name(u->dtype);
      for (int d = 0; d < depth; d++) wsb_puts(&body, "  ");
      wsb_printf(&body, "var %s: %s = %s;\n", name, tn, bidx);
      continue;
    }

    /* --- STORE: write to array or accumulator ----------------------- */
    if (u->op == POLY_OP_STORE) {
      char *target = wsm_get(&names, u->src[0]);
      char *val    = wsm_get(&names, u->src[1]);
      for (int d = 0; d < depth; d++) wsb_puts(&body, "  ");
      if (u->src[0]->op == POLY_OP_DEFINE_LOCAL)
        wsb_printf(&body, "%s = %s;\n", target, val);
      else
        wsb_printf(&body, "%s = %s;\n", target, val);
      continue;
    }

    /* --- CAST: type conversion -------------------------------------- */
    if (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) {
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      wsm_set(&names, u, strdup(name));

      char *src_s = wsm_get(&names, u->src[0]);
      const char *tn = wgsl_type_name(u->dtype);
      for (int d = 0; d < depth; d++) wsb_puts(&body, "  ");
      if (u->op == POLY_OP_BITCAST)
        wsb_printf(&body, "var %s: %s = bitcast<%s>(%s);\n", name, tn, tn, src_s);
      else
        wsb_printf(&body, "var %s: %s = %s(%s);\n", name, tn, tn, src_s);
      continue;
    }

    /* --- ALU ops: arithmetic expressions ---------------------------- */
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      char expr[512];
      const char *s0 = (u->n_src > 0) ? wsm_get(&names, u->src[0]) : "";
      const char *s1 = (u->n_src > 1) ? wsm_get(&names, u->src[1]) : "";
      const char *s2 = (u->n_src > 2) ? wsm_get(&names, u->src[2]) : "";
      render_alu_wgsl(expr, sizeof(expr), u->op, u->dtype, s0, s1, s2);

      char name[32];
      snprintf(name, sizeof(name), "alu%d", c_alu++);
      wsm_set(&names, u, strdup(name));

      const char *tn = wgsl_type_name(u->dtype);
      for (int d = 0; d < depth; d++) wsb_puts(&body, "  ");
      wsb_printf(&body, "var %s: %s = %s;\n", name, tn, expr);
      continue;
    }

    /* --- IF: conditional -------------------------------------------- */
    if (u->op == POLY_OP_IF) {
      char *cond_s = wsm_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++) wsb_puts(&body, "  ");
      wsb_printf(&body, "if (%s) {\n", cond_s);
      depth++;
      continue;
    }
  }

  /* ── Sort bindings by index ────────────────────────────────────────── */
  for (int i = 1; i < n_bindings; i++) {
    int ki = binding_indices[i];
    char *kn = binding_names[i];
    PolyDType kd = binding_dtypes[i];
    int j = i - 1;
    while (j >= 0 && binding_indices[j] > ki) {
      binding_indices[j+1] = binding_indices[j];
      binding_names[j+1] = binding_names[j];
      binding_dtypes[j+1] = binding_dtypes[j];
      j--;
    }
    binding_indices[j+1] = ki;
    binding_names[j+1] = kn;
    binding_dtypes[j+1] = kd;
  }

  /* ── Build complete source ─────────────────────────────────────────── */
  WgslStrBuf out;
  wsb_init(&out);

  /* buffer bindings */
  for (int i = 0; i < n_bindings; i++) {
    const char *tn = wgsl_type_name(binding_dtypes[i]);
    wsb_printf(&out, "@group(0) @binding(%d)\nvar<storage,read_write> %s: array<%s>;\n",
               binding_indices[i], binding_names[i], tn);
  }

  /* compute shader entry point */
  wsb_printf(&out, "@compute @workgroup_size(1)\nfn %s("
             "@builtin(global_invocation_id) gid: vec3<u32>) {\n",
             fn_name);
  wsb_puts(&out, body.buf);
  wsb_puts(&out, "}\n");

  /* cleanup */
  for (int i = 0; i < n_bindings; i++)
    free(binding_names[i]);
  free(body.buf);
  wsm_destroy(&names);

  return out.buf;
}
