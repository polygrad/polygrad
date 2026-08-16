/*
 * render_wgsl.c — WGSL compute shader renderer
 *
 * Walks linearized UOps (same input as render_c.c) and emits a WGSL
 * compute shader string. Buffers become storage bindings, loops use
 * WGSL syntax, WHERE maps to select().
 *
 * GPU ops: SPECIAL (gidx/lidx → workgroup_id/local_invocation_id),
 * BARRIER (workgroupBarrier), DEFINE_LOCAL (var<workgroup> shared arrays),
 * DEFINE_REG (register-local arrays), AFTER (passthrough).
 *
 * Parity target: tinygrad renderer/wgsl.py (WGSLRenderer)
 *
 * Key differences from C/CUDA renderer:
 *   - Binding 0 is reserved for INFINITY uniform
 *   - Buffer bindings start at @binding(1)
 *   - WHERE → select(false_val, true_val, cond)  (reversed args)
 *   - Workgroup shared memory: var<workgroup> (externalized before @compute)
 *   - Workgroup builtins: gindex (workgroup_id), lindex (local_invocation_id)
 *   - No float4/vector support (supports_float4=false)
 */

#define _POSIX_C_SOURCE 200809L

#include "codegen.h"
#include "bigint.h"
#include "engine/schedule.h" /* POLY_DEVICE_WEBGPU */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include "utils.h"

static int wgsl_launch_dim_upper_bound(PolyUOp *expr) {
  if (!expr) return 1;
  int64_t lo = 0, hi = 1;
  poly_uop_minmax(NULL, expr, &lo, &hi);
  if (hi <= 0) return 1;
  if (hi > INT32_MAX) return INT32_MAX;
  return (int)hi;
}

/* String builder (same as render_c.c) */

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

/* Pointer → string hash map (same as render_c.c) */

typedef struct {
  PolyUOp **keys;
  char **vals;
  int cap;
} WgslStrMap;

static void wsm_init(WgslStrMap *m, int n) {
  m->cap = (n < 4) ? 16 : n * 3;
  m->keys = calloc(m->cap, sizeof(PolyUOp *));
  m->vals = calloc(m->cap, sizeof(char *));
}

static void wsm_set(WgslStrMap *m, PolyUOp *key, char *val) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
  while (m->keys[h] && m->keys[h] != key)
    h = (h + 1) % m->cap;
  if (m->keys[h] == key) free(m->vals[h]);
  m->keys[h] = key;
  m->vals[h] = val;
}

static char *wsm_get(WgslStrMap *m, PolyUOp *key) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
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

typedef struct {
  char *name;
  int index;
  PolyDType dtype;
  bool is_buffer;
} WgslBinding;

static bool wgsl_binding_append(
    WgslBinding **bindings,
    int *n_bindings,
    int *cap_bindings,
    const char *name,
    PolyDType dtype,
    bool is_buffer
) {
  if (!bindings || !n_bindings || !cap_bindings || !name) return false;
  if (*n_bindings >= *cap_bindings) {
    int new_cap = (*cap_bindings > 0) ? (*cap_bindings * 2) : 16;
    WgslBinding *new_bindings = realloc(*bindings, (size_t)new_cap * sizeof(WgslBinding));
    if (!new_bindings) return false;
    *bindings = new_bindings;
    *cap_bindings = new_cap;
  }
  char *dup = strdup(name);
  if (!dup) return false;
  (*bindings)[*n_bindings] = (WgslBinding){
      .name = dup,
      .index = *n_bindings,
      .dtype = dtype,
      .is_buffer = is_buffer,
  };
  (*n_bindings)++;
  return true;
}

static void wgsl_bindings_free(WgslBinding *bindings, int n_bindings) {
  for (int i = 0; i < n_bindings; i++)
    free(bindings[i].name);
  free(bindings);
}

/* WGSL type name */

static const char *wgsl_type_name(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  /* f16 for half (tinygrad: dtypes.half -> "f16") */
  if (s.priority == POLY_FLOAT16.priority && s.bitsize == POLY_FLOAT16.bitsize) return "f16";
  if (poly_dtype_is_float(dt)) return "f32";
  if (poly_dtype_is_bool(dt)) return "bool";
  /* Sub-4-byte types: char/short map to i32/u32 (tinygrad type_map) */
  if (poly_dtype_is_unsigned(dt)) return "u32";
  return "i32";
}

static bool wgsl_dtype_is_packed_storage(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  int itemsize = poly_dtype_itemsize(s);
  /* tinygrad WGSL packs sub-32-bit non-half storage through atomic<u32>.
   * WGSL scalar bool is valid, but bool storage buffers are not host-shareable,
   * so bool buffers must use the same byte-lane packing path as uint8. */
  return itemsize > 0 && itemsize < 4 && (poly_dtype_is_int(s) || poly_dtype_is_bool(s));
}

static const char *wgsl_buffer_type_name(PolyDType dt) {
  return wgsl_dtype_is_packed_storage(dt) ? "atomic<u32>" : wgsl_type_name(dt);
}

static bool wgsl_packed_params(PolyDType dt, int *itemsize, int *elems, unsigned *mask) {
  if (!wgsl_dtype_is_packed_storage(dt)) return false;
  int is = poly_dtype_itemsize(poly_dtype_scalar(dt));
  if (is != 1 && is != 2) return false;
  if (itemsize) *itemsize = is;
  if (elems) *elems = 4 / is;
  if (mask) *mask = (is == 1) ? 0xFFu : 0xFFFFu;
  return true;
}

static bool wgsl_index_uses_packed_buffer(PolyUOp *idx_uop, PolyDType *buf_dt_out) {
  if (!idx_uop || idx_uop->op != POLY_OP_INDEX || idx_uop->n_src < 2) return false;
  PolyDType buf_dt = poly_dtype_scalar(idx_uop->src[0]->dtype);
  if (!wgsl_dtype_is_packed_storage(buf_dt)) return false;
  if (buf_dt_out) *buf_dt_out = buf_dt;
  return true;
}

static bool wgsl_make_packed_load_expr(
    WgslStrMap *names,
    PolyUOp *idx_uop,
    PolyDType load_dt,
    char *out,
    size_t out_sz
) {
  PolyDType buf_dt;
  if (!wgsl_index_uses_packed_buffer(idx_uop, &buf_dt)) return false;

  int itemsize = 0, elems = 0;
  unsigned mask = 0;
  if (!wgsl_packed_params(buf_dt, &itemsize, &elems, &mask)) return false;

  char *buf_s = wsm_get(names, idx_uop->src[0]);
  char *idx_s = wsm_get(names, idx_uop->src[1]);
  if (!buf_s || !idx_s) return false;

  char raw[512];
  snprintf(
      raw, sizeof(raw), "((atomicLoad(&%s[(%s/%d)]) >> ((u32(%s)%%%du)*%du)) & 0x%Xu)", buf_s,
      idx_s, elems, idx_s, elems, 8u * (unsigned)itemsize, mask
  );

  PolyDType scalar = poly_dtype_scalar(load_dt);
  if (poly_dtype_is_bool(scalar)) {
    snprintf(out, out_sz, "(%s != 0u)", raw);
  } else if (!poly_dtype_is_unsigned(scalar) && poly_dtype_is_int(scalar)) {
    int sext = (itemsize == 1) ? 24 : 16;
    snprintf(out, out_sz, "((i32(%s)<<%d)>>%d)", raw, sext, sext);
  } else {
    snprintf(out, out_sz, "%s", raw);
  }
  return true;
}

static bool wgsl_emit_packed_store(
    WgslStrBuf *body,
    WgslStrMap *names,
    PolyUOp *idx_uop,
    const char *val,
    int depth
) {
  PolyDType buf_dt;
  if (!wgsl_index_uses_packed_buffer(idx_uop, &buf_dt)) return false;

  int itemsize = 0, elems = 0;
  unsigned mask = 0;
  if (!wgsl_packed_params(buf_dt, &itemsize, &elems, &mask)) return false;

  char *buf_s = wsm_get(names, idx_uop->src[0]);
  char *idx_s = wsm_get(names, idx_uop->src[1]);
  if (!buf_s || !idx_s || !val) return false;

  char val_u32[512];
  if (poly_dtype_is_bool(poly_dtype_scalar(buf_dt)))
    snprintf(val_u32, sizeof(val_u32), "select(0u, 1u, %s)", val);
  else
    snprintf(val_u32, sizeof(val_u32), "u32(%s)", val);

  unsigned shift_bits = 8u * (unsigned)itemsize;
  for (int d = 0; d < depth; d++)
    wsb_puts(body, "  ");
  wsb_printf(
      body, "atomicAnd(&%s[(%s/%d)], ((0x%Xu << ((u32(%s)%%%du)*%uu)) ^ 0xFFFFFFFFu));\n", buf_s,
      idx_s, elems, mask, idx_s, elems, shift_bits
  );

  for (int d = 0; d < depth; d++)
    wsb_puts(body, "  ");
  wsb_printf(
      body, "atomicAdd(&%s[(%s/%d)], ((%s & 0x%Xu) << ((u32(%s)%%%du)*%uu)));\n", buf_s, idx_s,
      elems, val_u32, mask, idx_s, elems, shift_bits
  );
  return true;
}

static bool wgsl_uop_tree_contains_target(PolyUOp *u, PolyUOp *target) {
  if (!u || !target) return false;
  if (u == target) return true;
  for (int i = 0; i < u->n_src; i++)
    if (wgsl_uop_tree_contains_target(u->src[i], target)) return true;
  return false;
}

static PolyUOp *wgsl_pick_lane_uop(PolyUOp *u, int lane) {
  if (!u || u->dtype.count <= 1) return u;
  if (lane < 0) lane = 0;
  if (u->op == POLY_OP_VECTORIZE || u->op == POLY_OP_VCONST) {
    if (u->n_src <= 0) return NULL;
    if (lane >= u->n_src) lane = u->n_src - 1;
    return u->src[lane];
  }
  return NULL;
}

static int wgsl_infer_gated_load_lane(PolyUOp *idx_expr, PolyUOp *vec_gate) {
  if (!idx_expr || !vec_gate || vec_gate->dtype.count <= 1) return -1;
  int found = -1;
  for (int lane = 0; lane < vec_gate->dtype.count; lane++) {
    PolyUOp *gate_lane = wgsl_pick_lane_uop(vec_gate, lane);
    if (!gate_lane) continue;
    if (!wgsl_uop_tree_contains_target(idx_expr, gate_lane)) continue;
    if (found != -1 && found != lane) return -1;
    found = lane;
  }
  return found;
}

static int wgsl_next_vector_lane(
    PolyUOp *key,
    int lanes,
    PolyUOp **lane_keys,
    int *lane_next,
    int *n_lane_keys
) {
  if (!key || lanes <= 0 || !lane_keys || !lane_next || !n_lane_keys) return -1;
  for (int i = 0; i < *n_lane_keys; i++) {
    if (lane_keys[i] != key) continue;
    int lane = lane_next[i];
    lane_next[i] = (lane_next[i] + 1) % lanes;
    return lane;
  }
  if (*n_lane_keys >= 128) return -1;
  lane_keys[*n_lane_keys] = key;
  lane_next[*n_lane_keys] = 1 % lanes;
  (*n_lane_keys)++;
  return 0;
}

static int wgsl_pos_mod_i64(int64_t x, int mod) {
  if (mod <= 0) return 0;
  int64_t r = x % mod;
  if (r < 0) r += mod;
  return (int)r;
}

static bool wgsl_const_i64(PolyUOp *u, int64_t *out) {
  if (!u) return false;
  if ((u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) && u->n_src > 0)
    return wgsl_const_i64(u->src[0], out);
  if (u->op != POLY_OP_CONST || poly_dtype_is_float(poly_dtype_scalar(u->dtype))) return false;
  if (out) *out = u->arg.i;
  return true;
}

static bool wgsl_expr_mod_const(PolyUOp *u, int mod, int *out) {
  if (!u || mod <= 0) return false;
  int64_t c0 = 0, c1 = 0;
  int r0 = 0, r1 = 0;
  switch (u->op) {
  case POLY_OP_CONST:
    if (!wgsl_const_i64(u, &c0)) return false;
    if (out) *out = wgsl_pos_mod_i64(c0, mod);
    return true;
  case POLY_OP_RANGE:
  case POLY_OP_SPECIAL:
  case POLY_OP_DEFINE_VAR:
  case POLY_OP_PARAM:
    if (out) *out = 0;
    return true;
  case POLY_OP_CAST:
  case POLY_OP_BITCAST:
    return u->n_src > 0 && wgsl_expr_mod_const(u->src[0], mod, out);
  case POLY_OP_ADD:
    if (u->n_src < 2 || !wgsl_expr_mod_const(u->src[0], mod, &r0) ||
        !wgsl_expr_mod_const(u->src[1], mod, &r1))
      return false;
    if (out) *out = (r0 + r1) % mod;
    return true;
  case POLY_OP_SUB:
    if (u->n_src < 2 || !wgsl_expr_mod_const(u->src[0], mod, &r0) ||
        !wgsl_expr_mod_const(u->src[1], mod, &r1))
      return false;
    if (out) *out = wgsl_pos_mod_i64((int64_t)r0 - r1, mod);
    return true;
  case POLY_OP_MUL:
    if (u->n_src < 2) return false;
    if (wgsl_const_i64(u->src[0], &c0) && (c0 % mod) == 0) {
      if (out) *out = 0;
      return true;
    }
    if (wgsl_const_i64(u->src[1], &c1) && (c1 % mod) == 0) {
      if (out) *out = 0;
      return true;
    }
    if (!wgsl_expr_mod_const(u->src[0], mod, &r0) ||
        !wgsl_expr_mod_const(u->src[1], mod, &r1))
      return false;
    if (out) *out = (int)(((int64_t)r0 * r1) % mod);
    return true;
  case POLY_OP_MULACC: {
    if (u->n_src < 3) return false;
    int racc = 0;
    if (!wgsl_expr_mod_const(u->src[2], mod, &racc)) return false;
    if ((wgsl_const_i64(u->src[0], &c0) && (c0 % mod) == 0) ||
        (wgsl_const_i64(u->src[1], &c1) && (c1 % mod) == 0)) {
      if (out) *out = racc;
      return true;
    }
    if (!wgsl_expr_mod_const(u->src[0], mod, &r0) ||
        !wgsl_expr_mod_const(u->src[1], mod, &r1))
      return false;
    if (out) *out = (int)(((int64_t)r0 * r1 + racc) % mod);
    return true;
  }
  case POLY_OP_SHL:
    if (u->n_src < 2 || !wgsl_const_i64(u->src[1], &c1) || c1 < 0 || c1 >= 62) return false;
    if ((((int64_t)1 << c1) % mod) == 0) {
      if (out) *out = 0;
      return true;
    }
    if (!wgsl_expr_mod_const(u->src[0], mod, &r0)) return false;
    if (out) *out = (int)(((int64_t)r0 * ((int64_t)1 << c1)) % mod);
    return true;
  default:
    return false;
  }
}

static int wgsl_value_lane_count(PolyUOp *u) {
  while (u && (u->op == POLY_OP_COPY || u->op == POLY_OP_UNROLL) && u->n_src > 0)
    u = u->src[0];
  if (!u) return 1;
  if ((u->op == POLY_OP_VECTORIZE || u->op == POLY_OP_VCONST) && u->n_src > 1) return u->n_src;
  if (u->dtype.count > 1) return u->dtype.count;
  return 1;
}

static int wgsl_store_target_lane(PolyUOp *target, int lanes) {
  if (lanes <= 1) return 0;
  PolyUOp *idx = poly_find_memory_slice_through_cast(target);
  if (!idx || idx->n_src < 2) return 0;
  int lane = 0;
  return wgsl_expr_mod_const(idx->src[1], lanes, &lane) ? lane : 0;
}

static bool wgsl_wraps_unroll(PolyUOp *u) {
  if (!u) return false;
  if (u->op == POLY_OP_UNROLL) return true;
  if (u->op == POLY_OP_COPY && u->n_src > 0) return wgsl_wraps_unroll(u->src[0]);
  return false;
}

static char *wgsl_render_lane_expr(WgslStrMap *names, PolyUOp *u, int lane) {
  if (!u) return strdup("0");
  if (u->op == POLY_OP_COPY && u->n_src > 0) return wgsl_render_lane_expr(names, u->src[0], lane);
  if (u->op == POLY_OP_UNROLL && u->n_src > 0) return wgsl_render_lane_expr(names, u->src[0], lane);
  if ((u->op == POLY_OP_VECTORIZE || u->op == POLY_OP_VCONST) && u->n_src > 0) {
    int pick = wgsl_pos_mod_i64(lane, u->n_src);
    char *s = wsm_get(names, u->src[pick]);
    return strdup(s ? s : "0");
  }
  char *s = wsm_get(names, u);
  if (!s) return strdup("0");
  if (u->dtype.count > 1) {
    char expr[256];
    snprintf(expr, sizeof(expr), "%s[%d]", s, wgsl_pos_mod_i64(lane, u->dtype.count));
    return strdup(expr);
  }
  return strdup(s);
}

/* WGSL float constant */

static char *render_float_const_wgsl(double v, char *buf, int cap) {
  /* Use enough digits to round-trip float32 constants through text. */
  snprintf(buf, cap, "%.9g", (double)(float)v);
  /* ensure decimal point (WGSL requires it for f32 literals) */
  if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E')) {
    int len = (int)strlen(buf);
    if (len + 2 < cap) {
      buf[len] = '.';
      buf[len + 1] = '0';
      buf[len + 2] = '\0';
    }
  }
  /* no 'f' suffix in WGSL */
  return buf;
}

/* WGSL ALU expression */

static void render_alu_wgsl(
    char *buf,
    int cap,
    PolyOps op,
    PolyDType dtype,
    const char *s0,
    const char *s1,
    const char *s2
) {
  switch (op) {
  /* unary */
  case POLY_OP_NEG:
    if (poly_dtype_is_bool(dtype))
      /* Raw NEG(bool) is typed bool identity after arithmetic negation and
       * bool conversion. Standard logical NOT is CMPNE(x,true). */
      snprintf(buf, cap, "(%s)", s0);
    else if (poly_dtype_is_unsigned(dtype))
      /* WGSL doesn't support unary minus on unsigned (tinygrad wgsl.py:69) */
      snprintf(buf, cap, "(0-%s)", s0);
    else
      snprintf(buf, cap, "(-%s)", s0);
    break;
  case POLY_OP_SQRT:
    snprintf(buf, cap, "sqrt(%s)", s0);
    break;
  case POLY_OP_TRUNC:
    snprintf(buf, cap, "trunc(%s)", s0);
    break;
  case POLY_OP_EXP2:
    snprintf(buf, cap, "exp2(%s)", s0);
    break;
  case POLY_OP_LOG2:
    snprintf(buf, cap, "log2(%s)", s0);
    break;
  case POLY_OP_SIN:
    snprintf(buf, cap, "sin(%s)", s0);
    break;
  case POLY_OP_RECIPROCAL:
    snprintf(buf, cap, "(1.0/%s)", s0);
    break;
  /* binary */
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
    snprintf(buf, cap, "max(%s,%s)", s0, s1);
    break;
  case POLY_OP_POW:
    snprintf(buf, cap, "pow(%s,%s)", s0, s1);
    break;
  /* ternary */
  case POLY_OP_WHERE:
    snprintf(buf, cap, "select(%s,%s,%s)", s2, s1, s0);
    break;
  case POLY_OP_MULACC:
    snprintf(buf, cap, "((%s*%s)+%s)", s0, s1, s2);
    break;
  default:
    snprintf(buf, cap, "/* unknown op %d */0", op);
    break;
  }
}

/* WGSL Renderer */

char *poly_render_wgsl(PolyUOp **uops, int n, const char *fn_name) {
  WgslStrBuf body;
  wsb_init(&body);

  /* Function-scope declarations buffer (WGSL has block scoping, so
   * variables used across loop boundaries must be declared at function scope).
   * Mirrors render_c.c's decls buffer pattern. */
  WgslStrBuf decls;
  wsb_init(&decls);

  WgslStrMap names;
  wsm_init(&names, n);

  /* Kernel parameter bindings: PARAM (storage) and DEFINE_VAR (uniform).
   * Match tinygrad exactly here:
   *   - cstyle.py collects PARAM/DEFINE_VAR into one ordered `bufs` list
   *   - wgsl.py assigns bindings sequentially from that list
   * So WGSL binding numbers follow encounter order, not PARAM.arg. */
  WgslBinding *bindings = NULL;
  int n_bindings = 0;
  int cap_bindings = 0;

  /* prefix counters */
  int c_val = 0, c_alu = 0, c_cast = 0, c_acc = 0;
  int depth = 1;

  /* Workgroup shared memory declarations (externalized before @compute).
   * Tinygrad: var<workgroup> lines are hoisted out of the kernel body
   * (wgsl.py render_kernel lines 105-106). */
  char extern_locals[16][256];
  int n_extern_locals = 0;

  /* Pre-scan: extract local dims from SPECIAL ops, detect f16 usage */
  int local_dims[3] = {1, 1, 1};
  bool has_local_dims = false;
  bool uses_f16 = false;
  PolyUOp *gated_lane_keys[128];
  int gated_lane_next[128];
  int n_gated_lane_keys = 0;

  for (int i = 0; i < n; i++) {
    /* Local dims: SPECIAL("lidxN") → local_dims[N] = bound */
    if (uops[i]->op == POLY_OP_SPECIAL && uops[i]->arg.str && uops[i]->arg.str[0] == 'l') {
      int slen = (int)strlen(uops[i]->arg.str);
      int dim_idx = (slen > 0) ? uops[i]->arg.str[slen - 1] - '0' : 0;
      if (dim_idx >= 0 && dim_idx < 3 && uops[i]->n_src > 0) {
        local_dims[dim_idx] = wgsl_launch_dim_upper_bound(uops[i]->src[0]);
        has_local_dims = true;
      }
    }
    /* f16: check all UOps for half dtype */
    PolyDType s = poly_dtype_scalar(uops[i]->dtype);
    if (s.priority == POLY_FLOAT16.priority && s.bitsize == POLY_FLOAT16.bitsize) uses_f16 = true;
  }

  /* Main render loop */

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];

    /* --- SINK/NOOP/GROUP: skip ---------------------------------------- */
    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP) continue;

    /* --- PARAM: storage buffer binding -------------------------------- */
    if (u->op == POLY_OP_PARAM) {
      char name[32];
      snprintf(name, sizeof(name), "data%lld", (long long)u->arg.i);
      wsm_set(&names, u, strdup(name));

      if (!wgsl_binding_append(
              &bindings, &n_bindings, &cap_bindings, name, poly_dtype_scalar(u->dtype), true
          ))
        goto fail;
      continue;
    }

    /* --- DEFINE_VAR: scalar uniform binding --------------------------- */
    if (u->op == POLY_OP_DEFINE_VAR) {
      const char *vname = u->arg.kind == POLY_ARG_DEFINE_VAR ? u->arg.define_var.name
                                                             : (u->arg.str ? u->arg.str : "var");
      wsm_set(&names, u, strdup(vname));

      if (!wgsl_binding_append(
              &bindings, &n_bindings, &cap_bindings, vname, poly_dtype_scalar(u->dtype), false
          ))
        goto fail;
      continue;
    }

    /* --- SPECIAL: GPU thread/workgroup index (tinygrad gpudims.py) ---- */
    if (u->op == POLY_OP_SPECIAL) {
      const char *sname = u->arg.str ? u->arg.str : "gidx0";
      wsm_set(&names, u, strdup(sname));

      /* Dimension: last char of name (gidx0→x, gidx1→y, gidx2→z) */
      int slen = (int)strlen(sname);
      int dim_idx = (slen > 0) ? sname[slen - 1] - '0' : 0;
      if (dim_idx < 0 || dim_idx > 2) dim_idx = 0;
      char dim_char = "xyz"[dim_idx];

      for (int d = 0; d < depth; d++)
        wsb_puts(&body, "  ");
      if (sname[0] == 'l') {
        /* Local index: lindex (local_invocation_id)
         * tinygrad: i32(lindex.x) */
        wsb_printf(&body, "var %s: i32 = i32(lindex.%c);\n", sname, dim_char);
      } else {
        /* Global index: gindex (workgroup_id)
         * tinygrad: i32(gindex.x) — note: workgroup_id, NOT global_invocation_id */
        wsb_printf(&body, "var %s: i32 = i32(gindex.%c);\n", sname, dim_char);
      }

      /* Bounds check for global indices only.
       * Local threads must all run (barrier requires it). */
      if (sname[0] != 'l') {
        char *bound = wsm_get(&names, u->src[0]);
        if (bound) {
          for (int d = 0; d < depth; d++)
            wsb_puts(&body, "  ");
          wsb_printf(&body, "if (%s >= %s) { return; }\n", sname, bound);
        }
      }
      continue;
    }

    /* --- BARRIER: workgroup synchronization (tinygrad: "workgroupBarrier();") */
    if (u->op == POLY_OP_BARRIER) {
      for (int d = 0; d < depth; d++)
        wsb_puts(&body, "  ");
      wsb_puts(&body, "workgroupBarrier();\n");
      continue;
    }

    /* --- CONST: inline literal ---------------------------------------- */
    if (u->op == POLY_OP_CONST) {
      char val[128];
      if (poly_dtype_is_float(u->dtype)) {
        PolyDType s = poly_dtype_scalar(u->dtype);
        if (isinf(u->arg.f)) {
          /* Use INFINITY uniform (tinygrad wgsl.py:109) */
          snprintf(val, sizeof(val), u->arg.f > 0 ? "INFINITY" : "(-INFINITY)");
        } else if (isnan(u->arg.f)) {
          /* Use nan() function (tinygrad wgsl.py:108) */
          snprintf(val, sizeof(val), "nan()");
        } else if (s.priority == POLY_FLOAT16.priority && s.bitsize == POLY_FLOAT16.bitsize) {
          /* f16 const: cast from f32 literal */
          char f32buf[64];
          render_float_const_wgsl(u->arg.f, f32buf, sizeof(f32buf));
          snprintf(val, sizeof(val), "f16(%s)", f32buf);
        } else {
          render_float_const_wgsl(u->arg.f, val, sizeof(val));
        }
      } else if (poly_dtype_is_bool(u->dtype)) {
        snprintf(val, sizeof(val), "%s", u->arg.b ? "true" : "false");
      } else if (poly_dtype_is_unsigned(u->dtype)) {
        /* Unsigned consts: negative → bitcast, positive → Nu suffix
         * (tinygrad wgsl.py:72) */
        bool negative = u->arg.kind == POLY_ARG_BIGINT ? u->arg.bigint.sign < 0
                                                       : u->arg.i < 0;
        if (negative) {
          char *decimal = poly_arg_integer_to_decimal(u->arg);
          if (!decimal) return NULL;
          snprintf(val, sizeof(val), "bitcast<u32>(%s)", decimal);
          free(decimal);
        } else {
          snprintf(
              val, sizeof(val), "%uu",
              (unsigned)(uint32_t)poly_arg_integer_to_u64_mod(u->arg)
          );
        }
      } else {
        snprintf(
            val, sizeof(val), "%d",
            (int32_t)poly_arg_integer_to_u64_mod(u->arg)
        );
      }
      wsm_set(&names, u, strdup(val));
      continue;
    }

    /* --- INDEX: array indexing (buf[idx]) ----------------------------- */
    if (u->op == POLY_OP_INDEX) {
      char *buf_s = wsm_get(&names, u->src[0]);
      char *idx_s = wsm_get(&names, u->src[1]);
      char expr[256];
      snprintf(expr, sizeof(expr), "%s[%s]", buf_s, idx_s);
      wsm_set(&names, u, strdup(expr));
      continue;
    }

    /* --- SHRINK: late codegen memory slice --------------------------- */
    if (u->op == POLY_OP_SHRINK) {
      char *buf_s = wsm_get(&names, u->src[0]);
      char *idx_s = wsm_get(&names, u->src[1]);
      char expr[256];
      snprintf(expr, sizeof(expr), "%s[%s]", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      wsm_set(&names, u, strdup(expr));
      continue;
    }

    /* --- RANGE: for loop --------------------------------------------- */
    if (u->op == POLY_OP_RANGE) {
      char name[32];
      snprintf(name, sizeof(name), "ridx%lld", (long long)poly_range_axis_id(u->arg));
      wsm_set(&names, u, strdup(name));

      char *bound = wsm_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++)
        wsb_puts(&body, "  ");
      wsb_printf(&body, "for (var %s: i32 = 0; %s < %s; %s++) {\n", name, name, bound, name);
      depth++;
      continue;
    }

    /* --- END / ENDIF: close brace ------------------------------------ */
    if (u->op == POLY_OP_END || u->op == POLY_OP_ENDIF) {
      depth--;
      for (int d = 0; d < depth; d++)
        wsb_puts(&body, "  ");
      wsb_puts(&body, "}\n");
      continue;
    }

    /* Pinned tinygrad/renderer/wgsl.py renders the LOCAL-address-space
     * BUFFER produced by pm_add_buffers_local as workgroup storage. Keep the
     * legacy DEFINE_LOCAL spelling until the vocabulary debt is retired. */
    if (u->op == POLY_OP_DEFINE_LOCAL ||
        (u->op == POLY_OP_BUFFER && poly_program_memory_is(u, POLY_ADDR_LOCAL))) {
      if (u->op == POLY_OP_BUFFER) {
        /* Workgroup shared memory array.
         * tinygrad: var<workgroup> smemN: array<type, SIZE>;
         * Externalized before @compute (wgsl.py render_kernel lines 105-106). */
        char name[32];
        snprintf(name, sizeof(name), "smem%lld", (long long)poly_program_buffer_slot(u));
        wsm_set(&names, u, strdup(name));

        int64_t smem_size = poly_program_buffer_size(u);
        const char *base_tn = wgsl_type_name(poly_dtype_scalar(poly_program_buffer_dtype(u)));
        if (n_extern_locals < 16) {
          snprintf(
              extern_locals[n_extern_locals], 256, "var<workgroup> %s: array<%s,%lld>;", name,
              base_tn, (long long)smem_size
          );
          n_extern_locals++;
        }
      } else {
        /* Scalar accumulator (non-pointer dtype).
         * Used by pm_reduce for reduction accumulators. */
        char name[32];
        snprintf(name, sizeof(name), "acc%d", c_acc++);
        wsm_set(&names, u, strdup(name));

        char initval[64];
        if (u->arg.kind == POLY_ARG_FLOAT)
          render_float_const_wgsl(u->arg.f, initval, sizeof(initval));
        else
          snprintf(initval, sizeof(initval), "0.0");

        const char *tn = wgsl_type_name(u->dtype);
        wsb_printf(&decls, "  var %s: %s;\n", name, tn);
        for (int d = 0; d < depth; d++)
          wsb_puts(&body, "  ");
        wsb_printf(&body, "%s = %s;\n", name, initval);
      }
      continue;
    }

    /* --- register-local array ---------------------------------------- */
    if (u->op == POLY_OP_DEFINE_REG ||
        (u->op == POLY_OP_BUFFER && poly_program_memory_is(u, POLY_ADDR_REG))) {
      char name[32];
      snprintf(name, sizeof(name), "r%lld", (long long)poly_program_buffer_slot(u));
      wsm_set(&names, u, strdup(name));

      /* tinygrad: var rN: array<type, SIZE>; (wgsl.py:75) */
      int64_t reg_size = poly_program_buffer_size(u);
      const char *base_tn = wgsl_type_name(poly_dtype_scalar(poly_program_buffer_dtype(u)));
      wsb_printf(&decls, "  var %s: array<%s,%lld>;\n", name, base_tn, (long long)reg_size);
      continue;
    }

    /* --- AFTER: passthrough (alias src[0] name) ---------------------- */
    if (u->op == POLY_OP_AFTER) {
      char *src_name = wsm_get(&names, u->src[0]);
      if (src_name) wsm_set(&names, u, strdup(src_name));
      continue;
    }

    /* --- COPY / UNROLL: transparent placement and expansion wrappers -- */
    if ((u->op == POLY_OP_COPY || u->op == POLY_OP_UNROLL) && u->n_src > 0) {
      char *src_name = wsm_get(&names, u->src[0]);
      if (src_name) wsm_set(&names, u, strdup(src_name));
      continue;
    }

    /* --- LOAD: read from array --------------------------------------- */
    if (u->op == POLY_OP_LOAD) {
      char name[32];
      snprintf(name, sizeof(name), "val%d", c_val++);
      wsm_set(&names, u, strdup(name));

      char *bidx = wsm_get(&names, u->src[0]);
      const char *tn = wgsl_type_name(u->dtype);
      PolyUOp *idx_uop = poly_find_index_through_cast(u->src[0]);
      char packed_load[512];
      const char *load_expr = bidx;
      if (idx_uop &&
          wgsl_make_packed_load_expr(&names, idx_uop, u->dtype, packed_load, sizeof(packed_load)))
        load_expr = packed_load;
      for (int d = 0; d < depth; d++)
        wsb_puts(&body, "  ");

      wsb_printf(&decls, "  var %s: %s;\n", name, tn);
      /* Gated load: select(alt, load, gate) -- tinygrad WGSL parity */
      PolyUOp *gate_uop =
          (u->n_src >= 3 && poly_dtype_is_bool(poly_dtype_scalar(u->src[2]->dtype)))
              ? u->src[2]
              : ((idx_uop && idx_uop->n_src >= 3 &&
                  poly_dtype_is_bool(poly_dtype_scalar(idx_uop->src[2]->dtype)))
                     ? idx_uop->src[2]
                     : NULL);
      if (gate_uop && u->n_src >= 2) {
        PolyUOp *alt_uop = u->src[1];
        if (u->dtype.count == 1 &&
            ((gate_uop && gate_uop->dtype.count > 1) || (alt_uop && alt_uop->dtype.count > 1))) {
          int lane = wgsl_infer_gated_load_lane(idx_uop->src[1], gate_uop);
          int lanes = 0;
          if (gate_uop && gate_uop->dtype.count > lanes) lanes = gate_uop->dtype.count;
          if (alt_uop && alt_uop->dtype.count > lanes) lanes = alt_uop->dtype.count;
          if (lane < 0 && lanes > 1) {
            PolyUOp *lane_key = (gate_uop && gate_uop->dtype.count > 1)
                                    ? gate_uop
                                    : ((alt_uop && alt_uop->dtype.count > 1) ? alt_uop : NULL);
            lane = wgsl_next_vector_lane(
                lane_key, lanes, gated_lane_keys, gated_lane_next, &n_gated_lane_keys
            );
          }
          if (lane >= 0) {
            PolyUOp *lane_gate = wgsl_pick_lane_uop(gate_uop, lane);
            PolyUOp *lane_alt = wgsl_pick_lane_uop(alt_uop, lane);
            if (lane_gate) gate_uop = lane_gate;
            if (lane_alt) alt_uop = lane_alt;
          }
        }
        char *gate_s = wsm_get(&names, gate_uop);
        char *alt_s = wsm_get(&names, alt_uop);
        wsb_printf(&body, "%s = select(%s, %s, %s);\n", name, alt_s, load_expr, gate_s);
      } else if (gate_uop) {
        if (u->dtype.count == 1 && gate_uop && gate_uop->dtype.count > 1) {
          int lane = wgsl_infer_gated_load_lane(idx_uop->src[1], gate_uop);
          if (lane < 0) {
            lane = wgsl_next_vector_lane(
                gate_uop, gate_uop->dtype.count, gated_lane_keys, gated_lane_next,
                &n_gated_lane_keys
            );
          }
          if (lane >= 0) {
            PolyUOp *lane_gate = wgsl_pick_lane_uop(gate_uop, lane);
            if (lane_gate) gate_uop = lane_gate;
          }
        }
        char *gate_s = wsm_get(&names, gate_uop);
        wsb_printf(&body, "%s = select(%s(0), %s, %s);\n", name, tn, load_expr, gate_s);
      } else {
        wsb_printf(&body, "%s = %s;\n", name, load_expr);
      }
      continue;
    }

    /* --- GEP: extract element from vector/array (devectorized access) -- */
    if (u->op == POLY_OP_GEP) {
      if (u->n_src < 1) {
        wsm_set(&names, u, strdup("0"));
        continue;
      }
      char *src_s = wsm_get(&names, u->src[0]);
      if (!src_s) src_s = "0";
      if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n == 1) {
        /* Single-lane GEP: src[i] */
        char expr[256];
        snprintf(expr, sizeof(expr), "%s[%lld]", src_s, (long long)u->arg.int_tuple.vals[0]);
        wsm_set(&names, u, strdup(expr));
      } else if (u->arg.kind == POLY_ARG_INT) {
        char expr[256];
        snprintf(expr, sizeof(expr), "%s[%lld]", src_s, (long long)u->arg.i);
        wsm_set(&names, u, strdup(expr));
      } else {
        /* Identity GEP or unsupported: alias source */
        wsm_set(&names, u, strdup(src_s));
      }
      continue;
    }

    /* --- VECTORIZE: vector constructor (scalar after devectorize) ----- */
    if (u->op == POLY_OP_VECTORIZE || u->op == POLY_OP_VCONST) {
      if (u->n_src == 1) {
        /* Single source: alias */
        char *s = wsm_get(&names, u->src[0]);
        wsm_set(&names, u, strdup(s ? s : "0"));
      } else if (u->n_src > 1) {
        /* Multi-source: vec constructor. Shouldn't appear for scalar WGSL
         * (supports_float4=false), but handle for robustness. */
        const char *tn = wgsl_type_name(poly_dtype_scalar(u->dtype));
        WgslStrBuf vexpr;
        wsb_init(&vexpr);
        wsb_printf(&vexpr, "vec%d<%s>(", u->n_src, tn);
        for (int j = 0; j < u->n_src; j++) {
          char *s = wsm_get(&names, u->src[j]);
          wsb_printf(&vexpr, "%s%s", j > 0 ? "," : "", s ? s : "0");
        }
        wsb_puts(&vexpr, ")");
        wsm_set(&names, u, vexpr.buf);
      } else {
        wsm_set(&names, u, strdup("0"));
      }
      continue;
    }

    /* --- STORE: write to array or accumulator ------------------------ */
    if (u->op == POLY_OP_STORE) {
      char *target = wsm_get(&names, u->src[0]);
      char *val = wsm_get(&names, u->src[1]);
      char *owned_val = NULL;
      if (wgsl_wraps_unroll(u->src[1])) {
        int lanes = wgsl_value_lane_count(u->src[1]);
        int lane = wgsl_store_target_lane(u->src[0], lanes);
        owned_val = wgsl_render_lane_expr(&names, u->src[1], lane);
        val = owned_val;
      }

      PolyUOp *store_idx = poly_find_index_through_cast(u->src[0]);
      if (!wgsl_emit_packed_store(&body, &names, store_idx, val, depth)) {
        for (int d = 0; d < depth; d++)
          wsb_puts(&body, "  ");
        wsb_printf(&body, "%s = %s;\n", target, val);
      }

      free(owned_val);
      continue;
    }

    /* --- CAST / BITCAST: type conversion ----------------------------- */
    if (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) {
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      wsm_set(&names, u, strdup(name));

      char *src_s = wsm_get(&names, u->src[0]);
      const char *tn = wgsl_type_name(u->dtype);
      PolyDType dst_s = poly_dtype_scalar(u->dtype);
      PolyDType src_dt = poly_dtype_scalar(u->src[0]->dtype);

      char expr[256];
      if (u->op == POLY_OP_BITCAST) {
        /* BITCAST specializations matching tinygrad wgsl.py:76-84 */
        if (dst_s.priority == POLY_FLOAT16.priority && dst_s.bitsize == POLY_FLOAT16.bitsize) {
          snprintf(expr, sizeof(expr), "bitcast<vec2<f16>>(%s)[0]", src_s);
        } else if (dst_s.priority == POLY_UINT8.priority && dst_s.bitsize == 8) {
          snprintf(expr, sizeof(expr), "bitcast<u32>(%s&0xFF)", src_s);
        } else if (dst_s.priority == POLY_INT8.priority && dst_s.bitsize == 8) {
          snprintf(expr, sizeof(expr), "((i32(%s&0xFF)<<24)>>24)", src_s);
        } else if (dst_s.priority == POLY_UINT16.priority && dst_s.bitsize == 16) {
          if (src_dt.priority == POLY_FLOAT16.priority && src_dt.bitsize == POLY_FLOAT16.bitsize)
            snprintf(expr, sizeof(expr), "bitcast<u32>(vec2<f16>(%s,0))", src_s);
          else
            snprintf(expr, sizeof(expr), "bitcast<u32>(%s&0xFFFF)", src_s);
        } else if (dst_s.priority == POLY_INT16.priority && dst_s.bitsize == 16) {
          if (src_dt.priority == POLY_FLOAT16.priority && src_dt.bitsize == POLY_FLOAT16.bitsize)
            snprintf(expr, sizeof(expr), "bitcast<i32>(vec2<f16>(%s,0))", src_s);
          else
            snprintf(expr, sizeof(expr), "((i32(%s&0xFFFF)<<16)>>16)", src_s);
        } else {
          snprintf(expr, sizeof(expr), "bitcast<%s>(%s)", tn, src_s);
        }
      } else {
        /* CAST: type conversion */
        snprintf(expr, sizeof(expr), "%s(%s)", tn, src_s);
      }

      wsb_printf(&decls, "  var %s: %s;\n", name, tn);
      for (int d = 0; d < depth; d++)
        wsb_puts(&body, "  ");
      wsb_printf(&body, "%s = %s;\n", name, expr);
      continue;
    }

    /* --- IF: conditional --------------------------------------------- */
    if (u->op == POLY_OP_IF) {
      char *cond_s = wsm_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++)
        wsb_puts(&body, "  ");
      wsb_printf(&body, "if (%s) {\n", cond_s);
      depth++;
      continue;
    }

    /* --- ALU ops: arithmetic expressions ----------------------------- */
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
      wsb_printf(&decls, "  var %s: %s;\n", name, tn);
      for (int d = 0; d < depth; d++)
        wsb_puts(&body, "  ");
      wsb_printf(&body, "%s = %s;\n", name, expr);
      continue;
    }
  }

  /* Sort bindings by index */
  for (int i = 1; i < n_bindings; i++) {
    WgslBinding key = bindings[i];
    int j = i - 1;
    while (j >= 0 && bindings[j].index > key.index) {
      bindings[j + 1] = bindings[j];
      j--;
    }
    bindings[j + 1] = key;
  }

  /* Build complete source */
  WgslStrBuf out;
  wsb_init(&out);

  /* Preamble: f16 extension, nan() function, INFINITY uniform.
   * Matches tinygrad WGSLRenderer.render_kernel lines 107-109. */
  if (uses_f16) wsb_puts(&out, "enable f16;\n");
  wsb_puts(&out, "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\n");
  wsb_puts(&out, "@group(0) @binding(0)\nvar<uniform> INFINITY : f32;\n");

  /* Externalized workgroup shared memory declarations.
   * These must appear before the @compute function (tinygrad wgsl.py:105-106). */
  for (int i = 0; i < n_extern_locals; i++) {
    wsb_printf(&out, "%s\n", extern_locals[i]);
  }

  /* Parameter bindings: sequential indices starting at 1 (binding 0 = INFINITY).
   * Storage buffers use var<storage,read_write>, scalar vars use var<uniform>. */
  for (int i = 0; i < n_bindings; i++) {
    const char *tn = bindings[i].is_buffer ? wgsl_buffer_type_name(bindings[i].dtype)
                                           : wgsl_type_name(bindings[i].dtype);
    if (bindings[i].is_buffer) {
      wsb_printf(
          &out, "@group(0) @binding(%d)\nvar<storage,read_write> %s: array<%s>;\n",
          bindings[i].index + 1, bindings[i].name, tn
      );
    } else {
      wsb_printf(
          &out, "@group(0) @binding(%d)\nvar<uniform> %s: %s;\n", bindings[i].index + 1,
          bindings[i].name, tn
      );
    }
  }

  /* Compute shader entry point.
   * tinygrad: @builtin(workgroup_id) gindex, @builtin(local_invocation_id) lindex
   * workgroup_size from collected local dims. */
  if (has_local_dims)
    wsb_printf(
        &out, "@compute @workgroup_size(%d,%d,%d)\nfn %s(", local_dims[0], local_dims[1],
        local_dims[2], fn_name
    );
  else
    wsb_printf(&out, "@compute @workgroup_size(1)\nfn %s(", fn_name);

  wsb_puts(&out, "@builtin(workgroup_id) gindex: vec3<u32>,");
  wsb_puts(&out, "@builtin(local_invocation_id) lindex: vec3<u32>) {\n");
  /* Function-scope declarations first, then body (WGSL block scoping). */
  wsb_puts(&out, decls.buf);
  wsb_puts(&out, body.buf);
  wsb_puts(&out, "}\n");

  /* cleanup */
  wgsl_bindings_free(bindings, n_bindings);
  free(decls.buf);
  free(body.buf);
  wsm_destroy(&names);

  return out.buf;

fail:
  wgsl_bindings_free(bindings, n_bindings);
  free(decls.buf);
  free(body.buf);
  wsm_destroy(&names);
  return NULL;
}

/* WGSL extra matcher (tinygrad wgsl_matcher, wgsl.py:40-53) */

/* Rule: SHL/SHR with non-u32 shift amount → cast shift to u32.
 * WGSL spec requires shift amounts to be u32. tinygrad wgsl.py:49-50. */
static PolyUOp *rule_wgsl_shift_u32(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (u->n_src < 2) return NULL;
  PolyUOp *amount = u->src[1];
  /* Already u32? No rewrite. */
  if (poly_dtype_eq(poly_dtype_scalar(amount->dtype), POLY_UINT32)) return NULL;
  /* Cast shift amount to u32 */
  PolyUOp *cast_amount = poly_uop1(ctx, POLY_OP_CAST, POLY_UINT32, amount, poly_arg_none());
  return poly_uop2(ctx, u->op, u->dtype, u->src[0], cast_amount, u->arg);
}

/* Rule: unsupported/mixed bool ALU → cast to i32, apply op, cast back to bool.
 * tinygrad wgsl.py:41-42 handles bool CMPLT/XOR this way. Polygrad can also
 * produce mixed bool/int equality after packed bool storage loads; normalize
 * those before WGSL so expressions like `0u != bool_val` never reach render. */
static PolyUOp *rule_wgsl_bool_alu(PolyCtx *ctx, PolyUOp *u, const PolyBindings *bindings) {
  (void)bindings;
  if (u->n_src < 2) return NULL;
  bool src0_bool = poly_dtype_is_bool(poly_dtype_scalar(u->src[0]->dtype));
  bool src1_bool = poly_dtype_is_bool(poly_dtype_scalar(u->src[1]->dtype));
  if (!src0_bool && !src1_bool) return NULL;
  if ((u->op == POLY_OP_CMPEQ || u->op == POLY_OP_CMPNE) && src0_bool && src1_bool)
    return NULL;
  PolyUOp *a = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, u->src[0], poly_arg_none());
  PolyUOp *b = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, u->src[1], poly_arg_none());
  PolyDType result_dt =
      (u->op == POLY_OP_XOR) ? POLY_INT32 : ((u->dtype.count > 1) ? poly_dtype_vec(POLY_BOOL, u->dtype.count) : POLY_BOOL);
  PolyUOp *result = poly_uop2(ctx, u->op, result_dt, a, b, poly_arg_none());
  if (poly_dtype_is_bool(poly_dtype_scalar(result->dtype))) return result;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_BOOL, result, poly_arg_none());
}

static _Thread_local PolyPatternMatcher *g_pm_wgsl_extra = NULL;

PolyPatternMatcher *poly_pm_wgsl_extra(void) {
  if (g_pm_wgsl_extra) return g_pm_wgsl_extra;

  PolyOpSet shift_set = {{0, 0}};
  shift_set = poly_opset_add(shift_set, POLY_OP_SHL);
  shift_set = poly_opset_add(shift_set, POLY_OP_SHR);

  PolyOpSet bool_alu_set = {{0, 0}};
  bool_alu_set = poly_opset_add(bool_alu_set, POLY_OP_CMPLT);
  bool_alu_set = poly_opset_add(bool_alu_set, POLY_OP_CMPEQ);
  bool_alu_set = poly_opset_add(bool_alu_set, POLY_OP_CMPNE);
  bool_alu_set = poly_opset_add(bool_alu_set, POLY_OP_XOR);

  PolyRule rules[] = {
      /* WGSL shift amounts must be u32 (tinygrad wgsl.py:49-50) */
      {poly_pat_ops(shift_set, NULL, 0, NULL), rule_wgsl_shift_u32},
      /* WGSL bool ALU normalization: tinygrad wgsl.py:41-42 plus mixed equality guard. */
      {poly_pat_ops(bool_alu_set, NULL, 0, NULL), rule_wgsl_bool_alu},
  };

  g_pm_wgsl_extra =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_wgsl_extra;
}

/* WebGPU linearizer */

/* Linearize a kernel for WebGPU: full codegen pipeline with GPU dims.
 *
 * Pipeline: full_rewrite_to_sink_ex (with device=WEBGPU, triggering
 * add_gpudims + group_for_reduce) → apply_control_flow → linearize.
 *
 * WebGPU constraints (matching tinygrad WGSLRenderer):
 *   - supports_float4=false → devectorize=1
 *   - local_max=(256,256,64) → gpu_block_size=256
 *   - No MULACC/THREEFRY hardware support
 *   - No tensor cores
 *   - extra_matcher: shift u32 normalization, bool CMPLT/XOR */
PolyUOp *poly_rewrite_webgpu(PolyCtx *ctx, PolyUOp *sink) {
  PolyRewriteOpts opts = {
      .optimize = poly_kernel_optimize_enabled(sink),
      .devectorize = 1, /* supports_float4=false: full devectorize */
      .beam_width = 0,
      .caps =
          {
              .has_mulacc = false,
              .has_threefry = false,
              .has_exp2 = true,
              .has_log2 = true,
              .has_sin = true,
              /* Pinned WGSLRenderer inherits RECIPROCAL and does not
               * advertise FDIV (renderer/wgsl.py:56-66). */
              .has_fdiv = false,
              .has_int64 = false,
              .has_local = true,
              .global_max = {65535, 65535, 65535},
              .local_max = {256, 256, 64},
              .max_vec_width = 1, /* supports_float4=false: scalar loads, no vec folding */
          },
      .device = POLY_DEVICE_WEBGPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      /* Pinned codegen/__init__.py:120-140 runs pm_dtype_decomps before the
       * WGSL renderer's final extra_matcher. WGSL has no native BF16. */
      .dtype_matcher = poly_pm_bf16_non_native(),
      .extra_matcher = poly_pm_wgsl_extra(),
      .gpu_block_size = 256, /* WebGPU local_max[0] */
  };
  return poly_full_rewrite_to_sink_ex(ctx, sink, opts);
}

PolyUOp **poly_linearize_webgpu(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  sink = poly_rewrite_webgpu(ctx, sink);
  /* apply_control_flow already called inside full_rewrite_to_sink_ex (codegen.c:5674) */
  return poly_linearize_rewritten(ctx, sink, n_out);
}
