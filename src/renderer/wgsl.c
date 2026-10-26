/*
 * renderer/wgsl.c — current Tinygrad renderer/wgsl.py port
 *
 * Walks linearized UOps and emits a WGSL
 * compute shader string. Buffers become storage bindings, loops use
 * WGSL syntax, WHERE maps to select().
 *
 * GPU ops: SPECIAL (gidx/lidx → workgroup_id/local_invocation_id),
 * BARRIER (workgroupBarrier), BUFFER(LOCAL) workgroup arrays, BUFFER(REG)
 * register-local arrays, and AFTER passthrough.
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

#include "codegen/codegen.h"
#include "runtime_webgpu.h"
#include "renderer/cstyle.h"
#include "uop/ops.h"
#include "bigint.h"
#include "engine/schedule.h" /* POLY_DEVICE_WEBGPU */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <limits.h>
#include "utils.h"

static int wgsl_launch_dim_upper_bound(PolyUOp *expr) {
  if (!expr) return 1;
  int64_t lo = 0, hi = 1;
  poly_uop_minmax(NULL, expr, &lo, &hi);
  if (hi <= 0) return 1;
  if (hi > INT32_MAX) return INT32_MAX;
  return (int)hi;
}

/* C storage for Tinygrad's string-renderer output. */

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

/* C storage for Tinygrad's per-UOp rendered-expression dictionary. */

typedef struct {
  PolyUOp **keys;
  char **vals;
  int cap;
  bool failed;
} WgslStrMap;

static void wsm_init(WgslStrMap *m, int n) {
  m->failed = false;
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
  if (!val) m->failed = true;
}

static char *wsm_get(WgslStrMap *m, PolyUOp *key) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
  while (m->keys[h]) {
    if (m->keys[h] == key && m->vals[h]) return m->vals[h];
    if (m->keys[h] == key) break;
    h = (h + 1) % m->cap;
  }
  if (!m->failed)
    fprintf(stderr, "polygrad: renderer operand has no expression (%s)\n", poly_op_name(key->op));
  m->failed = true;
  /* Keep formatting safe while unwinding; failed maps can never publish source.
   * Unlike a zero fallback, this sentinel is not a rendered value. */
  return "";
}

static void wsm_destroy(WgslStrMap *m) {
  for (int i = 0; i < m->cap; i++)
    if (m->vals[i]) free(m->vals[i]);
  free(m->keys);
  free(m->vals);
}

typedef struct {
  char *name;
  PolyUOp *uop;
  bool is_buffer;
} WgslBinding;

/* C rows for WGSLRenderer.render_kernel's ordered (name, (UOp, mutable)) bufs. */
static bool wgsl_binding_append(
    WgslBinding **bindings,
    int *n_bindings,
    int *cap_bindings,
    const char *name,
    PolyUOp *uop,
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
      .uop = uop,
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
  PolyDType s = dt;
  /* f16 for half (tinygrad: dtypes.half -> "f16") */
  if (s.priority == POLY_FLOAT16.priority && s.bitsize == POLY_FLOAT16.bitsize) return "f16";
  if (poly_dtype_is_float(dt)) return "f32";
  if (poly_dtype_is_bool(dt)) return "bool";
  /* Sub-4-byte types: char/short map to i32/u32 (tinygrad type_map) */
  if (poly_dtype_is_unsigned(dt)) return "u32";
  return "i32";
}

static bool wgsl_dtype_is_packed_storage(PolyDType dt) {
  PolyDType s = dt;
  int itemsize = poly_dtype_itemsize(s);
  /* tinygrad WGSL packs sub-32-bit non-half storage through atomic<u32>.
   * WGSL scalar bool is valid, but bool storage buffers are not host-shareable,
   * so bool buffers must use the same byte-lane packing path as uint8. */
  return itemsize > 0 && itemsize < 4 && (poly_dtype_is_int(s) || poly_dtype_is_bool(s));
}

/* Pinned renderer/wgsl.py:is_packed. LOAD's address space is ALU: the
 * backing buffer owns storage, including behind AFTER/view wrappers. */
static bool wgsl_is_packed(PolyCtx *ctx, PolyUOp *u) {
  if (!u) return false;
  PolyDType dt = u->op == POLY_OP_STORE && u->n_src > 1 ? u->src[1]->dtype : u->dtype;
  if (!wgsl_dtype_is_packed_storage(dt)) return false;
  PolyAddrSpace space;
  return poly_uop_addrspace(poly_uop_buf_uop(ctx, u), &space) && space != POLY_ADDR_REG;
}

/* Pinned WGSLRenderer.buf_map and _packed_size share is_packed with accesses. */
static const char *wgsl_buffer_type_name(PolyCtx *ctx, PolyUOp *u) {
  return wgsl_is_packed(ctx, u) ? "atomic<u32>" : wgsl_type_name(u->dtype);
}

/* Pinned renderer/wgsl.py:_packed_size. */
static int64_t wgsl_packed_size(PolyCtx *ctx, PolyUOp *u) {
  int64_t size = poly_program_buffer_size(u);
  if (!wgsl_is_packed(ctx, u)) return size;
  int elems = 4 / poly_dtype_itemsize(u->dtype);
  /* ceildiv without overflowing a valid maximum storage extent. */
  return size / elems + (size % elems != 0);
}

static bool wgsl_packed_params(PolyDType dt, int *itemsize, int *elems, unsigned *mask) {
  if (!wgsl_dtype_is_packed_storage(dt)) return false;
  int is = poly_dtype_itemsize(dt);
  if (is != 1 && is != 2) return false;
  if (itemsize) *itemsize = is;
  if (elems) *elems = 4 / is;
  if (mask) *mask = (is == 1) ? 0xFFu : 0xFFFFu;
  return true;
}

/* Pinned packed_load: field width and signed extension follow the LOAD dtype. */
static char *wgsl_make_packed_load_expr(
    PolyCtx *ctx,
    WgslStrMap *names,
    PolyUOp *load,
    PolyUOp *idx_uop
) {
  if (!idx_uop || !wgsl_is_packed(ctx, load)) return NULL;

  int itemsize = 0, elems = 0;
  unsigned mask = 0;
  if (!wgsl_packed_params(load->dtype, &itemsize, &elems, &mask)) return NULL;

  char *buf_s = wsm_get(names, idx_uop->src[0]);
  char *idx_s = wsm_get(names, idx_uop->src[1]);
  if (!buf_s || !idx_s) return NULL;

  char *raw = poly_strfmt(
      "((atomicLoad(&%s[(%s/%d)]) >> ((u32(%s)%%%du)*%du)) & 0x%Xu)", buf_s, idx_s, elems, idx_s,
      elems, 8u * (unsigned)itemsize, mask
  );
  if (!raw) return NULL;

  char *out;
  PolyDType scalar = load->dtype;
  if (poly_dtype_is_bool(scalar)) {
    out = poly_strfmt("(%s != 0u)", raw);
  } else if (!poly_dtype_is_unsigned(scalar) && poly_dtype_is_int(scalar)) {
    int sext = (itemsize == 1) ? 24 : 16;
    out = poly_strfmt("((i32(%s)<<%d)>>%d)", raw, sext, sext);
  } else {
    return raw;
  }
  free(raw);
  return out;
}

/* Pinned packed_store and string_rewrite: select field width from the value,
 * and never perform packed read-modify-write on register storage. */
static bool wgsl_emit_packed_store(
    PolyCtx *ctx,
    WgslStrBuf *body,
    WgslStrMap *names,
    PolyUOp *store,
    PolyUOp *idx_uop,
    const char *val,
    int depth
) {
  if (!idx_uop || !wgsl_is_packed(ctx, store)) return false;
  PolyDType value_dt = store->src[1]->dtype;

  int itemsize = 0, elems = 0;
  unsigned mask = 0;
  if (!wgsl_packed_params(value_dt, &itemsize, &elems, &mask)) return false;

  char *buf_s = wsm_get(names, idx_uop->src[0]);
  char *idx_s = wsm_get(names, idx_uop->src[1]);
  if (!buf_s || !idx_s || !val) return false;

  char *val_u32 = poly_strfmt(poly_dtype_is_bool(value_dt) ? "select(0u, 1u, %s)" : "u32(%s)", val);
  if (!val_u32) return false;

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
  free(val_u32);
  return true;
}

static bool wgsl_uop_tree_contains_target(PolyUOp *u, PolyUOp *target) {
  if (!u || !target) return false;
  if (u == target) return true;
  for (int i = 0; i < u->n_src; i++)
    if (wgsl_uop_tree_contains_target(u->src[i], target)) return true;
  return false;
}

static PolyUOp *wgsl_pick_lane_uop(PolyCtx *ctx, PolyUOp *u, int lane) {
  if (!u || poly_uop_max_numel(ctx, u) <= 1) return u;
  if (lane < 0) lane = 0;
  if (u->op == POLY_OP_STACK) {
    if (u->n_src <= 0) return NULL;
    if (lane >= u->n_src) lane = u->n_src - 1;
    return u->src[lane];
  }
  return NULL;
}

static int wgsl_infer_gated_load_lane(PolyCtx *ctx, PolyUOp *idx_expr, PolyUOp *vec_gate) {
  int64_t lanes = poly_uop_max_numel(ctx, vec_gate);
  if (!idx_expr || !vec_gate || lanes <= 1 || lanes > INT_MAX) return -1;
  int found = -1;
  for (int lane = 0; lane < lanes; lane++) {
    PolyUOp *gate_lane = wgsl_pick_lane_uop(ctx, vec_gate, lane);
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

/* WGSL float constant */

static char *render_float_const_wgsl(double v, char *buf, int cap) {
  /* Match the pin's str(c.val): round-trip the promoted double, not just f32.
   * Nine digits round FLT_MAX above its range as a WGSL abstract-float. */
  snprintf(buf, cap, "%.17g", (double)(float)v);
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

/* PG-DIV-008: pinned WGSLRenderer maps 8/16-bit integers to i32/u32 without
 * truncating CAST or ALU results. Preserve dtype.truncate/PythonProgram's
 * typed value before any later widening, not only at packed STORE. */
static void wgsl_emit_typed_expr(WgslStrBuf *out, PolyDType dtype, const char *expr) {
  if (poly_dtype_is_int(dtype) && (dtype.bitsize == 8 || dtype.bitsize == 16)) {
    if (poly_dtype_is_unsigned(dtype))
      wsb_printf(out, "(u32(%s) & %uu)", expr, (1u << dtype.bitsize) - 1);
    else
      wsb_printf(out, "((i32(%s) << %uu) >> %uu)", expr, 32 - dtype.bitsize, 32 - dtype.bitsize);
  } else
    wsb_puts(out, expr);
}

/* Current renderer/cstyle.py base_rewrite and renderer/wgsl.py render
 * CAST(strong, CONST(weak/bool)) directly as a destination-typed literal. */
static char *wgsl_render_const_literal(PolyUOp *c, PolyDType dtype) {
  if (!c || c->op != POLY_OP_CONST) return NULL;
  PolyDType scalar = dtype;
  char val[192];
  if (poly_dtype_is_float(scalar)) {
    if (isinf(c->arg.f)) {
      snprintf(val, sizeof(val), c->arg.f > 0 ? "INFINITY" : "(-INFINITY)");
    } else if (isnan(c->arg.f)) {
      snprintf(val, sizeof(val), "nan()");
    } else if (scalar.priority == POLY_FLOAT16.priority && scalar.bitsize == 16) {
      char f32buf[64];
      render_float_const_wgsl(c->arg.f, f32buf, sizeof(f32buf));
      snprintf(val, sizeof(val), "f16(%s)", f32buf);
    } else {
      render_float_const_wgsl(c->arg.f, val, sizeof(val));
    }
  } else if (poly_dtype_is_bool(scalar)) {
    snprintf(val, sizeof(val), "%s", c->arg.b ? "true" : "false");
  } else if (poly_dtype_is_unsigned(scalar)) {
    bool negative = c->arg.kind == POLY_ARG_BIGINT ? c->arg.bigint.sign < 0 : c->arg.i < 0;
    if (negative) {
      char *decimal = poly_arg_integer_to_decimal(c->arg);
      if (!decimal) return NULL;
      snprintf(val, sizeof(val), "bitcast<u32>(%s)", decimal);
      free(decimal);
    } else {
      snprintf(val, sizeof(val), "%uu", (unsigned)(uint32_t)poly_arg_integer_to_u64_mod(c->arg));
    }
  } else {
    snprintf(val, sizeof(val), "%d", (int32_t)poly_arg_integer_to_u64_mod(c->arg));
  }
  WgslStrBuf out;
  wsb_init(&out);
  wgsl_emit_typed_expr(&out, dtype, val);
  return out.buf;
}

/* WGSL ALU expression */

static char *render_alu_wgsl(
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
      return poly_strfmt("(%s)", s0);
    else if (poly_dtype_is_unsigned(dtype))
      /* WGSL doesn't support unary minus on unsigned (tinygrad wgsl.py:69) */
      return poly_strfmt("(0-%s)", s0);
    else
      return poly_strfmt("(-%s)", s0);
  case POLY_OP_SQRT:
    return poly_strfmt("sqrt(%s)", s0);
  case POLY_OP_TRUNC:
    return poly_strfmt("trunc(%s)", s0);
  case POLY_OP_EXP2:
    return poly_strfmt("exp2(%s)", s0);
  case POLY_OP_LOG2:
    return poly_strfmt("log2(%s)", s0);
  case POLY_OP_SIN:
    return poly_strfmt("sin(%s)", s0);
  case POLY_OP_RECIPROCAL:
    return poly_strfmt("(1.0/%s)", s0);
  /* binary */
  case POLY_OP_ADD:
    return poly_strfmt("(%s+%s)", s0, s1);
  case POLY_OP_SUB:
    return poly_strfmt("(%s-%s)", s0, s1);
  case POLY_OP_MUL:
    return poly_strfmt("(%s*%s)", s0, s1);
  case POLY_OP_FDIV:
    return poly_strfmt("(%s/%s)", s0, s1);
  case POLY_OP_IDIV:
    return poly_strfmt("(%s/%s)", s0, s1);
  case POLY_OP_MOD:
    return poly_strfmt("(%s%%%s)", s0, s1);
  case POLY_OP_SHL:
    return poly_strfmt("(%s<<%s)", s0, s1);
  case POLY_OP_SHR:
    return poly_strfmt("(%s>>%s)", s0, s1);
  case POLY_OP_AND:
    return poly_strfmt("(%s&%s)", s0, s1);
  case POLY_OP_OR:
    return poly_strfmt("(%s|%s)", s0, s1);
  case POLY_OP_XOR:
    return poly_strfmt("(%s^%s)", s0, s1);
  case POLY_OP_CMPLT:
    return poly_strfmt("(%s<%s)", s0, s1);
  case POLY_OP_CMPNE:
    return poly_strfmt("(%s!=%s)", s0, s1);
  case POLY_OP_CMPEQ:
    return poly_strfmt("(%s==%s)", s0, s1);
  case POLY_OP_MAX:
    return poly_strfmt("max(%s,%s)", s0, s1);
  case POLY_OP_POW:
    return poly_strfmt("pow(%s,%s)", s0, s1);
  /* ternary */
  case POLY_OP_WHERE:
    return poly_strfmt("select(%s,%s,%s)", s2, s1, s0);
  case POLY_OP_MULACC:
    return poly_strfmt("((%s*%s)+%s)", s0, s1, s2);
  default:
    return NULL;
  }
}

/* Takes ownership of expr. The linearizer determines declaration scope; only
 * pinned single-use expressions may be deferred to their consumer. */
static void wgsl_render_value(
    PolyCtx *ctx,
    WgslStrMap *names,
    WgslStrBuf *body,
    PolyUOp *u,
    char *expr,
    bool inline_expr,
    const char *prefix,
    int *counter,
    int depth
) {
  /* PG-DIV-039: Chrome limits WGSL expression depth to 512. Count syntax
   * punctuation, not just parentheses: associative inlining can flatten a DAG
   * into a long left-associated chain. Leave room for the consuming operation
   * and its other operands. Materialization preserves every operation/order. */
  if (expr && inline_expr) {
    int syntax = 0;
    for (const char *c = expr; *c; c++) {
      if (strchr("()[]+-*/%&|^!~<>=,", *c) && ++syntax > 128) {
        inline_expr = false;
        break;
      }
    }
  }
  if (!expr || inline_expr) {
    wsm_set(names, u, expr);
    return;
  }
  char name[32];
  snprintf(name, sizeof(name), "%s%d", prefix, (*counter)++);
  wsm_set(names, u, strdup(name));
  for (int d = 0; d < depth; d++)
    wsb_puts(body, "  ");
  if (u->op == POLY_OP_STACK && poly_uop_max_numel(ctx, u) > 1)
    wsb_printf(body, "var %s = %s;\n", name, expr);
  else
    wsb_printf(body, "var %s: %s = %s;\n", name, wgsl_type_name(u->dtype), expr);
  free(expr);
}

/* WGSL Renderer */

/* Pinned WGSLRenderer.string_rewrite and render_kernel, with C string storage. */
char *poly_render_wgsl(PolyCtx *ctx, PolyUOp **uops, int n, const char *fn_name) {
  WgslStrBuf body;
  wsb_init(&body);

  WgslStrMap names;
  wsm_init(&names, n);

  /* Kernel parameter bindings use ParamArg.addrspace.
   * Match tinygrad exactly here:
   *   - cstyle.py collects PARAMs into one ordered `bufs` list
   *   - wgsl.py assigns bindings sequentially from that list
   * So WGSL binding numbers follow encounter order, not PARAM.arg. */
  WgslBinding *bindings = NULL;
  int n_bindings = 0;
  int cap_bindings = 0;
  /* Pinned CStyleLanguage._render counts edges, including repeated sources. */
  int *child_count = poly_render_child_counts(uops, n);
  if (!child_count) goto fail;
  bool expand_ssa = poly_getenv_flag("EXPAND_SSA");

  /* prefix counters */
  int c_val = 0, c_alu = 0, c_cast = 0;
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
    PolyDType s = uops[i]->dtype;
    if (s.priority == POLY_FLOAT16.priority && s.bitsize == POLY_FLOAT16.bitsize) uses_f16 = true;
  }

  /* Main render loop */

  for (int i = 0; i < n; i++) {
    if (names.failed) goto fail;
    PolyUOp *u = uops[i];

    /* --- SINK/NOOP/GROUP: skip ---------------------------------------- */
    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP) continue;

    /* Current cstyle uses ALU address space for scalar uniforms and GLOBAL
     * for storage buffers; both remain one numbered PARAM sequence. */
    if (u->op == POLY_OP_PARAM) {
      int64_t slot = poly_program_buffer_slot(u);
      if (slot < 0) goto fail;
      char name[1024];
      if (!poly_render_param_name(ctx, u, name, sizeof(name))) goto fail;
      wsm_set(&names, u, strdup(name));

      if (!wgsl_binding_append(
              &bindings, &n_bindings, &cap_bindings, name, u, !poly_uop_is_alu_param(u)
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
      char *literal = wgsl_render_const_literal(u, u->dtype);
      if (!literal) goto fail;
      wsm_set(&names, u, literal);
      continue;
    }

    /* --- INDEX: array indexing (buf[idx]) ----------------------------- */
    if (u->op == POLY_OP_INDEX) {
      char *buf_s = wsm_get(&names, u->src[0]);
      char *idx_s = wsm_get(&names, u->src[1]);
      char *expr = poly_strfmt("%s[%s]", buf_s, idx_s);
      if (!expr) goto fail;
      wsm_set(&names, u, expr);
      continue;
    }

    /* --- SHRINK: late codegen memory slice --------------------------- */
    if (u->op == POLY_OP_SHRINK) {
      char *buf_s = wsm_get(&names, u->src[0]);
      char *idx_s = wsm_get(&names, u->src[1]);
      char *expr = poly_strfmt("%s[%s]", buf_s, idx_s);
      if (!expr) goto fail;
      wsm_set(&names, u, expr);
      continue;
    }

    /* --- RANGE: for loop --------------------------------------------- */
    if (u->op == POLY_OP_RANGE) {
      char *range = poly_uop_range_str(u->arg);
      char *name = malloc(strlen(range) + 6);
      snprintf(name, strlen(range) + 6, "%cidx%s", poly_uop_axis_letter(u->arg), range);
      free(range);
      wsm_set(&names, u, name);

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

    /* tinygrad@2026-08-22/a9069c177a9d renderer/wgsl.py renders LOCAL
     * BUFFER storage as a module-scope workgroup array. */
    if (u->op == POLY_OP_BUFFER && poly_program_memory_is(u, POLY_ADDR_LOCAL)) {
      char name[32];
      snprintf(name, sizeof(name), "smem%lld", (long long)poly_program_buffer_slot(u));
      wsm_set(&names, u, strdup(name));

      int64_t smem_size = wgsl_packed_size(ctx, u);
      const char *base_tn = wgsl_buffer_type_name(ctx, u);
      if (n_extern_locals < 16) {
        snprintf(
            extern_locals[n_extern_locals], 256, "var<workgroup> %s: array<%s,%lld>;", name,
            base_tn, (long long)smem_size
        );
        n_extern_locals++;
      }
      continue;
    }

    /* --- register-local array ---------------------------------------- */
    if (u->op == POLY_OP_BUFFER && poly_program_memory_is(u, POLY_ADDR_REG)) {
      char name[32];
      snprintf(name, sizeof(name), "r%lld", (long long)poly_program_buffer_slot(u));
      wsm_set(&names, u, strdup(name));

      /* tinygrad: var rN: array<type, SIZE>; (wgsl.py:75) */
      int64_t reg_size = wgsl_packed_size(ctx, u);
      const char *base_tn = wgsl_buffer_type_name(ctx, u);
      for (int d = 0; d < depth; d++)
        wsb_puts(&body, "  ");
      wsb_printf(&body, "var %s: array<%s,%lld>;\n", name, base_tn, (long long)reg_size);
      continue;
    }

    /* --- AFTER: passthrough (alias src[0] name) ---------------------- */
    if (u->op == POLY_OP_AFTER) {
      char *src_name = wsm_get(&names, u->src[0]);
      if (src_name) wsm_set(&names, u, strdup(src_name));
      continue;
    }

    /* --- COPY: transparent placement wrapper ------------------------- */
    if (u->op == POLY_OP_COPY && u->n_src > 0) {
      char *src_name = wsm_get(&names, u->src[0]);
      if (src_name) wsm_set(&names, u, strdup(src_name));
      continue;
    }

    /* --- LOAD: read from array --------------------------------------- */
    if (u->op == POLY_OP_LOAD) {
      char *bidx = wsm_get(&names, u->src[0]);
      const char *tn = wgsl_type_name(u->dtype);
      PolyUOp *idx_uop = poly_as_index(u->src[0]);
      char *packed_load = NULL;
      const char *load_expr = bidx;
      if (idx_uop && wgsl_is_packed(ctx, u)) {
        packed_load = wgsl_make_packed_load_expr(ctx, &names, u, idx_uop);
        if (!packed_load) goto fail;
        load_expr = packed_load;
      }
      char *expr;
      /* Gated load: select(alt, load, gate) -- tinygrad WGSL parity */
      PolyUOp *gate_uop =
          (u->n_src >= 3 && poly_dtype_is_bool(u->src[2]->dtype))
              ? u->src[2]
              : ((idx_uop && idx_uop->n_src >= 3 && poly_dtype_is_bool(idx_uop->src[2]->dtype))
                     ? idx_uop->src[2]
                     : NULL);
      if (gate_uop && u->n_src >= 2) {
        PolyUOp *alt_uop = u->src[1];
        int64_t u_lanes = poly_uop_max_numel(ctx, u);
        int64_t gate_lanes = gate_uop ? poly_uop_max_numel(ctx, gate_uop) : 1;
        int64_t alt_lanes = alt_uop ? poly_uop_max_numel(ctx, alt_uop) : 1;
        if (u_lanes == 1 && (gate_lanes > 1 || alt_lanes > 1)) {
          int lane = wgsl_infer_gated_load_lane(ctx, idx_uop->src[1], gate_uop);
          int64_t lanes = gate_lanes > alt_lanes ? gate_lanes : alt_lanes;
          if (lane < 0 && lanes > 1) {
            PolyUOp *lane_key = gate_lanes > 1 ? gate_uop : (alt_lanes > 1 ? alt_uop : NULL);
            lane = wgsl_next_vector_lane(
                lane_key, (int)lanes, gated_lane_keys, gated_lane_next, &n_gated_lane_keys
            );
          }
          if (lane >= 0) {
            PolyUOp *lane_gate = wgsl_pick_lane_uop(ctx, gate_uop, lane);
            PolyUOp *lane_alt = wgsl_pick_lane_uop(ctx, alt_uop, lane);
            if (lane_gate) gate_uop = lane_gate;
            if (lane_alt) alt_uop = lane_alt;
          }
        }
        char *gate_s = wsm_get(&names, gate_uop);
        char *alt_s = wsm_get(&names, alt_uop);
        expr = poly_strfmt("select(%s, %s, %s)", alt_s, load_expr, gate_s);
      } else if (gate_uop) {
        int64_t u_lanes = poly_uop_max_numel(ctx, u);
        int64_t gate_lanes = gate_uop ? poly_uop_max_numel(ctx, gate_uop) : 1;
        if (u_lanes == 1 && gate_lanes > 1) {
          int lane = wgsl_infer_gated_load_lane(ctx, idx_uop->src[1], gate_uop);
          if (lane < 0) {
            lane = wgsl_next_vector_lane(
                gate_uop, (int)gate_lanes, gated_lane_keys, gated_lane_next, &n_gated_lane_keys
            );
          }
          if (lane >= 0) {
            PolyUOp *lane_gate = wgsl_pick_lane_uop(ctx, gate_uop, lane);
            if (lane_gate) gate_uop = lane_gate;
          }
        }
        char *gate_s = wsm_get(&names, gate_uop);
        expr = poly_strfmt("select(%s(0), %s, %s)", tn, load_expr, gate_s);
      } else {
        expr = strdup(load_expr);
      }
      free(packed_load);
      PolyAddrSpace space;
      bool inline_load =
          child_count[i] == 1 && poly_uop_addrspace(u->src[0], &space) && space == POLY_ADDR_REG;
      wgsl_render_value(ctx, &names, &body, u, expr, inline_load, "val", &c_val, depth);
      continue;
    }

    /* --- VECTORIZE: vector constructor (scalar after devectorize) ----- */
    if (u->op == POLY_OP_STACK) {
      if (u->n_src == 1) {
        /* Single source: alias */
        char *s = wsm_get(&names, u->src[0]);
        wsm_set(&names, u, strdup(s));
      } else if (u->n_src > 1) {
        /* Multi-source: vec constructor. Shouldn't appear for scalar WGSL
         * (supports_float4=false), but handle for robustness. */
        const char *tn = wgsl_type_name(u->dtype);
        WgslStrBuf vexpr;
        wsb_init(&vexpr);
        wsb_printf(&vexpr, "vec%d<%s>(", u->n_src, tn);
        for (int j = 0; j < u->n_src; j++) {
          char *s = wsm_get(&names, u->src[j]);
          wsb_printf(&vexpr, "%s%s", j > 0 ? "," : "", s);
        }
        wsb_puts(&vexpr, ")");
        wgsl_render_value(
            ctx, &names, &body, u, vexpr.buf, child_count[i] == 1 && !expand_ssa, "cast", &c_cast,
            depth
        );
      } else {
        wsm_set(&names, u, strdup("0"));
      }
      continue;
    }

    /* --- STORE: write to array or accumulator ------------------------ */
    if (u->op == POLY_OP_STORE) {
      char *target = wsm_get(&names, u->src[0]);
      char *val = wsm_get(&names, u->src[1]);

      PolyUOp *store_idx = poly_as_index(u->src[0]);
      if (store_idx && wgsl_is_packed(ctx, u)) {
        if (!wgsl_emit_packed_store(ctx, &body, &names, u, store_idx, val, depth)) goto fail;
      } else {
        for (int d = 0; d < depth; d++)
          wsb_puts(&body, "  ");
        wsb_printf(&body, "%s = %s;\n", target, val);
      }

      continue;
    }

    /* --- CAST / BITCAST: type conversion ----------------------------- */
    if (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) {
      if (u->op == POLY_OP_CAST && u->n_src == 1 && u->src[0] && u->src[0]->op == POLY_OP_CONST &&
          (poly_dtype_is_weak(u->src[0]->dtype) || poly_dtype_is_bool(u->src[0]->dtype)) &&
          poly_uop_max_numel(ctx, u) == 1) {
        char *literal = wgsl_render_const_literal(u->src[0], u->dtype);
        if (!literal) goto fail;
        wsm_set(&names, u, literal);
        continue;
      }
      char *src_s = wsm_get(&names, u->src[0]);
      const char *tn = wgsl_type_name(u->dtype);
      PolyDType dst_s = u->dtype;
      PolyDType src_dt = u->src[0]->dtype;

      char *expr;
      if (u->op == POLY_OP_BITCAST) {
        /* BITCAST specializations matching tinygrad wgsl.py:76-84 */
        if (dst_s.priority == POLY_FLOAT16.priority && dst_s.bitsize == POLY_FLOAT16.bitsize) {
          expr = poly_strfmt("bitcast<vec2<f16>>(%s)[0]", src_s);
        } else if (dst_s.priority == POLY_UINT8.priority && dst_s.bitsize == 8) {
          expr = poly_strfmt("bitcast<u32>(%s&0xFF)", src_s);
        } else if (dst_s.priority == POLY_INT8.priority && dst_s.bitsize == 8) {
          expr = poly_strfmt("((i32(%s&0xFF)<<24)>>24)", src_s);
        } else if (dst_s.priority == POLY_UINT16.priority && dst_s.bitsize == 16) {
          if (src_dt.priority == POLY_FLOAT16.priority && src_dt.bitsize == POLY_FLOAT16.bitsize)
            expr = poly_strfmt("bitcast<u32>(vec2<f16>(%s,0))", src_s);
          else
            expr = poly_strfmt("bitcast<u32>(%s&0xFFFF)", src_s);
        } else if (dst_s.priority == POLY_INT16.priority && dst_s.bitsize == 16) {
          if (src_dt.priority == POLY_FLOAT16.priority && src_dt.bitsize == POLY_FLOAT16.bitsize)
            expr = poly_strfmt("bitcast<i32>(vec2<f16>(%s,0))", src_s);
          else
            expr = poly_strfmt("((i32(%s&0xFFFF)<<16)>>16)", src_s);
        } else {
          expr = poly_strfmt("bitcast<%s>(%s)", tn, src_s);
        }
      } else {
        /* PG-DIV-008: u32(float) clamps negatives before masking. Narrow
         * integer conversions stage through i32 so signed values can wrap. */
        bool narrow_float =
            poly_dtype_is_float(src_dt) && poly_dtype_is_int(dst_s) && dst_s.bitsize < 32;
        expr = poly_strfmt("%s(%s)", narrow_float ? "i32" : tn, src_s);
      }

      if (!expr) goto fail;
      if (u->op == POLY_OP_CAST && poly_dtype_is_int(u->dtype) && u->dtype.bitsize < 32) {
        WgslStrBuf typed;
        wsb_init(&typed);
        wgsl_emit_typed_expr(&typed, u->dtype, expr);
        free(expr);
        expr = typed.buf;
      }
      bool inline_cast = (u->op != POLY_OP_CAST || poly_uop_max_numel(ctx, u) == 1) &&
                         child_count[i] == 1 && !expand_ssa;
      wgsl_render_value(ctx, &names, &body, u, expr, inline_cast, "cast", &c_cast, depth);
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
      const char *sources[3] = {"", "", ""};
      char *stripped[3] = {0};
      for (int j = 0; j < u->n_src && j < 3; j++) {
        sources[j] = wsm_get(&names, u->src[j]);
        /* PG-DIV-008 wraps narrow ALU in truncation, so its outer operator
         * is no longer the original associative ALU. Preserve those parentheses. */
        if (u->src[j]->op == u->op &&
            !(poly_dtype_is_int(u->src[j]->dtype) && u->src[j]->dtype.bitsize < 32) &&
            (u->op == POLY_OP_ADD || u->op == POLY_OP_MUL || u->op == POLY_OP_XOR ||
             u->op == POLY_OP_OR || u->op == POLY_OP_AND)) {
          stripped[j] = poly_render_strip_parens(sources[j]);
          if (stripped[j]) sources[j] = stripped[j];
        }
      }
      char *expr = render_alu_wgsl(u->op, u->dtype, sources[0], sources[1], sources[2]);
      for (int j = 0; j < 3; j++)
        free(stripped[j]);
      if (!expr) goto fail;
      if (poly_dtype_is_int(u->dtype) && u->dtype.bitsize < 32) {
        WgslStrBuf typed;
        wsb_init(&typed);
        wgsl_emit_typed_expr(&typed, u->dtype, expr);
        free(expr);
        expr = typed.buf;
      }
      wgsl_render_value(
          ctx, &names, &body, u, expr, u->op != POLY_OP_WHERE && child_count[i] == 1 && !expand_ssa,
          "alu", &c_alu, depth
      );
      continue;
    }
  }

  /* Build complete source */
  if (names.failed) goto fail;
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

  /* Pinned render_kernel binds in encounter order (binding 0 = INFINITY).
   * Storage buffers use var<storage,read_write>, scalar vars use var<uniform>. */
  for (int i = 0; i < n_bindings; i++) {
    const char *tn = bindings[i].is_buffer ? wgsl_buffer_type_name(ctx, bindings[i].uop)
                                           : wgsl_type_name(bindings[i].uop->dtype);
    if (bindings[i].is_buffer) {
      wsb_printf(
          &out, "@group(0) @binding(%d)\nvar<storage,read_write> %s: array<%s>;\n", i + 1,
          bindings[i].name, tn
      );
    } else {
      wsb_printf(
          &out, "@group(0) @binding(%d)\nvar<uniform> %s: %s;\n", i + 1, bindings[i].name, tn
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
  wsb_puts(&out, body.buf);
  wsb_puts(&out, "}\n");

  /* cleanup */
  wgsl_bindings_free(bindings, n_bindings);
  free(child_count);
  free(body.buf);
  wsm_destroy(&names);

  return out.buf;

fail:
  wgsl_bindings_free(bindings, n_bindings);
  free(child_count);
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
  if (poly_dtype_eq(amount->dtype, POLY_UINT32)) return NULL;
  /* Cast shift amount to u32 */
  PolyUOp *cast_amount = poly_uop1(ctx, POLY_OP_CAST, POLY_UINT32, amount, poly_arg_none());
  if (u->op == POLY_OP_SHL) {
    /* Pinned wgsl_matcher shifts unsigned bits, then restores the value dtype. */
    PolyUOp *bits = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT32, u->src[0], poly_arg_none());
    PolyUOp *shift = poly_uop2(ctx, u->op, POLY_UINT32, bits, cast_amount, u->arg);
    return poly_uop1(ctx, POLY_OP_BITCAST, u->dtype, shift, poly_arg_none());
  }
  return poly_uop2(ctx, u->op, u->dtype, u->src[0], cast_amount, u->arg);
}

/* Rule: unsupported/mixed bool ALU → cast to i32, apply op, cast back to bool.
 * tinygrad wgsl.py:41-42 handles bool CMPLT/XOR this way. Polygrad can also
 * produce mixed bool/int equality after packed bool storage loads; normalize
 * those before WGSL so expressions like `0u != bool_val` never reach render. */
static PolyUOp *rule_wgsl_bool_alu(PolyCtx *ctx, PolyUOp *u, const PolyBindings *bindings) {
  (void)bindings;
  if (u->n_src < 2) return NULL;
  bool src0_bool = poly_dtype_is_bool(u->src[0]->dtype);
  bool src1_bool = poly_dtype_is_bool(u->src[1]->dtype);
  if (!src0_bool && !src1_bool) return NULL;
  if ((u->op == POLY_OP_CMPEQ || u->op == POLY_OP_CMPNE) && src0_bool && src1_bool) return NULL;
  PolyUOp *a = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, u->src[0], poly_arg_none());
  PolyUOp *b = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, u->src[1], poly_arg_none());
  PolyDType result_dt = (u->op == POLY_OP_XOR) ? POLY_INT32 : POLY_BOOL;
  PolyUOp *result = poly_uop2(ctx, u->op, result_dt, a, b, poly_arg_none());
  if (poly_dtype_is_bool(result->dtype)) return result;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_BOOL, result, poly_arg_none());
}

static _Thread_local PolyPatternMatcher *g_pm_wgsl_extra = NULL;

/* Pinned wgsl.py:is_nan: WGSL permits finite-math optimization of x != x.
 * Both self-comparison forms must inspect bits after dtype decomposition. */
static PolyUOp *rule_wgsl_self_compare(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (u->n_src != 2 || u->src[0] != u->src[1]) return NULL;
  PolyUOp *x = u->src[0];
  PolyDType uint;
  int64_t mask, infinity;
  if (poly_dtype_eq(x->dtype, POLY_FLOAT16)) {
    uint = POLY_UINT16;
    mask = 0x7fff;
    infinity = 0x7c00;
  } else if (poly_dtype_eq(x->dtype, POLY_FLOAT32)) {
    uint = POLY_UINT32;
    mask = 0x7fffffff;
    infinity = 0x7f800000;
  } else {
    return NULL;
  }
  PolyUOp *bits = poly_uop1(ctx, POLY_OP_BITCAST, uint, x, poly_arg_none());
  PolyUOp *signless = poly_uop2(
      ctx, POLY_OP_AND, uint, bits, poly_uop0(ctx, POLY_OP_CONST, uint, poly_arg_int(mask)),
      poly_arg_none()
  );
  PolyUOp *isnan = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, poly_uop0(ctx, POLY_OP_CONST, uint, poly_arg_int(infinity)),
      signless, poly_arg_none()
  );
  if (u->op == POLY_OP_CMPNE) return isnan;
  return poly_uop2(
      ctx, POLY_OP_CMPNE, POLY_BOOL, isnan,
      poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true)), poly_arg_none()
  );
}

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

  PolyOpSet self_compare_set = {{0, 0}};
  self_compare_set = poly_opset_add(self_compare_set, POLY_OP_CMPEQ);
  self_compare_set = poly_opset_add(self_compare_set, POLY_OP_CMPNE);

  PolyRule rules[] = {
      {poly_upat_ops(self_compare_set, NULL, 0, NULL), rule_wgsl_self_compare},
      /* WGSL shift amounts must be u32 (tinygrad wgsl.py:49-50) */
      {poly_upat_ops(shift_set, NULL, 0, NULL), rule_wgsl_shift_u32},
      /* WGSL bool ALU normalization: tinygrad wgsl.py:41-42 plus mixed equality guard. */
      {poly_upat_ops(bool_alu_set, NULL, 0, NULL), rule_wgsl_bool_alu},
  };

  g_pm_wgsl_extra =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_wgsl_extra;
}

/* WebGPU linearizer */

/* Linearize a kernel for WebGPU: full codegen pipeline with GPU dims.
 *
 * Pipeline: full_rewrite_to_sink_ex (with device=WEBGPU, triggering
 * add_gpudims + pm_reduce_local) → apply_control_flow → linearize.
 *
 * WebGPU constraints (matching tinygrad WGSLRenderer):
 *   - No MULACC/THREEFRY hardware support
 *   - No tensor cores
 *   - extra_matcher: shift u32 normalization, bool CMPLT/XOR, NaN self-comparison */
PolyUOp *poly_rewrite_webgpu(PolyCtx *ctx, PolyUOp *sink) {
  PolyRewriteOpts opts = {
      .optimize = poly_kernel_optimize_enabled(sink),
      .beam_width = poly_kernel_beam(sink),
      .caps =
          {
              .device = "WEBGPU",
              .has_mulacc = false,
              .has_threefry = false,
              .has_exp2 = true,
              .has_log2 = true,
              .has_sin = true,
              /* Pinned WGSLRenderer inherits RECIPROCAL and does not
               * advertise FDIV (renderer/wgsl.py:56-66). */
              .has_fdiv = false,
              /* Tinygrad 2026-08-22/a9069c177a9d WGSLRenderer.supported_dtypes
               * includes half only when shader-f16 is in Target.arch. */
              .supports_float16 = poly_webgpu_supports_float16(),
              .supports_bfloat16 = false,
              .has_int64 = false,
              .has_local = true,
              .global_max = {65535, 65535, 65535},
              .local_max = {256, 256, 64},
              .max_vec_width = 1, /* supports_float4=false: scalar loads, no vec folding */
          },
      .device = POLY_DEVICE_WEBGPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .extra_matcher = poly_pm_wgsl_extra(),
  };
  return poly_full_rewrite_to_sink_ex(ctx, sink, opts);
}

PolyUOp **poly_linearize_webgpu(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  sink = poly_rewrite_webgpu(ctx, sink);
  /* apply_control_flow already called inside full_rewrite_to_sink_ex (codegen.c:5674) */
  return poly_do_linearize(ctx, sink, n_out);
}
