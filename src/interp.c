/*
 * interp.c -- Interpreter backend for polygrad
 *
 * Walks a linearized UOp sequence, evaluating each op in order.
 * RANGE/END pairs become explicit C loops via recursive region evaluation.
 * All scalar types are stored in a 64-bit value union (InterpVal).
 */

#include "interp.h"
#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* ── Value representation ────────────────────────────────────────────── */

typedef union {
  int64_t  i;
  uint64_t u;
  double   f;
  void    *p;
} InterpVal;

static InterpVal iv_int(int64_t v)  { return (InterpVal){.i = v}; }
static InterpVal iv_uint(uint64_t v){ return (InterpVal){.u = v}; }
static InterpVal iv_flt(double v)   { return (InterpVal){.f = v}; }
static InterpVal iv_ptr(void *v)    { return (InterpVal){.p = v}; }

/* Read a value as float, respecting dtype */
static double as_float(InterpVal v, PolyDType dt) {
  if (poly_dtype_is_float(dt)) return v.f;
  if (poly_dtype_is_unsigned(dt)) return (double)v.u;
  return (double)v.i;
}

/* Read a value as int64, respecting dtype */
static int64_t as_int(InterpVal v, PolyDType dt) {
  if (poly_dtype_is_float(dt)) return (int64_t)v.f;
  if (poly_dtype_is_unsigned(dt)) return (int64_t)v.u;
  return v.i;
}

static uint64_t as_uint(InterpVal v, PolyDType dt) {
  if (poly_dtype_is_float(dt)) return (uint64_t)v.f;
  if (poly_dtype_is_unsigned(dt)) return v.u;
  return (uint64_t)v.i;
}

/* ── Memory access helpers ───────────────────────────────────────────── */

static InterpVal mem_load(void *ptr, PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  int bs = s.bitsize;
  bool is_flt = poly_dtype_is_float(s);
  bool is_uns = poly_dtype_is_unsigned(s);

  if (is_flt && bs == 32)  { float v;    memcpy(&v, ptr, 4); return iv_flt(v); }
  if (is_flt && bs == 64)  { double v;   memcpy(&v, ptr, 8); return iv_flt(v); }
  if (!is_flt && !is_uns && bs == 32) { int32_t v;  memcpy(&v, ptr, 4); return iv_int(v); }
  if (is_uns && bs == 32)  { uint32_t v; memcpy(&v, ptr, 4); return iv_uint(v); }
  if (!is_flt && !is_uns && bs == 64) { int64_t v;  memcpy(&v, ptr, 8); return iv_int(v); }
  if (is_uns && bs == 64)  { uint64_t v; memcpy(&v, ptr, 8); return iv_uint(v); }
  if (!is_flt && !is_uns && bs == 8)  { int8_t v;   memcpy(&v, ptr, 1); return iv_int(v); }
  if (is_uns && bs == 8)   { uint8_t v;  memcpy(&v, ptr, 1); return iv_uint(v); }
  if (!is_flt && !is_uns && bs == 16) { int16_t v;  memcpy(&v, ptr, 2); return iv_int(v); }
  if (is_uns && bs == 16)  { uint16_t v; memcpy(&v, ptr, 2); return iv_uint(v); }
  if (poly_dtype_is_bool(s)) { uint8_t v; memcpy(&v, ptr, 1); return iv_int(v ? 1 : 0); }
  /* float16/bfloat16 */
  if (is_flt && bs == 16 && s.name && strcmp(s.name, "__bf16") == 0) {
    uint16_t bits; memcpy(&bits, ptr, 2);
    uint32_t f32 = (uint32_t)bits << 16;
    float fv; memcpy(&fv, &f32, 4);
    return iv_flt(fv);
  }
  if (is_flt && bs == 16) {
    uint16_t bits; memcpy(&bits, ptr, 2);
    /* IEEE 754 half -> float conversion */
    uint32_t sign = (uint32_t)(bits >> 15) << 31;
    uint32_t exp  = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
      if (mant == 0) f32 = sign;
      else { /* subnormal */ float sv = (float)mant / 1024.0f * (1.0f / 16384.0f); return iv_flt(sign ? -sv : sv); }
    } else if (exp == 31) { f32 = sign | 0x7F800000 | (mant << 13); }
    else { f32 = sign | ((exp + 112) << 23) | (mant << 13); }
    float fv; memcpy(&fv, &f32, 4);
    return iv_flt(fv);
  }
  /* bfloat16 already handled above */
  return iv_int(0);
}

static void mem_store(void *ptr, InterpVal v, PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  int bs = s.bitsize;
  bool is_flt = poly_dtype_is_float(s);
  bool is_uns = poly_dtype_is_unsigned(s);

  if (is_flt && bs == 32)  { float sv = (float)v.f; memcpy(ptr, &sv, 4); return; }
  if (is_flt && bs == 64)  { memcpy(ptr, &v.f, 8); return; }
  if (!is_flt && !is_uns && bs == 32) { int32_t sv = (int32_t)v.i; memcpy(ptr, &sv, 4); return; }
  if (is_uns && bs == 32)  { uint32_t sv = (uint32_t)v.u; memcpy(ptr, &sv, 4); return; }
  if (!is_flt && !is_uns && bs == 64) { memcpy(ptr, &v.i, 8); return; }
  if (is_uns && bs == 64)  { memcpy(ptr, &v.u, 8); return; }
  if (!is_flt && !is_uns && bs == 8)  { int8_t sv = (int8_t)v.i; memcpy(ptr, &sv, 1); return; }
  if (is_uns && bs == 8)   { uint8_t sv = (uint8_t)v.u; memcpy(ptr, &sv, 1); return; }
  if (!is_flt && !is_uns && bs == 16) { int16_t sv = (int16_t)v.i; memcpy(ptr, &sv, 2); return; }
  if (is_uns && bs == 16)  { uint16_t sv = (uint16_t)v.u; memcpy(ptr, &sv, 2); return; }
  if (poly_dtype_is_bool(s)) { uint8_t sv = v.i ? 1 : 0; memcpy(ptr, &sv, 1); return; }
  if (is_flt && bs == 16 && s.name && strcmp(s.name, "__bf16") == 0) {
    float fv = (float)v.f;
    uint32_t f32; memcpy(&f32, &fv, 4);
    uint16_t bf = (uint16_t)(f32 >> 16);
    memcpy(ptr, &bf, 2);
    return;
  }
  if (is_flt && bs == 16) {
    float fv = (float)v.f;
    uint32_t f32; memcpy(&f32, &fv, 4);
    uint32_t sign = (f32 >> 16) & 0x8000;
    int32_t exp = ((f32 >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f32 >> 13) & 0x3FF;
    uint16_t h;
    if (exp <= 0) h = (uint16_t)sign;
    else if (exp >= 31) h = (uint16_t)(sign | 0x7C00);
    else h = (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
    memcpy(ptr, &h, 2);
    return;
  }
  if (0) { /* bfloat16 already handled above */
    return;
  }
}

/* ── ALU evaluation ──────────────────────────────────────────────────── */

static InterpVal eval_alu(PolyOps op, PolyDType dt, InterpVal *srcs, int n_src) {
  bool is_flt = poly_dtype_is_float(dt);
  bool is_bool = poly_dtype_is_bool(dt);

  /* For comparison ops, use source dtype, not result dtype (which is bool) */
  PolyDType src_dt = dt;
  if (is_bool && n_src > 0) {
    /* Comparisons: result is bool but operands have their own dtype.
     * We don't have the source dtype here, so infer from the value.
     * For safety, treat comparison operands as float if the value looks
     * like it. This is handled by the caller passing source dtypes. */
  }
  (void)src_dt;

  double a = srcs[0].f, b = n_src > 1 ? srcs[1].f : 0, c = n_src > 2 ? srcs[2].f : 0;
  int64_t ai = srcs[0].i, bi = n_src > 1 ? srcs[1].i : 0;
  uint64_t au = srcs[0].u, bu = n_src > 1 ? srcs[1].u : 0;

  switch (op) {
  /* Unary */
  case POLY_OP_NEG:        return is_flt ? iv_flt(-a) : iv_int(-ai);
  case POLY_OP_SQRT:       return iv_flt(sqrt(a));
  case POLY_OP_RECIPROCAL: return iv_flt(1.0 / a);
  case POLY_OP_EXP2:       return iv_flt(exp2(a));
  case POLY_OP_LOG2:       return iv_flt(log2(a));
  case POLY_OP_SIN:        return iv_flt(sin(a));
  case POLY_OP_TRUNC:      return iv_flt(trunc(a));

  /* Binary float */
  case POLY_OP_ADD:  return is_flt ? iv_flt(a + b) : iv_int(ai + bi);
  case POLY_OP_SUB:  return is_flt ? iv_flt(a - b) : iv_int(ai - bi);
  case POLY_OP_MUL:  return is_flt ? iv_flt(a * b) : iv_int(ai * bi);
  case POLY_OP_FDIV: return iv_flt(a / b);
  case POLY_OP_IDIV: return bi != 0 ? iv_int(ai / bi) : iv_int(0);
  case POLY_OP_MOD:  return bi != 0 ? iv_int(ai % bi) : iv_int(0);
  case POLY_OP_MAX:  return is_flt ? iv_flt(a > b ? a : b) : iv_int(ai > bi ? ai : bi);
  case POLY_OP_POW:  return iv_flt(pow(a, b));

  /* Bitwise */
  case POLY_OP_SHL: return iv_uint(au << (bu & 63));
  case POLY_OP_SHR: return iv_uint(au >> (bu & 63));
  case POLY_OP_AND: return iv_uint(au & bu);
  case POLY_OP_OR:  return iv_uint(au | bu);
  case POLY_OP_XOR: return iv_uint(au ^ bu);

  /* Comparison (result is bool/int) */
  case POLY_OP_CMPLT: return is_flt ? iv_int(a < b ? 1 : 0) : iv_int(ai < bi ? 1 : 0);
  case POLY_OP_CMPNE: return is_flt ? iv_int(a != b ? 1 : 0) : iv_int(ai != bi ? 1 : 0);
  case POLY_OP_CMPEQ: return is_flt ? iv_int(a == b ? 1 : 0) : iv_int(ai == bi ? 1 : 0);

  /* Ternary */
  case POLY_OP_WHERE:  return srcs[0].i ? srcs[1] : srcs[2];
  case POLY_OP_MULACC: return is_flt ? iv_flt(a * b + c) : iv_int(ai * bi + srcs[2].i);

  /* THREEFRY: pass through as identity for interpreter
   * (decomposed by pm_decomp before linearization when no native support) */
  case POLY_OP_THREEFRY: return iv_uint(au ^ bu);

  default:
    fprintf(stderr, "polygrad: interp: unhandled ALU op %s\n", poly_op_name(op));
    return iv_int(0);
  }
}

/* ── Lookup helpers ──────────────────────────────────────────────────── */

/* Build a UOp pointer -> linearized index map.
 * Uses a simple open-addressing hash table. */
typedef struct {
  PolyUOp **keys;
  int *vals;
  int cap;
} UOpIndexMap;

static UOpIndexMap uop_index_map_new(int n) {
  int cap = n < 16 ? 16 : n * 2;
  UOpIndexMap m;
  m.cap = cap;
  m.keys = calloc((size_t)cap, sizeof(PolyUOp *));
  m.vals = calloc((size_t)cap, sizeof(int));
  return m;
}

static void uop_index_map_set(UOpIndexMap *m, PolyUOp *key, int val) {
  uint64_t h = ((uint64_t)(uintptr_t)key * 0x9E3779B97F4A7C15ULL) >> 32;
  for (int i = 0; i < m->cap; i++) {
    int idx = (int)((h + (uint64_t)i) % (uint64_t)m->cap);
    if (!m->keys[idx]) { m->keys[idx] = key; m->vals[idx] = val; return; }
    if (m->keys[idx] == key) { m->vals[idx] = val; return; }
  }
}

static int uop_index_map_get(const UOpIndexMap *m, PolyUOp *key) {
  uint64_t h = ((uint64_t)(uintptr_t)key * 0x9E3779B97F4A7C15ULL) >> 32;
  for (int i = 0; i < m->cap; i++) {
    int idx = (int)((h + (uint64_t)i) % (uint64_t)m->cap);
    if (!m->keys[idx]) return -1;
    if (m->keys[idx] == key) return m->vals[idx];
  }
  return -1;
}

static void uop_index_map_free(UOpIndexMap *m) {
  free(m->keys);
  free(m->vals);
}

/* Find matching END for a RANGE at position `start` */
static int find_matching_end(PolyUOp **lin, int n, int range_pos) {
  PolyUOp *range = lin[range_pos];
  for (int i = range_pos + 1; i < n; i++) {
    if (lin[i]->op == POLY_OP_END && lin[i]->n_src > 1 && lin[i]->src[1] == range)
      return i;
  }
  return n; /* fallback: end of sequence */
}

/* ── Region interpreter ──────────────────────────────────────────────── */

/*
 * Interpret a region of the linearized list [start, end).
 * Returns 0 on success, <0 on error.
 */
static int interp_region(PolyUOp **lin, int n_lin, int start, int end,
                         InterpVal *vals, void **args, int n_args,
                         const UOpIndexMap *idx_map) {
  for (int i = start; i < end; i++) {
    PolyUOp *u = lin[i];

    switch (u->op) {

    case POLY_OP_PARAM:
      vals[i] = iv_ptr(args[u->arg.i]);
      break;

    case POLY_OP_DEFINE_VAR: {
      /* DEFINE_VAR params come after buffer params.
       * The value is passed as int* in args. */
      int *vp = (int *)args[u->arg.i];
      vals[i] = iv_int(vp ? *vp : 0);
      break;
    }

    case POLY_OP_CONST:
      if (poly_dtype_is_float(u->dtype))
        vals[i] = iv_flt(u->arg.f);
      else if (poly_dtype_is_bool(u->dtype))
        vals[i] = iv_int(u->arg.b ? 1 : 0);
      else
        vals[i] = iv_int(u->arg.i);
      break;

    case POLY_OP_DEFINE_REG: {
      /* Allocate a small buffer for the register accumulator.
       * DEFINE_REG dtype is ptr<scalar, REG>. */
      PolyDType base = poly_dtype_scalar(u->dtype);
      int sz = poly_dtype_itemsize(base);
      if (sz < 4) sz = 4;
      void *reg_buf = calloc(1, (size_t)sz);
      vals[i] = iv_ptr(reg_buf);
      break;
    }

    case POLY_OP_RANGE: {
      /* Get loop bound from src[0] */
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      if (src0 < 0) { fprintf(stderr, "polygrad: interp: RANGE src0 not found\n"); return -1; }
      int bound = (int)as_int(vals[src0], u->src[0]->dtype);
      int end_pos = find_matching_end(lin, n_lin, i);

      for (int ridx = 0; ridx < bound; ridx++) {
        vals[i] = iv_int(ridx);
        int rc = interp_region(lin, n_lin, i + 1, end_pos, vals, args, n_args, idx_map);
        if (rc < 0) return rc;
      }
      i = end_pos; /* skip past END */
      break;
    }

    case POLY_OP_END:
      /* Handled by RANGE recursion. If we reach here, it's a no-op. */
      break;

    case POLY_OP_AFTER: {
      /* Pass through src[0]'s value. Dependencies are structural. */
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      vals[i] = vals[src0];
      break;
    }

    case POLY_OP_INDEX: {
      /* Pointer arithmetic: base + offset * element_size */
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      int src1 = uop_index_map_get(idx_map, u->src[1]);
      if (src0 < 0 || src1 < 0) {
        fprintf(stderr, "polygrad: interp: INDEX src not found\n");
        return -1;
      }
      PolyDType base_dt = poly_dtype_scalar(u->dtype);
      int itemsize = poly_dtype_itemsize(base_dt);
      if (itemsize < 1) itemsize = 1;
      char *base_ptr = (char *)vals[src0].p;
      int64_t offset = as_int(vals[src1], u->src[1]->dtype);
      vals[i] = iv_ptr(base_ptr + offset * itemsize);
      break;
    }

    case POLY_OP_LOAD: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      if (src0 < 0) { fprintf(stderr, "polygrad: interp: LOAD src not found\n"); return -1; }
      vals[i] = mem_load(vals[src0].p, u->dtype);
      break;
    }

    case POLY_OP_STORE: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      int src1 = uop_index_map_get(idx_map, u->src[1]);
      if (src0 < 0 || src1 < 0) {
        fprintf(stderr, "polygrad: interp: STORE src not found\n");
        return -1;
      }
      /* Determine store dtype from the target pointer's scalar type */
      PolyDType store_dt;
      if (u->src[0]->op == POLY_OP_DEFINE_LOCAL)
        store_dt = poly_dtype_scalar(u->src[0]->dtype);
      else
        store_dt = poly_dtype_scalar(u->src[0]->dtype);
      mem_store(vals[src0].p, vals[src1], store_dt);
      break;
    }

    case POLY_OP_DEFINE_LOCAL: {
      /* Stack-allocated accumulator scalar */
      PolyDType base = poly_dtype_scalar(u->dtype);
      int sz = poly_dtype_itemsize(base);
      if (sz < 4) sz = 4;
      void *buf = calloc(1, (size_t)sz);
      vals[i] = iv_ptr(buf);
      break;
    }

    case POLY_OP_CAST: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      PolyDType src_dt = u->src[0]->dtype;
      if (poly_dtype_is_float(u->dtype)) {
        vals[i] = iv_flt(as_float(vals[src0], src_dt));
      } else if (poly_dtype_is_unsigned(u->dtype)) {
        vals[i] = iv_uint(as_uint(vals[src0], src_dt));
      } else {
        vals[i] = iv_int(as_int(vals[src0], src_dt));
      }
      break;
    }

    case POLY_OP_BITCAST: {
      /* Reinterpret bits: same bit pattern, different type interpretation.
       * Must go through a raw byte copy since InterpVal stores int64/double
       * which have different bit layouts than int32/float32. */
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      PolyDType src_dt = poly_dtype_scalar(u->src[0]->dtype);
      PolyDType dst_dt = poly_dtype_scalar(u->dtype);
      int src_bs = src_dt.bitsize;
      int dst_bs = dst_dt.bitsize;

      if (src_bs == 32 && dst_bs == 32) {
        /* int32 <-> float32 bitcast */
        uint32_t bits;
        if (poly_dtype_is_float(src_dt)) {
          float fv = (float)vals[src0].f;
          memcpy(&bits, &fv, 4);
        } else {
          bits = (uint32_t)vals[src0].i;
        }
        if (poly_dtype_is_float(dst_dt)) {
          float fv; memcpy(&fv, &bits, 4);
          vals[i] = iv_flt(fv);
        } else {
          vals[i] = iv_int((int32_t)bits);
        }
      } else if (src_bs == 64 && dst_bs == 64) {
        /* int64 <-> float64 bitcast */
        uint64_t bits;
        if (poly_dtype_is_float(src_dt)) {
          memcpy(&bits, &vals[src0].f, 8);
        } else {
          bits = vals[src0].u;
        }
        if (poly_dtype_is_float(dst_dt)) {
          double fv; memcpy(&fv, &bits, 8);
          vals[i] = iv_flt(fv);
        } else {
          vals[i] = iv_uint(bits);
        }
      } else {
        /* Fallback: copy raw union bytes */
        vals[i] = vals[src0];
      }
      break;
    }

    case POLY_OP_SINK:
      /* Terminal node, no action needed */
      break;

    case POLY_OP_NOOP:
    case POLY_OP_GROUP:
      break;

    default: {
      /* ALU operations */
      if (u->op >= POLY_OP_EXP2 && u->op <= POLY_OP_MULACC) {
        InterpVal src_vals[3];
        PolyDType alu_dt = u->dtype;

        /* For comparison ops, use source dtype for evaluation */
        bool is_comparison = (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPNE ||
                              u->op == POLY_OP_CMPEQ);
        if (is_comparison && u->n_src > 0)
          alu_dt = u->src[0]->dtype;

        for (int j = 0; j < u->n_src && j < 3; j++) {
          int sj = uop_index_map_get(idx_map, u->src[j]);
          if (sj >= 0) src_vals[j] = vals[sj];
          else src_vals[j] = iv_int(0);
        }
        vals[i] = eval_alu(u->op, alu_dt, src_vals, u->n_src);
      } else {
        fprintf(stderr, "polygrad: interp: unhandled op %s (%d)\n",
                poly_op_name(u->op), u->op);
      }
      break;
    }
    }
  }
  return 0;
}

/* ── Public API ──────────────────────────────────────────────────────── */

int poly_interp_eval(PolyUOp **lin, int n_lin, void **args, int n_args) {
  if (!lin || n_lin <= 0) return -1;

  InterpVal *vals = calloc((size_t)n_lin, sizeof(InterpVal));
  if (!vals) return -1;

  /* Build UOp pointer -> index map for O(1) source lookup */
  UOpIndexMap idx_map = uop_index_map_new(n_lin);
  for (int i = 0; i < n_lin; i++)
    uop_index_map_set(&idx_map, lin[i], i);

  int ret = interp_region(lin, n_lin, 0, n_lin, vals, args, n_args, &idx_map);

  uop_index_map_free(&idx_map);

  /* Free any DEFINE_REG / DEFINE_LOCAL allocations */
  for (int i = 0; i < n_lin; i++) {
    if ((lin[i]->op == POLY_OP_DEFINE_REG || lin[i]->op == POLY_OP_DEFINE_LOCAL)
        && vals[i].p) {
      free(vals[i].p);
    }
  }

  free(vals);
  return ret;
}
