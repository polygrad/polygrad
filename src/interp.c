/*
 * interp.c -- Interpreter backend for polygrad
 *
 * Design: linearize-then-interpret.
 *
 * The interpreter receives a linearized UOp sequence (the output of
 * poly_linearize, which includes the full codegen pipeline:
 * sym -> pm_reduce -> pm_decomp -> pm_transcendental -> pm_add_control_flow).
 *
 * This means the interpreter sees the SAME decomposed ops as the C renderer:
 * EXP2 is a 73-op Horner polynomial, LOG2 is polynomial + ilogb2k, etc.
 * The advantage is that the interpreter validates the entire codegen pipeline,
 * not just the scheduling. If CPU and INTERP produce the same result, we know
 * scheduling + codegen + rendering are all correct.
 *
 * The interpreter does NOT need an external compiler (no fork, no clang,
 * no dlopen). It runs anywhere the C core compiles, including Emscripten.
 *
 * Value model: InterpLane (scalar union) + InterpVal (lane-array).
 * Scalar UOps (count=1) use an inline lane with zero allocation overhead.
 * Vector UOps (count>1) use slices from a pre-allocated lane arena.
 * This matches tinygrad's Python interpreter which represents values as
 * lists of scalars with broadcasting.
 *
 * Execution model:
 *   - The linearized list is a flat sequence of UOps in execution order.
 *   - RANGE/END pairs define loops. When the interpreter hits a RANGE,
 *     it finds the matching END and recursively interprets the region
 *     [RANGE+1, END) for each iteration. Nested loops work naturally.
 *   - Each UOp produces one InterpVal stored in a per-UOp value table.
 *   - Memory ops (LOAD/STORE) access buffers via typed memcpy per lane.
 *   - BITCAST preserves bit patterns across int32/float32 and int64/float64
 *     boundaries using explicit memcpy.
 */

#include "interp.h"
#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Lane value (scalar) ─────────────────────────────────────────────── */

typedef union {
  int64_t  i;
  uint64_t u;
  double   f;
  void    *p;
} InterpLane;

static InterpLane il_int(int64_t v)  { return (InterpLane){.i = v}; }
static InterpLane il_uint(uint64_t v){ return (InterpLane){.u = v}; }
static InterpLane il_flt(double v)   { return (InterpLane){.f = v}; }
static InterpLane il_ptr(void *v)    { return (InterpLane){.p = v}; }

/* ── Vector value (multi-lane) ───────────────────────────────────────── */
/*
 * Each linearized UOp produces one InterpVal.
 * Scalar UOps (count=1) store their value in inline0 with lanes == &inline0.
 * Vector UOps (count>1) use a pre-allocated slice from the lane arena.
 * This avoids per-op malloc while supporting arbitrary vector widths.
 */

typedef struct {
  uint16_t   count;     /* lane count, 1 for scalar */
  InterpLane inline0;   /* storage for count==1 (no allocation) */
  InterpLane *lanes;    /* == &inline0 when count==1, else arena slice */
} InterpVal;

/* Initialize a scalar InterpVal */
static InterpVal iv_scalar(InterpLane lane) {
  InterpVal v;
  v.count = 1;
  v.inline0 = lane;
  v.lanes = &v.inline0;
  return v;
}

/* After copying an InterpVal, fix up the lanes pointer if it's scalar.
 * Required because the inline0 pointer becomes stale after copy. */
static void iv_fixup(InterpVal *v) {
  if (v->count == 1)
    v->lanes = &v->inline0;
}

/* Get lane k from val with scalar broadcast */
static InterpLane iv_get(const InterpVal *v, int k) {
  return v->lanes[v->count == 1 ? 0 : k];
}

/* Set lane k of val */
static void iv_set(InterpVal *v, int k, InterpLane lane) {
  v->lanes[k] = lane;
}

/* ── Scalar conversion helpers ───────────────────────────────────────── */

static double as_float(InterpLane v, PolyDType dt) {
  if (poly_dtype_is_float(dt)) return v.f;
  if (poly_dtype_is_unsigned(dt)) return (double)v.u;
  return (double)v.i;
}

static int64_t as_int(InterpLane v, PolyDType dt) {
  if (poly_dtype_is_float(dt)) return (int64_t)v.f;
  if (poly_dtype_is_unsigned(dt)) return (int64_t)v.u;
  return v.i;
}

static uint64_t as_uint(InterpLane v, PolyDType dt) {
  if (poly_dtype_is_float(dt)) return (uint64_t)v.f;
  if (poly_dtype_is_unsigned(dt)) return v.u;
  return (uint64_t)v.i;
}

/* ── Memory access (per scalar lane) ─────────────────────────────────── */
/*
 * Type dispatch uses poly_dtype_is_float/is_unsigned + bitsize instead
 * of poly_dtype_eq, because codegen-rewritten UOps may carry dtypes
 * that are structurally equivalent to POLY_FLOAT32 but not pointer-equal
 * (e.g. different ptr_size or count fields from intermediate rewrites).
 */

static InterpLane mem_load_scalar(void *ptr, PolyDType s) {
  int bs = s.bitsize;
  bool is_flt = poly_dtype_is_float(s);
  bool is_uns = poly_dtype_is_unsigned(s);

  if (is_flt && bs == 32)             { float v;    memcpy(&v, ptr, 4); return il_flt(v); }
  if (is_flt && bs == 64)             { double v;   memcpy(&v, ptr, 8); return il_flt(v); }
  if (!is_flt && !is_uns && bs == 32) { int32_t v;  memcpy(&v, ptr, 4); return il_int(v); }
  if (is_uns && bs == 32)             { uint32_t v; memcpy(&v, ptr, 4); return il_uint(v); }
  if (!is_flt && !is_uns && bs == 64) { int64_t v;  memcpy(&v, ptr, 8); return il_int(v); }
  if (is_uns && bs == 64)             { uint64_t v; memcpy(&v, ptr, 8); return il_uint(v); }
  if (!is_flt && !is_uns && bs == 8)  { int8_t v;   memcpy(&v, ptr, 1); return il_int(v); }
  if (is_uns && bs == 8)              { uint8_t v;  memcpy(&v, ptr, 1); return il_uint(v); }
  if (!is_flt && !is_uns && bs == 16) { int16_t v;  memcpy(&v, ptr, 2); return il_int(v); }
  if (is_uns && bs == 16)             { uint16_t v; memcpy(&v, ptr, 2); return il_uint(v); }
  if (poly_dtype_is_bool(s))          { uint8_t v;  memcpy(&v, ptr, 1); return il_int(v ? 1 : 0); }

  /* bfloat16: top 16 bits of float32 */
  if (is_flt && bs == 16 && s.name && strcmp(s.name, "__bf16") == 0) {
    uint16_t bits; memcpy(&bits, ptr, 2);
    uint32_t f32 = (uint32_t)bits << 16;
    float fv; memcpy(&fv, &f32, 4);
    return il_flt(fv);
  }
  /* float16: IEEE 754 half-precision */
  if (is_flt && bs == 16) {
    uint16_t bits; memcpy(&bits, ptr, 2);
    uint32_t sign = (uint32_t)(bits >> 15) << 31;
    uint32_t exp  = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
      if (mant == 0) f32 = sign;
      else { float sv = (float)mant / 1024.0f * (1.0f / 16384.0f); return il_flt(sign ? -sv : sv); }
    } else if (exp == 31) { f32 = sign | 0x7F800000 | (mant << 13); }
    else { f32 = sign | ((exp + 112) << 23) | (mant << 13); }
    float fv; memcpy(&fv, &f32, 4);
    return il_flt(fv);
  }
  return il_int(0);
}

static void mem_store_scalar(void *ptr, InterpLane v, PolyDType s) {
  int bs = s.bitsize;
  bool is_flt = poly_dtype_is_float(s);
  bool is_uns = poly_dtype_is_unsigned(s);

  if (is_flt && bs == 32)             { float sv = (float)v.f; memcpy(ptr, &sv, 4); return; }
  if (is_flt && bs == 64)             { memcpy(ptr, &v.f, 8); return; }
  if (!is_flt && !is_uns && bs == 32) { int32_t sv = (int32_t)v.i; memcpy(ptr, &sv, 4); return; }
  if (is_uns && bs == 32)             { uint32_t sv = (uint32_t)v.u; memcpy(ptr, &sv, 4); return; }
  if (!is_flt && !is_uns && bs == 64) { memcpy(ptr, &v.i, 8); return; }
  if (is_uns && bs == 64)             { memcpy(ptr, &v.u, 8); return; }
  if (!is_flt && !is_uns && bs == 8)  { int8_t sv = (int8_t)v.i; memcpy(ptr, &sv, 1); return; }
  if (is_uns && bs == 8)              { uint8_t sv = (uint8_t)v.u; memcpy(ptr, &sv, 1); return; }
  if (!is_flt && !is_uns && bs == 16) { int16_t sv = (int16_t)v.i; memcpy(ptr, &sv, 2); return; }
  if (is_uns && bs == 16)             { uint16_t sv = (uint16_t)v.u; memcpy(ptr, &sv, 2); return; }
  if (poly_dtype_is_bool(s))          { uint8_t sv = v.i ? 1 : 0; memcpy(ptr, &sv, 1); return; }

  /* bfloat16 */
  if (is_flt && bs == 16 && s.name && strcmp(s.name, "__bf16") == 0) {
    float fv = (float)v.f;
    uint32_t f32; memcpy(&f32, &fv, 4);
    uint16_t bf = (uint16_t)(f32 >> 16);
    memcpy(ptr, &bf, 2);
    return;
  }
  /* float16 */
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
}

/* ── ALU evaluation (scalar, per-lane) ───────────────────────────────── */
/*
 * Evaluates a single ALU op on scalar lane operands.
 * dt is the effective dtype for the operation:
 *   - For most ops, this is the UOp's own dtype (scalar).
 *   - For comparisons (CMPLT, CMPNE, CMPEQ), the caller passes the
 *     source dtype instead, since the result dtype is bool but the
 *     comparison semantics depend on the operand type (float vs int).
 */

static InterpLane eval_alu(PolyOps op, PolyDType dt, InterpLane *srcs, int n_src) {
  bool is_flt = poly_dtype_is_float(dt);

  double a = srcs[0].f, b = n_src > 1 ? srcs[1].f : 0, c = n_src > 2 ? srcs[2].f : 0;
  int64_t ai = srcs[0].i, bi = n_src > 1 ? srcs[1].i : 0;
  uint64_t au = srcs[0].u, bu = n_src > 1 ? srcs[1].u : 0;

  switch (op) {
  /* Unary */
  case POLY_OP_NEG:
    if (poly_dtype_is_bool(poly_dtype_scalar(dt))) return il_int(srcs[0].i ? 0 : 1);
    return is_flt ? il_flt(-a) : il_int(-ai);
  case POLY_OP_SQRT:       return il_flt(sqrt(a));
  case POLY_OP_RECIPROCAL: return il_flt(1.0 / a);
  case POLY_OP_EXP2:       return il_flt(exp2(a));
  case POLY_OP_LOG2:       return il_flt(log2(a));
  case POLY_OP_SIN:        return il_flt(sin(a));
  case POLY_OP_TRUNC:      return il_flt(trunc(a));

  /* Binary arithmetic — wrapping add/sub/mul to avoid signed overflow UB.
   * tinygrad's Python uses arbitrary-precision ints; in C we wrap via unsigned cast. */
  case POLY_OP_ADD:  return is_flt ? il_flt(a + b) : il_int((int64_t)((uint64_t)ai + (uint64_t)bi));
  case POLY_OP_SUB:  return is_flt ? il_flt(a - b) : il_int((int64_t)((uint64_t)ai - (uint64_t)bi));
  case POLY_OP_MUL:  return is_flt ? il_flt(a * b) : il_int((int64_t)((uint64_t)ai * (uint64_t)bi));
  case POLY_OP_FDIV: return il_flt(a / b);
  case POLY_OP_IDIV: return bi != 0 ? il_int(ai / bi) : il_int(0);
  case POLY_OP_MOD:  return bi != 0 ? il_int(ai % bi) : il_int(0);
  case POLY_OP_MAX:  return is_flt ? il_flt(a > b ? a : b) : il_int(ai > bi ? ai : bi);
  case POLY_OP_POW:  return il_flt(pow(a, b));

  /* Bitwise (always unsigned semantics) */
  case POLY_OP_SHL: return il_uint(au << (bu & 63));
  case POLY_OP_SHR: return il_uint(au >> (bu & 63));
  case POLY_OP_AND: return il_uint(au & bu);
  case POLY_OP_OR:  return il_uint(au | bu);
  case POLY_OP_XOR: return il_uint(au ^ bu);

  /* Comparison (result is 0 or 1) */
  case POLY_OP_CMPLT: return is_flt ? il_int(a < b ? 1 : 0) : il_int(ai < bi ? 1 : 0);
  case POLY_OP_CMPNE: return is_flt ? il_int(a != b ? 1 : 0) : il_int(ai != bi ? 1 : 0);
  case POLY_OP_CMPEQ: return is_flt ? il_int(a == b ? 1 : 0) : il_int(ai == bi ? 1 : 0);

  /* Ternary */
  case POLY_OP_WHERE:  return srcs[0].i ? srcs[1] : srcs[2];
  case POLY_OP_MULACC: return is_flt ? il_flt(a * b + c) : il_int(ai * bi + srcs[2].i);

  /* THREEFRY is decomposed by pm_decomp before linearization.
   * If it somehow survives, XOR is a reasonable fallback. */
  case POLY_OP_THREEFRY: return il_uint(au ^ bu);

  default:
    fprintf(stderr, "polygrad: interp: unhandled ALU op %s\n", poly_op_name(op));
    return il_int(0);
  }
}

/* Truncate integer lane to dtype width (matches alu.c truncate_result). */
static InterpLane interp_truncate_lane(InterpLane v, PolyDType dt) {
  if (poly_dtype_is_float(dt) || poly_dtype_is_bool(dt)) return v;
  int bits = dt.bitsize;
  if (bits <= 0 || bits >= 64) return v;
  if (poly_dtype_is_unsigned(dt)) {
    v.u &= ((uint64_t)1 << bits) - 1;
  } else {
    uint64_t mask = ((uint64_t)1 << bits) - 1;
    v.i &= (int64_t)mask;
    if (v.i & ((int64_t)1 << (bits - 1)))
      v.i |= ~(int64_t)mask; /* sign extend */
  }
  return v;
}

/* BITCAST one lane: reinterpret bits without value conversion. */
static InterpLane bitcast_lane(InterpLane src, PolyDType src_dt, PolyDType dst_dt) {
  if (src_dt.bitsize == 32 && dst_dt.bitsize == 32) {
    uint32_t bits;
    if (poly_dtype_is_float(src_dt)) {
      float fv = (float)src.f;
      memcpy(&bits, &fv, 4);
    } else {
      bits = (uint32_t)src.i;
    }
    if (poly_dtype_is_float(dst_dt)) {
      float fv; memcpy(&fv, &bits, 4);
      return il_flt(fv);
    } else {
      return il_int((int32_t)bits);
    }
  } else if (src_dt.bitsize == 64 && dst_dt.bitsize == 64) {
    uint64_t bits;
    if (poly_dtype_is_float(src_dt))
      memcpy(&bits, &src.f, 8);
    else
      bits = src.u;
    if (poly_dtype_is_float(dst_dt)) {
      double fv; memcpy(&fv, &bits, 8);
      return il_flt(fv);
    } else {
      return il_uint(bits);
    }
  }
  /* Mismatched sizes: pass through (best effort) */
  return src;
}

/* CAST one lane: value conversion with truncation. */
static InterpLane cast_lane(InterpLane src, PolyDType src_dt, PolyDType dst_dt) {
  if (poly_dtype_is_float(dst_dt))
    return il_flt(as_float(src, src_dt));
  else if (poly_dtype_is_unsigned(dst_dt))
    return interp_truncate_lane(il_uint(as_uint(src, src_dt)), dst_dt);
  else
    return interp_truncate_lane(il_int(as_int(src, src_dt)), dst_dt);
}

/* ── UOp index map ───────────────────────────────────────────────────── */

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

/* Find the END node that closes a given RANGE node at position range_pos */
static int find_matching_end(PolyUOp **lin, int n, int range_pos) {
  PolyUOp *range = lin[range_pos];
  for (int i = range_pos + 1; i < n; i++) {
    if (lin[i]->op == POLY_OP_END && lin[i]->n_src > 1 && lin[i]->src[1] == range)
      return i;
  }
  return n;
}

/* ── Region interpreter ──────────────────────────────────────────────── */

static int interp_region(PolyUOp **lin, int n_lin, int start, int end,
                         InterpVal *vals, void **args, int n_args,
                         const UOpIndexMap *idx_map, InterpLane *arena) {
  for (int i = start; i < end; i++) {
    PolyUOp *u = lin[i];

    switch (u->op) {

    case POLY_OP_PARAM:
      if (u->arg.i >= 0 && u->arg.i < n_args) {
        vals[i] = iv_scalar(il_ptr(args[u->arg.i]));
        iv_fixup(&vals[i]);
      } else {
        fprintf(stderr, "polygrad: interp: PARAM index %lld out of range (n_args=%d)\n",
                (long long)u->arg.i, n_args);
        return -1;
      }
      break;

    case POLY_OP_DEFINE_VAR: {
      int n_buf_params = 0, var_ord = 0;
      for (int j = 0; j < i; j++) {
        if (lin[j]->op == POLY_OP_PARAM) n_buf_params++;
        if (lin[j]->op == POLY_OP_DEFINE_VAR) var_ord++;
      }
      int slot = n_buf_params + var_ord;
      if (slot >= 0 && slot < n_args && args[slot]) {
        int *vp = (int *)args[slot];
        vals[i] = iv_scalar(il_int(*vp));
      } else {
        vals[i] = iv_scalar(il_int(0));
      }
      iv_fixup(&vals[i]);
      break;
    }

    case POLY_OP_CONST: {
      InterpLane lane;
      if (poly_dtype_is_float(u->dtype))
        lane = il_flt(u->arg.f);
      else if (poly_dtype_is_bool(u->dtype))
        lane = il_int(u->arg.b ? 1 : 0);
      else
        lane = il_int(u->arg.i);
      /* Fill all lanes with the same value */
      int cnt = u->dtype.count > 0 ? u->dtype.count : 1;
      vals[i].count = (uint16_t)cnt;
      if (cnt == 1) {
        vals[i].inline0 = lane;
        vals[i].lanes = &vals[i].inline0;
      } else {
        /* lanes pointer was set during arena pre-alloc */
        for (int k = 0; k < cnt; k++)
          vals[i].lanes[k] = lane;
      }
      break;
    }

    case POLY_OP_DEFINE_REG: {
      PolyDType base = poly_dtype_scalar(u->dtype);
      int sz = poly_dtype_itemsize(base);
      if (sz < 1) sz = 1;
      /* Include vector width in allocation size */
      int cnt = base.count > 1 ? base.count : 1;
      vals[i] = iv_scalar(il_ptr(calloc(1, (size_t)(sz * cnt))));
      iv_fixup(&vals[i]);
      break;
    }

    case POLY_OP_RANGE: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      if (src0 < 0) { fprintf(stderr, "polygrad: interp: RANGE bound not found\n"); return -1; }
      int bound = (int)as_int(iv_get(&vals[src0], 0), u->src[0]->dtype);
      int end_pos = find_matching_end(lin, n_lin, i);

      for (int ridx = 0; ridx < bound; ridx++) {
        vals[i] = iv_scalar(il_int(ridx));
        iv_fixup(&vals[i]);
        int rc = interp_region(lin, n_lin, i + 1, end_pos, vals, args, n_args, idx_map, arena);
        if (rc < 0) return rc;
      }
      i = end_pos; /* skip past END */
      break;
    }

    case POLY_OP_END:
      break;

    case POLY_OP_AFTER: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      /* Deep copy: replicate lanes, fix pointer */
      vals[i].count = vals[src0].count;
      if (vals[i].count == 1) {
        vals[i].inline0 = vals[src0].lanes[0];
        vals[i].lanes = &vals[i].inline0;
      } else {
        /* lanes pointer was pre-assigned during arena setup */
        for (int k = 0; k < vals[i].count; k++)
          vals[i].lanes[k] = vals[src0].lanes[k];
      }
      break;
    }

    case POLY_OP_INDEX: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      int src1 = uop_index_map_get(idx_map, u->src[1]);
      if (src0 < 0 || src1 < 0) {
        fprintf(stderr, "polygrad: interp: INDEX src not found\n");
        return -1;
      }
      PolyDType base_dt = poly_dtype_scalar(u->dtype);
      int itemsize = poly_dtype_itemsize(base_dt);
      if (itemsize < 1) itemsize = 1;
      /* Include vector count in stride for vector loads */
      int vec_count = u->dtype.count > 1 ? u->dtype.count : 1;
      char *base_ptr = (char *)iv_get(&vals[src0], 0).p;
      int64_t offset = as_int(iv_get(&vals[src1], 0), u->src[1]->dtype);
      vals[i] = iv_scalar(il_ptr(base_ptr + offset * itemsize * vec_count));
      iv_fixup(&vals[i]);
      break;
    }

    case POLY_OP_LOAD: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      if (src0 < 0) { fprintf(stderr, "polygrad: interp: LOAD src not found\n"); return -1; }

      /* Gated load check */
      PolyUOp *ld_idx = poly_find_index_through_cast(u->src[0]);
      if (ld_idx && ld_idx->n_src >= 3) {
        int gate_i = uop_index_map_get(idx_map, ld_idx->src[2]);
        if (gate_i >= 0 && !as_int(iv_get(&vals[gate_i], 0), ld_idx->src[2]->dtype)) {
          /* Gate is false: use alt value or zero */
          int cnt = u->dtype.count > 0 ? u->dtype.count : 1;
          vals[i].count = (uint16_t)cnt;
          if (cnt == 1) vals[i].lanes = &vals[i].inline0;
          if (u->n_src >= 2) {
            int alt_i = uop_index_map_get(idx_map, u->src[1]);
            if (alt_i >= 0) {
              for (int k = 0; k < cnt; k++)
                iv_set(&vals[i], k, iv_get(&vals[alt_i], k));
            } else {
              for (int k = 0; k < cnt; k++)
                iv_set(&vals[i], k, il_flt(0.0));
            }
          } else {
            for (int k = 0; k < cnt; k++)
              iv_set(&vals[i], k, il_flt(0.0));
          }
          break;
        }
      }

      /* Load dt.count contiguous scalars */
      PolyDType scalar_dt = poly_dtype_scalar(u->dtype);
      int cnt = u->dtype.count > 0 ? u->dtype.count : 1;
      int scalar_size = poly_dtype_itemsize(scalar_dt);
      if (scalar_size < 1) scalar_size = 1;
      char *ptr = (char *)iv_get(&vals[src0], 0).p;

      vals[i].count = (uint16_t)cnt;
      if (cnt == 1) vals[i].lanes = &vals[i].inline0;
      for (int k = 0; k < cnt; k++)
        iv_set(&vals[i], k, mem_load_scalar(ptr + k * scalar_size, scalar_dt));
      break;
    }

    case POLY_OP_STORE: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      int src1 = uop_index_map_get(idx_map, u->src[1]);
      if (src0 < 0 || src1 < 0) {
        fprintf(stderr, "polygrad: interp: STORE src not found\n");
        return -1;
      }
      PolyDType store_dt = poly_dtype_scalar(u->src[0]->dtype);
      int scalar_size = poly_dtype_itemsize(store_dt);
      if (scalar_size < 1) scalar_size = 1;
      char *ptr = (char *)iv_get(&vals[src0], 0).p;
      int cnt = vals[src1].count;
      for (int k = 0; k < cnt; k++)
        mem_store_scalar(ptr + k * scalar_size, iv_get(&vals[src1], k), store_dt);
      break;
    }

    case POLY_OP_DEFINE_LOCAL: {
      PolyDType base = poly_dtype_scalar(u->dtype);
      int sz = poly_dtype_itemsize(base);
      if (sz < 1) sz = 1;
      int cnt = base.count > 1 ? base.count : 1;
      vals[i] = iv_scalar(il_ptr(calloc(1, (size_t)(sz * cnt))));
      iv_fixup(&vals[i]);
      break;
    }

    case POLY_OP_CAST: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      PolyDType src_dt = poly_dtype_scalar(u->src[0]->dtype);
      PolyDType dst_dt = poly_dtype_scalar(u->dtype);
      int dst_cnt = u->dtype.count > 0 ? u->dtype.count : 1;
      int src_cnt = vals[src0].count;

      /* Pointer CAST: pass through (used for type coercion on INDEX) */
      if (u->dtype.is_ptr) {
        vals[i] = vals[src0];
        iv_fixup(&vals[i]);
        break;
      }

      vals[i].count = (uint16_t)dst_cnt;
      if (dst_cnt == 1) vals[i].lanes = &vals[i].inline0;

      for (int k = 0; k < dst_cnt; k++) {
        InterpLane lane = iv_get(&vals[src0], k < src_cnt ? k : 0);
        iv_set(&vals[i], k, cast_lane(lane, src_dt, dst_dt));
      }
      break;
    }

    case POLY_OP_BITCAST: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      PolyDType src_dt = poly_dtype_scalar(u->src[0]->dtype);
      PolyDType dst_dt = poly_dtype_scalar(u->dtype);
      int dst_cnt = u->dtype.count > 0 ? u->dtype.count : 1;
      int src_cnt = vals[src0].count;

      vals[i].count = (uint16_t)dst_cnt;
      if (dst_cnt == 1) vals[i].lanes = &vals[i].inline0;

      for (int k = 0; k < dst_cnt; k++) {
        InterpLane lane = iv_get(&vals[src0], k < src_cnt ? k : 0);
        iv_set(&vals[i], k, bitcast_lane(lane, src_dt, dst_dt));
      }
      break;
    }

    case POLY_OP_GEP: {
      /* Lane extraction from vector source.
       * arg is an int_tuple of lane indices.
       * src[0] is the vector to extract from. */
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      if (src0 < 0) { vals[i] = iv_scalar(il_int(0)); iv_fixup(&vals[i]); break; }

      int n_idxs = (u->arg.kind == POLY_ARG_INT_TUPLE) ? u->arg.int_tuple.n : 0;

      if (n_idxs == 1) {
        /* Single lane extraction: out = src.lanes[idx] */
        int lane_idx = (int)u->arg.int_tuple.vals[0];
        InterpLane lane = iv_get(&vals[src0], lane_idx);
        vals[i] = iv_scalar(lane);
        iv_fixup(&vals[i]);
      } else if (n_idxs > 1) {
        /* Multi-lane GEP: out.lanes[k] = src.lanes[idxs[k]] */
        vals[i].count = (uint16_t)n_idxs;
        if (n_idxs == 1) vals[i].lanes = &vals[i].inline0;
        /* lanes pointer from arena pre-alloc */
        for (int k = 0; k < n_idxs; k++) {
          int lane_idx = (int)u->arg.int_tuple.vals[k];
          iv_set(&vals[i], k, iv_get(&vals[src0], lane_idx));
        }
      } else {
        /* Fallback: passthrough */
        vals[i] = vals[src0];
        iv_fixup(&vals[i]);
      }
      break;
    }

    case POLY_OP_VECTORIZE: {
      /* Construct vector from scalar sources.
       * out.count = n_src, out.lanes[k] = src[k].lanes[0] */
      int cnt = u->n_src;
      if (cnt < 1) cnt = 1;
      vals[i].count = (uint16_t)cnt;
      if (cnt == 1) vals[i].lanes = &vals[i].inline0;
      for (int k = 0; k < cnt; k++) {
        int sk = uop_index_map_get(idx_map, u->src[k]);
        iv_set(&vals[i], k, (sk >= 0) ? iv_get(&vals[sk], 0) : il_int(0));
      }
      break;
    }

    case POLY_OP_SINK:
    case POLY_OP_NOOP:
    case POLY_OP_GROUP:
      break;

    default: {
      /* ALU operations: dispatch to eval_alu with lane loop */
      if (u->op >= POLY_OP_EXP2 && u->op <= POLY_OP_MULACC) {
        PolyDType alu_dt = poly_dtype_scalar(u->dtype);
        int out_cnt = u->dtype.count > 0 ? u->dtype.count : 1;

        /* Comparisons produce bool but operate on source dtype */
        bool is_cmp = (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPNE ||
                       u->op == POLY_OP_CMPEQ);
        if (is_cmp && u->n_src > 0)
          alu_dt = poly_dtype_scalar(u->src[0]->dtype);

        /* Resolve source indices */
        int src_idx[3] = {-1, -1, -1};
        for (int j = 0; j < u->n_src && j < 3; j++)
          src_idx[j] = uop_index_map_get(idx_map, u->src[j]);

        vals[i].count = (uint16_t)out_cnt;
        if (out_cnt == 1) vals[i].lanes = &vals[i].inline0;

        /* Lane loop: broadcast scalar sources */
        for (int k = 0; k < out_cnt; k++) {
          InterpLane lane_srcs[3];
          for (int j = 0; j < u->n_src && j < 3; j++)
            lane_srcs[j] = (src_idx[j] >= 0) ? iv_get(&vals[src_idx[j]], k) : il_int(0);
          iv_set(&vals[i], k,
                 interp_truncate_lane(eval_alu(u->op, alu_dt, lane_srcs, u->n_src),
                                      poly_dtype_scalar(u->dtype)));
        }
      } else {
        fprintf(stderr, "polygrad: interp: unhandled op %s (%d)\n",
                poly_op_name(u->op), u->op);
        return -1;
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
  (void)n_args;

  if (getenv("POLY_DUMP_KERNELS")) {
    fprintf(stderr, "=== INTERP KERNEL (%d ops) ===\n", n_lin);
    for (int i = 0; i < n_lin; i++) {
      PolyUOp *u = lin[i];
      fprintf(stderr, "  [%3d] %-16s dt=%s count=%d nsrc=%d",
              i, poly_op_name(u->op), u->dtype.name ? u->dtype.name : "?",
              u->dtype.count, u->n_src);
      if (u->op == POLY_OP_CONST) {
        if (poly_dtype_is_float(u->dtype)) fprintf(stderr, " val=%g", u->arg.f);
        else fprintf(stderr, " val=%lld", (long long)u->arg.i);
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "=== END ===\n");
  }

  /* Pre-scan: count total vector lanes needed for arena allocation */
  int total_vec_lanes = 0;
  for (int i = 0; i < n_lin; i++) {
    int cnt = lin[i]->dtype.count;
    if (cnt > 1)
      total_vec_lanes += cnt;
    /* VECTORIZE: output count = n_src (may differ from dtype.count) */
    if (lin[i]->op == POLY_OP_VECTORIZE && lin[i]->n_src > 1)
      total_vec_lanes += lin[i]->n_src;
  }

  InterpVal *vals = calloc((size_t)n_lin, sizeof(InterpVal));
  if (!vals) return -1;

  /* Allocate lane arena: one contiguous block for all vector UOps */
  InterpLane *arena = NULL;
  if (total_vec_lanes > 0) {
    arena = calloc((size_t)total_vec_lanes, sizeof(InterpLane));
    if (!arena) { free(vals); return -1; }
  }

  /* Assign arena slices to vector UOps */
  int arena_offset = 0;
  for (int i = 0; i < n_lin; i++) {
    int cnt = lin[i]->dtype.count;
    /* VECTORIZE uses n_src as count */
    if (lin[i]->op == POLY_OP_VECTORIZE && lin[i]->n_src > 1)
      cnt = lin[i]->n_src;
    if (cnt > 1) {
      vals[i].count = (uint16_t)cnt;
      vals[i].lanes = arena + arena_offset;
      arena_offset += cnt;
    } else {
      vals[i].count = 1;
      vals[i].lanes = &vals[i].inline0;
    }
  }

  UOpIndexMap idx_map = uop_index_map_new(n_lin);
  for (int i = 0; i < n_lin; i++)
    uop_index_map_set(&idx_map, lin[i], i);

  int ret = interp_region(lin, n_lin, 0, n_lin, vals, args, n_args, &idx_map, arena);

  uop_index_map_free(&idx_map);

  /* Free register/local accumulator allocations */
  for (int i = 0; i < n_lin; i++) {
    if ((lin[i]->op == POLY_OP_DEFINE_REG || lin[i]->op == POLY_OP_DEFINE_LOCAL)
        && vals[i].lanes[0].p)
      free(vals[i].lanes[0].p);
  }

  free(arena);
  free(vals);
  return ret;
}
