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
 * Execution model:
 *   - The linearized list is a flat sequence of UOps in execution order.
 *   - RANGE/END pairs define loops. When the interpreter hits a RANGE,
 *     it finds the matching END and recursively interprets the region
 *     [RANGE+1, END) for each iteration. Nested loops work naturally.
 *   - Each UOp produces one value stored in a per-UOp value table (InterpVal).
 *   - Values are typed: int64 for integers, double for floats, void* for pointers.
 *   - Memory ops (LOAD/STORE) access buffers via typed memcpy.
 *   - BITCAST preserves bit patterns across int32/float32 and int64/float64
 *     boundaries using explicit memcpy (since InterpVal fields have different
 *     C representations).
 */

#include "interp.h"
#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Value representation ────────────────────────────────────────────── */
/*
 * Each linearized UOp produces one InterpVal. The active union field
 * depends on the UOp's dtype:
 *   - float types: .f (double, holds float32 or float64 values)
 *   - signed int types: .i (int64_t)
 *   - unsigned int types: .u (uint64_t)
 *   - pointer types (PARAM, INDEX, DEFINE_REG): .p (void*)
 */

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

static double as_float(InterpVal v, PolyDType dt) {
  if (poly_dtype_is_float(dt)) return v.f;
  if (poly_dtype_is_unsigned(dt)) return (double)v.u;
  return (double)v.i;
}

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

/* ── Memory access ───────────────────────────────────────────────────── */
/*
 * Type dispatch uses poly_dtype_is_float/is_unsigned + bitsize instead
 * of poly_dtype_eq, because codegen-rewritten UOps may carry dtypes
 * that are structurally equivalent to POLY_FLOAT32 but not pointer-equal
 * (e.g. different ptr_size or count fields from intermediate rewrites).
 */

static InterpVal mem_load(void *ptr, PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  int bs = s.bitsize;
  bool is_flt = poly_dtype_is_float(s);
  bool is_uns = poly_dtype_is_unsigned(s);

  if (is_flt && bs == 32)             { float v;    memcpy(&v, ptr, 4); return iv_flt(v); }
  if (is_flt && bs == 64)             { double v;   memcpy(&v, ptr, 8); return iv_flt(v); }
  if (!is_flt && !is_uns && bs == 32) { int32_t v;  memcpy(&v, ptr, 4); return iv_int(v); }
  if (is_uns && bs == 32)             { uint32_t v; memcpy(&v, ptr, 4); return iv_uint(v); }
  if (!is_flt && !is_uns && bs == 64) { int64_t v;  memcpy(&v, ptr, 8); return iv_int(v); }
  if (is_uns && bs == 64)             { uint64_t v; memcpy(&v, ptr, 8); return iv_uint(v); }
  if (!is_flt && !is_uns && bs == 8)  { int8_t v;   memcpy(&v, ptr, 1); return iv_int(v); }
  if (is_uns && bs == 8)              { uint8_t v;  memcpy(&v, ptr, 1); return iv_uint(v); }
  if (!is_flt && !is_uns && bs == 16) { int16_t v;  memcpy(&v, ptr, 2); return iv_int(v); }
  if (is_uns && bs == 16)             { uint16_t v; memcpy(&v, ptr, 2); return iv_uint(v); }
  if (poly_dtype_is_bool(s))          { uint8_t v;  memcpy(&v, ptr, 1); return iv_int(v ? 1 : 0); }

  /* bfloat16: top 16 bits of float32 */
  if (is_flt && bs == 16 && s.name && strcmp(s.name, "__bf16") == 0) {
    uint16_t bits; memcpy(&bits, ptr, 2);
    uint32_t f32 = (uint32_t)bits << 16;
    float fv; memcpy(&fv, &f32, 4);
    return iv_flt(fv);
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
      else { float sv = (float)mant / 1024.0f * (1.0f / 16384.0f); return iv_flt(sign ? -sv : sv); }
    } else if (exp == 31) { f32 = sign | 0x7F800000 | (mant << 13); }
    else { f32 = sign | ((exp + 112) << 23) | (mant << 13); }
    float fv; memcpy(&fv, &f32, 4);
    return iv_flt(fv);
  }
  return iv_int(0);
}

static void mem_store(void *ptr, InterpVal v, PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
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

/* ── ALU evaluation ──────────────────────────────────────────────────── */
/*
 * Evaluates a single ALU op on scalar operands.
 * dt is the effective dtype for the operation:
 *   - For most ops, this is the UOp's own dtype.
 *   - For comparisons (CMPLT, CMPNE, CMPEQ), the caller passes the
 *     source dtype instead, since the result dtype is bool but the
 *     comparison semantics depend on the operand type (float vs int).
 */

static InterpVal eval_alu(PolyOps op, PolyDType dt, InterpVal *srcs, int n_src) {
  bool is_flt = poly_dtype_is_float(dt);

  double a = srcs[0].f, b = n_src > 1 ? srcs[1].f : 0, c = n_src > 2 ? srcs[2].f : 0;
  int64_t ai = srcs[0].i, bi = n_src > 1 ? srcs[1].i : 0;
  uint64_t au = srcs[0].u, bu = n_src > 1 ? srcs[1].u : 0;

  switch (op) {
  /* Unary */
  case POLY_OP_NEG:
    if (poly_dtype_is_bool(poly_dtype_scalar(dt))) return iv_int(srcs[0].i ? 0 : 1);
    return is_flt ? iv_flt(-a) : iv_int(-ai);
  case POLY_OP_SQRT:       return iv_flt(sqrt(a));
  case POLY_OP_RECIPROCAL: return iv_flt(1.0 / a);
  case POLY_OP_EXP2:       return iv_flt(exp2(a));
  case POLY_OP_LOG2:       return iv_flt(log2(a));
  case POLY_OP_SIN:        return iv_flt(sin(a));
  case POLY_OP_TRUNC:      return iv_flt(trunc(a));

  /* Binary arithmetic — wrapping add/sub/mul to avoid signed overflow UB.
   * tinygrad's Python uses arbitrary-precision ints; in C we wrap via unsigned cast. */
  case POLY_OP_ADD:  return is_flt ? iv_flt(a + b) : iv_int((int64_t)((uint64_t)ai + (uint64_t)bi));
  case POLY_OP_SUB:  return is_flt ? iv_flt(a - b) : iv_int((int64_t)((uint64_t)ai - (uint64_t)bi));
  case POLY_OP_MUL:  return is_flt ? iv_flt(a * b) : iv_int((int64_t)((uint64_t)ai * (uint64_t)bi));
  case POLY_OP_FDIV: return iv_flt(a / b);
  case POLY_OP_IDIV: return bi != 0 ? iv_int(ai / bi) : iv_int(0);
  case POLY_OP_MOD:  return bi != 0 ? iv_int(ai % bi) : iv_int(0);
  case POLY_OP_MAX:  return is_flt ? iv_flt(a > b ? a : b) : iv_int(ai > bi ? ai : bi);
  case POLY_OP_POW:  return iv_flt(pow(a, b));

  /* Bitwise (always unsigned semantics) */
  case POLY_OP_SHL: return iv_uint(au << (bu & 63));
  case POLY_OP_SHR: return iv_uint(au >> (bu & 63));
  case POLY_OP_AND: return iv_uint(au & bu);
  case POLY_OP_OR:  return iv_uint(au | bu);
  case POLY_OP_XOR: return iv_uint(au ^ bu);

  /* Comparison (result is 0 or 1) */
  case POLY_OP_CMPLT: return is_flt ? iv_int(a < b ? 1 : 0) : iv_int(ai < bi ? 1 : 0);
  case POLY_OP_CMPNE: return is_flt ? iv_int(a != b ? 1 : 0) : iv_int(ai != bi ? 1 : 0);
  case POLY_OP_CMPEQ: return is_flt ? iv_int(a == b ? 1 : 0) : iv_int(ai == bi ? 1 : 0);

  /* Ternary */
  case POLY_OP_WHERE:  return srcs[0].i ? srcs[1] : srcs[2];
  case POLY_OP_MULACC: return is_flt ? iv_flt(a * b + c) : iv_int(ai * bi + srcs[2].i);

  /* THREEFRY is decomposed by pm_decomp before linearization.
   * If it somehow survives, XOR is a reasonable fallback. */
  case POLY_OP_THREEFRY: return iv_uint(au ^ bu);

  default:
    fprintf(stderr, "polygrad: interp: unhandled ALU op %s\n", poly_op_name(op));
    return iv_int(0);
  }
}

/* ── UOp index map ───────────────────────────────────────────────────── */
/*
 * Maps UOp pointers to their position in the linearized array.
 * Used for O(1) source lookup: when UOp[i] references UOp[j] via
 * src[], we need to find j's index to read vals[j].
 * Simple open-addressing hash table with golden-ratio hash.
 */

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
/*
 * Interprets a contiguous region [start, end) of the linearized list.
 * RANGE nodes recurse into the loop body; all other ops execute in order.
 * vals[] is shared across all recursive calls -- PARAM/CONST values set
 * before a RANGE remain visible inside the loop body.
 */

static int interp_region(PolyUOp **lin, int n_lin, int start, int end,
                         InterpVal *vals, void **args, int n_args,
                         const UOpIndexMap *idx_map) {
  for (int i = start; i < end; i++) {
    PolyUOp *u = lin[i];

    switch (u->op) {

    case POLY_OP_PARAM:
      if (u->arg.i >= 0 && u->arg.i < n_args)
        vals[i] = iv_ptr(args[u->arg.i]);
      else {
        fprintf(stderr, "polygrad: interp: PARAM index %lld out of range (n_args=%d)\n",
                (long long)u->arg.i, n_args);
        return -1;
      }
      break;

    case POLY_OP_DEFINE_VAR: {
      if (u->arg.i >= 0 && u->arg.i < n_args && args[u->arg.i]) {
        int *vp = (int *)args[u->arg.i];
        vals[i] = iv_int(*vp);
      } else {
        /* Var not bound (e.g. DEFINE_VAR beyond provided args). Default to 0. */
        vals[i] = iv_int(0);
      }
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
      /* Register accumulator: allocate a small buffer for scalar storage.
       * DEFINE_REG dtype is ptr<scalar, REG>; the buffer is indexed at 0. */
      PolyDType base = poly_dtype_scalar(u->dtype);
      int sz = poly_dtype_itemsize(base);
      if (sz < 4) sz = 4;
      vals[i] = iv_ptr(calloc(1, (size_t)sz));
      break;
    }

    case POLY_OP_RANGE: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      if (src0 < 0) { fprintf(stderr, "polygrad: interp: RANGE bound not found\n"); return -1; }
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
      /* Loop close is handled by RANGE's recursion. No-op here. */
      break;

    case POLY_OP_AFTER: {
      /* Ordering node: pass through src[0]'s value.
       * Dependencies (src[1:]) are structural, not semantic. */
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      vals[i] = vals[src0];
      break;
    }

    case POLY_OP_INDEX: {
      /* Pointer arithmetic: base_ptr + offset * element_size */
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
      /* Gated load: LOAD(INDEX(buf, idx, gate), alt) or LOAD(CAST(INDEX(..., gate)), alt) */
      PolyUOp *ld_idx = poly_find_index_through_cast(u->src[0]);
      if (ld_idx && ld_idx->n_src >= 3) {
        int gate_i = uop_index_map_get(idx_map, ld_idx->src[2]);
        if (gate_i >= 0 && !as_int(vals[gate_i], ld_idx->src[2]->dtype)) {
          /* Gate is false: use alt value (src[1]) or zero */
          if (u->n_src >= 2) {
            int alt_i = uop_index_map_get(idx_map, u->src[1]);
            vals[i] = (alt_i >= 0) ? vals[alt_i] : iv_flt(0.0);
          } else {
            vals[i] = iv_flt(0.0);
          }
          break;
        }
      }
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
      PolyDType store_dt = poly_dtype_scalar(u->src[0]->dtype);
      mem_store(vals[src0].p, vals[src1], store_dt);
      break;
    }

    case POLY_OP_DEFINE_LOCAL: {
      /* Stack-allocated accumulator scalar (like DEFINE_REG but simpler) */
      PolyDType base = poly_dtype_scalar(u->dtype);
      int sz = poly_dtype_itemsize(base);
      if (sz < 4) sz = 4;
      vals[i] = iv_ptr(calloc(1, (size_t)sz));
      break;
    }

    case POLY_OP_CAST: {
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      PolyDType src_dt = u->src[0]->dtype;
      if (poly_dtype_is_float(u->dtype))
        vals[i] = iv_flt(as_float(vals[src0], src_dt));
      else if (poly_dtype_is_unsigned(u->dtype))
        vals[i] = iv_uint(as_uint(vals[src0], src_dt));
      else
        vals[i] = iv_int(as_int(vals[src0], src_dt));
      break;
    }

    case POLY_OP_BITCAST: {
      /* Reinterpret bits without value conversion.
       * InterpVal stores int64/double which have different bit layouts
       * than int32/float32, so we must go through explicit memcpy. */
      int src0 = uop_index_map_get(idx_map, u->src[0]);
      PolyDType src_dt = poly_dtype_scalar(u->src[0]->dtype);
      PolyDType dst_dt = poly_dtype_scalar(u->dtype);

      if (src_dt.bitsize == 32 && dst_dt.bitsize == 32) {
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
      } else if (src_dt.bitsize == 64 && dst_dt.bitsize == 64) {
        uint64_t bits;
        if (poly_dtype_is_float(src_dt))
          memcpy(&bits, &vals[src0].f, 8);
        else
          bits = vals[src0].u;
        if (poly_dtype_is_float(dst_dt)) {
          double fv; memcpy(&fv, &bits, 8);
          vals[i] = iv_flt(fv);
        } else {
          vals[i] = iv_uint(bits);
        }
      } else {
        vals[i] = vals[src0];
      }
      break;
    }

    case POLY_OP_SINK:
    case POLY_OP_NOOP:
    case POLY_OP_GROUP:
      break;

    default: {
      /* ALU operations: dispatch to eval_alu */
      if (u->op >= POLY_OP_EXP2 && u->op <= POLY_OP_MULACC) {
        InterpVal src_vals[3];
        PolyDType alu_dt = u->dtype;

        /* Comparisons produce bool but operate on source dtype */
        bool is_cmp = (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPNE ||
                       u->op == POLY_OP_CMPEQ);
        if (is_cmp && u->n_src > 0)
          alu_dt = u->src[0]->dtype;

        for (int j = 0; j < u->n_src && j < 3; j++) {
          int sj = uop_index_map_get(idx_map, u->src[j]);
          src_vals[j] = (sj >= 0) ? vals[sj] : iv_int(0);
        }
        vals[i] = eval_alu(u->op, alu_dt, src_vals, u->n_src);
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

  InterpVal *vals = calloc((size_t)n_lin, sizeof(InterpVal));
  if (!vals) return -1;

  UOpIndexMap idx_map = uop_index_map_new(n_lin);
  for (int i = 0; i < n_lin; i++)
    uop_index_map_set(&idx_map, lin[i], i);

  int ret = interp_region(lin, n_lin, 0, n_lin, vals, args, n_args, &idx_map);

  uop_index_map_free(&idx_map);

  /* Free register/local accumulator allocations */
  for (int i = 0; i < n_lin; i++) {
    if ((lin[i]->op == POLY_OP_DEFINE_REG || lin[i]->op == POLY_OP_DEFINE_LOCAL)
        && vals[i].p)
      free(vals[i].p);
  }

  free(vals);
  return ret;
}
