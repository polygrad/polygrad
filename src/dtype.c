/*
 * dtype.c — DType system
 *
 * Mirrors tinygrad's dtype.py: predefined scalar types, vectorization,
 * pointer types, type classification helpers.
 */

#include "polygrad.h"
#include <string.h>

/* Predefined scalar dtypes */
/* priority, bitsize, name, fmt, count, is_ptr, addrspace, vcount, ptr_size */

const PolyDType POLY_VOID = {-1, 0, "void", 0, 1, false, 0, 0, 0};
const PolyDType POLY_INDEX = {0, 800, "weakint", 0, 1, false, 0, 0, 0};
const PolyDType POLY_BOOL = {0, 1, "bool", '?', 1, false, 0, 0, 0};
const PolyDType POLY_INT8 = {1, 8, "signed char", 'b', 1, false, 0, 0, 0};
const PolyDType POLY_UINT8 = {2, 8, "unsigned char", 'B', 1, false, 0, 0, 0};
const PolyDType POLY_INT16 = {3, 16, "short", 'h', 1, false, 0, 0, 0};
const PolyDType POLY_UINT16 = {4, 16, "unsigned short", 'H', 1, false, 0, 0, 0};
const PolyDType POLY_INT32 = {5, 32, "int", 'i', 1, false, 0, 0, 0};
const PolyDType POLY_UINT32 = {6, 32, "unsigned int", 'I', 1, false, 0, 0, 0};
const PolyDType POLY_INT64 = {7, 64, "long", 'q', 1, false, 0, 0, 0};
const PolyDType POLY_UINT64 = {8, 64, "unsigned long", 'Q', 1, false, 0, 0, 0};
const PolyDType POLY_FLOAT16 = {11, 16, "__fp16", 'e', 1, false, 0, 0, 0};
const PolyDType POLY_BFLOAT16 = {12, 16, "__bf16", 0, 1, false, 0, 0, 0};
const PolyDType POLY_FLOAT32 = {13, 32, "float", 'f', 1, false, 0, 0, 0};
const PolyDType POLY_FLOAT64 = {14, 64, "double", 'd', 1, false, 0, 0, 0};

/* FFI-friendly dtype lookup: id -> PolyDType. The id ordering matches the
 * _DTYPE_IDS dict in py/polygrad/_ffi.py and js/src/ffi.js. */
static const PolyDType *_dtype_table[] = {
    &POLY_VOID,    &POLY_BOOL,     &POLY_INT8,    &POLY_UINT8,   &POLY_INT16,
    &POLY_UINT16,  &POLY_INT32,    &POLY_UINT32,  &POLY_INT64,   &POLY_UINT64,
    &POLY_FLOAT16, &POLY_BFLOAT16, &POLY_FLOAT32, &POLY_FLOAT64,
};
#define N_DTYPE_TABLE ((int)(sizeof(_dtype_table) / sizeof(_dtype_table[0])))

int poly_dtype_count(void) { return N_DTYPE_TABLE; }

bool poly_dtype_by_id(int id, PolyDType *out) {
  if (!out || id < 0 || id >= N_DTYPE_TABLE) return false;
  *out = *_dtype_table[id];
  return true;
}

int poly_dtype_id_by_name(const char *name) {
  if (!name || !name[0]) return -1;
  if (strcmp(name, "void") == 0) return 0;
  if (strcmp(name, "bool") == 0) return 1;
  if (strcmp(name, "int8") == 0 || strcmp(name, "signed char") == 0) return 2;
  if (strcmp(name, "uint8") == 0 || strcmp(name, "unsigned char") == 0) return 3;
  if (strcmp(name, "int16") == 0 || strcmp(name, "short") == 0) return 4;
  if (strcmp(name, "uint16") == 0 || strcmp(name, "unsigned short") == 0) return 5;
  if (strcmp(name, "int32") == 0 || strcmp(name, "int") == 0) return 6;
  if (strcmp(name, "uint32") == 0 || strcmp(name, "unsigned int") == 0) return 7;
  if (strcmp(name, "int64") == 0 || strcmp(name, "long") == 0) return 8;
  if (strcmp(name, "uint64") == 0 || strcmp(name, "unsigned long") == 0) return 9;
  if (strcmp(name, "float16") == 0 || strcmp(name, "__fp16") == 0) return 10;
  if (strcmp(name, "bfloat16") == 0 || strcmp(name, "__bf16") == 0) return 11;
  if (strcmp(name, "float32") == 0 || strcmp(name, "float") == 0) return 12;
  if (strcmp(name, "float64") == 0 || strcmp(name, "double") == 0) return 13;
  return -1;
}

bool poly_dtype_eq(PolyDType a, PolyDType b) {
  return a.priority == b.priority && a.bitsize == b.bitsize && a.count == b.count &&
         a.is_ptr == b.is_ptr && a.addrspace == b.addrspace && a.vcount == b.vcount &&
         a.ptr_size == b.ptr_size &&
         (a.name == b.name || (a.name && b.name && strcmp(a.name, b.name) == 0));
}

static bool dtype_is_weakint_like(PolyDType dt) {
  if (dt.is_ptr || !dt.name || dt.priority != POLY_INDEX.priority) return false;
  return strcmp(dt.name, POLY_INDEX.name) == 0;
}

bool poly_dtype_is_float(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return s.priority >= 9 && s.priority <= 14; /* fp8 through float64 */
}

bool poly_dtype_is_index(PolyDType dt) {
  /* tinygrad dtypes are interned, so `dtype.scalar() is dtypes.weakint`
   * recognizes all weak-index vector/scalar forms. Polygrad stores dtypes as
   * value structs, and some late index rewrites can carry a weakint name with
   * non-canonical bit metadata. Treat the weakint name+priority as the stable
   * identity and lower it before rendering, matching tinygrad's
   * pm_lower_index_dtype boundary. */
  return dtype_is_weakint_like(poly_dtype_scalar(dt));
}

bool poly_dtype_is_int(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return (s.priority >= 1 && s.priority <= 8) || poly_dtype_is_index(s);
}

bool poly_dtype_is_unsigned(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return s.priority == 2 || s.priority == 4 || s.priority == 6 || s.priority == 8;
}

bool poly_dtype_is_bool(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return s.priority == 0 && s.bitsize == 1;
}

PolyDType poly_dtype_scalar(PolyDType dt) {
  if (dtype_is_weakint_like(dt)) return POLY_INDEX;
  if (dt.count == 1) return dt;
  /* Match tinygrad DType.scalar(): vector dtypes return the canonical scalar
   * dtype, including its fmt field. */
  PolyDType s = dt;
  s.count = 1;
  s.bitsize = dt.bitsize / dt.count;
  if (dtype_is_weakint_like(s)) return POLY_INDEX;
  if (!s.is_ptr) {
    for (int i = 0; i < N_DTYPE_TABLE; i++) {
      const PolyDType *canon = _dtype_table[i];
      if (canon->priority == s.priority && canon->bitsize == s.bitsize &&
          canon->count == 1 && canon->name && s.name && strcmp(canon->name, s.name) == 0)
        return *canon;
    }
  }
  return s;
}

PolyDType poly_dtype_vec(PolyDType dt, int sz) {
  if (sz == 1 || poly_dtype_eq(dt, POLY_VOID)) return dt;
  PolyDType v = dt;
  v.bitsize = dt.bitsize * sz;
  v.count = sz;
  v.fmt = 0; /* no format for vector types */
  return v;
}

PolyDType poly_dtype_ptr(PolyDType dt, int64_t size, PolyAddrSpace addrspace) {
  PolyDType p = dt;
  p.is_ptr = true;
  p.addrspace = addrspace;
  p.vcount = 1;
  p.ptr_size = size;
  return p;
}

int poly_dtype_itemsize(PolyDType dt) {
  return (dt.bitsize + 7) / 8;
}

const char *poly_dtype_name(PolyDType dt) {
  return dt.name;
}
