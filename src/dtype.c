/*
 * dtype.c — DType system
 *
 * Mirrors tinygrad's dtype.py: predefined scalar types, vectorization,
 * pointer types, type classification helpers.
 */

#include "polygrad.h"
#include <string.h>

/* ── Predefined scalar dtypes ─────────────────────────────────────────── */
/* priority, bitsize, name, fmt, count, is_ptr, addrspace, vcount, ptr_size */

const PolyDType POLY_VOID    = { -1,   0, "void",           0,   1, false, 0, 0, 0 };
const PolyDType POLY_INDEX   = { -1, 800, "index",          0,   1, false, 0, 0, 0 };
const PolyDType POLY_BOOL    = {  0,   1, "bool",          '?',  1, false, 0, 0, 0 };
const PolyDType POLY_INT8    = {  1,   8, "signed char",   'b',  1, false, 0, 0, 0 };
const PolyDType POLY_UINT8   = {  2,   8, "unsigned char", 'B',  1, false, 0, 0, 0 };
const PolyDType POLY_INT16   = {  3,  16, "short",         'h',  1, false, 0, 0, 0 };
const PolyDType POLY_UINT16  = {  4,  16, "unsigned short",'H',  1, false, 0, 0, 0 };
const PolyDType POLY_INT32   = {  5,  32, "int",           'i',  1, false, 0, 0, 0 };
const PolyDType POLY_UINT32  = {  6,  32, "unsigned int",  'I',  1, false, 0, 0, 0 };
const PolyDType POLY_INT64   = {  7,  64, "long",          'q',  1, false, 0, 0, 0 };
const PolyDType POLY_UINT64  = {  8,  64, "unsigned long", 'Q',  1, false, 0, 0, 0 };
const PolyDType POLY_FLOAT16 = { 11,  16, "half",          'e',  1, false, 0, 0, 0 };
const PolyDType POLY_BFLOAT16= { 12,  16, "__bf16",         0,   1, false, 0, 0, 0 };
const PolyDType POLY_FLOAT32 = { 13,  32, "float",         'f',  1, false, 0, 0, 0 };
const PolyDType POLY_FLOAT64 = { 14,  64, "double",        'd',  1, false, 0, 0, 0 };

bool poly_dtype_eq(PolyDType a, PolyDType b) {
  return a.priority == b.priority && a.bitsize == b.bitsize &&
         a.count == b.count && a.is_ptr == b.is_ptr &&
         a.addrspace == b.addrspace && a.vcount == b.vcount &&
         a.ptr_size == b.ptr_size &&
         (a.name == b.name || (a.name && b.name && strcmp(a.name, b.name) == 0));
}

bool poly_dtype_is_float(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return s.priority >= 9 && s.priority <= 14;  /* fp8 through float64 */
}

bool poly_dtype_is_int(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return (s.priority >= 1 && s.priority <= 8) || poly_dtype_eq(s, POLY_INDEX);
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
  if (dt.count == 1) return dt;
  /* return the base scalar type by matching priority+name */
  PolyDType s = dt;
  s.count = 1;
  s.bitsize = dt.bitsize / dt.count;
  return s;
}

PolyDType poly_dtype_vec(PolyDType dt, int sz) {
  if (sz == 1 || poly_dtype_eq(dt, POLY_VOID)) return dt;
  PolyDType v = dt;
  v.bitsize = dt.bitsize * sz;
  v.count = sz;
  v.fmt = 0;  /* no format for vector types */
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
