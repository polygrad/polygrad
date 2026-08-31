/*
 * dtype.c — DType system
 *
 * Mirrors current tinygrad dtype.py scalar DTypes. Shape and storage metadata
 * belong to UOps and ParamArg.
 */

#include "polygrad.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Predefined scalar dtypes */
/* priority, bitsize, name, fmt */

const PolyDType POLY_VOID = {-1, 0, "void", 0};
const PolyDType POLY_WEAKINT = {0, 800, "weakint", 0};
const PolyDType POLY_BOOL = {0, 1, "bool", '?'};
const PolyDType POLY_INT8 = {1, 8, "signed char", 'b'};
const PolyDType POLY_UINT8 = {2, 8, "unsigned char", 'B'};
const PolyDType POLY_INT16 = {3, 16, "short", 'h'};
const PolyDType POLY_UINT16 = {4, 16, "unsigned short", 'H'};
const PolyDType POLY_INT32 = {5, 32, "int", 'i'};
const PolyDType POLY_UINT32 = {6, 32, "unsigned int", 'I'};
const PolyDType POLY_INT64 = {7, 64, "long", 'q'};
const PolyDType POLY_UINT64 = {8, 64, "unsigned long", 'Q'};
const PolyDType POLY_WEAKFLOAT = {9, 800, "weakfloat", 0};
const PolyDType POLY_FP8E4M3 = {10, 8, "float8_e4m3", 0};
const PolyDType POLY_FP8E5M2 = {11, 8, "float8_e5m2", 0};
const PolyDType POLY_FP8E4M3FNUZ = {10, 8, "float8_e4m3fnuz", 0};
const PolyDType POLY_FP8E5M2FNUZ = {11, 8, "float8_e5m2fnuz", 0};
const PolyDType POLY_FLOAT16 = {12, 16, "__fp16", 'e'};
const PolyDType POLY_BFLOAT16 = {13, 16, "__bf16", 0};
const PolyDType POLY_FLOAT32 = {14, 32, "float", 'f'};
const PolyDType POLY_FLOAT64 = {15, 64, "double", 'd'};

/* FFI-friendly dtype lookup: id -> PolyDType. The id ordering matches the
 * _DTYPE_IDS dict in py/polygrad/_ffi.py and js/src/ffi.js. */
static const PolyDType *_dtype_table[] = {
    &POLY_VOID,      &POLY_BOOL,     &POLY_INT8,    &POLY_UINT8,       &POLY_INT16,
    &POLY_UINT16,    &POLY_INT32,    &POLY_UINT32,  &POLY_INT64,       &POLY_UINT64,
    &POLY_FLOAT16,   &POLY_BFLOAT16, &POLY_FLOAT32, &POLY_FLOAT64,     &POLY_WEAKINT,
    &POLY_WEAKFLOAT, &POLY_FP8E4M3,  &POLY_FP8E5M2, &POLY_FP8E4M3FNUZ, &POLY_FP8E5M2FNUZ,
};
#define N_DTYPE_TABLE ((int)(sizeof(_dtype_table) / sizeof(_dtype_table[0])))

int poly_dtype_count(void) {
  return N_DTYPE_TABLE;
}

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
  if (strcmp(name, "weakint") == 0) return 14;
  if (strcmp(name, "weakfloat") == 0) return 15;
  if (strcmp(name, "fp8e4m3") == 0 || strcmp(name, "float8_e4m3") == 0) return 16;
  if (strcmp(name, "fp8e5m2") == 0 || strcmp(name, "float8_e5m2") == 0) return 17;
  if (strcmp(name, "fp8e4m3fnuz") == 0 || strcmp(name, "float8_e4m3fnuz") == 0) return 18;
  if (strcmp(name, "fp8e5m2fnuz") == 0 || strcmp(name, "float8_e5m2fnuz") == 0) return 19;
  return -1;
}

bool poly_dtype_eq(PolyDType a, PolyDType b) {
  return a.priority == b.priority && a.bitsize == b.bitsize &&
         (a.name == b.name || (a.name && b.name && strcmp(a.name, b.name) == 0));
}

static bool dtype_is_weakint_like(PolyDType dt) {
  if (!dt.name || dt.priority != POLY_WEAKINT.priority) return false;
  return strcmp(dt.name, POLY_WEAKINT.name) == 0;
}

static bool dtype_is_weakfloat_like(PolyDType dt) {
  if (!dt.name || dt.priority != POLY_WEAKFLOAT.priority) return false;
  return strcmp(dt.name, POLY_WEAKFLOAT.name) == 0;
}

bool poly_dtype_is_float(PolyDType dt) {
  return dtype_is_weakfloat_like(dt) || poly_dtype_eq(dt, POLY_FP8E4M3) ||
         poly_dtype_eq(dt, POLY_FP8E5M2) || poly_dtype_eq(dt, POLY_FP8E4M3FNUZ) ||
         poly_dtype_eq(dt, POLY_FP8E5M2FNUZ) ||
         (dt.priority >= POLY_FLOAT16.priority && dt.priority <= POLY_FLOAT64.priority);
}

bool poly_dtype_is_fp8(PolyDType dt) {
  return poly_dtype_eq(dt, POLY_FP8E4M3) || poly_dtype_eq(dt, POLY_FP8E5M2) ||
         poly_dtype_eq(dt, POLY_FP8E4M3FNUZ) || poly_dtype_eq(dt, POLY_FP8E5M2FNUZ);
}

bool poly_dtype_is_fp8_fnuz(PolyDType dt) {
  return poly_dtype_eq(dt, POLY_FP8E4M3FNUZ) || poly_dtype_eq(dt, POLY_FP8E5M2FNUZ);
}

bool poly_dtype_is_index(PolyDType dt) {
  /* Polygrad stores DTypes by value, so non-canonical weak metadata can reach
   * the current pm_lower_index_dtype boundary. Name and priority identify it. */
  return dtype_is_weakint_like(dt);
}

bool poly_dtype_is_weak(PolyDType dt) {
  return dtype_is_weakint_like(dt) || dtype_is_weakfloat_like(dt);
}

bool poly_dtype_is_int(PolyDType dt) {
  return (dt.priority >= 1 && dt.priority <= 8) || poly_dtype_is_index(dt);
}

bool poly_dtype_is_unsigned(PolyDType dt) {
  return dt.priority == 2 || dt.priority == 4 || dt.priority == 6 || dt.priority == 8;
}

bool poly_dtype_is_bool(PolyDType dt) {
  return dt.priority == 0 && dt.bitsize == 1;
}

/* Current tinygrad dtype.py:165-168. Weak values have a kind but no storage
 * width; these helpers commit them only at an explicit storage/consumer
 * boundary and derive the weak kind used by scalar promotion. */
PolyDType poly_dtype_strong(PolyDType dt) {
  if (dtype_is_weakint_like(dt)) return POLY_INT32;
  if (dtype_is_weakfloat_like(dt)) return POLY_FLOAT32;
  return dt;
}

PolyDType poly_dtype_weak(PolyDType dt) {
  if (poly_dtype_is_float(dt)) return POLY_WEAKFLOAT;
  if (poly_dtype_is_int(dt)) return POLY_WEAKINT;
  return dt;
}

/*
 * Current tinygrad dtype.py:171-188 defines the JAX-style promotion lattice
 * and least_upper_dtype over scalar dtypes.
 */
bool poly_dtype_least_upper(PolyDType a, PolyDType b, PolyDType *out) {
  if (!out) return false;
  if (poly_dtype_eq(a, b)) {
    *out = a;
    return true;
  }

  enum {
    PROMO_BOOL,
    PROMO_WEAKINT,
    PROMO_INT8,
    PROMO_UINT8,
    PROMO_INT16,
    PROMO_UINT16,
    PROMO_INT32,
    PROMO_UINT32,
    PROMO_INT64,
    PROMO_UINT64,
    PROMO_WEAKFLOAT,
    PROMO_FP8E4M3,
    PROMO_FP8E5M2,
    PROMO_FP8E4M3FNUZ,
    PROMO_FP8E5M2FNUZ,
    PROMO_FLOAT16,
    PROMO_BFLOAT16,
    PROMO_FLOAT32,
    PROMO_FLOAT64,
    PROMO_COUNT,
  };
  static const PolyDType *const types[PROMO_COUNT] = {
      &POLY_BOOL,      &POLY_WEAKINT,  &POLY_INT8,    &POLY_UINT8,       &POLY_INT16,
      &POLY_UINT16,    &POLY_INT32,    &POLY_UINT32,  &POLY_INT64,       &POLY_UINT64,
      &POLY_WEAKFLOAT, &POLY_FP8E4M3,  &POLY_FP8E5M2, &POLY_FP8E4M3FNUZ, &POLY_FP8E5M2FNUZ,
      &POLY_FLOAT16,   &POLY_BFLOAT16, &POLY_FLOAT32, &POLY_FLOAT64,
  };
  static const uint32_t parents[PROMO_COUNT] = {
      [PROMO_BOOL] = 1u << PROMO_WEAKINT,
      [PROMO_WEAKINT] = (1u << PROMO_INT8) | (1u << PROMO_UINT8),
      [PROMO_INT8] = 1u << PROMO_INT16,
      [PROMO_UINT8] = (1u << PROMO_INT16) | (1u << PROMO_UINT16),
      [PROMO_INT16] = 1u << PROMO_INT32,
      [PROMO_UINT16] = (1u << PROMO_INT32) | (1u << PROMO_UINT32),
      [PROMO_INT32] = 1u << PROMO_INT64,
      [PROMO_UINT32] = (1u << PROMO_INT64) | (1u << PROMO_UINT64),
      [PROMO_INT64] = 1u << PROMO_WEAKFLOAT,
      [PROMO_UINT64] = 1u << PROMO_WEAKFLOAT,
      [PROMO_WEAKFLOAT] = (1u << PROMO_FP8E4M3) | (1u << PROMO_FP8E5M2) |
                          (1u << PROMO_FP8E4M3FNUZ) | (1u << PROMO_FP8E5M2FNUZ),
      [PROMO_FP8E4M3] = (1u << PROMO_FLOAT16) | (1u << PROMO_BFLOAT16),
      [PROMO_FP8E5M2] = (1u << PROMO_FLOAT16) | (1u << PROMO_BFLOAT16),
      [PROMO_FP8E4M3FNUZ] = (1u << PROMO_FLOAT16) | (1u << PROMO_BFLOAT16),
      [PROMO_FP8E5M2FNUZ] = (1u << PROMO_FLOAT16) | (1u << PROMO_BFLOAT16),
      [PROMO_FLOAT16] = 1u << PROMO_FLOAT32,
      [PROMO_BFLOAT16] = 1u << PROMO_FLOAT32,
      [PROMO_FLOAT32] = 1u << PROMO_FLOAT64,
  };

  int ai = -1, bi = -1;
  for (int i = 0; i < PROMO_COUNT; i++) {
    if (poly_dtype_eq(a, *types[i])) ai = i;
    if (poly_dtype_eq(b, *types[i])) bi = i;
  }
  if (ai < 0 || bi < 0) return false;

  uint32_t closures[2] = {1u << ai, 1u << bi};
  for (int c = 0; c < 2; c++) {
    for (;;) {
      uint32_t expanded = closures[c];
      for (int i = 0; i < PROMO_COUNT; i++)
        if (closures[c] & (1u << i)) expanded |= parents[i];
      if (expanded == closures[c]) break;
      closures[c] = expanded;
    }
  }
  uint32_t common = closures[0] & closures[1];
  for (int i = 0; i < PROMO_COUNT; i++) {
    if (!(common & (1u << i))) continue;
    *out = *types[i];
    return true;
  }
  return false;
}

/* Current tinygrad dtype.py:187-188. Transcendental ALU ops preserve an
 * existing floating dtype, promote weakint to weakfloat, and otherwise meet
 * the input with default_float. Tensor-stage dtypes are scalar in current
 * tinygrad; keep Polygrad's still-live late vector form lane-preserving while
 * that separate compiler representation is migrated. */
bool poly_dtype_least_upper_float(PolyDType dt, PolyDType *out) {
  if (!out) return false;
  PolyDType scalar = dt;
  if (poly_dtype_eq(scalar, POLY_WEAKINT)) {
    scalar = POLY_WEAKFLOAT;
  } else if (poly_dtype_is_float(scalar)) {
    *out = dt;
    return true;
  } else if (!poly_dtype_least_upper(scalar, POLY_FLOAT32, &scalar)) {
    return false;
  }
  *out = scalar;
  return true;
}

/* Current tinygrad dtype.py:212-216.  SUM_DTYPE applies only to the
 * floating default; integer and unsigned accumulators retain their exact
 * current Tinygrad floors. */
bool poly_sum_acc_dtype(PolyDType dt, PolyDType *out) {
  if (!out) return false;
  PolyDType floor;
  if (poly_dtype_is_unsigned(dt)) {
    floor = POLY_UINT32;
  } else if (poly_dtype_is_int(dt) || poly_dtype_is_bool(dt)) {
    floor = POLY_INT32;
  } else {
    floor = POLY_FLOAT32;
    const char *name = getenv("SUM_DTYPE");
    if (name && name[0]) {
      int id = poly_dtype_id_by_name(name);
      if (id < 0 || !poly_dtype_by_id(id, &floor)) return false;
    }
  }
  return poly_dtype_least_upper(dt, floor, out);
}

/* Pinned tinygrad dtype.py:256-270. Return whether dt1 preserves every value
 * representable by dt0. Exact equality and bool sources are lossless before
 * tinygrad's scalar-only target table is consulted. */
bool poly_dtype_can_lossless_cast(PolyDType dt0, PolyDType dt1) {
  if (poly_dtype_eq(dt0, dt1)) return true;
  if (poly_dtype_eq(dt0, POLY_BOOL)) return true;

#define DT_IS(dt, type) poly_dtype_eq((dt), (type))
  if (poly_dtype_is_index(dt1))
    return poly_dtype_is_int(dt0) && !poly_dtype_is_index(dt0) && !poly_dtype_is_bool(dt0);
  if (DT_IS(dt1, POLY_FLOAT64))
    return DT_IS(dt0, POLY_FLOAT32) || DT_IS(dt0, POLY_FLOAT16) || DT_IS(dt0, POLY_BFLOAT16) ||
           DT_IS(dt0, POLY_FP8E4M3) || DT_IS(dt0, POLY_FP8E5M2) || DT_IS(dt0, POLY_FP8E4M3FNUZ) ||
           DT_IS(dt0, POLY_FP8E5M2FNUZ) || DT_IS(dt0, POLY_UINT32) || DT_IS(dt0, POLY_UINT16) ||
           DT_IS(dt0, POLY_UINT8) || DT_IS(dt0, POLY_INT32) || DT_IS(dt0, POLY_INT16) ||
           DT_IS(dt0, POLY_INT8);
  if (DT_IS(dt1, POLY_FLOAT32))
    return DT_IS(dt0, POLY_FLOAT16) || DT_IS(dt0, POLY_BFLOAT16) || DT_IS(dt0, POLY_UINT16) ||
           DT_IS(dt0, POLY_FP8E4M3) || DT_IS(dt0, POLY_FP8E5M2) || DT_IS(dt0, POLY_FP8E4M3FNUZ) ||
           DT_IS(dt0, POLY_FP8E5M2FNUZ) || DT_IS(dt0, POLY_UINT8) || DT_IS(dt0, POLY_INT16) ||
           DT_IS(dt0, POLY_INT8);
  if (DT_IS(dt1, POLY_FLOAT16))
    return DT_IS(dt0, POLY_FP8E4M3) || DT_IS(dt0, POLY_FP8E5M2) || DT_IS(dt0, POLY_FP8E4M3FNUZ) ||
           DT_IS(dt0, POLY_FP8E5M2FNUZ) || DT_IS(dt0, POLY_UINT8) || DT_IS(dt0, POLY_INT8);
  if (DT_IS(dt1, POLY_UINT64))
    return DT_IS(dt0, POLY_UINT32) || DT_IS(dt0, POLY_UINT16) || DT_IS(dt0, POLY_UINT8);
  if (DT_IS(dt1, POLY_UINT32)) return DT_IS(dt0, POLY_UINT16) || DT_IS(dt0, POLY_UINT8);
  if (DT_IS(dt1, POLY_UINT16)) return DT_IS(dt0, POLY_UINT8);
  if (DT_IS(dt1, POLY_INT64))
    return DT_IS(dt0, POLY_UINT32) || DT_IS(dt0, POLY_UINT16) || DT_IS(dt0, POLY_UINT8) ||
           DT_IS(dt0, POLY_INT32) || DT_IS(dt0, POLY_INT16) || DT_IS(dt0, POLY_INT8);
  if (DT_IS(dt1, POLY_INT32))
    return DT_IS(dt0, POLY_UINT16) || DT_IS(dt0, POLY_UINT8) || DT_IS(dt0, POLY_INT16) ||
           DT_IS(dt0, POLY_INT8);
  if (DT_IS(dt1, POLY_INT16)) return DT_IS(dt0, POLY_UINT8) || DT_IS(dt0, POLY_INT8);
#undef DT_IS
  return false;
}

/* Current tinygrad dtype.py:228-291 float_to_fp8/fp8_to_float. */
typedef struct {
  int bias;
  int sig_bits;
  uint8_t mant_mask;
  uint64_t min_denorm_half;
  uint64_t ovf_threshold;
  uint8_t max_norm;
  uint64_t min_norm;
} PolyFP8Config;

static PolyFP8Config fp8_config(PolyDType dtype) {
  if (poly_dtype_eq(dtype, POLY_FP8E4M3))
    return (PolyFP8Config
    ){7,
      4,
      0x7,
      UINT64_C(0x3f50000000000000),
      UINT64_C(0x407d000000000000),
      0x7e,
      UINT64_C(0x3f90000000000000)};
  if (poly_dtype_eq(dtype, POLY_FP8E5M2))
    return (PolyFP8Config
    ){15,
      3,
      0x3,
      UINT64_C(0x3ee0000000000000),
      UINT64_C(0x40ee000000000000) - 1,
      0x7b,
      UINT64_C(0x3f10000000000000)};
  if (poly_dtype_eq(dtype, POLY_FP8E4M3FNUZ))
    return (PolyFP8Config
    ){8,
      4,
      0x7,
      UINT64_C(0x3f40000000000000),
      UINT64_C(0x406f000000000000) - 1,
      0x7f,
      UINT64_C(0x3f80000000000000)};
  return (PolyFP8Config
  ){16,
    3,
    0x3,
    UINT64_C(0x3ed0000000000000),
    UINT64_C(0x40ee000000000000) - 1,
    0x7f,
    UINT64_C(0x3f00000000000000)};
}

uint8_t poly_float_to_fp8(double x, PolyDType dtype) {
  bool fnuz = poly_dtype_is_fp8_fnuz(dtype);
  if (fnuz && !isfinite(x)) return 0x80;
  if (fnuz && x == 0.0) return 0x00;
  if (poly_dtype_eq(dtype, POLY_FP8E4M3) && !isfinite(x)) return copysign(1.0, x) > 0 ? 0x7f : 0xff;
  if (poly_dtype_eq(dtype, POLY_FP8E5M2) && !isfinite(x))
    return (uint8_t)((copysign(1.0, x) > 0 ? 0 : 0x80) | (isinf(x) ? 0x7c : 0x7f));

  PolyFP8Config cfg = fp8_config(dtype);
  uint64_t xbits;
  memcpy(&xbits, &x, sizeof(xbits));
  uint64_t half_ulp = UINT64_C(1) << (52 - cfg.sig_bits);
  uint8_t sign = (uint8_t)(((xbits >> 63) & 1u) << 7);
  int exp = (int)((xbits >> 52) & 0x7ffu) - 1023 + cfg.bias;
  uint64_t mantissa = (xbits >> (53 - cfg.sig_bits)) & cfg.mant_mask;
  uint64_t absx = xbits & UINT64_C(0x7fffffffffffffff);
  uint64_t res;
  if (absx <= cfg.min_denorm_half) {
    res = 0;
  } else if (absx > cfg.ovf_threshold) {
    res = cfg.max_norm;
  } else if (absx >= cfg.min_norm) {
    res = ((uint64_t)exp << (cfg.sig_bits - 1)) | mantissa;
    uint64_t round_bits = xbits & ((half_ulp << 1) - 1);
    if (round_bits > half_ulp || (round_bits == half_ulp && (mantissa & 1))) res++;
  } else {
    int shift = 1 - exp;
    mantissa |= UINT64_C(1) << (cfg.sig_bits - 1);
    res = mantissa >> shift;
    uint64_t half = half_ulp << shift;
    uint64_t round_bits = (xbits | (UINT64_C(1) << 52)) & ((half << 1) - 1);
    if (round_bits > half || (round_bits == half && (res & 1))) res++;
  }
  return (uint8_t)(fnuz && res == 0 ? 0 : res | sign);
}

double poly_fp8_to_float(uint8_t x, PolyDType dtype) {
  bool fnuz = poly_dtype_is_fp8_fnuz(dtype);
  if (fnuz && x == 0x80) return NAN;
  if ((x & 0x7f) == 0) return x & 0x80 ? -0.0 : 0.0;

  PolyFP8Config cfg = fp8_config(dtype);
  int mant_bits = cfg.sig_bits - 1, exp_bits = 8 - cfg.sig_bits;
  int exp_max = (1 << exp_bits) - 1, mant_max = (1 << mant_bits) - 1;
  int sign = (x >> 7) & 1, exp = (x >> mant_bits) & exp_max, mantissa = x & mant_max;
  if (!fnuz && exp == exp_max) {
    if (poly_dtype_eq(dtype, POLY_FP8E5M2))
      return copysign(mantissa ? NAN : INFINITY, sign ? -1.0 : 1.0);
    if (mantissa == mant_max) return NAN;
  }
  double val = exp == 0 ? ((double)mantissa / (mant_max + 1)) * ldexp(1.0, 1 - cfg.bias)
                        : (1.0 + (double)mantissa / (mant_max + 1)) * ldexp(1.0, exp - cfg.bias);
  return sign ? -val : val;
}

int poly_dtype_itemsize(PolyDType dt) {
  return (dt.bitsize + 7) / 8;
}

const char *poly_dtype_name(PolyDType dt) {
  return dt.name;
}
