#ifndef POLY_BIGINT_H
#define POLY_BIGINT_H

#include "polygrad.h"

/* Heap-owned arithmetic value used while folding. PolyArg stores the same
 * canonical signed-magnitude limbs as an immutable arena-owned view.
 *
 * Constructors/arithmetic require zero-initialized, empty output owners,
 * distinct from inputs (and from each other for quotient/remainder). They do
 * not replace populated outputs or support in-place aliasing. Free outputs
 * after either success or failure. To update a value, build a fresh candidate,
 * then free/move only on success, as for Python's immutable integer results. */
typedef struct {
  int sign;
  size_t n_limbs;
  uint32_t *limbs;
} PolyInt;

void poly_int_init(PolyInt *v);
void poly_int_free(PolyInt *v);
bool poly_int_from_i64(PolyInt *out, int64_t value);
bool poly_int_from_arg(PolyInt *out, PolyArg arg);
bool poly_int_from_decimal(PolyInt *out, const char *decimal);
bool poly_int_copy(PolyInt *out, const PolyInt *value);
bool poly_int_to_i64(const PolyInt *value, int64_t *out);
uint64_t poly_int_to_u64_mod(const PolyInt *value);
double poly_int_to_double(const PolyInt *value);
char *poly_int_to_decimal(const PolyInt *value);
int poly_int_cmp(const PolyInt *a, const PolyInt *b);
bool poly_int_is_zero(const PolyInt *value);
bool poly_int_is_negative(const PolyInt *value);

bool poly_int_neg(PolyInt *out, const PolyInt *a);
bool poly_int_add(PolyInt *out, const PolyInt *a, const PolyInt *b);
bool poly_int_sub(PolyInt *out, const PolyInt *a, const PolyInt *b);
bool poly_int_mul(PolyInt *out, const PolyInt *a, const PolyInt *b);
bool poly_int_shl(PolyInt *out, const PolyInt *a, uint64_t shift);
bool poly_int_shr(PolyInt *out, const PolyInt *a, uint64_t shift);
bool poly_int_bitwise(PolyInt *out, PolyOps op, const PolyInt *a, const PolyInt *b);
bool poly_int_divmod(
    PolyInt *quotient,
    PolyInt *remainder,
    const PolyInt *a,
    const PolyInt *b,
    bool floor_mode
);
bool poly_int_pow(PolyInt *out, const PolyInt *base, const PolyInt *exponent);
bool poly_int_truncate(PolyInt *out, const PolyInt *value, int bits, bool is_unsigned);

/* Return an exact PolyArg view. The returned BIGINT limbs alias value and must
 * be copied by poly_uop before value is freed. */
PolyArg poly_int_as_arg(const PolyInt *value);

/* Read helpers for durable UOp arguments. */
bool poly_arg_integer_to_i64(PolyArg arg, int64_t *out);
uint64_t poly_arg_integer_to_u64_mod(PolyArg arg);
double poly_arg_integer_to_double(PolyArg arg);
char *poly_arg_integer_to_decimal(PolyArg arg);
int poly_arg_integer_cmp(PolyArg a, PolyArg b, bool *ok);
int poly_arg_integer_cmp_float(PolyArg integer, double value);
bool poly_arg_python_numeric_eq(PolyArg a, PolyArg b);
int poly_arg_python_numeric_cmp(PolyArg a, PolyArg b, bool *ok);

#endif
