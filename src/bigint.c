#include "bigint.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void normalize(PolyInt *v) {
  while (v->n_limbs > 0 && v->limbs[v->n_limbs - 1] == 0)
    v->n_limbs--;
  if (v->n_limbs == 0) {
    free(v->limbs);
    v->limbs = NULL;
    v->sign = 0;
  }
}

static bool alloc_limbs(PolyInt *v, size_t n) {
  v->limbs = n ? calloc(n, sizeof(uint32_t)) : NULL;
  v->n_limbs = v->limbs ? n : 0;
  v->sign = 0;
  return n == 0 || v->limbs != NULL;
}

void poly_int_init(PolyInt *v) {
  if (v) memset(v, 0, sizeof(*v));
}

void poly_int_free(PolyInt *v) {
  if (!v) return;
  free(v->limbs);
  memset(v, 0, sizeof(*v));
}

bool poly_int_copy(PolyInt *out, const PolyInt *value) {
  if (!out || !value || !alloc_limbs(out, value->n_limbs)) return false;
  if (value->n_limbs) memcpy(out->limbs, value->limbs, value->n_limbs * sizeof(uint32_t));
  out->sign = value->sign;
  return true;
}

bool poly_int_from_i64(PolyInt *out, int64_t value) {
  if (!out) return false;
  uint64_t magnitude = value < 0 ? (uint64_t)(-(value + 1)) + UINT64_C(1) : (uint64_t)value;
  size_t n = magnitude > UINT32_MAX ? 2 : magnitude ? 1 : 0;
  if (!alloc_limbs(out, n)) return false;
  if (n > 0) out->limbs[0] = (uint32_t)magnitude;
  if (n > 1) out->limbs[1] = (uint32_t)(magnitude >> 32);
  out->sign = magnitude ? (value < 0 ? -1 : 1) : 0;
  return true;
}

bool poly_int_from_arg(PolyInt *out, PolyArg arg) {
  if (!out) return false;
  if (arg.kind == POLY_ARG_INT) return poly_int_from_i64(out, arg.i);
  if (arg.kind == POLY_ARG_BOOL) return poly_int_from_i64(out, arg.b ? 1 : 0);
  if (arg.kind != POLY_ARG_BIGINT || arg.bigint.n_limbs == 0 || !arg.bigint.limbs) return false;
  if (!alloc_limbs(out, arg.bigint.n_limbs)) return false;
  memcpy(out->limbs, arg.bigint.limbs, arg.bigint.n_limbs * sizeof(uint32_t));
  out->sign = arg.bigint.sign < 0 ? -1 : 1;
  normalize(out);
  return true;
}

static int cmp_abs(const PolyInt *a, const PolyInt *b) {
  if (a->n_limbs != b->n_limbs) return a->n_limbs < b->n_limbs ? -1 : 1;
  for (size_t i = a->n_limbs; i-- > 0;) {
    if (a->limbs[i] != b->limbs[i]) return a->limbs[i] < b->limbs[i] ? -1 : 1;
  }
  return 0;
}

int poly_int_cmp(const PolyInt *a, const PolyInt *b) {
  if (a->sign != b->sign) return a->sign < b->sign ? -1 : 1;
  if (a->sign == 0) return 0;
  int cmp = cmp_abs(a, b);
  return a->sign < 0 ? -cmp : cmp;
}

bool poly_int_is_zero(const PolyInt *value) {
  return !value || value->sign == 0 || value->n_limbs == 0;
}

bool poly_int_is_negative(const PolyInt *value) {
  return value && value->sign < 0;
}

bool poly_int_to_i64(const PolyInt *value, int64_t *out) {
  if (!value || !out || value->n_limbs > 2) return false;
  uint64_t magnitude = value->n_limbs > 0 ? value->limbs[0] : 0;
  if (value->n_limbs > 1) magnitude |= (uint64_t)value->limbs[1] << 32;
  if (value->sign >= 0) {
    if (magnitude > INT64_MAX) return false;
    *out = (int64_t)magnitude;
    return true;
  }
  if (magnitude > (UINT64_C(1) << 63)) return false;
  *out = magnitude == (UINT64_C(1) << 63) ? INT64_MIN : -(int64_t)magnitude;
  return true;
}

uint64_t poly_int_to_u64_mod(const PolyInt *value) {
  if (!value) return 0;
  uint64_t magnitude = value->n_limbs > 0 ? value->limbs[0] : 0;
  if (value->n_limbs > 1) magnitude |= (uint64_t)value->limbs[1] << 32;
  return value->sign < 0 ? UINT64_C(0) - magnitude : magnitude;
}

double poly_int_to_double(const PolyInt *value) {
  if (!value) return 0.0;
  double out = 0.0;
  for (size_t i = value->n_limbs; i-- > 0;)
    out = ldexp(out, 32) + value->limbs[i];
  return value->sign < 0 ? -out : out;
}

static bool add_abs(PolyInt *out, const PolyInt *a, const PolyInt *b) {
  size_t n = a->n_limbs > b->n_limbs ? a->n_limbs : b->n_limbs;
  if (!alloc_limbs(out, n + 1)) return false;
  uint64_t carry = 0;
  for (size_t i = 0; i < n; i++) {
    uint64_t sum = carry;
    if (i < a->n_limbs) sum += a->limbs[i];
    if (i < b->n_limbs) sum += b->limbs[i];
    out->limbs[i] = (uint32_t)sum;
    carry = sum >> 32;
  }
  out->limbs[n] = (uint32_t)carry;
  out->sign = 1;
  normalize(out);
  return true;
}

/* Requires |a| >= |b|. */
static bool sub_abs(PolyInt *out, const PolyInt *a, const PolyInt *b) {
  if (!alloc_limbs(out, a->n_limbs)) return false;
  uint64_t borrow = 0;
  for (size_t i = 0; i < a->n_limbs; i++) {
    uint64_t av = a->limbs[i];
    uint64_t bv = (i < b->n_limbs ? b->limbs[i] : 0) + borrow;
    out->limbs[i] = (uint32_t)(av - bv);
    borrow = av < bv;
  }
  out->sign = 1;
  normalize(out);
  return true;
}

bool poly_int_neg(PolyInt *out, const PolyInt *a) {
  if (!poly_int_copy(out, a)) return false;
  out->sign = -out->sign;
  return true;
}

bool poly_int_add(PolyInt *out, const PolyInt *a, const PolyInt *b) {
  if (a->sign == 0) return poly_int_copy(out, b);
  if (b->sign == 0) return poly_int_copy(out, a);
  if (a->sign == b->sign) {
    if (!add_abs(out, a, b)) return false;
    out->sign = a->sign;
    return true;
  }
  int cmp = cmp_abs(a, b);
  if (cmp == 0) return alloc_limbs(out, 0);
  if (cmp > 0) {
    if (!sub_abs(out, a, b)) return false;
    out->sign = a->sign;
  } else {
    if (!sub_abs(out, b, a)) return false;
    out->sign = b->sign;
  }
  return true;
}

bool poly_int_sub(PolyInt *out, const PolyInt *a, const PolyInt *b) {
  PolyInt neg = {0};
  if (!poly_int_neg(&neg, b)) return false;
  bool ok = poly_int_add(out, a, &neg);
  poly_int_free(&neg);
  return ok;
}

bool poly_int_mul(PolyInt *out, const PolyInt *a, const PolyInt *b) {
  if (poly_int_is_zero(a) || poly_int_is_zero(b)) return alloc_limbs(out, 0);
  if (!alloc_limbs(out, a->n_limbs + b->n_limbs)) return false;
  for (size_t i = 0; i < a->n_limbs; i++) {
    uint64_t carry = 0;
    for (size_t j = 0; j < b->n_limbs; j++) {
      size_t k = i + j;
      uint64_t cur = (uint64_t)a->limbs[i] * b->limbs[j] + out->limbs[k] + carry;
      out->limbs[k] = (uint32_t)cur;
      carry = cur >> 32;
    }
    size_t k = i + b->n_limbs;
    while (carry && k < out->n_limbs) {
      uint64_t cur = (uint64_t)out->limbs[k] + carry;
      out->limbs[k++] = (uint32_t)cur;
      carry = cur >> 32;
    }
  }
  out->sign = a->sign == b->sign ? 1 : -1;
  normalize(out);
  return true;
}

static size_t bit_length(const PolyInt *a) {
  if (!a || a->n_limbs == 0) return 0;
  uint32_t top = a->limbs[a->n_limbs - 1];
  return (a->n_limbs - 1) * 32 + (size_t)(32 - __builtin_clz(top));
}

bool poly_int_shl(PolyInt *out, const PolyInt *a, uint64_t shift) {
  if (poly_int_is_zero(a)) return alloc_limbs(out, 0);
  if (shift > SIZE_MAX - bit_length(a)) return false;
  size_t words = (size_t)(shift / 32), bits = (size_t)(shift % 32);
  if (!alloc_limbs(out, a->n_limbs + words + (bits ? 1 : 0))) return false;
  uint64_t carry = 0;
  for (size_t i = 0; i < a->n_limbs; i++) {
    uint64_t cur = ((uint64_t)a->limbs[i] << bits) | carry;
    out->limbs[i + words] = (uint32_t)cur;
    carry = cur >> 32;
  }
  if (bits) out->limbs[a->n_limbs + words] = (uint32_t)carry;
  out->sign = a->sign;
  normalize(out);
  return true;
}

static bool shr_abs(PolyInt *out, const PolyInt *a, uint64_t shift, bool *discarded) {
  if (discarded) *discarded = false;
  /* Python rshift consumes the whole count. Compare before narrowing: on
   * wasm32, 2**37 bits otherwise wraps to zero size_t limb words. */
  if (shift / 32 >= a->n_limbs) {
    if (discarded) *discarded = !poly_int_is_zero(a);
    return alloc_limbs(out, 0);
  }
  size_t words = (size_t)(shift / 32), bits = (size_t)(shift % 32);
  if (!alloc_limbs(out, a->n_limbs - words)) return false;
  if (discarded) {
    for (size_t i = 0; i < words; i++)
      if (a->limbs[i]) *discarded = true;
    if (bits && (a->limbs[words] & ((UINT32_C(1) << bits) - 1))) *discarded = true;
  }
  uint32_t carry = 0;
  for (size_t i = a->n_limbs; i-- > words;) {
    uint32_t cur = a->limbs[i];
    out->limbs[i - words] = bits ? (cur >> bits) | (carry << (32 - bits)) : cur;
    carry = bits ? cur & ((UINT32_C(1) << bits) - 1) : 0;
  }
  out->sign = 1;
  normalize(out);
  return true;
}

static bool add_small_abs(PolyInt *value, uint32_t add) {
  if (add == 0) return true;
  if (value->n_limbs == 0) {
    if (!alloc_limbs(value, 1)) return false;
    value->limbs[0] = add;
    value->sign = 1;
    return true;
  }
  uint64_t carry = add;
  for (size_t i = 0; i < value->n_limbs && carry; i++) {
    uint64_t cur = (uint64_t)value->limbs[i] + carry;
    value->limbs[i] = (uint32_t)cur;
    carry = cur >> 32;
  }
  if (carry) {
    uint32_t *grown = realloc(value->limbs, (value->n_limbs + 1) * sizeof(uint32_t));
    if (!grown) return false;
    value->limbs = grown;
    value->limbs[value->n_limbs++] = (uint32_t)carry;
  }
  return true;
}

bool poly_int_shr(PolyInt *out, const PolyInt *a, uint64_t shift) {
  bool discarded = false;
  if (!shr_abs(out, a, shift, &discarded)) return false;
  if (a->sign < 0) {
    if (discarded && !add_small_abs(out, 1)) {
      poly_int_free(out);
      return false;
    }
    out->sign = poly_int_is_zero(out) ? 0 : -1;
  }
  return true;
}

static bool get_bit(const PolyInt *a, size_t bit) {
  size_t word = bit / 32;
  return word < a->n_limbs && ((a->limbs[word] >> (bit % 32)) & 1u);
}

static bool set_bit(PolyInt *a, size_t bit) {
  size_t word = bit / 32;
  if (word >= a->n_limbs) return false;
  a->limbs[word] |= UINT32_C(1) << (bit % 32);
  return true;
}

static bool shl1_inplace(PolyInt *a) {
  if (a->n_limbs == 0) return true;
  uint64_t carry = 0;
  for (size_t i = 0; i < a->n_limbs; i++) {
    uint64_t cur = ((uint64_t)a->limbs[i] << 1) | carry;
    a->limbs[i] = (uint32_t)cur;
    carry = cur >> 32;
  }
  if (carry) {
    uint32_t *grown = realloc(a->limbs, (a->n_limbs + 1) * sizeof(uint32_t));
    if (!grown) return false;
    a->limbs = grown;
    a->limbs[a->n_limbs++] = (uint32_t)carry;
  }
  return true;
}

static bool divmod_abs(PolyInt *q, PolyInt *r, const PolyInt *a, const PolyInt *b) {
  if (poly_int_is_zero(b)) return false;
  size_t bits = bit_length(a);
  if (!alloc_limbs(q, (bits + 31) / 32)) return false;
  if (!alloc_limbs(r, 0)) {
    poly_int_free(q);
    return false;
  }
  for (size_t i = bits; i-- > 0;) {
    if (!shl1_inplace(r) || (get_bit(a, i) && !add_small_abs(r, 1))) goto fail;
    if (cmp_abs(r, b) >= 0) {
      PolyInt next = {0};
      if (!sub_abs(&next, r, b)) goto fail;
      poly_int_free(r);
      *r = next;
      if (!set_bit(q, i)) goto fail;
    }
  }
  q->sign = q->n_limbs ? 1 : 0;
  r->sign = r->n_limbs ? 1 : 0;
  normalize(q);
  normalize(r);
  return true;
fail:
  poly_int_free(q);
  poly_int_free(r);
  return false;
}

bool poly_int_divmod(
    PolyInt *quotient,
    PolyInt *remainder,
    const PolyInt *a,
    const PolyInt *b,
    bool floor_mode
) {
  if (!quotient || !remainder || !a || !b || poly_int_is_zero(b)) return false;
  PolyInt aa = {0}, bb = {0};
  if (!poly_int_copy(&aa, a) || !poly_int_copy(&bb, b)) goto fail;
  aa.sign = aa.n_limbs ? 1 : 0;
  bb.sign = bb.n_limbs ? 1 : 0;
  if (!divmod_abs(quotient, remainder, &aa, &bb)) goto fail;
  bool signs_differ = a->sign != b->sign;
  if (floor_mode && signs_differ && !poly_int_is_zero(remainder)) {
    if (!add_small_abs(quotient, 1)) goto fail_out;
    quotient->sign = -1;
    PolyInt adjusted = {0};
    if (!sub_abs(&adjusted, &bb, remainder)) goto fail_out;
    poly_int_free(remainder);
    *remainder = adjusted;
    remainder->sign = b->sign;
  } else {
    quotient->sign = poly_int_is_zero(quotient) ? 0 : signs_differ ? -1 : 1;
    remainder->sign = poly_int_is_zero(remainder) ? 0 : a->sign;
  }
  poly_int_free(&aa);
  poly_int_free(&bb);
  return true;
fail_out:
  poly_int_free(quotient);
  poly_int_free(remainder);
fail:
  poly_int_free(&aa);
  poly_int_free(&bb);
  return false;
}

static bool to_twos(uint32_t *dst, size_t n, const PolyInt *value) {
  memset(dst, 0, n * sizeof(uint32_t));
  size_t copy_n = value->n_limbs < n ? value->n_limbs : n;
  if (copy_n) memcpy(dst, value->limbs, copy_n * sizeof(uint32_t));
  if (value->sign >= 0) return true;
  uint64_t carry = 1;
  for (size_t i = 0; i < n; i++) {
    uint64_t cur = (uint64_t)(~dst[i]) + carry;
    dst[i] = (uint32_t)cur;
    carry = cur >> 32;
  }
  return true;
}

bool poly_int_bitwise(PolyInt *out, PolyOps op, const PolyInt *a, const PolyInt *b) {
  if (op != POLY_OP_AND && op != POLY_OP_OR && op != POLY_OP_XOR) return false;
  size_t bits = (bit_length(a) > bit_length(b) ? bit_length(a) : bit_length(b)) + 1;
  size_t n = (bits + 31) / 32;
  uint32_t *ta = calloc(n, sizeof(uint32_t)), *tb = calloc(n, sizeof(uint32_t));
  if (!ta || !tb || !alloc_limbs(out, n)) {
    free(ta);
    free(tb);
    return false;
  }
  to_twos(ta, n, a);
  to_twos(tb, n, b);
  for (size_t i = 0; i < n; i++)
    out->limbs[i] = op == POLY_OP_AND  ? ta[i] & tb[i]
                    : op == POLY_OP_OR ? ta[i] | tb[i]
                                       : ta[i] ^ tb[i];
  bool negative = (out->limbs[n - 1] >> ((bits - 1) % 32)) & 1u;
  if (bits % 32) out->limbs[n - 1] &= (UINT32_C(1) << (bits % 32)) - 1;
  if (negative) {
    uint64_t carry = 1;
    for (size_t i = 0; i < n; i++) {
      uint64_t cur = (uint64_t)(~out->limbs[i]) + carry;
      out->limbs[i] = (uint32_t)cur;
      carry = cur >> 32;
    }
    if (bits % 32) out->limbs[n - 1] &= (UINT32_C(1) << (bits % 32)) - 1;
    out->sign = -1;
  } else {
    out->sign = 1;
  }
  normalize(out);
  free(ta);
  free(tb);
  return true;
}

static bool to_u64_exact_nonnegative(const PolyInt *value, uint64_t *out) {
  if (!value || !out || value->sign < 0 || value->n_limbs > 2) return false;
  *out = value->n_limbs ? value->limbs[0] : 0;
  if (value->n_limbs > 1) *out |= (uint64_t)value->limbs[1] << 32;
  return true;
}

bool poly_int_pow(PolyInt *out, const PolyInt *base, const PolyInt *exponent) {
  uint64_t exp = 0;
  if (!to_u64_exact_nonnegative(exponent, &exp)) return false;
  PolyInt result = {0}, factor = {0};
  if (!poly_int_from_i64(&result, 1) || !poly_int_copy(&factor, base)) goto fail;
  while (exp) {
    if (exp & 1) {
      PolyInt next = {0};
      if (!poly_int_mul(&next, &result, &factor)) goto fail;
      poly_int_free(&result);
      result = next;
    }
    exp >>= 1;
    if (exp) {
      PolyInt next = {0};
      if (!poly_int_mul(&next, &factor, &factor)) goto fail;
      poly_int_free(&factor);
      factor = next;
    }
  }
  poly_int_free(&factor);
  *out = result;
  return true;
fail:
  poly_int_free(&result);
  poly_int_free(&factor);
  return false;
}

bool poly_int_truncate(PolyInt *out, const PolyInt *value, int bits, bool is_unsigned) {
  if (!out || !value || bits <= 0) return false;
  size_t n = ((size_t)bits + 31) / 32;
  uint32_t *twos = calloc(n, sizeof(uint32_t));
  if (!twos) return false;
  to_twos(twos, n, value);
  if (bits % 32) twos[n - 1] &= (UINT32_C(1) << (bits % 32)) - 1;
  bool negative = !is_unsigned && ((twos[(bits - 1) / 32] >> ((bits - 1) % 32)) & 1u);
  if (!alloc_limbs(out, n)) {
    free(twos);
    return false;
  }
  memcpy(out->limbs, twos, n * sizeof(uint32_t));
  if (negative) {
    uint64_t carry = 1;
    for (size_t i = 0; i < n; i++) {
      uint64_t cur = (uint64_t)(~out->limbs[i]) + carry;
      out->limbs[i] = (uint32_t)cur;
      carry = cur >> 32;
    }
    if (bits % 32) out->limbs[n - 1] &= (UINT32_C(1) << (bits % 32)) - 1;
    out->sign = -1;
  } else {
    out->sign = 1;
  }
  normalize(out);
  free(twos);
  return true;
}

static uint32_t div_small_inplace(PolyInt *value, uint32_t divisor) {
  uint64_t remainder = 0;
  for (size_t i = value->n_limbs; i-- > 0;) {
    uint64_t cur = (remainder << 32) | value->limbs[i];
    value->limbs[i] = (uint32_t)(cur / divisor);
    remainder = cur % divisor;
  }
  normalize(value);
  return (uint32_t)remainder;
}

char *poly_int_to_decimal(const PolyInt *value) {
  if (!value || poly_int_is_zero(value)) {
    char *zero = malloc(2);
    if (zero) memcpy(zero, "0", 2);
    return zero;
  }
  /* Each 32-bit limb needs at most ten decimal digits, plus sign and NUL.
   * Unlike bit_length * log10(2)'s integer approximation, this checked bound
   * cannot wrap on wasm32 for modest (~19KB) integer values. */
  if (value->n_limbs > (SIZE_MAX - 2) / 10) return NULL;
  size_t cap = value->n_limbs * 10 + 2;
  PolyInt tmp = {0};
  if (!poly_int_copy(&tmp, value)) return NULL;
  tmp.sign = 1;
  char *digits = malloc(cap);
  uint32_t *chunks = malloc((cap / 9 + 2) * sizeof(uint32_t));
  if (!digits || !chunks) {
    free(digits);
    free(chunks);
    poly_int_free(&tmp);
    return NULL;
  }
  size_t n_chunks = 0;
  while (!poly_int_is_zero(&tmp))
    chunks[n_chunks++] = div_small_inplace(&tmp, 1000000000u);
  size_t pos = 0;
  if (value->sign < 0) digits[pos++] = '-';
  pos += (size_t)snprintf(digits + pos, cap - pos, "%u", chunks[n_chunks - 1]);
  for (size_t i = n_chunks - 1; i-- > 0;)
    pos += (size_t)snprintf(digits + pos, cap - pos, "%09u", chunks[i]);
  digits[pos] = '\0';
  free(chunks);
  poly_int_free(&tmp);
  return digits;
}

static bool mul_small_inplace(PolyInt *value, uint32_t factor) {
  if (factor == 0 || poly_int_is_zero(value)) {
    poly_int_free(value);
    return alloc_limbs(value, 0);
  }
  uint64_t carry = 0;
  for (size_t i = 0; i < value->n_limbs; i++) {
    uint64_t cur = (uint64_t)value->limbs[i] * factor + carry;
    value->limbs[i] = (uint32_t)cur;
    carry = cur >> 32;
  }
  if (carry) {
    uint32_t *grown = realloc(value->limbs, (value->n_limbs + 1) * sizeof(uint32_t));
    if (!grown) return false;
    value->limbs = grown;
    value->limbs[value->n_limbs++] = (uint32_t)carry;
  }
  return true;
}

bool poly_int_from_decimal(PolyInt *out, const char *decimal) {
  if (!out || !decimal || !*decimal) return false;
  int sign = 1;
  if (*decimal == '-' || *decimal == '+') {
    if (*decimal++ == '-') sign = -1;
  }
  if (!*decimal || !alloc_limbs(out, 0)) return false;
  for (; *decimal; decimal++) {
    if (*decimal < '0' || *decimal > '9') {
      poly_int_free(out);
      return false;
    }
    if (!mul_small_inplace(out, 10) || !add_small_abs(out, (uint32_t)(*decimal - '0'))) {
      poly_int_free(out);
      return false;
    }
  }
  out->sign = poly_int_is_zero(out) ? 0 : sign;
  return true;
}

PolyArg poly_int_as_arg(const PolyInt *value) {
  int64_t small = 0;
  if (poly_int_to_i64(value, &small)) return poly_arg_int(small);
  return poly_arg_bigint(value->sign, value->limbs, (uint32_t)value->n_limbs);
}

bool poly_arg_integer_to_i64(PolyArg arg, int64_t *out) {
  PolyInt value = {0};
  if (!poly_int_from_arg(&value, arg)) return false;
  bool ok = poly_int_to_i64(&value, out);
  poly_int_free(&value);
  return ok;
}

uint64_t poly_arg_integer_to_u64_mod(PolyArg arg) {
  PolyInt value = {0};
  if (!poly_int_from_arg(&value, arg)) return 0;
  uint64_t out = poly_int_to_u64_mod(&value);
  poly_int_free(&value);
  return out;
}

double poly_arg_integer_to_double(PolyArg arg) {
  PolyInt value = {0};
  if (!poly_int_from_arg(&value, arg)) return 0.0;
  double out = poly_int_to_double(&value);
  poly_int_free(&value);
  return out;
}

char *poly_arg_integer_to_decimal(PolyArg arg) {
  PolyInt value = {0};
  if (!poly_int_from_arg(&value, arg)) return NULL;
  char *out = poly_int_to_decimal(&value);
  poly_int_free(&value);
  return out;
}

int poly_arg_integer_cmp(PolyArg a, PolyArg b, bool *ok) {
  PolyInt ia = {0}, ib = {0};
  bool valid = poly_int_from_arg(&ia, a) && poly_int_from_arg(&ib, b);
  int out = valid ? poly_int_cmp(&ia, &ib) : 0;
  poly_int_free(&ia);
  poly_int_free(&ib);
  if (ok) *ok = valid;
  return out;
}

int poly_arg_integer_cmp_float(PolyArg integer, double value) {
  /* Python compares arbitrary-size ints and binary64 values exactly. */
  if (isnan(value)) return 0;
  if (isinf(value)) return value < 0.0 ? 1 : -1;

  PolyInt lhs = {0};
  if (!poly_int_from_arg(&lhs, integer)) return 0;
  if (value == 0.0) {
    int ret = lhs.sign < 0 ? -1 : lhs.sign > 0 ? 1 : 0;
    poly_int_free(&lhs);
    return ret;
  }

  int float_sign = signbit(value) ? -1 : 1;
  if (lhs.sign != float_sign) {
    int ret = lhs.sign < float_sign ? -1 : 1;
    poly_int_free(&lhs);
    return ret;
  }
  lhs.sign = 1;

  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  uint64_t exponent = (bits >> 52) & UINT64_C(0x7ff);
  uint64_t significand = bits & UINT64_C(0x000fffffffffffff);
  int shift;
  if (exponent == 0) {
    shift = -1074;
  } else {
    significand |= UINT64_C(1) << 52;
    shift = (int)exponent - 1023 - 52;
  }

  PolyInt rhs = {0}, scaled = {0};
  bool ok = poly_int_from_i64(&rhs, (int64_t)significand);
  int magnitude_cmp = 0;
  if (ok && shift >= 0) {
    ok = poly_int_shl(&scaled, &rhs, (uint64_t)shift);
    if (ok) magnitude_cmp = poly_int_cmp(&lhs, &scaled);
  } else if (ok) {
    ok = poly_int_shl(&scaled, &lhs, (uint64_t)-shift);
    if (ok) magnitude_cmp = poly_int_cmp(&scaled, &rhs);
  }
  poly_int_free(&scaled);
  poly_int_free(&rhs);
  poly_int_free(&lhs);
  if (!ok) return 0;
  return float_sign < 0 ? -magnitude_cmp : magnitude_cmp;
}

bool poly_arg_python_numeric_eq(PolyArg a, PolyArg b) {
  /* Tinygrad UPat.match uses Python == for literal arguments. */
  bool a_int = a.kind == POLY_ARG_BOOL || a.kind == POLY_ARG_INT || a.kind == POLY_ARG_BIGINT;
  bool b_int = b.kind == POLY_ARG_BOOL || b.kind == POLY_ARG_INT || b.kind == POLY_ARG_BIGINT;
  if (a_int && b_int) {
    bool ok = false;
    return poly_arg_integer_cmp(a, b, &ok) == 0 && ok;
  }
  if (a_int && b.kind == POLY_ARG_FLOAT)
    return !isnan(b.f) && poly_arg_integer_cmp_float(a, b.f) == 0;
  if (a.kind == POLY_ARG_FLOAT && b_int)
    return !isnan(a.f) && poly_arg_integer_cmp_float(b, a.f) == 0;
  if (a.kind == POLY_ARG_FLOAT && b.kind == POLY_ARG_FLOAT)
    return (isnan(a.f) && isnan(b.f)) || a.f == b.f;
  return false;
}
