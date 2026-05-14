/*
 * sym.c — Symbolic simplification rules (phase 1)
 *
 * Mirrors tinygrad's symbolic_simple: self-folding, zero-folding,
 * constant folding, cast folding, basic identities.
 */

#include "pat.h"
#include "uop_cache_internal.h"
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include "utils.h"

/* Overflow-safe int64 helpers */

static bool i64_add_ok(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_add_overflow(a, b, out);
}
static bool i64_sub_ok(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_sub_overflow(a, b, out);
}
static bool i64_mul_ok(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_mul_overflow(a, b, out);
}
static bool i64_neg_ok(int64_t a, int64_t *out) {
  if (a == INT64_MIN) return false;
  *out = -a;
  return true;
}
static bool i64_shl_ok(int64_t a, int64_t shift, int64_t *out) {
  if (shift < 0 || shift >= 63 || a < 0) return false;
  if (a > (INT64_MAX >> shift)) return false;
  *out = a << shift;
  return true;
}

/* vmin/vmax bounds (port of tinygrad's UOp._min_max) *
 * Full port of tinygrad/uop/ops.py:856-897 (UOp._min_max). The switch in
 * tinygrad is gated on GroupOp.Binary (uop/__init__.py:111-112) with an
 * outer `not dtypes.is_float(self.dtype)` guard at line 858. Polygrad
 * mirrors the exact case ordering and guards; see inline tinygrad refs.
 *
 * Ops covered:
 *   Binary block (non-float dtype):
 *     ADD  line 860
 *     SUB  line 861
 *     AND  line 862 (int with non-neg const)
 *     MUL  line 864 (4-corner)
 *     SHL  line 866 (const rhs)
 *     SHR  line 867 (const rhs)
 *     MOD  lines 868-872 (three cases by divisor sign)
 *     IDIV lines 873-876 (sign-definite divisor only)
 *     XOR  line 877 (with -1 only: bitwise NOT)
 *     MAX  line 878
 *     CMPLT line 879
 *     CMPNE line 880
 *     OR   line 881 (bool dtype)
 *     AND  line 882 (bool dtype)
 *   Post-Binary:
 *     WHERE        line 884 (int dtype only)
 *     DEFINE_VAR   line 887
 *     RANGE/SPECIAL line 888 (src[0].vmax - 1 via identity (s-1).vmax)
 *     BIND         line 889 (passthrough src[0])
 *     UNROLL/VECTORIZE line 890 (min/max over srcs)
 *     CONST        line 891
 *     VCONST       line 892
 *     GEP          line 893 (passthrough src[0])
 *     CAST         lines 895-896 (monotone targets only: float/signed int)
 *
 * Not ported (deliberate divergences):
 *   - NEG: tinygrad's _min_max has no NEG case; falls through to dtype
 *     bounds. Polygrad does likewise for strict parity.
 *   - CMPEQ: in GroupOp.Binary but no dispatch case in tinygrad's switch
 *     (only CMPLT and CMPNE are handled). Falls through to dtype bounds.
 *   - BITCAST: not monotone, falls through to dtype bounds.
 *   - PARAM: tinygrad's rule reads src[2].arg/src[3].arg; polygrad's
 *     POLY_OP_PARAM (rangeify.c:1582) is a zero-src buffer index with
 *     no bounds info. Falls through to dtype bounds.
 *   - Float dtype: sentinel (INT64_MIN, INT64_MAX). Phase D never queries
 *     float bounds; callers must check poly_dtype_is_float first.
 *
 * Parity: test/parity_scripts/tg_minmax_gt.py captures the ground truth,
 * test/test_sym.c asserts every case verbatim. */

/* C-style integer division (truncates toward zero). Matches tinygrad's
 * helpers.py:58 cdiv: abs(x)//abs(y) * sign(x*y) if y != 0 else 0. */
static int64_t cdiv(int64_t x, int64_t y) {
  if (y == 0) return 0;
  int64_t ax = x < 0 ? -x : x;
  int64_t ay = y < 0 ? -y : y;
  int64_t q = ax / ay;
  return (x < 0) != (y < 0) ? -q : q;
}

static int64_t dtype_min(PolyDType dt) {
  if (poly_dtype_eq(dt, POLY_BOOL)) return 0;
  if (poly_dtype_eq(dt, POLY_INT8)) return INT8_MIN;
  if (poly_dtype_eq(dt, POLY_UINT8)) return 0;
  if (poly_dtype_eq(dt, POLY_INT16)) return INT16_MIN;
  if (poly_dtype_eq(dt, POLY_UINT16)) return 0;
  if (poly_dtype_eq(dt, POLY_INT32)) return INT32_MIN;
  if (poly_dtype_eq(dt, POLY_UINT32)) return 0;
  if (poly_dtype_eq(dt, POLY_INT64)) return INT64_MIN;
  if (poly_dtype_eq(dt, POLY_UINT64)) return 0;
  /* float / index / unknown: conservative */
  return INT64_MIN / 2;
}

static int64_t dtype_max(PolyDType dt) {
  if (poly_dtype_eq(dt, POLY_BOOL)) return 1;
  if (poly_dtype_eq(dt, POLY_INT8)) return INT8_MAX;
  if (poly_dtype_eq(dt, POLY_UINT8)) return UINT8_MAX;
  if (poly_dtype_eq(dt, POLY_INT16)) return INT16_MAX;
  if (poly_dtype_eq(dt, POLY_UINT16)) return UINT16_MAX;
  if (poly_dtype_eq(dt, POLY_INT32)) return INT32_MAX;
  if (poly_dtype_eq(dt, POLY_UINT32)) return UINT32_MAX;
  if (poly_dtype_eq(dt, POLY_INT64)) return INT64_MAX;
  if (poly_dtype_eq(dt, POLY_UINT64)) return INT64_MAX; /* clamped */
  return INT64_MAX / 2;
}

typedef struct MinMaxBox {
  int64_t lo, hi;
} MinMaxBox;


static int64_t min4(int64_t a, int64_t b, int64_t c, int64_t d) {
  int64_t x = a < b ? a : b;
  int64_t y = c < d ? c : d;
  return x < y ? x : y;
}
static int64_t max4(int64_t a, int64_t b, int64_t c, int64_t d) {
  int64_t x = a > b ? a : b;
  int64_t y = c > d ? c : d;
  return x > y ? x : y;
}
static int64_t i64_min(int64_t a, int64_t b) {
  return a < b ? a : b;
}
static int64_t i64_max(int64_t a, int64_t b) {
  return a > b ? a : b;
}

static void poly_uop_minmax_rec(
    PolyCtx *ctx,
    PolyUOp *u,
    int64_t *vmin,
    int64_t *vmax,
    PolyMap *memo
);

static void minmax_src(PolyCtx *ctx, PolyUOp *s, int64_t *lo, int64_t *hi, PolyMap *memo) {
  poly_uop_minmax_rec(ctx, s, lo, hi, memo);
}

/* Public single-call wrapper. Allocates a throwaway memo per call; Phase D
 * hot paths should use poly_uop_minmax_ex with a shared PolyUOpCache. */
void poly_uop_minmax(PolyCtx *ctx, PolyUOp *u, int64_t *vmin, int64_t *vmax) {
  PolyMap *memo = poly_map_new(64);
  poly_uop_minmax_rec(ctx, u, vmin, vmax, memo);
  poly_map_destroy(memo);
}

void poly_uop_minmax_ex(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOpCache *cache,
    int64_t *vmin,
    int64_t *vmax
) {
  PolyMap *memo = cache ? poly_uop_cache_minmax_map(cache) : NULL;
  if (memo)
    poly_uop_minmax_rec(ctx, u, vmin, vmax, memo);
  else
    poly_uop_minmax(ctx, u, vmin, vmax);
}

static void poly_uop_minmax_rec(
    PolyCtx *ctx,
    PolyUOp *u,
    int64_t *vmin,
    int64_t *vmax,
    PolyMap *memo
) {
  if (!u) {
    *vmin = 0;
    *vmax = 0;
    return;
  }

  uint32_t h = poly_ptr_hash(u);
  MinMaxBox *cached = memo ? (MinMaxBox *)poly_map_get(memo, h, u, poly_ptr_eq) : NULL;
  if (cached) {
    *vmin = cached->lo;
    *vmax = cached->hi;
    return;
  }

  /* CONST */
  if (u->op == POLY_OP_CONST) {
    if (poly_dtype_is_float(u->dtype)) {
      /* floats are tracked at int64 precision; matches prior behavior
       * and tinygrad's integer-bounds-only pattern matching. */
      *vmin = *vmax = (int64_t)u->arg.f;
    } else if (poly_dtype_eq(u->dtype, POLY_BOOL)) {
      *vmin = *vmax = u->arg.b ? 1 : 0;
    } else {
      *vmin = *vmax = u->arg.i;
    }
    goto done;
  }

  /* VCONST — child CONSTs in src[]. Min/max over lanes. */
  if (u->op == POLY_OP_VCONST && u->n_src > 0) {
    int64_t lo = INT64_MAX, hi = INT64_MIN;
    for (int i = 0; i < u->n_src; i++) {
      int64_t a, b;
      minmax_src(ctx, u->src[i], &a, &b, memo);
      if (a < lo) lo = a;
      if (b > hi) hi = b;
    }
    *vmin = lo;
    *vmax = hi;
    goto done;
  }

  /* DEFINE_VAR: (name, min_val, max_val) */
  if (u->op == POLY_OP_DEFINE_VAR && u->arg.kind == POLY_ARG_DEFINE_VAR) {
    *vmin = u->arg.define_var.min_val;
    *vmax = u->arg.define_var.max_val;
    goto done;
  }

  /* RANGE / SPECIAL: tinygrad ops.py:888
   *   if self.op in (Ops.RANGE, Ops.SPECIAL): return 0, (self.src[0]-1).vmax
   * Tinygrad constructs (src[0] - 1) as a real UOp and recursively queries
   * its vmax. SUB's rule (line 861) gives `s0_vmax - s1_vmin`; with
   * s1 = CONST(1) (vmin=vmax=1), that simplifies to `src[0].vmax - 1`,
   * which is mathematically identical and avoids the allocation. */
  if ((u->op == POLY_OP_RANGE || u->op == POLY_OP_SPECIAL) && u->n_src >= 1) {
    int64_t lo, hi;
    minmax_src(ctx, u->src[0], &lo, &hi, memo);
    *vmin = 0;
    *vmax = hi - 1;
    goto done;
  }

  /* BIND: passthrough src[0] (ignore bound value) */
  if (u->op == POLY_OP_BIND && u->n_src >= 1) {
    minmax_src(ctx, u->src[0], vmin, vmax, memo);
    goto done;
  }

  /* GEP: passthrough src[0] */
  if (u->op == POLY_OP_GEP && u->n_src >= 1) {
    minmax_src(ctx, u->src[0], vmin, vmax, memo);
    goto done;
  }

  /* UNROLL/VECTORIZE: min/max over all srcs */
  if ((u->op == POLY_OP_UNROLL || u->op == POLY_OP_VECTORIZE) && u->n_src > 0) {
    int64_t lo = INT64_MAX, hi = INT64_MIN;
    for (int i = 0; i < u->n_src; i++) {
      int64_t a, b;
      minmax_src(ctx, u->src[i], &a, &b, memo);
      if (a < lo) lo = a;
      if (b > hi) hi = b;
    }
    *vmin = lo;
    *vmax = hi;
    goto done;
  }

  /* Binary ops — gated on `not is_float` to match tinygrad ops.py:858.
   * Float-dtype binary ops fall through to the dtype-bounds default; their
   * NaN handling makes interval arithmetic unsafe. CMPLT/CMPNE on float
   * operands are dispatched separately below since their result is bool. */
  if (u->n_src == 2 && !poly_dtype_is_float(u->dtype)) {
    int64_t a0, a1, b0, b1;
    minmax_src(ctx, u->src[0], &a0, &a1, memo);
    minmax_src(ctx, u->src[1], &b0, &b1, memo);

    if (u->op == POLY_OP_ADD) {
      int64_t lo, hi;
      if (i64_add_ok(a0, b0, &lo) && i64_add_ok(a1, b1, &hi)) {
        *vmin = lo;
        *vmax = hi;
        goto done;
      }
      /* overflow: fall through */
    }
    if (u->op == POLY_OP_SUB) {
      int64_t lo, hi;
      if (i64_sub_ok(a0, b1, &lo) && i64_sub_ok(a1, b0, &hi)) {
        *vmin = lo;
        *vmax = hi;
        goto done;
      }
      /* overflow: fall through */
    }
    if (u->op == POLY_OP_MUL) {
      int64_t v0, v1, v2, v3;
      if (i64_mul_ok(a0, b0, &v0) && i64_mul_ok(a0, b1, &v1) && i64_mul_ok(a1, b0, &v2) &&
          i64_mul_ok(a1, b1, &v3)) {
        *vmin = min4(v0, v1, v2, v3);
        *vmax = max4(v0, v1, v2, v3);
        goto done;
      }
      /* overflow: fall through */
    }
    if (u->op == POLY_OP_MAX) {
      *vmin = i64_max(a0, b0);
      *vmax = i64_max(a1, b1);
      goto done;
    }
    if (u->op == POLY_OP_MOD) {
      /* tinygrad ops.py:868-872 */
      if (b0 == b1 && b0 > 0) {
        int64_t c = b0;
        int64_t lo = (a0 > 0) ? 0 : (a0 >= -c + 1 && a0 <= 0 ? a0 : -(c - 1));
        int64_t hi = (a1 < 0) ? 0 : (a1 >= 0 && a1 < c ? a1 : c - 1);
        *vmin = lo;
        *vmax = hi;
        goto done;
      }
      if (b0 > 0) {
        if (a0 >= 0) {
          *vmin = 0;
          *vmax = b1 - 1;
        } else if (a1 <= 0) {
          *vmin = -(b1 - 1);
          *vmax = 0;
        } else {
          *vmin = -(b1 - 1);
          *vmax = b1 - 1;
        }
        goto done;
      }
      if (b1 < 0) {
        int64_t m = -b0 - 1;
        if (a0 >= 0) {
          *vmin = 0;
          *vmax = m;
        } else if (a1 <= 0) {
          *vmin = -m;
          *vmax = 0;
        } else {
          *vmin = -m;
          *vmax = m;
        }
        goto done;
      }
    }
    if (u->op == POLY_OP_IDIV) {
      /* Only handle the case where the divisor sign is known */
      /* Tinygrad ops.py:875 uses `s1_vmin*s1_vmax>0` which can overflow
       * int64. The same-sign check below is equivalent and overflow-safe;
       * matches the idiom already used in fold_divmod_general. */
      if ((b0 > 0 && b1 > 0) || (b0 < 0 && b1 < 0)) {
        int64_t v0 = cdiv(a0, b0), v1 = cdiv(a0, b1);
        int64_t v2 = cdiv(a1, b0), v3 = cdiv(a1, b1);
        *vmin = min4(v0, v1, v2, v3);
        *vmax = max4(v0, v1, v2, v3);
        goto done;
      }
    }
    if (u->op == POLY_OP_SHL && b0 == b1 && b0 >= 0 && b0 < 63) {
      int64_t v0, v1;
      if (i64_shl_ok(a0, b0, &v0) && i64_shl_ok(a1, b0, &v1)) {
        *vmin = i64_min(v0, v1);
        *vmax = i64_max(v0, v1);
        goto done;
      }
      /* overflow or negative lhs: fall through */
    }
    if (u->op == POLY_OP_SHR && b0 == b1 && b0 >= 0 && b0 < 63) {
      *vmin = a0 >> b0;
      *vmax = a1 >> b0;
      goto done;
    }
    if (u->op == POLY_OP_XOR && b0 == b1 && b0 == -1) {
      /* ~x: bitwise not */
      *vmin = ~a1;
      *vmax = ~a0;
      goto done;
    }
    if (u->op == POLY_OP_AND && poly_dtype_is_int(u->dtype) && b0 == b1 && b0 >= 0) {
      /* tinygrad ops.py:862-863:
       *   if self.op is Ops.AND and dtypes.is_int(self.dtype)
       *      and s1_vmin == s1_vmax >= 0:
       *     return 0, s1_vmax if s0_vmin < 0 else min(s0_vmax, s1_vmax) */
      *vmin = 0;
      *vmax = (a0 < 0) ? b1 : i64_min(a1, b1);
      goto done;
    }
  }

  /* Bool binary ops (AND / OR on bool dtype) */
  if (u->n_src == 2 && poly_dtype_eq(u->dtype, POLY_BOOL)) {
    int64_t a0, a1, b0, b1;
    minmax_src(ctx, u->src[0], &a0, &a1, memo);
    minmax_src(ctx, u->src[1], &b0, &b1, memo);
    if (u->op == POLY_OP_AND) {
      *vmin = (a0 && b0) ? 1 : 0;
      *vmax = (a1 && b1) ? 1 : 0;
      goto done;
    }
    if (u->op == POLY_OP_OR) {
      *vmin = (a0 || b0) ? 1 : 0;
      *vmax = (a1 || b1) ? 1 : 0;
      goto done;
    }
  }

  /* Comparisons: always bool, regardless of operand dtype */
  if (u->n_src == 2 && (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPNE)) {
    int64_t a0, a1, b0, b1;
    minmax_src(ctx, u->src[0], &a0, &a1, memo);
    minmax_src(ctx, u->src[1], &b0, &b1, memo);
    if (u->op == POLY_OP_CMPLT) {
      *vmin = (a1 < b0) ? 1 : 0;
      *vmax = (a0 < b1) ? 1 : 0;
      goto done;
    }
    /* CMPNE */
    bool def_ne = (a1 < b0) || (b1 < a0);
    bool all_eq = (a0 == a1) && (b0 == b1) && (a0 == b0);
    *vmin = def_ne ? 1 : 0;
    *vmax = all_eq ? 0 : 1;
    goto done;
  }

  /* WHERE (int branches): min/max over both branches */
  if (u->op == POLY_OP_WHERE && u->n_src == 3 && poly_dtype_is_int(u->dtype)) {
    int64_t t0, t1, f0, f1;
    minmax_src(ctx, u->src[1], &t0, &t1, memo);
    minmax_src(ctx, u->src[2], &f0, &f1, memo);
    *vmin = i64_min(t0, f0);
    *vmax = i64_max(t1, f1);
    goto done;
  }

  /* CAST: clamp src[0] bounds to dtype range. Matches tinygrad ops.py:895 —
   * only monotone casts. Cast to bool/unsigned is not necessarily monotone;
   * fall through to dtype bounds for those. */
  if (u->op == POLY_OP_CAST && u->n_src >= 1) {
    bool monotone = poly_dtype_is_float(u->dtype) ||
                    (poly_dtype_is_int(u->dtype) && !poly_dtype_is_unsigned(u->dtype) &&
                     !poly_dtype_eq(u->dtype, POLY_BOOL));
    if (monotone) {
      int64_t a0, a1;
      minmax_src(ctx, u->src[0], &a0, &a1, memo);
      *vmin = i64_max(dtype_min(u->dtype), a0);
      *vmax = i64_min(a1, dtype_max(u->dtype));
      goto done;
    }
  }

  /* Fallback: dtype range */
  *vmin = dtype_min(u->dtype);
  *vmax = dtype_max(u->dtype);

done:
  if (memo && ctx) {
    MinMaxBox *box = poly_arena_alloc(poly_ctx_arena(ctx), sizeof(MinMaxBox), _Alignof(MinMaxBox));
    if (box) {
      box->lo = *vmin;
      box->hi = *vmax;
      poly_map_set(memo, h, u, box, poly_ptr_eq);
    }
  }
}

/* Rewrite callbacks */

/* Self-folding: return x */
static PolyUOp *rule_identity(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)root;
  return poly_bind(b, "x");
}

/* x+0 -> x, x*1 -> x, x^0 -> x, x//1 -> x */
/* All use rule_identity — pattern does the matching */

/* x//x -> 1 */
static PolyUOp *rule_div_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_int(ctx, poly_bind(b, "x"), 1);
}

/* x//-1 -> -x */
static PolyUOp *rule_div_neg1(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  return poly_uop1(ctx, POLY_OP_NEG, x->dtype, x, poly_arg_none());
}

/* Idempotent(x, x) -> x (OR, AND, MAX) */
static PolyUOp *rule_idempotent(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)root;
  return poly_bind(b, "x");
}

/* Zero-folding: x < x -> False */
static PolyUOp *rule_lt_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_bool(ctx, poly_bind(b, "x"), false);
}

/* x != x -> false (int/bool only; float NaN!=NaN is true) */
static PolyUOp *rule_cmpne_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  if (poly_dtype_is_float(x->dtype)) return NULL; /* NaN != NaN */
  return poly_const_like_bool(ctx, x, false);
}

/* x % x -> 0 */
static PolyUOp *rule_mod_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_int(ctx, poly_bind(b, "x"), 0);
}

/* x ^ x -> 0 */
static PolyUOp *rule_xor_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_int(ctx, poly_bind(b, "x"), 0);
}

/* x * 0 -> 0 (simplified: ignore nan/inf edge cases for now) */
static PolyUOp *rule_mul_zero(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  /* If x is a float const that is nan or inf, result should be nan */
  if (x->op == POLY_OP_CONST && x->arg.kind == POLY_ARG_FLOAT &&
      (isnan(x->arg.f) || isinf(x->arg.f)))
    return poly_const_like_float(ctx, x, NAN);
  return poly_const_like_int(ctx, root, 0);
}

/* bool * bool -> AND */
static PolyUOp *rule_bool_mul_to_and(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  if (!poly_dtype_is_bool(root->dtype)) return NULL;
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *y = poly_bind(b, "y");
  return poly_uop2(ctx, POLY_OP_AND, root->dtype, x, y, poly_arg_none());
}

/* bool + bool -> OR */
static PolyUOp *rule_bool_add_to_or(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  if (!poly_dtype_is_bool(root->dtype)) return NULL;
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *y = poly_bind(b, "y");
  return poly_uop2(ctx, POLY_OP_OR, root->dtype, x, y, poly_arg_none());
}

/* AND(x, false) -> false */
static PolyUOp *rule_and_zero(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_bool(ctx, poly_bind(b, "x"), false);
}

/* OR(x, true) -> true */
static PolyUOp *rule_or_one(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_bool(ctx, poly_bind(b, "x"), true);
}

/* ALU/variable min==max -> CONST.
 * tinygrad symbolic.py:
 *   (UPat({Ops.CMPLT, Ops.CMPNE, Ops.IDIV, Ops.MOD, Ops.DEFINE_VAR, Ops.BIND, Ops.SPECIAL}, name="x"),
 *    lambda x: x.const_like(x.vmin) if x.vmin == x.vmax else None)
 *   (UPat(Ops.RANGE, src=(UPat(Ops.CONST,)), name="x"), lambda x: x.const_like(x.vmin) if x.vmin == x.vmax else None)
 */
static PolyUOp *rule_const_when_minmax_point(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  int64_t vmin = 0, vmax = 0;
  poly_uop_minmax(ctx, root, &vmin, &vmax);
  if (vmin != vmax) return NULL;
  if (poly_dtype_eq(root->dtype, POLY_BOOL)) return poly_const_like_bool(ctx, root, vmin != 0);
  if (poly_dtype_is_float(root->dtype)) return poly_const_like_float(ctx, root, (double)vmin);
  return poly_const_like_int(ctx, root, vmin);
}

/* max folding
 *   maximum(x, y) -> x if x.vmin >= y.vmax else y if x.vmax <= y.vmin
 * tinygrad uop/symbolic.py:243 */
static PolyUOp *rule_max_fold(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_MAX || root->n_src != 2) return NULL;
  int64_t x_min = 0, x_max = 0, y_min = 0, y_max = 0;
  poly_uop_minmax(ctx, root->src[0], &x_min, &x_max);
  poly_uop_minmax(ctx, root->src[1], &y_min, &y_max);
  if (x_min >= y_max) return root->src[0];
  if (x_max <= y_min) return root->src[1];
  return NULL;
}

static bool const_lane_arg(PolyUOp *u, int lane, PolyArg *out) {
  if (!u || !out || lane < 0) return false;
  if (u->op == POLY_OP_CONST) {
    *out = u->arg;
    return true;
  }
  if (u->op != POLY_OP_VCONST) return false;
  if (lane < u->n_src && u->src[lane] && u->src[lane]->op == POLY_OP_CONST) {
    *out = u->src[lane]->arg;
    return true;
  }
  if (u->arg.kind == POLY_ARG_INT_TUPLE && lane < u->arg.int_tuple.n) {
    *out = poly_arg_int(u->arg.int_tuple.vals[lane]);
    return true;
  }
  return false;
}

static PolyUOp *build_vector_const_fold(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyOps op,
    int n_ops
) {
  if (!root || root->dtype.count <= 1 || root->dtype.count > 128 || n_ops < 1 || n_ops > 3) return NULL;
  PolyUOp *elts[128];
  PolyDType lane_dt = poly_dtype_scalar(root->dtype);
  for (int lane = 0; lane < root->dtype.count; lane++) {
    PolyArg lane_ops[3];
    for (int i = 0; i < n_ops; i++) {
      if (!const_lane_arg(root->src[i], lane, &lane_ops[i])) return NULL;
    }
    PolyArg lane_result = poly_exec_alu(op, lane_dt, lane_ops, n_ops);
    elts[lane] = poly_uop0(ctx, POLY_OP_CONST, lane_dt, lane_result);
  }
  /* Match tinygrad's lane-wise vector exec_alu. Keeping child CONST lanes here
   * preserves Invalid lanes instead of turning them into scalar zero. */
  return poly_uop(ctx, POLY_OP_VCONST, root->dtype, elts, root->dtype.count, poly_arg_none());
}

/* Constant folding: Unary(CONST) -> CONST */
static PolyUOp *rule_const_fold_unary(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  if (a->dtype.count > 1) return build_vector_const_fold(ctx, a, a->op, 1);
  /* Guard int64 NEG against INT64_MIN overflow */
  if (a->op == POLY_OP_NEG && poly_dtype_is_int(a->dtype) && a->dtype.bitsize == 64 &&
      a->src[0]->arg.kind == POLY_ARG_INT) {
    int64_t v;
    if (!i64_neg_ok(a->src[0]->arg.i, &v)) return NULL;
  }
  PolyArg operand = a->src[0]->arg;
  PolyArg result = poly_exec_alu(a->op, a->dtype, &operand, 1);
  return poly_const_like(ctx, a, result);
}

/* Constant folding: Binary(CONST, CONST) -> CONST */
static PolyUOp *rule_const_fold_binary(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  if (a->dtype.count > 1) return build_vector_const_fold(ctx, a, a->op, 2);
  /* Guard int64 ADD/SUB/MUL against overflow (UB in C for signed int64) */
  if (poly_dtype_is_int(a->dtype) && a->dtype.bitsize == 64 &&
      a->src[0]->arg.kind == POLY_ARG_INT && a->src[1]->arg.kind == POLY_ARG_INT) {
    int64_t av = a->src[0]->arg.i, bv = a->src[1]->arg.i, rv;
    if (a->op == POLY_OP_ADD && !i64_add_ok(av, bv, &rv)) return NULL;
    if (a->op == POLY_OP_SUB && !i64_sub_ok(av, bv, &rv)) return NULL;
    if (a->op == POLY_OP_MUL && !i64_mul_ok(av, bv, &rv)) return NULL;
  }
  PolyArg operands[2] = {a->src[0]->arg, a->src[1]->arg};
  PolyArg result = poly_exec_alu(a->op, a->dtype, operands, 2);
  return poly_const_like(ctx, a, result);
}

/* Constant folding: Ternary(CONST, CONST, CONST) -> CONST */
static PolyUOp *rule_const_fold_ternary(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  if (a->dtype.count > 1) return build_vector_const_fold(ctx, a, a->op, 3);
  PolyArg operands[3] = {a->src[0]->arg, a->src[1]->arg, a->src[2]->arg};
  PolyArg result = poly_exec_alu(a->op, a->dtype, operands, 3);
  return poly_const_like(ctx, a, result);
}

/* CAST(CONST) -> CONST with new dtype */
static PolyUOp *rule_cast_const(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *c = root->src[0];
  if (root->dtype.count > 1) return build_vector_const_fold(ctx, root, POLY_OP_CAST, 1);
  return poly_const_like(ctx, root, c->arg);
}

/* CAST(x, bool) -> CMPNE(x, 0) */
static PolyUOp *rule_cast_bool(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!poly_dtype_eq(root->dtype, POLY_BOOL)) return NULL;
  PolyUOp *x = root->src[0];
  if (poly_dtype_eq(x->dtype, POLY_BOOL)) return NULL; /* already bool, handled by noop */
  PolyUOp *zero = poly_const_like_int(ctx, x, 0);
  return poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, x, zero, poly_arg_none());
}

/* CAST/BITCAST same dtype -> identity */
static PolyUOp *rule_cast_noop(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (poly_dtype_eq(root->dtype, root->src[0]->dtype)) return root->src[0];
  return NULL;
}

/* NEG(NEG(x)) -> x */
static PolyUOp *rule_double_neg(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)root;
  return poly_bind(b, "x");
}

/* x / x -> 1 (float division) */
static PolyUOp *rule_fdiv_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_float(ctx, poly_bind(b, "x"), 1.0);
}

/* WHERE(cond, val, val) -> val */
static PolyUOp *rule_where_same(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)root;
  return poly_bind(b, "val");
}

/* WHERE(true/false_const, c0, c1) -> c0 or c1 */
static PolyUOp *rule_where_const_gate(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  PolyUOp *gate = poly_bind(b, "gate");
  PolyUOp *c0 = poly_bind(b, "c0");
  PolyUOp *c1 = poly_bind(b, "c1");
  if (gate->arg.kind == POLY_ARG_BOOL) return gate->arg.b ? c0 : c1;
  if (gate->arg.kind == POLY_ARG_INT) return gate->arg.i ? c0 : c1;
  return NULL;
}

static bool is_invalid_const_uop(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INVALID;
}

static bool is_true_const_uop(PolyUOp *u) {
  if (!u || u->op != POLY_OP_CONST) return false;
  if (!poly_dtype_eq(u->dtype, POLY_BOOL)) return false;
  return (u->arg.kind == POLY_ARG_BOOL && u->arg.b) ||
         (u->arg.kind == POLY_ARG_INT && u->arg.i != 0);
}

/* WHERE(CMPNE(cond, true), t, f) -> WHERE(cond, f, t)
 * Tinygrad symbolic.py:230-231:
 *   cond.logical_not().where(t, f) -> cond.where(f, t)
 * Keep the Invalid guard from tinygrad so we don't move Invalid into the
 * taken branch and perturb validity semantics. */
static PolyUOp *rule_where_logical_not(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *cond = poly_bind(b, "cond");
  PolyUOp *t = poly_bind(b, "t");
  PolyUOp *f = poly_bind(b, "f");
  if (!root || root->n_src != 3 || !is_true_const_uop(root->src[0]->src[1])) return NULL;
  if (is_invalid_const_uop(f)) return NULL;
  return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cond, f, t, poly_arg_none());
}

static PolyUOp *strip_casted_index_ptr(PolyUOp *u) {
  while (u && u->op == POLY_OP_CAST && u->n_src >= 1) u = u->src[0];
  return (u && u->op == POLY_OP_INDEX && u->n_src >= 2) ? u : NULL;
}

/* Tinygrad symbolic.py load/store folding:
 *   LOAD(INDEX(buf, Invalid)) -> const_like(0)
 *   STORE(INDEX(buf, Invalid), ...) -> NOOP
 * This is what removes dead masked lanes after devectorization. */
static PolyUOp *rule_fold_invalid_load_store(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || (root->op != POLY_OP_LOAD && root->op != POLY_OP_STORE) || root->n_src < 1) return NULL;
  PolyUOp *idx = strip_casted_index_ptr(root->src[0]);
  if (!idx || !is_invalid_const_uop(idx->src[1])) return NULL;
  if (root->op == POLY_OP_STORE) return poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  if (poly_dtype_eq(root->dtype, POLY_BOOL)) return poly_const_like_bool(ctx, root, false);
  if (poly_dtype_is_float(root->dtype)) return poly_const_like_float(ctx, root, 0.0);
  return poly_const_like_int(ctx, root, 0);
}

/* x + x -> x * 2 */
static PolyUOp *rule_add_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *two = poly_const_like_int(ctx, x, 2);
  return poly_uop2(ctx, POLY_OP_MUL, x->dtype, x, two, poly_arg_none());
}

/* ADD(ADD(a, x), x) -> ADD(a, MUL(x, 2))
 * Associative grouping of identical addends — handles the gradient
 * accumulation pattern where the same variable contributes via different
 * paths and ends up in nested ADDs. Commutative matching on both the
 * inner and outer ADD covers all orderings. */
static PolyUOp *rule_add_assoc_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *two = poly_const_like_int(ctx, x, 2);
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, x->dtype, x, two, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, root->dtype, a, mul, poly_arg_none());
}

static bool is_scalar_const_uop(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST;
}

static bool match_binary_one_const(PolyUOp *u, PolyUOp **out_x, PolyUOp **out_c) {
  if (!u || u->n_src != 2) return false;
  if (is_scalar_const_uop(u->src[0]) && !is_scalar_const_uop(u->src[1])) {
    *out_c = u->src[0];
    *out_x = u->src[1];
    return true;
  }
  if (is_scalar_const_uop(u->src[1]) && !is_scalar_const_uop(u->src[0])) {
    *out_c = u->src[1];
    *out_x = u->src[0];
    return true;
  }
  return false;
}

/* tinygrad symbolic.py:
 *   (x:weakint + c).cast(signed_int) -> x.cast(signed_int) + c.cast(signed_int)
 *
 * This turns Tensor.arange's `(range + 1).cast(int) + -1` class index into
 * `range.cast(int)`, exposing the sparse one-hot load-collapse rule. Keep it
 * scoped to POLY_INDEX sources and signed integer destinations, matching
 * dtypes.weakint -> dtypes.sints. */
static PolyUOp *rule_cast_index_add_const_to_add_casts(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!root || root->op != POLY_OP_CAST || root->n_src != 1) return NULL;
  if (!poly_dtype_is_int(root->dtype) || poly_dtype_is_unsigned(root->dtype) ||
      poly_dtype_is_bool(root->dtype))
    return NULL;

  PolyUOp *add = root->src[0];
  if (!add || add->op != POLY_OP_ADD || add->n_src != 2) return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(add->dtype), POLY_INDEX)) return NULL;

  PolyUOp *x = NULL;
  PolyUOp *c = NULL;
  if (!match_binary_one_const(add, &x, &c)) return NULL;

  PolyUOp *x_cast = poly_uop1(ctx, POLY_OP_CAST, root->dtype, x, poly_arg_none());
  PolyUOp *c_cast = poly_uop1(ctx, POLY_OP_CAST, root->dtype, c, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, root->dtype, x_cast, c_cast, poly_arg_none());
}

/* tinygrad symbolic.py:260-262
 *   x.alu(op, c1).alu(op, c2) -> x.alu(op, c1.alu(op, c2))
 * Implemented structurally for binary associative ops with one const in the
 * inner node and one const at the root, regardless of src ordering. */
static PolyUOp *rule_assoc_fold_consts(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 2) return NULL;
  if (!poly_opset_has(POLY_GROUP_ASSOCIATIVE, root->op)) return NULL;

  PolyUOp *inner = NULL, *c2 = NULL;
  if (!match_binary_one_const(root, &inner, &c2)) return NULL;
  if (!inner || inner->op != root->op || inner->n_src != 2) return NULL;

  PolyUOp *x = NULL, *c1 = NULL;
  if (!match_binary_one_const(inner, &x, &c1)) return NULL;

  PolyArg operands[2] = {c1->arg, c2->arg};
  PolyArg folded = poly_exec_alu(root->op, root->dtype, operands, 2);
  if (folded.kind == POLY_ARG_INVALID) return NULL;
  PolyUOp *fc = poly_const_like(ctx, c1, folded);
  return poly_uop2(ctx, root->op, root->dtype, x, fc, poly_arg_none());
}

/* tinygrad symbolic.py:
 *   y * (x + c) -> (y * x) + (y * c)
 *
 * tinygrad gates this rule to weak integer/index expressions. Applying it to
 * float arithmetic changes rounding behavior and also breaks useful vector ALU
 * structure, so keep the Polygrad port integer-only instead of distributing
 * general f32/f64 math such as linspace scaling.
 * Support both src orderings so we do not depend on a prior commutative flip
 * just to expose the add+const shape. */
static PolyUOp *rule_distribute_const_mul_over_add(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!root || root->op != POLY_OP_MUL || root->n_src != 2) return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(root->dtype), POLY_INDEX)) return NULL;

  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *c = root->src[swap];
    PolyUOp *add = root->src[swap ^ 1];
    if (!is_scalar_const_uop(c) || !add || add->op != POLY_OP_ADD || add->n_src != 2) continue;

    PolyUOp *x = NULL, *add_c = NULL;
    if (!match_binary_one_const(add, &x, &add_c)) continue;

    PolyArg operands[2] = {c->arg, add_c->arg};
    PolyArg folded = poly_exec_alu(POLY_OP_MUL, root->dtype, operands, 2);
    if (folded.kind == POLY_ARG_INVALID) continue;

    PolyUOp *scaled_x = poly_uop2(ctx, POLY_OP_MUL, root->dtype, x, c, poly_arg_none());
    PolyUOp *scaled_c = poly_const_like(ctx, add_c, folded);
    return poly_uop2(ctx, POLY_OP_ADD, root->dtype, scaled_x, scaled_c, poly_arg_none());
  }

  return NULL;
}

/* tinygrad symbolic.py:269-271
 *   (x + c) + y -> (x + y) + c
 *   (x * c) * y -> (x * y) * c
 * Keeps constants at the end of associative chains. This is visible in
 * optimized index expressions such as cross-entropy over a non-last axis. */
static PolyUOp *rule_move_const_to_end(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 2) return NULL;
  if (root->op != POLY_OP_ADD && root->op != POLY_OP_MUL) return NULL;

  PolyUOp *inner = root->src[0];
  PolyUOp *y = root->src[1];
  if (!inner || inner->op != root->op || inner->n_src != 2 || is_scalar_const_uop(y))
    return NULL;

  PolyUOp *x = NULL;
  PolyUOp *c = NULL;
  if (!match_binary_one_const(inner, &x, &c)) return NULL;
  PolyUOp *xy = poly_uop2(ctx, root->op, root->dtype, x, y, poly_arg_none());
  return poly_uop2(ctx, root->op, root->dtype, xy, c, poly_arg_none());
}

static int cmp_i64(int64_t a, int64_t b) {
  return (a > b) - (a < b);
}

static int cmp_u16(uint16_t a, uint16_t b) {
  return (a > b) - (a < b);
}

static int cmp_bool(bool a, bool b) {
  return (a > b) - (a < b);
}

static int cmp_cstr(const char *a, const char *b) {
  if (a == b) return 0;
  if (!a) return -1;
  if (!b) return 1;
  return strcmp(a, b);
}

static int dtype_tuplize_cmp(PolyDType a, PolyDType b) {
  if (poly_dtype_eq(a, b)) return 0;
  int ret;
  if ((ret = cmp_i64(a.priority, b.priority))) return ret;
  if ((ret = cmp_u16(a.bitsize, b.bitsize))) return ret;
  if ((ret = cmp_u16(a.count, b.count))) return ret;
  if ((ret = cmp_bool(a.is_ptr, b.is_ptr))) return ret;
  if ((ret = cmp_i64(a.addrspace, b.addrspace))) return ret;
  if ((ret = cmp_u16(a.vcount, b.vcount))) return ret;
  if ((ret = cmp_i64(a.ptr_size, b.ptr_size))) return ret;
  return cmp_cstr(a.name, b.name);
}

static int arg_int_tuple_cmp(const int64_t *a, int an, const int64_t *b, int bn) {
  int n = an < bn ? an : bn;
  for (int i = 0; i < n; i++) {
    int ret = cmp_i64(a[i], b[i]);
    if (ret) return ret;
  }
  return (an > bn) - (an < bn);
}

static int arg_pair_tuple_cmp(int64_t (*a)[2], int an, int64_t (*b)[2], int bn) {
  int n = an < bn ? an : bn;
  for (int i = 0; i < n; i++) {
    int ret = cmp_i64(a[i][0], b[i][0]);
    if (ret) return ret;
    ret = cmp_i64(a[i][1], b[i][1]);
    if (ret) return ret;
  }
  return (an > bn) - (an < bn);
}

static int arg_tuplize_cmp(PolyArg a, PolyArg b) {
  if (poly_arg_eq(a, b)) return 0;
  if (a.kind != b.kind) return (a.kind > b.kind) - (a.kind < b.kind);
  switch (a.kind) {
  case POLY_ARG_NONE:
  case POLY_ARG_INVALID:
    return 0;
  case POLY_ARG_INT:
    return cmp_i64(a.i, b.i);
  case POLY_ARG_FLOAT: {
    uint64_t av, bv;
    memcpy(&av, &a.f, sizeof(av));
    memcpy(&bv, &b.f, sizeof(bv));
    return (av > bv) - (av < bv);
  }
  case POLY_ARG_BOOL:
    return cmp_bool(a.b, b.b);
  case POLY_ARG_INT_TUPLE:
    return arg_int_tuple_cmp(a.int_tuple.vals, a.int_tuple.n, b.int_tuple.vals, b.int_tuple.n);
  case POLY_ARG_PAIR_TUPLE:
    return arg_pair_tuple_cmp(a.pair_tuple.pairs, a.pair_tuple.n, b.pair_tuple.pairs, b.pair_tuple.n);
  case POLY_ARG_STRING:
    return cmp_cstr(a.str, b.str);
  case POLY_ARG_OPS:
    return cmp_i64((int64_t)a.ops, (int64_t)b.ops);
  case POLY_ARG_REDUCE_AXIS: {
    int ret = cmp_i64((int64_t)a.reduce_axis.op, (int64_t)b.reduce_axis.op);
    if (ret) return ret;
    return arg_int_tuple_cmp(
        a.reduce_axis.axes, a.reduce_axis.n, b.reduce_axis.axes, b.reduce_axis.n
    );
  }
  case POLY_ARG_RANGE: {
    int ret = cmp_i64(a.range.axis_id, b.range.axis_id);
    if (ret) return ret;
    ret = cmp_i64((int64_t)a.range.axis_type, (int64_t)b.range.axis_type);
    if (ret) return ret;
    return arg_int_tuple_cmp(a.range.extra, a.range.n_extra, b.range.extra, b.range.n_extra);
  }
  case POLY_ARG_DEFINE_VAR: {
    int ret = cmp_cstr(a.define_var.name, b.define_var.name);
    if (ret) return ret;
    ret = cmp_i64(a.define_var.min_val, b.define_var.min_val);
    if (ret) return ret;
    return cmp_i64(a.define_var.max_val, b.define_var.max_val);
  }
  case POLY_ARG_BUFFERIZE_OPTS: {
    int ret = cmp_i64(a.bufferize_opts.device, b.bufferize_opts.device);
    if (ret) return ret;
    ret = cmp_i64((int64_t)a.bufferize_opts.addrspace, (int64_t)b.bufferize_opts.addrspace);
    if (ret) return ret;
    return cmp_bool(a.bufferize_opts.removable, b.bufferize_opts.removable);
  }
  }
  return 0;
}

static int uop_tuplize_cmp(PolyUOp *a, PolyUOp *b) {
  if (a == b) return 0;
  if (!a) return -1;
  if (!b) return 1;

  /* Port of tinygrad UOp.tuplize tuple comparison:
   *   (op.value, arg, dtype) + tuple(src.tuplize for src in src)
   * This is used only by the weak-index commutative rule below, so it stays
   * local to symbolic rewriting instead of becoming a general identity rule. */
  int ret = cmp_i64((int64_t)a->op, (int64_t)b->op);
  if (ret) return ret;
  ret = arg_tuplize_cmp(a->arg, b->arg);
  if (ret) return ret;
  ret = dtype_tuplize_cmp(a->dtype, b->dtype);
  if (ret) return ret;

  int n = a->n_src < b->n_src ? a->n_src : b->n_src;
  for (int i = 0; i < n; i++) {
    ret = uop_tuplize_cmp(a->src[i], b->src[i]);
    if (ret) return ret;
  }
  return (a->n_src > b->n_src) - (a->n_src < b->n_src);
}

/* tinygrad symbolic.py:
 *   UPat(GroupOp.Commutative, dtype=dtypes.weakint)
 *     -> reverse srcs if src[1].tuplize < src[0].tuplize
 *
 * Polygrad's weak index dtype is POLY_INDEX. Keep the rule scoped there; doing
 * this for ordinary numeric ALU can disturb vector math merging, matching the
 * warning in tinygrad's own comment. */
static PolyUOp *rule_commutative_index_tuplize_order(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!root || root->n_src != 2) return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(root->dtype), POLY_INDEX)) return NULL;
  if (!poly_opset_has(POLY_GROUP_COMMUTATIVE, root->op)) return NULL;
  if (uop_tuplize_cmp(root->src[1], root->src[0]) >= 0) return NULL;
  return poly_uop2(ctx, root->op, root->dtype, root->src[1], root->src[0], root->arg);
}

static int64_t uop_const_factor(PolyUOp *u);

static bool is_flat_index_term_uop(PolyUOp *u) {
  if (!u) return false;
  if (u->op == POLY_OP_RANGE || u->op == POLY_OP_SPECIAL || u->op == POLY_OP_DEFINE_VAR)
    return true;
  if ((u->op == POLY_OP_MUL || u->op == POLY_OP_SHL) && u->n_src == 2)
    return is_flat_index_term_uop(u->src[0]) || is_flat_index_term_uop(u->src[1]);
  return false;
}

/* tinygrad's reshape/indexing path simplifies weakint flat indexes at movement
 * construction time. Polygrad represents these as POLY_INDEX, so keep the
 * canonical innermost-first grouping scoped to integer index-like ADD chains. */
static PolyUOp *rule_flat_index_outer_term_to_end(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_ADD || root->n_src != 2) return NULL;
  if (!poly_dtype_is_int(root->dtype) || poly_dtype_is_bool(root->dtype)) return NULL;

  PolyUOp *inner = root->src[0];
  PolyUOp *last = root->src[1];
  if (!inner || inner->op != POLY_OP_ADD || inner->n_src != 2) return NULL;
  if (!is_flat_index_term_uop(inner->src[0]) || !is_flat_index_term_uop(inner->src[1]) ||
      !is_flat_index_term_uop(last))
    return NULL;

  int64_t outer_factor = uop_const_factor(inner->src[0]);
  int64_t middle_factor = uop_const_factor(inner->src[1]);
  int64_t last_factor = uop_const_factor(last);
  if (!(outer_factor > middle_factor && middle_factor >= last_factor)) return NULL;

  PolyUOp *new_inner =
      poly_uop2(ctx, POLY_OP_ADD, root->dtype, inner->src[1], last, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, root->dtype, new_inner, inner->src[0], poly_arg_none());
}

/* fold_divmod helpers (port of tinygrad divandmod.py) */

/* Split ADD chain into flat list of additive terms */
static int split_add_terms(PolyUOp *u, PolyUOp **terms, int max) {
  if (max <= 0) return 0;
  if (u->op == POLY_OP_ADD && u->n_src == 2) {
    int n = split_add_terms(u->src[0], terms, max);
    return n + split_add_terms(u->src[1], terms + n, max - n);
  }
  terms[0] = u;
  return 1;
}

/* Get constant factor of a UOp (MUL(x,3)→3, CONST(5)→5, x→1) */
static int64_t uop_const_factor(PolyUOp *u) {
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INT) return u->arg.i;
  if (u->op == POLY_OP_MUL && u->n_src == 2) {
    if (u->src[1]->op == POLY_OP_CONST && u->src[1]->arg.kind == POLY_ARG_INT)
      return u->src[1]->arg.i * uop_const_factor(u->src[0]);
    if (u->src[0]->op == POLY_OP_CONST && u->src[0]->arg.kind == POLY_ARG_INT)
      return u->src[0]->arg.i * uop_const_factor(u->src[1]);
  }
  return 1;
}

/* Divide UOp by constant factor: MUL(x,6)/3 → MUL(x,2) */
static PolyUOp *uop_divides(PolyCtx *ctx, PolyUOp *u, int64_t f) {
  if (f == 1) return u;
  if (f == 0) return NULL;
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INT)
    return poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_int(u->arg.i / f));
  if (u->op == POLY_OP_MUL && u->n_src == 2) {
    if (u->src[1]->op == POLY_OP_CONST && u->src[1]->arg.kind == POLY_ARG_INT) {
      int64_t c = u->src[1]->arg.i;
      if (c % f == 0) {
        if (c / f == 1) return u->src[0];
        return poly_uop2(
            ctx, POLY_OP_MUL, u->dtype, u->src[0],
            poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_int(c / f)), poly_arg_none()
        );
      }
    }
    if (u->src[0]->op == POLY_OP_CONST && u->src[0]->arg.kind == POLY_ARG_INT) {
      int64_t c = u->src[0]->arg.i;
      if (c % f == 0) {
        if (c / f == 1) return u->src[1];
        return poly_uop2(
            ctx, POLY_OP_MUL, u->dtype,
            poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_int(c / f)), u->src[1], poly_arg_none()
        );
      }
    }
  }
  return NULL;
}

/* Floor division (Python-style //) */
static int64_t floordiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a ^ b) < 0 && q * b != a) q--;
  return q;
}

/* Floor mod (Python-style %) */
static int64_t cmod(int64_t a, int64_t b) {
  return a - floordiv(a, b) * b;
}

/* GCD (Euclidean) */
static int64_t gcd64(int64_t a, int64_t b) {
  a = a < 0 ? -a : a;
  b = b < 0 ? -b : b;
  while (b) {
    int64_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

/* Check if a CONST UOp's value is exactly divisible by c. Returns quotient or 0. */
static int64_t uop_divides_const(PolyUOp *u, int64_t c) {
  if (c == 0) return 0;
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INT && u->arg.i % c == 0)
    return u->arg.i / c;
  return 0;
}

/* fold_divmod_general (port of tinygrad divandmod.py) */

static PolyUOp *fold_divmod_general(PolyCtx *ctx, PolyUOp *root) {
  if (root->n_src != 2 || !poly_dtype_is_int(root->dtype)) return NULL;
  PolyUOp *x = root->src[0], *y = root->src[1];
  int64_t x_min, x_max, y_min, y_max;
  poly_uop_minmax(ctx, x, &x_min, &x_max);
  poly_uop_minmax(ctx, y, &y_min, &y_max);

  /* 1. cancel_divmod: all corners give same quotient */
  /* Use same-sign check instead of y_min*y_max>0 to avoid int64 overflow */
  if ((y_min > 0 && y_max > 0) || (y_min < 0 && y_max < 0)) {
    int64_t q00 = cdiv(x_min, y_min), q01 = cdiv(x_min, y_max);
    int64_t q10 = cdiv(x_max, y_min), q11 = cdiv(x_max, y_max);
    if (q00 == q01 && q00 == q10 && q00 == q11) {
      int64_t q = q00;
      if (root->op == POLY_OP_MOD) {
        if (q == 0) return x;
        PolyUOp *qc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(q));
        PolyUOp *qy = poly_uop2(ctx, POLY_OP_MUL, root->dtype, qc, y, poly_arg_none());
        return poly_uop2(
            ctx, POLY_OP_ADD, root->dtype, x,
            poly_uop1(ctx, POLY_OP_NEG, root->dtype, qy, poly_arg_none()), poly_arg_none()
        );
      }
      return poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(q));
    }
  }

  /* Constant positive denominator required for remaining rules */
  if (y->op != POLY_OP_CONST || y->arg.i <= 0) return NULL;
  int64_t c = y->arg.i;

  /* 2. nested_div_mod: (x%(k*c))//c → (x//c)%k, (x%(k*c))%c → x%c */
  if (x->op == POLY_OP_MOD && x->n_src == 2) {
    int64_t k = uop_divides_const(x->src[1], c);
    if (k > 0) {
      if (root->op == POLY_OP_IDIV) {
        PolyUOp *d = poly_uop2(ctx, POLY_OP_IDIV, root->dtype, x->src[0], y, poly_arg_none());
        PolyUOp *kc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(k));
        return poly_uop2(ctx, POLY_OP_MOD, root->dtype, d, kc, poly_arg_none());
      }
      return poly_uop2(ctx, POLY_OP_MOD, root->dtype, x->src[0], y, poly_arg_none());
    }
  }

  /* 3. remove_nested_mod: (a%4 + b)%2 → (a+b)%2 when x >= 0 */
  if (root->op == POLY_OP_MOD && x_min >= 0) {
    PolyUOp *sum_terms[16];
    int n_sum = split_add_terms(x, sum_terms, 16);
    if (n_sum > 0 && n_sum <= 15) {
      bool changed = false;
      for (int i = 0; i < n_sum; i++) {
        if (sum_terms[i]->op == POLY_OP_MOD && sum_terms[i]->n_src == 2 &&
            uop_divides_const(sum_terms[i]->src[1], c) > 0) {
          sum_terms[i] = sum_terms[i]->src[0];
          changed = true;
        }
      }
      if (changed) {
        PolyUOp *new_x = sum_terms[0];
        for (int i = 1; i < n_sum; i++)
          new_x = poly_uop2(ctx, POLY_OP_ADD, root->dtype, new_x, sum_terms[i], poly_arg_none());
        int64_t nx_min, nx_max;
        poly_uop_minmax(ctx, new_x, &nx_min, &nx_max);
        if (nx_min >= 0) return poly_uop2(ctx, POLY_OP_MOD, root->dtype, new_x, y, poly_arg_none());
      }
    }
  }

  if (x_min < 0) return NULL;

  /* Split x into additive terms */
  PolyUOp *terms[16];
  int n_terms = split_add_terms(x, terms, 16);
  if (n_terms == 0 || n_terms > 15) return NULL;

  /* Separate additive constant from non-constant terms */
  int64_t additive_const = 0;
  PolyUOp *nc_terms[16];
  int64_t nc_factors[16];
  int n_nc = 0;
  for (int i = 0; i < n_terms; i++) {
    if (terms[i]->op == POLY_OP_CONST && terms[i]->arg.kind == POLY_ARG_INT) {
      additive_const += terms[i]->arg.i;
    } else {
      nc_terms[n_nc] = terms[i];
      nc_factors[n_nc] = uop_const_factor(nc_terms[n_nc]);
      n_nc++;
    }
  }

  /* Compute remainders: rems[i] = min(f%c, f%c-c, key=abs) */
  int64_t rems[16];
  for (int i = 0; i < n_nc; i++) {
    int64_t r = nc_factors[i] % c;
    int64_t r2 = r - c;
    int64_t a1 = r < 0 ? -r : r, a2 = r2 < 0 ? -r2 : r2;
    rems[i] = (a1 <= a2) ? r : r2;
  }

  /* Get base for each term: base[i] = nc_terms[i] / nc_factors[i] */
  PolyUOp *bases[16];
  int64_t base_mins[16], base_maxs[16];
  for (int i = 0; i < n_nc; i++) {
    bases[i] = uop_divides(ctx, nc_terms[i], nc_factors[i]);
    if (!bases[i]) return NULL;
    poly_uop_minmax(ctx, bases[i], &base_mins[i], &base_maxs[i]);
  }

  /* 4. fold_binary_numerator: single non-const term with range of 2 */
  if (n_nc == 1 && base_maxs[0] - base_mins[0] == 1) {
    int64_t y1 = (root->op == POLY_OP_MOD) ? cmod(nc_factors[0] * base_mins[0] + additive_const, c)
                                           : cdiv(nc_factors[0] * base_mins[0] + additive_const, c);
    int64_t y2 = (root->op == POLY_OP_MOD) ? cmod(nc_factors[0] * base_maxs[0] + additive_const, c)
                                           : cdiv(nc_factors[0] * base_maxs[0] + additive_const, c);
    /* result = (y2-y1)*(v-v_min) + y1 */
    int64_t slope = y2 - y1;
    PolyUOp *v_off = poly_uop2(
        ctx, POLY_OP_ADD, root->dtype, bases[0],
        poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(-base_mins[0])), poly_arg_none()
    );
    PolyUOp *r;
    if (slope == 0) {
      r = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(y1));
    } else if (slope == 1) {
      r = poly_uop2(
          ctx, POLY_OP_ADD, root->dtype, v_off,
          poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(y1)), poly_arg_none()
      );
    } else {
      PolyUOp *sc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(slope));
      r = poly_uop2(
          ctx, POLY_OP_ADD, root->dtype,
          poly_uop2(ctx, POLY_OP_MUL, root->dtype, sc, v_off, poly_arg_none()),
          poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(y1)), poly_arg_none()
      );
    }
    return r;
  }

  /* Compute bounds of rem = sum(rems[i]*base[i]) + additive_const%c */
  int64_t crem = additive_const % c;
  int64_t rem_lo = crem, rem_hi = crem;
  for (int i = 0; i < n_nc; i++) {
    int64_t v0 = rems[i] * base_mins[i], v1 = rems[i] * base_maxs[i];
    int64_t lo = v0 < v1 ? v0 : v1, hi = v0 < v1 ? v1 : v0;
    rem_lo += lo;
    rem_hi += hi;
  }

  /* Check: rem range fits in one c-interval */
  if (floordiv(rem_lo, c) != floordiv(rem_hi, c)) {
    /* 5. gcd_with_remainder: factor out GCD of all factors and c */
    if (x_min >= 0 && n_nc > 0) {
      int64_t g = c;
      for (int i = 0; i < n_nc; i++) {
        int64_t af = nc_factors[i] < 0 ? -nc_factors[i] : nc_factors[i];
        g = gcd64(g, af);
      }
      if (g > 1) {
        /* Rebuild numerator / g */
        int64_t new_c = c / g;
        PolyUOp *new_x = NULL;
        for (int i = 0; i < n_nc; i++) {
          PolyUOp *divided = uop_divides(ctx, nc_terms[i], g);
          if (!divided) goto skip_gcd;
          new_x = new_x ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, new_x, divided, poly_arg_none())
                        : divided;
        }
        /* Add (additive_const/g) % new_c */
        int64_t ac_g = additive_const / g;
        int64_t ac_rem = cmod(ac_g, new_c);
        if (ac_rem != 0 || !new_x) {
          PolyUOp *ac_uop = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(ac_rem));
          new_x = new_x ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, new_x, ac_uop, poly_arg_none())
                        : ac_uop;
        }
        int64_t nx_min, nx_max;
        poly_uop_minmax(ctx, new_x, &nx_min, &nx_max);
        if (nx_min >= 0) {
          PolyUOp *new_y = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(new_c));
          if (root->op == POLY_OP_MOD) {
            PolyUOp *inner =
                poly_uop2(ctx, POLY_OP_MOD, root->dtype, new_x, new_y, poly_arg_none());
            PolyUOp *gc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(g));
            PolyUOp *scaled = poly_uop2(ctx, POLY_OP_MUL, root->dtype, inner, gc, poly_arg_none());
            int64_t const_rem = additive_const % g;
            if (const_rem != 0) {
              PolyUOp *cr = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(const_rem));
              return poly_uop2(ctx, POLY_OP_ADD, root->dtype, scaled, cr, poly_arg_none());
            }
            return scaled;
          }
          PolyUOp *inner = poly_uop2(ctx, POLY_OP_IDIV, root->dtype, new_x, new_y, poly_arg_none());
          int64_t const_div = additive_const / c;
          if (const_div != 0) {
            PolyUOp *cd = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(const_div));
            return poly_uop2(ctx, POLY_OP_ADD, root->dtype, inner, cd, poly_arg_none());
          }
          return inner;
        }
      }
    }
  skip_gcd:
    return NULL;
  }

  if (root->op == POLY_OP_MOD) {
    /* return rem - floordiv(rem_lo, c) * c */
    int64_t offset = floordiv(rem_lo, c) * c;
    int64_t const_val = crem - offset;

    /* Build rem expression */
    PolyUOp *result = NULL;
    for (int i = 0; i < n_nc; i++) {
      if (rems[i] == 0) continue;
      PolyUOp *term;
      if (rems[i] == 1) {
        term = bases[i];
      } else {
        PolyUOp *rc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(rems[i]));
        term = poly_uop2(ctx, POLY_OP_MUL, root->dtype, rc, bases[i], poly_arg_none());
      }
      result =
          result ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, result, term, poly_arg_none()) : term;
    }
    if (const_val != 0 || !result) {
      PolyUOp *cc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(const_val));
      result = result ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, result, cc, poly_arg_none()) : cc;
    }
    return result;
  }

  /* IDIV: sum((f-r)//c * base[i]) + (additive_const - crem + floordiv(rem_lo,c)*c)//c */
  int64_t const_part = (additive_const - crem + floordiv(rem_lo, c) * c) / c;
  PolyUOp *result = NULL;
  for (int i = 0; i < n_nc; i++) {
    int64_t coeff = (nc_factors[i] - rems[i]) / c;
    if (coeff == 0) continue;
    PolyUOp *term;
    if (coeff == 1) {
      term = bases[i];
    } else {
      PolyUOp *cc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(coeff));
      term = poly_uop2(ctx, POLY_OP_MUL, root->dtype, cc, bases[i], poly_arg_none());
    }
    result =
        result ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, result, term, poly_arg_none()) : term;
  }
  if (const_part != 0 || !result) {
    PolyUOp *cc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(const_part));
    result = result ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, result, cc, poly_arg_none()) : cc;
  }
  return result;
}

static PolyUOp *rule_cancel_divmod(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  return fold_divmod_general(ctx, root);
}

/* fold_add_divmod_recombine: dynamic divmod cancellation on ADD.
 * Port of tinygrad symbolic.py:28-48.  Replaces the old hardcoded pattern.
 * Handles:
 *   (base%div)*mul + (base//div)*(div*mul) -> base*mul
 *   ((base//d)%div)*mul + (base//(d*div))*(div*mul) -> (base//d)*mul
 *   ((base//div)%d)*div + base%div -> base%(div*d)
 */
static PolyUOp *rule_fold_add_divmod_recombine(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_ADD) return NULL;
  /* only on int types (tinygrad gates on dtypes.weakint/index) */
  if (poly_dtype_is_float(root->dtype) || root->dtype.bitsize == 1) return NULL;

  PolyUOp *terms[32];
  int n = split_add_terms(root, terms, 32);
  if (n < 2) return NULL;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = terms[i];
    PolyUOp *base = NULL;
    int64_t div_val = 0, mul_val = 0;

    /* Match: base%div (mul=1) */
    if (u->op == POLY_OP_MOD && u->n_src == 2 && u->src[1]->op == POLY_OP_CONST &&
        u->src[1]->arg.kind == POLY_ARG_INT) {
      base = u->src[0];
      div_val = u->src[1]->arg.i;
      mul_val = 1;
    }
    /* Match: (base%div)*mul */
    else if (u->op == POLY_OP_MUL && u->n_src == 2 && u->src[1]->op == POLY_OP_CONST &&
             u->src[1]->arg.kind == POLY_ARG_INT) {
      PolyUOp *m = u->src[0];
      if (m->op == POLY_OP_MOD && m->n_src == 2 && m->src[1]->op == POLY_OP_CONST &&
          m->src[1]->arg.kind == POLY_ARG_INT) {
        base = m->src[0];
        div_val = m->src[1]->arg.i;
        mul_val = u->src[1]->arg.i;
      }
    }
    if (!base || div_val == 0) continue;

    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      PolyUOp *v = terms[j];
      /* v must be MUL(q, div*mul) */
      if (v->op != POLY_OP_MUL || v->n_src != 2 || v->src[1]->op != POLY_OP_CONST ||
          v->src[1]->arg.kind != POLY_ARG_INT || v->src[1]->arg.i != div_val * mul_val)
        continue;
      PolyUOp *q = v->src[0];
      bool exact = false;

      /* (base%div)*mul + (base//div)*(div*mul) -> base*mul */
      if (q->op == POLY_OP_IDIV && q->n_src == 2 && q->src[1]->op == POLY_OP_CONST &&
          q->src[1]->arg.kind == POLY_ARG_INT && q->src[1]->arg.i == div_val && q->src[0] == base) {
        exact = true;
      }
      /* ((base//d)%div)*mul + (base//(d*div))*(div*mul) -> (base//d)*mul */
      if (!exact && base->op == POLY_OP_IDIV && base->n_src == 2 &&
          base->src[1]->op == POLY_OP_CONST && base->src[1]->arg.kind == POLY_ARG_INT) {
        if (q->op == POLY_OP_IDIV && q->n_src == 2 && q->src[1]->op == POLY_OP_CONST &&
            q->src[1]->arg.kind == POLY_ARG_INT && q->src[0] == base->src[0] &&
            q->src[1]->arg.i == base->src[1]->arg.i * div_val) {
          exact = true;
        }
      }
      if (exact) {
        /* result = base * mul + sum of remaining terms */
        PolyUOp *result;
        if (mul_val == 1) {
          result = base;
        } else {
          PolyUOp *mc = poly_const_like_int(ctx, root, mul_val);
          PolyUOp *ms[2] = {base, mc};
          result = poly_uop(ctx, POLY_OP_MUL, root->dtype, ms, 2, poly_arg_none());
        }
        for (int k = 0; k < n; k++) {
          if (k == i || k == j) continue;
          PolyUOp *as[2] = {result, terms[k]};
          result = poly_uop(ctx, POLY_OP_ADD, root->dtype, as, 2, poly_arg_none());
        }
        return result;
      }

      /* ((base//div)%d)*div + base%div -> base%(div*d) */
      if (mul_val == 1 && div_val > 0 && q->op == POLY_OP_MOD && q->n_src == 2 &&
          q->src[1]->op == POLY_OP_CONST && q->src[1]->arg.kind == POLY_ARG_INT) {
        int64_t d = q->src[1]->arg.i;
        if (d > 0 && q->src[0]->op == POLY_OP_IDIV && q->src[0]->n_src == 2 &&
            q->src[0]->src[0] == base && q->src[0]->src[1]->op == POLY_OP_CONST &&
            q->src[0]->src[1]->arg.kind == POLY_ARG_INT && q->src[0]->src[1]->arg.i == div_val) {
          PolyUOp *new_mod_c = poly_const_like_int(ctx, root, div_val * d);
          PolyUOp *ms[2] = {base, new_mod_c};
          PolyUOp *result = poly_uop(ctx, POLY_OP_MOD, root->dtype, ms, 2, poly_arg_none());
          for (int k = 0; k < n; k++) {
            if (k == i || k == j) continue;
            PolyUOp *as[2] = {result, terms[k]};
            result = poly_uop(ctx, POLY_OP_ADD, root->dtype, as, 2, poly_arg_none());
          }
          return result;
        }
      }
    }
  }
  return NULL;
}

/* (x * y) / y → x.  Ref: tinygrad symbolic_simple line 90 */
static PolyUOp *rule_mul_fdiv_cancel(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_bind(b, "x");
}

/* WHERE(a, WHERE(b, c, d), d) → WHERE(AND(a, b), c, d).  Ref: tinygrad line 117 */
static PolyUOp *rule_nested_where(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  PolyUOp *bb = poly_bind(b, "b");
  PolyUOp *c = poly_bind(b, "c");
  PolyUOp *d = poly_bind(b, "d");
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, a, bb, poly_arg_none());
  return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cond, c, d, poly_arg_none());
}

/* VECTORIZE(CONST...) -> VCONST(CONST...) */
static PolyUOp *rule_vectorize_const_fold(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_VECTORIZE || root->n_src <= 0) return NULL;
  for (int i = 0; i < root->n_src; i++) {
    if (root->src[i]->op != POLY_OP_CONST && root->src[i]->op != POLY_OP_VCONST) return NULL;
  }
  return poly_uop(ctx, POLY_OP_VCONST, root->dtype, root->src, root->n_src, poly_arg_none());
}

/* GEP(VECTORIZE(...)) -> VECTORIZE(select...) or scalar select */
static PolyUOp *rule_gep_vectorize(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_GEP || root->n_src < 1) return NULL;
  PolyUOp *vec = root->src[0];
  if (!vec || vec->op != POLY_OP_VECTORIZE || vec->n_src <= 0) return NULL;

  if (root->arg.kind == POLY_ARG_INT) {
    int64_t idx = root->arg.i;
    if (idx < 0 || idx >= vec->n_src) return NULL;
    return vec->src[idx];
  }
  if (root->arg.kind != POLY_ARG_INT_TUPLE || root->arg.int_tuple.n <= 0) return NULL;
  int n = root->arg.int_tuple.n;
  if (n == 1) {
    int64_t idx = root->arg.int_tuple.vals[0];
    if (idx < 0 || idx >= vec->n_src) return NULL;
    return vec->src[idx];
  }
  PolyUOp **elts = calloc((size_t)n, sizeof(*elts));
  if (!elts) return NULL;
  for (int i = 0; i < n; i++) {
    int64_t idx = root->arg.int_tuple.vals[i];
    if (idx < 0 || idx >= vec->n_src) {
      free(elts);
      return NULL;
    }
    elts[i] = vec->src[idx];
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, root->dtype, elts, n, poly_arg_none());
  free(elts);
  return ret;
}

/* GEP(CONST/VCONST) -> selected CONST(s). */
static PolyUOp *rule_gep_const(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_GEP || root->n_src < 1) return NULL;
  PolyUOp *c = root->src[0];
  if (!c) return NULL;
  if (c->op == POLY_OP_CONST) {
    if (c->dtype.count <= 1) return c;
    if (root->arg.kind == POLY_ARG_INT)
      return poly_uop0(ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype), c->arg);
    if (root->arg.kind != POLY_ARG_INT_TUPLE || root->arg.int_tuple.n <= 0) return NULL;
    if (root->arg.int_tuple.n == 1)
      return poly_uop0(ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype), c->arg);
    return poly_uop0(ctx, POLY_OP_CONST, root->dtype, c->arg);
  }
  if (c->op != POLY_OP_VCONST) return NULL;

  if (root->arg.kind == POLY_ARG_INT) {
    int64_t idx = root->arg.i;
    if (idx < 0) return NULL;
    if (c->n_src > idx) return c->src[idx];
    if (c->arg.kind == POLY_ARG_INT_TUPLE && idx < c->arg.int_tuple.n)
      return poly_uop0(
          ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype), poly_arg_int(c->arg.int_tuple.vals[idx])
      );
    return NULL;
  }
  if (root->arg.kind != POLY_ARG_INT_TUPLE || root->arg.int_tuple.n <= 0) return NULL;
  int n = root->arg.int_tuple.n;
  if (n == 1) {
    int64_t idx = root->arg.int_tuple.vals[0];
    if (idx < 0) return NULL;
    if (c->n_src > idx) return c->src[idx];
    if (c->arg.kind == POLY_ARG_INT_TUPLE && idx < c->arg.int_tuple.n)
      return poly_uop0(
          ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype), poly_arg_int(c->arg.int_tuple.vals[idx])
      );
    return NULL;
  }
  PolyUOp **elts = calloc((size_t)n, sizeof(*elts));
  if (!elts) return NULL;
  for (int i = 0; i < n; i++) {
    int64_t idx = root->arg.int_tuple.vals[i];
    if (idx < 0) {
      free(elts);
      return NULL;
    }
    if (c->n_src > idx)
      elts[i] = c->src[idx];
    else if (c->arg.kind == POLY_ARG_INT_TUPLE && idx < c->arg.int_tuple.n)
      elts[i] = poly_uop0(
          ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype), poly_arg_int(c->arg.int_tuple.vals[idx])
      );
    else {
      free(elts);
      return NULL;
    }
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, root->dtype, elts, n, poly_arg_none());
  free(elts);
  return ret;
}

/* GEP in natural order is identity. */
static PolyUOp *rule_gep_identity(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (root->op != POLY_OP_GEP || root->n_src < 1) return NULL;
  PolyUOp *src = root->src[0];
  if (!src || src->dtype.is_ptr) return NULL;

  if (root->arg.kind == POLY_ARG_INT) {
    if (src->dtype.count == 1 && root->arg.i == 0) return src;
    return NULL;
  }
  if (root->arg.kind != POLY_ARG_INT_TUPLE || root->arg.int_tuple.n <= 0) return NULL;
  int n = root->arg.int_tuple.n;
  if (src->dtype.count == 1 && n == 1 && root->arg.int_tuple.vals[0] == 0) return src;
  if (n != src->dtype.count) return NULL;
  for (int i = 0; i < n; i++) {
    if (root->arg.int_tuple.vals[i] != i) return NULL;
  }
  return src;
}

/* GEP through ALU: GEP(ALU(a,b), i) → ALU(GEP(a,i), GEP(b,i))
 * (tinygrad symbolic.py:192-194, gep_pushing) */
static PolyUOp *rule_gep_through_alu(PolyCtx *ctx, PolyUOp *gep, const PolyBindings *b) {
  (void)b;
  if (gep->op != POLY_OP_GEP || gep->n_src < 1) return NULL;
  if (gep->dtype.is_ptr) return NULL;
  /* Only push GEP through integer ALU (index arithmetic).
   * tinygrad gates on dtype=dtypes.index; polygrad uses POLY_INT32 for indices. */
  if (!poly_dtype_is_int(poly_dtype_scalar(gep->dtype))) return NULL;
  PolyUOp *alu = gep->src[0];
  if (!alu || alu->dtype.is_ptr) return NULL;
  if (!poly_opset_has(POLY_GROUP_ALU, alu->op) && alu->op != POLY_OP_CAST &&
      alu->op != POLY_OP_BITCAST)
    return NULL;
  if (alu->dtype.count <= 1) return NULL; /* already scalar */
  /* Build new ALU with GEP pushed to each source */
  PolyUOp *srcs[8];
  if (alu->n_src > 8) return NULL;
  int gep_count = 1;
  if (gep->arg.kind == POLY_ARG_INT_TUPLE) gep_count = gep->arg.int_tuple.n;
  PolyDType new_dt = (gep_count > 1) ? poly_dtype_vec(poly_dtype_scalar(alu->dtype), gep_count)
                                     : poly_dtype_scalar(alu->dtype);
  for (int i = 0; i < alu->n_src; i++) {
    PolyUOp *s = alu->src[i];
    if (s->dtype.count > 1) {
      PolyDType s_new_dt = (gep_count > 1) ? poly_dtype_vec(poly_dtype_scalar(s->dtype), gep_count)
                                           : poly_dtype_scalar(s->dtype);
      srcs[i] = poly_uop1(ctx, POLY_OP_GEP, s_new_dt, s, gep->arg);
    } else {
      srcs[i] = s; /* scalar source passes through */
    }
  }
  if (alu->n_src == 1) return poly_uop1(ctx, alu->op, new_dt, srcs[0], alu->arg);
  if (alu->n_src == 2) return poly_uop2(ctx, alu->op, new_dt, srcs[0], srcs[1], alu->arg);
  if (alu->n_src == 3) return poly_uop3(ctx, alu->op, new_dt, srcs[0], srcs[1], srcs[2], alu->arg);
  return poly_uop(ctx, alu->op, new_dt, srcs, alu->n_src, alu->arg);
}

/* Tinygrad symbolic.py: VCAT cannot be rendered directly, so expand it into
 * VECTORIZE of per-lane GEPs early enough for later scalarization passes. */
static PolyUOp *rule_vcat_to_vectorize(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  if (!x || x->op != POLY_OP_VCAT || x->n_src <= 0 || x->dtype.is_ptr) return NULL;
  int total = 0;
  for (int i = 0; i < x->n_src; i++) {
    int cnt = x->src[i]->dtype.count > 0 ? x->src[i]->dtype.count : 1;
    if (cnt > INT32_MAX - total) return NULL;
    total += cnt;
  }
  if (total <= 0) return NULL;
  PolyUOp **elts = calloc((size_t)total, sizeof(*elts));
  if (!elts) return NULL;
  int p = 0;
  for (int i = 0; i < x->n_src; i++) {
    PolyUOp *src = x->src[i];
    int cnt = src->dtype.count > 0 ? src->dtype.count : 1;
    for (int j = 0; j < cnt; j++) {
      elts[p++] = poly_uop1(
          ctx, POLY_OP_GEP, poly_dtype_scalar(src->dtype), src, poly_arg_int(j)
      );
    }
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, x->dtype, elts, p, poly_arg_none());
  free(elts);
  return ret;
}

/* VECTORIZE(GEP(x, a0), GEP(x, a1), ...) where all GEPs share same source x
 * → x.gep((a0, a1, ...))  (collapse to single GEP with tuple arg)
 * Port of tinygrad symbolic.py:199:
 *   (UPat(Ops.VECTORIZE, src=UPat(Ops.GEP, src=(UPat.var("x"),))),
 *    lambda v,x: x.gep(tuple(get_single_element(i.arg) for i in v.src))) */
static PolyUOp *rule_vectorize_same_gep(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_VECTORIZE || root->n_src < 2) return NULL;

  /* All sources must be single-lane GEP from the same source UOp */
  PolyUOp *base = NULL;
  for (int i = 0; i < root->n_src; i++) {
    PolyUOp *s = root->src[i];
    if (!s || s->op != POLY_OP_GEP || s->n_src < 1) return NULL;
    /* Accept both INT arg and INT_TUPLE with n=1 (tinygrad uses tuple form) */
    if (s->arg.kind != POLY_ARG_INT &&
        !(s->arg.kind == POLY_ARG_INT_TUPLE && s->arg.int_tuple.n == 1))
      return NULL;
    if (i == 0)
      base = s->src[0];
    else if (s->src[0] != base)
      return NULL;
  }
  if (!base) return NULL;

  /* Collect lane indices into tuple (stack — poly_uop copies to arena) */
  int n = root->n_src;
  int64_t *lanes = calloc((size_t)n, sizeof(*lanes));
  if (!lanes) return NULL;
  for (int i = 0; i < n; i++) {
    PolyUOp *s = root->src[i];
    lanes[i] = (s->arg.kind == POLY_ARG_INT) ? s->arg.i : s->arg.int_tuple.vals[0];
  }

  /* Create single GEP with tuple arg: base.gep((lane0, lane1, ...))
   * Then rule_gep_identity handles (0,1,...,N-1) → identity. */
  PolyArg tup;
  tup.kind = POLY_ARG_INT_TUPLE;
  tup.int_tuple.vals = lanes;
  tup.int_tuple.n = n;
  PolyUOp *ret = poly_uop1(ctx, POLY_OP_GEP, root->dtype, base, tup);
  free(lanes);
  return ret;
}

/* tinygrad symbolic.py: clean up singleton GROUP wrappers that appear after
 * store splitting/devectorization. Keeping GROUP(x) structurally distinct adds
 * a spurious late node compared to tinygrad's linearize path. */
static PolyUOp *rule_group_singleton(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (!root || root->op != POLY_OP_GROUP || root->n_src != 1) return NULL;
  return root->src[0];
}

/* GEP pushing PM (for combined devec pass) */

static PolyPatternMatcher *g_pm_gep_pushing = NULL;

PolyPatternMatcher *poly_pm_gep_pushing(void) {
  if (g_pm_gep_pushing) return g_pm_gep_pushing;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_vectorize},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_const},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_identity},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_through_alu},
      /* VECTORIZE(GEP(x,a0), GEP(x,a1), ...) → x.gep((a0,a1,...))
       * tinygrad symbolic.py:199 */
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, NULL), rule_vectorize_same_gep},
  };
  g_pm_gep_pushing = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_gep_pushing;
}

/* Build the symbolic_simple PatternMatcher */

static PolyPatternMatcher *g_symbolic_simple = NULL;
static PolyPatternMatcher *g_symbolic = NULL;

PolyPatternMatcher *poly_symbolic_simple(void) {
  if (g_symbolic_simple) return g_symbolic_simple;

  /* Exclude THREEFRY from binary const fold */
  PolyOpSet binary_no_threefry = POLY_GROUP_BINARY;
  binary_no_threefry.bits[POLY_OP_THREEFRY / 64] &= ~((uint64_t)1 << (POLY_OP_THREEFRY % 64));

  /* CAST | BITCAST set */
  PolyOpSet cast_set =
      poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CAST), POLY_OP_BITCAST);

  PolyRule rules[] = {
      /* -- Bool algebra (must come before generic ADD/MUL rules) -- */
      /* bool * bool -> AND */
      {poly_pat_op2(POLY_OP_MUL, poly_pat_any("x"), poly_pat_any("y"), NULL), rule_bool_mul_to_and},
      /* bool + bool -> OR */
      {poly_pat_op2(POLY_OP_ADD, poly_pat_any("x"), poly_pat_any("y"), NULL), rule_bool_add_to_or},

      /* -- Self-folding -- */
      /* x + 0 -> x */
      {poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL),
       rule_identity},
      /* x + 0.0 -> x (float) */
      {poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_float(0.0)), NULL),
       rule_identity},
      /* x * 1 -> x */
      {poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(1)), NULL),
       rule_identity},
      /* x * 1.0 -> x (float) */
      {poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"), poly_pat_const_val(poly_arg_float(1.0)), NULL),
       rule_identity},
      /* AND(x, true) -> x */
      {poly_pat_op2c(POLY_OP_AND, poly_pat_any("x"), poly_pat_const_val(poly_arg_bool(true)), NULL),
       rule_identity},
      /* AND(x, false) -> false */
      {poly_pat_op2c(
           POLY_OP_AND, poly_pat_any("x"), poly_pat_const_val(poly_arg_bool(false)), NULL
       ),
       rule_and_zero},
      /* OR(x, false) -> x */
      {poly_pat_op2c(POLY_OP_OR, poly_pat_any("x"), poly_pat_const_val(poly_arg_bool(false)), NULL),
       rule_identity},
      /* OR(x, true) -> true */
      {poly_pat_op2c(POLY_OP_OR, poly_pat_any("x"), poly_pat_const_val(poly_arg_bool(true)), NULL),
       rule_or_one},
      /* x // x -> 1 */
      {poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_div_self},
      /* x // 1 -> x */
      {poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(1)), NULL),
       rule_identity},
      /* x // -1 -> -x */
      {poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(-1)), NULL),
       rule_div_neg1},
      /* Idempotent(x, x) -> x */
      {poly_pat_ops2(POLY_GROUP_IDEMPOTENT, poly_pat_any("x"), poly_pat_any("x"), NULL),
       rule_idempotent},

      /* -- Zero-folding -- */
      /* x < x -> False */
      {poly_pat_op2(POLY_OP_CMPLT, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_lt_self},
      /* x != x -> False (int/bool only) */
      {poly_pat_op2(POLY_OP_CMPNE, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_cmpne_self},
      /* x % x -> 0 */
      {poly_pat_op2(POLY_OP_MOD, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_mod_self},
      /* x ^ 0 -> x */
      {poly_pat_op2c(POLY_OP_XOR, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL),
       rule_identity},
      /* x ^ x -> 0 */
      {poly_pat_op2(POLY_OP_XOR, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_xor_self},
      /* x & 0 -> 0 (tinygrad symbolic.py:98) */
      {poly_pat_op2c(POLY_OP_AND, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL),
       rule_mul_zero},
      /* x * 0 -> 0 */
      {poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL),
       rule_mul_zero},
      /* x * 0.0 -> 0 (float) */
      {poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"), poly_pat_const_val(poly_arg_float(0.0)), NULL),
       rule_mul_zero},

      /* -- Constant folding -- */
      /* Unary(CONST) -> CONST */
      {poly_pat_ops1(POLY_GROUP_UNARY, poly_pat_cvar(NULL), "a"), rule_const_fold_unary},
      /* Binary(CONST, CONST) -> CONST (excl. THREEFRY) */
      {poly_pat_ops2(binary_no_threefry, poly_pat_cvar(NULL), poly_pat_cvar(NULL), "a"),
       rule_const_fold_binary},
      /* Ternary(CONST, CONST, CONST) -> CONST */
      {poly_pat_ops3(
           POLY_GROUP_TERNARY, poly_pat_cvar(NULL), poly_pat_cvar(NULL), poly_pat_cvar(NULL), "a"
       ),
       rule_const_fold_ternary},
      /* VECTORIZE(CONST...) -> VCONST */
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, NULL), rule_vectorize_const_fold},
      /* GEP simplifications */
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_vectorize},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_const},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_identity},
      /* VCAT -> VECTORIZE(GEP...) */
      {poly_pat_op(POLY_OP_VCAT, NULL, 0, "x"), rule_vcat_to_vectorize},
      /* GROUP(x) -> x */
      {poly_pat_op(POLY_OP_GROUP, NULL, 0, NULL), rule_group_singleton},
      /* VECTORIZE(GEP(x,a0), ...) → x.gep((a0,...)) is in gep_pushing (tinygrad symbolic.py:199).
       * Then rule_gep_identity handles the (0,1,...,N-1) → identity case. */

      /* -- Cast folding -- */
      /* CAST(CONST) -> CONST */
      {poly_pat_op1(POLY_OP_CAST, poly_pat_cvar("c"), NULL), rule_cast_const},
      /* CAST/BITCAST same dtype -> identity */
      {poly_pat_ops(cast_set, NULL, 0, NULL), rule_cast_noop},
      /* CAST(x, bool) -> CMPNE(x, 0) */
      {poly_pat_op1(POLY_OP_CAST, poly_pat_any("x"), NULL), rule_cast_bool},

      /* -- Double negation -- */
      /* NEG(NEG(x)) -> x */
      {poly_pat_op1(POLY_OP_NEG, poly_pat_op1(POLY_OP_NEG, poly_pat_any("x"), NULL), NULL),
       rule_double_neg},

      /* -- Division identities -- */
      /* x / x -> 1 (float) */
      {poly_pat_op2(POLY_OP_FDIV, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_fdiv_self},
      /* (x * y) / y -> x */
      {poly_pat_op2(
           POLY_OP_FDIV, poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"), poly_pat_any("y"), NULL),
           poly_pat_any("y"), NULL
       ),
       rule_mul_fdiv_cancel},

      /* -- Where folding -- */
      /* WHERE(CMPNE(cond, true), t, f) -> WHERE(cond, f, t) */
      {poly_pat_op3(
           POLY_OP_WHERE,
           poly_pat_op2c(
               POLY_OP_CMPNE, poly_pat_dtype("cond", (PolyDType[]){POLY_BOOL}, 1),
               poly_pat_any("trueish"), NULL
           ),
           poly_pat_any("t"), poly_pat_any("f"), NULL
       ),
       rule_where_logical_not},
      /* WHERE(a, WHERE(b, c, d), d) -> WHERE(AND(a, b), c, d) */
      {poly_pat_op3(
           POLY_OP_WHERE, poly_pat_any("a"),
           poly_pat_op3(
               POLY_OP_WHERE, poly_pat_any("b"), poly_pat_any("c"), poly_pat_any("d"), NULL
           ),
           poly_pat_any("d"), NULL
       ),
       rule_nested_where},
      /* WHERE(cond, val, val) -> val */
      {poly_pat_op3(
           POLY_OP_WHERE, poly_pat_any(NULL), poly_pat_any("val"), poly_pat_any("val"), NULL
       ),
       rule_where_same},
      /* LOAD/STORE with Invalid index folds away */
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "x")), rule_fold_invalid_load_store},
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_STORE, NULL, 0, "x")), rule_fold_invalid_load_store},
      /* WHERE(const_gate, c0, c1) -> c0 or c1 */
      {poly_pat_op3(
           POLY_OP_WHERE, poly_pat_cvar("gate"), poly_pat_any("c0"), poly_pat_any("c1"), NULL
       ),
       rule_where_const_gate},

      /* -- vmin/vmax const folding -- */
      {poly_pat_ops(
           poly_opset_add(
               poly_opset_add(
                   poly_opset_add(
                       poly_opset_add(
                           poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CMPLT), POLY_OP_CMPNE
                       ),
                       POLY_OP_IDIV
                   ),
                   POLY_OP_MOD
               ),
               POLY_OP_DEFINE_VAR
           ),
           NULL, 0, "x"
       ),
       rule_const_when_minmax_point},
      {poly_pat_ops(
           poly_opset_add(
               poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_BIND), POLY_OP_SPECIAL
           ),
           NULL, 0, "x"
       ),
       rule_const_when_minmax_point},
      {poly_pat_op1(POLY_OP_RANGE, poly_pat_op(POLY_OP_CONST, NULL, 0, NULL), "x"),
       rule_const_when_minmax_point},
      {poly_pat_op2(POLY_OP_MAX, poly_pat_any("x"), poly_pat_any("y"), NULL), rule_max_fold},

      /* -- Combine terms -- */
      /* x + x -> x * 2 */
      {poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_add_self},
      /* ADD(ADD(a, x), x) -> ADD(a, MUL(x, 2)) */
      {poly_pat_op2c(
           POLY_OP_ADD, poly_pat_op2c(POLY_OP_ADD, poly_pat_any("a"), poly_pat_any("x"), NULL),
           poly_pat_any("x"), NULL
       ),
       rule_add_assoc_self},

      /* -- divmod recombine (dynamic): handles all divmod cancel patterns -- */
      /* (base%div)*mul + (base//div)*(div*mul) -> base*mul and variants */
      {poly_pat_op(POLY_OP_ADD, NULL, 0, NULL), rule_fold_add_divmod_recombine},

      /* -- cancel_divmod: MOD/IDIV simplification via vmin/vmax -- */
      {poly_pat_ops2(
           poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_MOD), POLY_OP_IDIV),
           poly_pat_any(NULL), poly_pat_any(NULL), NULL
       ),
       rule_cancel_divmod},
  };

  int n = sizeof(rules) / sizeof(rules[0]);
  g_symbolic_simple = poly_pm_new(rules, n);
  return g_symbolic_simple;
}

PolyPatternMatcher *poly_symbolic(void) {
  if (g_symbolic) return g_symbolic;
  PolyRule rules[] = {
      {poly_pat_ops2(POLY_GROUP_COMMUTATIVE, poly_pat_any(NULL), poly_pat_any(NULL), "x"),
       rule_commutative_index_tuplize_order},
      {poly_pat_op1(POLY_OP_CAST, poly_pat_any("x"), "cast"),
       rule_cast_index_add_const_to_add_casts},
      {poly_pat_ops2(POLY_GROUP_ASSOCIATIVE, poly_pat_any(NULL), poly_pat_any(NULL), "alu"),
       rule_assoc_fold_consts},
      {poly_pat_op(POLY_OP_MUL, NULL, 0, "mul"), rule_distribute_const_mul_over_add},
      {poly_pat_op(POLY_OP_ADD, NULL, 0, "add"), rule_move_const_to_end},
      {poly_pat_op(POLY_OP_MUL, NULL, 0, "mul"), rule_move_const_to_end},
      {poly_pat_op(POLY_OP_ADD, NULL, 0, "add"), rule_flat_index_outer_term_to_end},
  };
  PolyPatternMatcher *extra = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  PolyPatternMatcher *with_extra = poly_pm_concat(poly_symbolic_simple(), extra);
  poly_pm_destroy(extra); /* concat copies rules; the temporary matcher is no longer needed. */
  g_symbolic = poly_pm_concat(with_extra, poly_pm_gep_pushing());
  poly_pm_destroy(with_extra);
  return g_symbolic;
}
