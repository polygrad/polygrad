/*
 * tensor.c -- Composed tensor ops (elementwise, reduction, creation, etc.)
 *
 * These are higher-level ops built from the core UOp primitives.
 * Mechanical move from frontend.c -- no behavior changes.
 */

#define _GNU_SOURCE
#include "tensor.h"
#include "frontend.h"
#include "scheduler.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

/* ── Dtype table for FFI (used by poly_cast_by_id) ───────────────────── */

static const PolyDType *_dtype_table_ffi[] = {
  &POLY_VOID, &POLY_BOOL, &POLY_INT8, &POLY_UINT8,
  &POLY_INT16, &POLY_UINT16, &POLY_INT32, &POLY_UINT32,
  &POLY_INT64, &POLY_UINT64, &POLY_FLOAT16, &POLY_BFLOAT16,
  &POLY_FLOAT32, &POLY_FLOAT64,
};
#define N_DTYPE_FFI ((int)(sizeof(_dtype_table_ffi) / sizeof(_dtype_table_ffi[0])))

/* ── Internal helpers ────────────────────────────────────────────────── */

/* Helper: float constant matching the dtype of a given UOp.
 * For float inputs: creates a constant with the same float dtype.
 * For non-float inputs (comparisons producing bool): defaults to float32. */
static inline PolyUOp *cf(PolyCtx *ctx, PolyUOp *ref, double v) {
  PolyDType dt = poly_dtype_scalar(ref->dtype);
  if (poly_dtype_is_float(dt))
    return poly_const_typed(ctx, dt, v);
  return poly_const_float(ctx, v);
}

/* Helper: const with explicit dtype -- use in special-math ops for dtype correctness */
static inline PolyUOp *cdt(PolyCtx *ctx, PolyDType dt, double v) {
  return poly_const_typed(ctx, dt, v);
}

int64_t poly_shape_numel_checked(const int64_t *shape, int ndim) {
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return -1;
  if (ndim == 0) return 1;
  if (!shape) return -1;
  int64_t n = 1;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] <= 0) return -1;
    if (n > INT64_MAX / shape[i]) return -1;
    n *= shape[i];
  }
  return n;
}

bool poly_shape_equal(const int64_t *a, int a_ndim, const int64_t *b, int b_ndim) {
  if (!a || !b || a_ndim != b_ndim) return false;
  for (int i = 0; i < a_ndim; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

static bool shape_equal_except_axis(const int64_t *full, int full_ndim,
                                    const int64_t *reduced, int reduced_ndim,
                                    int axis) {
  if (!full || !reduced || full_ndim != reduced_ndim + 1) return false;
  if (axis < 0) axis += full_ndim;
  if (axis < 0 || axis >= full_ndim) return false;
  for (int i = 0, j = 0; i < full_ndim; i++) {
    if (i == axis) continue;
    if (full[i] != reduced[j++]) return false;
  }
  return true;
}

static PolyUOp *make_const_buffer_tensor(PolyCtx *ctx, PolyDType dt,
                                         const void *src, size_t elem_size,
                                         const int64_t *shape, int ndim) {
  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel <= 0 || !src || elem_size == 0) return NULL;
  if (numel > (1LL << 20)) {
    fprintf(stderr, "polygrad: large constant-backed tensor (%lld elems); "
            "prefer explicit bindings/device-generated paths for hot loops\n",
            (long long)numel);
  }
  if ((size_t)numel > SIZE_MAX / elem_size) return NULL;
  size_t nbytes = (size_t)numel * elem_size;
  void *copy = malloc(nbytes);
  if (!copy) return NULL;
  memcpy(copy, src, nbytes);

  PolyUOp *buf = poly_buffer(ctx, dt, numel);
  poly_const_registry_add(ctx, buf, copy);
  if (ndim == 1) return buf;
  return poly_reshape(ctx, buf, (int64_t *)shape, ndim);
}

static PolyUOp *make_const_f32_tensor(PolyCtx *ctx, const float *src,
                                      const int64_t *shape, int ndim) {
  return make_const_buffer_tensor(ctx, POLY_FLOAT32, src, sizeof(float), shape, ndim);
}

static PolyUOp *make_const_u32_tensor(PolyCtx *ctx, const uint32_t *src,
                                      const int64_t *shape, int ndim) {
  return make_const_buffer_tensor(ctx, POLY_UINT32, src, sizeof(uint32_t), shape, ndim);
}

/* ── Constant buffer registry (ctx-scoped) ─────────────────────────────
 *
 * Some additive frontend creation helpers (arange/full/rand/...) return
 * BUFFER-backed tensors. We retain host data here and auto-bind it when a
 * caller omits bindings for those buffers.
 */

typedef struct {
  PolyCtx *ctx;
  PolyUOp *buf;
  void *data;
} PolyConstBindingEntry;

static PolyConstBindingEntry *g_const_bindings = NULL;
static int g_const_bindings_n = 0;
static int g_const_bindings_cap = 0;

void poly_const_registry_add(PolyCtx *ctx, PolyUOp *buf, void *data) {
  if (!ctx || !buf || !data) return;
  for (int i = 0; i < g_const_bindings_n; i++) {
    if (g_const_bindings[i].ctx == ctx && g_const_bindings[i].buf == buf) {
      free(g_const_bindings[i].data);
      g_const_bindings[i].data = data;
      return;
    }
  }
  if (g_const_bindings_n == g_const_bindings_cap) {
    int new_cap = (g_const_bindings_cap == 0) ? 64 : (g_const_bindings_cap * 2);
    PolyConstBindingEntry *nb = realloc(g_const_bindings, (size_t)new_cap * sizeof(*nb));
    if (!nb) {
      free(data);
      return;
    }
    g_const_bindings = nb;
    g_const_bindings_cap = new_cap;
  }
  g_const_bindings[g_const_bindings_n++] = (PolyConstBindingEntry){
    .ctx = ctx, .buf = buf, .data = data
  };
}

void *poly_const_registry_lookup(PolyCtx *ctx, PolyUOp *buf) {
  if (!ctx || !buf) return NULL;
  for (int i = 0; i < g_const_bindings_n; i++) {
    if (g_const_bindings[i].ctx == ctx && g_const_bindings[i].buf == buf)
      return g_const_bindings[i].data;
  }
  return NULL;
}

bool poly_const_registry_has(PolyCtx *ctx, PolyUOp *buf) {
  return poly_const_registry_lookup(ctx, buf) != NULL;
}

void poly_const_registry_cleanup(PolyCtx *ctx) {
  if (g_const_bindings_n > 0) {
    int wr = 0;
    for (int i = 0; i < g_const_bindings_n; i++) {
      if (g_const_bindings[i].ctx == ctx) {
        free(g_const_bindings[i].data);
        continue;
      }
      if (wr != i) g_const_bindings[wr] = g_const_bindings[i];
      wr++;
    }
    g_const_bindings_n = wr;
  }
}

/* ── Internal: compute output shape for a single-axis reduction ──────── */

static void reduce_output_shape(const int64_t *shape, int ndim, int axis,
                                int keepdim, int64_t *out_shape, int *out_ndim) {
  if (axis < 0) axis += ndim;
  int on = 0;
  for (int i = 0; i < ndim; i++) {
    if (i == axis) {
      if (keepdim) out_shape[on++] = 1;
    } else {
      out_shape[on++] = shape[i];
    }
  }
  if (on == 0) { out_shape[0] = 1; on = 1; }
  *out_ndim = on;
}

/* Internal: do a single-axis reduce and optionally reshape away the axis */
static PolyUOp *do_reduce(PolyCtx *ctx, PolyOps reduce_op, PolyUOp *x,
                           const int64_t *shape, int ndim, int axis, int keepdim,
                           int64_t *out_shape, int *out_ndim) {
  if (axis < 0) axis += ndim;
  int64_t axes[] = { axis };
  PolyUOp *r = poly_reduce_axis(ctx, reduce_op, x, axes, 1);
  reduce_output_shape(shape, ndim, axis, keepdim, out_shape, out_ndim);
  if (!keepdim) {
    r = poly_reshape(ctx, r, out_shape, *out_ndim);
  }
  return r;
}

static PolyUOp *reshape_logical_input(PolyCtx *ctx, PolyUOp *x,
                                      const int64_t *shape, int ndim) {
  if (!ctx || !x || (ndim > 0 && !shape)) return NULL;
  if (ndim == 0) return poly_reshape(ctx, x, NULL, 0);
  return poly_reshape(ctx, x, (int64_t *)shape, ndim);
}

/* Internal: read shape from UOp into local arrays */
static int uop_shape(PolyCtx *ctx, PolyUOp *u, int64_t *out_shape) {
  int ndim = poly_uop_ndim(ctx, u);
  if (ndim > 0) {
    const int64_t *dims = poly_uop_dims(ctx, u);
    if (dims) memcpy(out_shape, dims, ndim * sizeof(int64_t));
  }
  return ndim;
}

/* ── erf tau helper ──────────────────────────────────────────────────── */

/* A&S 7.1.26: tau(|x|) = t * P(t) * exp(-x^2) where t = 1/(1+p*|x|).
 * erf(x) = sign(x) * (1 - tau(|x|)).
 * erfc(x) = tau(x) for x >= 0, 2 - tau(|x|) for x < 0.
 * Computing tau directly avoids the 1-erf(x) cancellation in erfc. */
static PolyUOp *erf_tau(PolyCtx *ctx, PolyUOp *ax, PolyDType dt) {
  PolyUOp *t = poly_alu1(ctx, POLY_OP_RECIPROCAL,
             poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.0),
             poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.3275911), ax)));
  PolyUOp *p = cdt(ctx, dt, 1.061405429);
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, -1.453152027), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.421413741), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, -0.284496736), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 0.254829592), poly_alu2(ctx, POLY_OP_MUL, t, p));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, ax, ax);
  PolyUOp *e = poly_exp(ctx, poly_alu1(ctx, POLY_OP_NEG, x2));
  return poly_alu2(ctx, POLY_OP_MUL, t, poly_alu2(ctx, POLY_OP_MUL, p, e));
}

/* ── lgamma Lanczos helper ───────────────────────────────────────────── */

static PolyUOp *poly_lgamma_forward_lanczos(PolyCtx *ctx, PolyUOp *x, PolyDType dt) {
  /* Lanczos approximation with reflection. */
  const double g = 7.0;
  const double c0 = 0.99999999999980993;
  const double c[8] = {
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7
  };

  PolyUOp *xm1 = poly_alu2(ctx, POLY_OP_SUB, x, cdt(ctx, dt, 1.0));
  PolyUOp *a = cdt(ctx, dt, c0);
  for (int i = 0; i < 8; i++) {
    PolyUOp *den = poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, (double)(i + 1)));
    a = poly_alu2(ctx, POLY_OP_ADD, a,
         poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, c[i]), den));
  }
  PolyUOp *t = poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, g + 0.5));
  PolyUOp *lg_pos = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 0.91893853320467274178),
                 poly_alu2(ctx, POLY_OP_ADD,
                   poly_alu2(ctx, POLY_OP_MUL,
                     poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, 0.5)),
                     poly_log(ctx, t)),
                   poly_alu2(ctx, POLY_OP_SUB, poly_log(ctx, a), t)));

  PolyUOp *one_minus_x = poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 1.0), x);
  PolyUOp *xm1r = poly_alu2(ctx, POLY_OP_SUB, one_minus_x, cdt(ctx, dt, 1.0));
  PolyUOp *ar = cdt(ctx, dt, c0);
  for (int i = 0; i < 8; i++) {
    PolyUOp *den = poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, (double)(i + 1)));
    ar = poly_alu2(ctx, POLY_OP_ADD, ar,
         poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, c[i]), den));
  }
  PolyUOp *tr = poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, g + 0.5));
  PolyUOp *lg_ref_base = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 0.91893853320467274178),
                      poly_alu2(ctx, POLY_OP_ADD,
                        poly_alu2(ctx, POLY_OP_MUL,
                          poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, 0.5)),
                          poly_log(ctx, tr)),
                        poly_alu2(ctx, POLY_OP_SUB, poly_log(ctx, ar), tr)));
  PolyUOp *sinpix = poly_sin(ctx, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, M_PI), x));
  PolyUOp *lg_ref = poly_alu2(ctx, POLY_OP_SUB,
                  poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, log(M_PI)),
                    poly_log(ctx, poly_abs(ctx, sinpix))),
                  lg_ref_base);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0.5));
  return poly_alu3(ctx, POLY_OP_WHERE, cond, lg_ref, lg_pos);
}

/* ── tri_mask helper ─────────────────────────────────────────────────── */

static PolyUOp *tri_mask(PolyCtx *ctx, const int64_t *shape, int diagonal, bool upper) {
  if (!shape || shape[0] <= 0 || shape[1] <= 0) return NULL;
  int64_t rows = shape[0], cols = shape[1];
  if ((size_t)rows > SIZE_MAX / (size_t)cols) return NULL;
  size_t numel = (size_t)rows * (size_t)cols;
  float *m = malloc(numel * sizeof(float));
  if (!m) return NULL;
  for (int64_t i = 0; i < rows; i++) {
    for (int64_t j = 0; j < cols; j++) {
      bool keep = upper ? (j >= i + diagonal) : (j <= i + diagonal);
      m[(size_t)i * (size_t)cols + (size_t)j] = keep ? 1.0f : 0.0f;
    }
  }
  PolyUOp *mask = make_const_f32_tensor(ctx, m, shape, 2);
  free(m);
  return mask;
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Public functions                                                     */
/* ══════════════════════════════════════════════════════════════════════ */

/* ── Broadcasting (tinygrad _broadcasted / _broadcast_to) ────────────── */

PolyUOp *poly_broadcast_to(PolyCtx *ctx, PolyUOp *x,
                           const int64_t *shape, int ndim) {
  if (!ctx || !x || !shape || ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  int64_t x_shape[POLY_MAX_DIMS];
  int x_ndim = uop_shape(ctx, x, x_shape);
  if (x_ndim < 0) return x;  /* shapeless (CONST scalar) -- pass through */
  if (ndim < x_ndim) return NULL;  /* can't broadcast to fewer dims */

  /* Already matching */
  if (x_ndim == ndim) {
    bool same = true;
    for (int i = 0; i < ndim; i++)
      if (x_shape[i] != shape[i]) { same = false; break; }
    if (same) return x;
  }

  /* Left-pad with 1s to match ndim (tinygrad _align_left) */
  int64_t aligned[POLY_MAX_DIMS];
  int pad = ndim - x_ndim;
  for (int i = 0; i < pad; i++) aligned[i] = 1;
  for (int i = 0; i < x_ndim; i++) aligned[pad + i] = x_shape[i];

  /* Validate: each aligned dim must be 1 or equal to target */
  for (int i = 0; i < ndim; i++) {
    if (aligned[i] != shape[i] && aligned[i] != 1) {
      fprintf(stderr, "poly_broadcast_to: incompatible dim %d: %lld vs %lld\n",
              i, (long long)aligned[i], (long long)shape[i]);
      return NULL;
    }
  }

  PolyUOp *r = poly_reshape(ctx, x, aligned, ndim);

  /* Expand where aligned[i]==1 and shape[i]>1 */
  bool need_expand = false;
  for (int i = 0; i < ndim; i++)
    if (aligned[i] != shape[i]) { need_expand = true; break; }
  if (need_expand)
    r = poly_expand(ctx, r, (int64_t *)shape, ndim);
  return r;
}

bool poly_broadcast_pair(PolyCtx *ctx, PolyUOp **a, PolyUOp **b,
                         int64_t *out_shape, int *out_ndim) {
  int64_t sa[POLY_MAX_DIMS], sb[POLY_MAX_DIMS];
  int na = uop_shape(ctx, *a, sa);
  int nb = uop_shape(ctx, *b, sb);

  /* Scalars or shapeless -- no broadcast needed */
  if (na <= 0 && nb <= 0) { *out_ndim = 0; return true; }
  if (na <= 0) { na = 0; }
  if (nb <= 0) { nb = 0; }

  /* Compute broadcast shape (tinygrad _broadcast_shape) */
  int nd = na > nb ? na : nb;
  if (nd > POLY_MAX_DIMS) { *out_ndim = 0; return false; }
  int pa = nd - na, pb = nd - nb;
  for (int i = 0; i < nd; i++) {
    int64_t da = (i >= pa) ? sa[i - pa] : 1;
    int64_t db = (i >= pb) ? sb[i - pb] : 1;
    if (da != db && da != 1 && db != 1) {
      fprintf(stderr, "poly_broadcast_pair: incompatible shapes at dim %d: %lld vs %lld\n",
              i, (long long)da, (long long)db);
      *out_ndim = 0;
      return false;
    }
    out_shape[i] = da > db ? da : db;
  }
  *out_ndim = nd;

  *a = poly_broadcast_to(ctx, *a, out_shape, nd);
  *b = poly_broadcast_to(ctx, *b, out_shape, nd);
  return true;
}

/* ── Broadcasting binary ops ──────────────────────────────────────────── */

PolyUOp *poly_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  poly_broadcast_pair(ctx, &a, &b, shape, &ndim);
  return poly_alu2(ctx, POLY_OP_ADD, a, b);
}

PolyUOp *poly_sub(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  poly_broadcast_pair(ctx, &a, &b, shape, &ndim);
  return poly_alu2(ctx, POLY_OP_SUB, a, b);
}

PolyUOp *poly_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  poly_broadcast_pair(ctx, &a, &b, shape, &ndim);
  return poly_alu2(ctx, POLY_OP_MUL, a, b);
}

PolyUOp *poly_div(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  poly_broadcast_pair(ctx, &a, &b, shape, &ndim);
  return poly_alu2(ctx, POLY_OP_FDIV, a, b);
}

/* ── Contiguous (realize barrier) ──────────────────────────────────── */

PolyUOp *poly_contiguous(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop1(ctx, POLY_OP_CONTIGUOUS, x->dtype, x, poly_arg_none());
}

/* ── Math ────────────────────────────────────────────────────────────── */

PolyUOp *poly_exp(PolyCtx *ctx, PolyUOp *x) {
  /* exp(x) = exp2(x * (1/ln2)) */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  return poly_alu1(ctx, POLY_OP_EXP2,
    poly_alu2(ctx, POLY_OP_MUL, x, cdt(ctx, dt, 1.0 / M_LN2)));
}

PolyUOp *poly_log(PolyCtx *ctx, PolyUOp *x) {
  /* log(x) = log2(x) * ln2 */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  return poly_alu2(ctx, POLY_OP_MUL,
    poly_alu1(ctx, POLY_OP_LOG2, x), cdt(ctx, dt, M_LN2));
}

PolyUOp *poly_log1p(PolyCtx *ctx, PolyUOp *x) {
  /* Stable near zero: log(1+x) ~ x - x^2/2 + x^3/3 */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *small = poly_alu2(ctx, POLY_OP_CMPLT, ax, cdt(ctx, dt, 1e-4));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x2, x);
  PolyUOp *poly = poly_alu2(ctx, POLY_OP_ADD, x,
                 poly_alu2(ctx, POLY_OP_ADD,
                   poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -0.5), x2),
                   poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0/3.0), x3)));
  PolyUOp *direct = poly_log(ctx, poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.0), x));
  return poly_alu3(ctx, POLY_OP_WHERE, small, poly, direct);
}

PolyUOp *poly_expm1(PolyCtx *ctx, PolyUOp *x) {
  /* Stable near zero: expm1(x) ~ x + x^2/2 + x^3/6 */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *small = poly_alu2(ctx, POLY_OP_CMPLT, ax, cdt(ctx, dt, 1e-4));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x2, x);
  PolyUOp *poly = poly_alu2(ctx, POLY_OP_ADD, x,
                 poly_alu2(ctx, POLY_OP_ADD,
                   poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.5), x2),
                   poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0/6.0), x3)));
  PolyUOp *direct = poly_alu2(ctx, POLY_OP_SUB, poly_exp(ctx, x), cdt(ctx, dt, 1.0));
  return poly_alu3(ctx, POLY_OP_WHERE, small, poly, direct);
}

PolyUOp *poly_sin(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu1(ctx, POLY_OP_SIN, x);
}

PolyUOp *poly_cos(PolyCtx *ctx, PolyUOp *x) {
  /* cos(x) = sin(pi/2 - x) */
  return poly_alu1(ctx, POLY_OP_SIN,
    poly_alu2(ctx, POLY_OP_SUB, cf(ctx, x, M_PI / 2.0), x));
}

PolyUOp *poly_tan(PolyCtx *ctx, PolyUOp *x) {
  /* tan(x) = sin(x) / cos(x) */
  return poly_alu2(ctx, POLY_OP_FDIV, poly_sin(ctx, x), poly_cos(ctx, x));
}

PolyUOp *poly_erf(PolyCtx *ctx, PolyUOp *x) {
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *sign = poly_sign(ctx, x);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *tau = erf_tau(ctx, ax, dt);
  return poly_alu2(ctx, POLY_OP_MUL, sign, poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 1.0), tau));
}

PolyUOp *poly_erfc(PolyCtx *ctx, PolyUOp *x) {
  /* erfc(x) = tau(|x|) for x >= 0, 2 - tau(|x|) for x < 0.
   * No 1-erf(x) cancellation -- tau is computed directly. */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *tau = erf_tau(ctx, ax, dt);
  PolyUOp *neg = poly_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0.0));
  PolyUOp *erfc_neg = poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 2.0), tau);
  return poly_alu3(ctx, POLY_OP_WHERE, neg, erfc_neg, tau);
}

PolyUOp *poly_erfinv(PolyCtx *ctx, PolyUOp *x) {
  /* Winitzki approximation (a=0.147). */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *a = cdt(ctx, dt, 0.147);
  PolyUOp *one = cdt(ctx, dt, 1.0);
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *ln = poly_log(ctx, poly_alu2(ctx, POLY_OP_SUB, one, x2));
  PolyUOp *term1 = poly_alu2(ctx, POLY_OP_ADD,
                   poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, 2.0/(M_PI*0.147)), one),
                   poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.5), ln));
  PolyUOp *term2 = poly_alu2(ctx, POLY_OP_FDIV, ln, a);
  PolyUOp *inside = poly_alu2(ctx, POLY_OP_SUB,
                    poly_alu2(ctx, POLY_OP_MUL, term1, term1), term2);
  PolyUOp *root = poly_alu1(ctx, POLY_OP_SQRT,
                  poly_alu2(ctx, POLY_OP_SUB, poly_alu1(ctx, POLY_OP_SQRT, inside), term1));
  return poly_alu2(ctx, POLY_OP_MUL, poly_sign(ctx, x), root);
}

PolyUOp *poly_ndtri(PolyCtx *ctx, PolyUOp *x) {
  /* ndtri(p) = sqrt(2) * erfinv(2p-1) */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *arg = poly_alu2(ctx, POLY_OP_SUB,
                poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 2.0), x), cdt(ctx, dt, 1.0));
  return poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, sqrt(2.0)), poly_erfinv(ctx, arg));
}

PolyUOp *poly_digamma(PolyCtx *ctx, PolyUOp *x) {
  /* First-order asymptotic with recurrence to x>=6. */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *acc = cdt(ctx, dt, 0.0);
  PolyUOp *xx = x;
  for (int i = 0; i < 6; i++) {
    PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, xx, cdt(ctx, dt, 6.0));
    PolyUOp *inv = poly_alu1(ctx, POLY_OP_RECIPROCAL, xx);
    acc = poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_SUB, acc, inv), acc);
    xx = poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_ADD, xx, cdt(ctx, dt, 1.0)), xx);
  }
  PolyUOp *inv = poly_alu1(ctx, POLY_OP_RECIPROCAL, xx);
  PolyUOp *inv2 = poly_alu2(ctx, POLY_OP_MUL, inv, inv);
  PolyUOp *inv4 = poly_alu2(ctx, POLY_OP_MUL, inv2, inv2);
  PolyUOp *inv6 = poly_alu2(ctx, POLY_OP_MUL, inv4, inv2);
  PolyUOp *asym = poly_alu2(ctx, POLY_OP_ADD, poly_log(ctx, xx),
                poly_alu2(ctx, POLY_OP_ADD,
                  poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -0.5), inv),
                  poly_alu2(ctx, POLY_OP_ADD,
                    poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -1.0/12.0), inv2),
                    poly_alu2(ctx, POLY_OP_ADD,
                      poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0/120.0), inv4),
                      poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -1.0/252.0), inv6)))));
  return poly_alu2(ctx, POLY_OP_ADD, acc, asym);
}

PolyUOp *poly_lgamma(PolyCtx *ctx, PolyUOp *x) {
  /* Explicit VJP override:
   * y = detach(f(x)) + (x - detach(x))*digamma(x) */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *fwd = poly_lgamma_forward_lanczos(ctx, x, dt);
  PolyUOp *dx = poly_uop1(ctx, POLY_OP_DETACH, x->dtype, x, poly_arg_none());
  PolyUOp *df = poly_uop1(ctx, POLY_OP_DETACH, fwd->dtype, fwd, poly_arg_none());
  PolyUOp *delta = poly_alu2(ctx, POLY_OP_SUB, x, dx);
  PolyUOp *forced = poly_alu2(ctx, POLY_OP_MUL, delta, poly_digamma(ctx, x));
  return poly_alu2(ctx, POLY_OP_ADD, df, forced);
}

PolyUOp *poly_sigmoid(PolyCtx *ctx, PolyUOp *x) {
  /* sigmoid(x) = (1 + exp2(x * (-1/ln2)))^-1 */
  PolyUOp *scaled = poly_alu2(ctx, POLY_OP_MUL, x, cf(ctx, x, -1.0 / M_LN2));
  PolyUOp *e = poly_alu1(ctx, POLY_OP_EXP2, scaled);
  return poly_alu1(ctx, POLY_OP_RECIPROCAL,
    poly_alu2(ctx, POLY_OP_ADD, cf(ctx, x, 1.0), e));
}

PolyUOp *poly_tanh_act(PolyCtx *ctx, PolyUOp *x) {
  /* tanh(x) = 2*sigmoid(2x) - 1 */
  PolyUOp *two_x = poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 2.0), x);
  return poly_alu2(ctx, POLY_OP_SUB,
    poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 2.0), poly_sigmoid(ctx, two_x)),
    cf(ctx, x, 1.0));
}

PolyUOp *poly_abs(PolyCtx *ctx, PolyUOp *x) {
  /* abs(x) = x * sign(x) */
  return poly_alu2(ctx, POLY_OP_MUL, x, poly_sign(ctx, x));
}

PolyUOp *poly_sign(PolyCtx *ctx, PolyUOp *x) {
  /* sign(x) = ne(x,0).where(lt(x,0).where(-1, 1), 0) + x*0
   * The +x*0 preserves NaN (NaN*0=NaN, NaN+0=NaN) */
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *is_nonzero = poly_alu2(ctx, POLY_OP_CMPNE, x, zero);
  PolyUOp *is_neg = poly_alu2(ctx, POLY_OP_CMPLT, x, zero);
  PolyUOp *neg_or_pos = poly_alu3(ctx, POLY_OP_WHERE, is_neg, cf(ctx, x, -1.0), cf(ctx, x, 1.0));
  PolyUOp *result = poly_alu3(ctx, POLY_OP_WHERE, is_nonzero, neg_or_pos, zero);
  /* +x*0 to propagate NaN */
  return poly_alu2(ctx, POLY_OP_ADD, result,
    poly_alu2(ctx, POLY_OP_MUL, x, zero));
}

PolyUOp *poly_square(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL, x, x);
}

PolyUOp *poly_rsqrt(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu1(ctx, POLY_OP_RECIPROCAL, poly_alu1(ctx, POLY_OP_SQRT, x));
}

PolyUOp *poly_ceil(PolyCtx *ctx, PolyUOp *x) {
  /* ceil(x) = (x > (b=trunc(x))).where(b+1, b) */
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, b, x); /* b < x = x > b */
  return poly_alu3(ctx, POLY_OP_WHERE, cond,
    poly_alu2(ctx, POLY_OP_ADD, b, cf(ctx, x, 1.0)), b);
}

PolyUOp *poly_floor(PolyCtx *ctx, PolyUOp *x) {
  /* floor(x) = (x < (b=trunc(x))).where(b-1, b) */
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, b); /* x < b */
  return poly_alu3(ctx, POLY_OP_WHERE, cond,
    poly_alu2(ctx, POLY_OP_SUB, b, cf(ctx, x, 1.0)), b);
}

PolyUOp *poly_round_f(PolyCtx *ctx, PolyUOp *x) {
  /* round(x) with banker's rounding (round half to even):
   * (x > 0) == (trunc(x/2) == trunc(trunc(x)/2)) ? ceil(x-0.5) : floor(x+0.5) */
  PolyUOp *half = cf(ctx, x, 0.5);
  PolyUOp *two = cf(ctx, x, 2.0);
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *x_gt_0 = poly_alu2(ctx, POLY_OP_CMPLT, cf(ctx, x, 0.0), x);
  PolyUOp *b_half = poly_alu2(ctx, POLY_OP_FDIV, b, two);
  PolyUOp *x_half = poly_alu2(ctx, POLY_OP_FDIV, x, two);
  PolyUOp *trunc_b_half = poly_alu1(ctx, POLY_OP_TRUNC, b_half);
  PolyUOp *trunc_x_half = poly_alu1(ctx, POLY_OP_TRUNC, x_half);
  PolyUOp *halves_eq = poly_eq(ctx, trunc_b_half, trunc_x_half);
  PolyUOp *cond = poly_eq(ctx, x_gt_0, halves_eq);
  return poly_alu3(ctx, POLY_OP_WHERE, cond,
    poly_ceil(ctx, poly_alu2(ctx, POLY_OP_SUB, x, half)),
    poly_floor(ctx, poly_alu2(ctx, POLY_OP_ADD, x, half)));
}

PolyUOp *poly_isinf(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *x_minus_x = poly_alu2(ctx, POLY_OP_SUB, x, x);
  PolyUOp *not_nan = poly_eq(ctx, x, x); /* true if not NaN */
  PolyUOp *sub_nan = poly_ne(ctx, x_minus_x, x_minus_x); /* true if x-x is NaN (inf case) */
  PolyUOp *not_zero = poly_ne(ctx, x, zero);
  /* all three must be true: use AND via MUL on bool-like values */
  PolyUOp *t1 = poly_alu2(ctx, POLY_OP_MUL, not_nan, sub_nan);
  return poly_alu2(ctx, POLY_OP_MUL, t1, not_zero);
}

PolyUOp *poly_isnan(PolyCtx *ctx, PolyUOp *x) {
  /* isnan(x) = (x != x) -- IEEE 754 */
  return poly_alu2(ctx, POLY_OP_CMPNE, x, x);
}

/* ── Activations ─────────────────────────────────────────────────────── */

PolyUOp *poly_relu(PolyCtx *ctx, PolyUOp *x) {
  /* relu(x) = where(0 < x, x, 0) */
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, zero, x);
  return poly_alu3(ctx, POLY_OP_WHERE, cond, x, zero);
}

PolyUOp *poly_relu6(PolyCtx *ctx, PolyUOp *x) {
  /* relu6(x) = relu(x) - relu(x - 6) */
  return poly_alu2(ctx, POLY_OP_SUB,
    poly_relu(ctx, x),
    poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, x, cf(ctx, x, 6.0))));
}

PolyUOp *poly_leaky_relu(PolyCtx *ctx, PolyUOp *x, double neg_slope) {
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, cf(ctx, x, 0.0));
  return poly_alu3(ctx, POLY_OP_WHERE, cond,
    poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, neg_slope), x), x);
}

PolyUOp *poly_gelu(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x, poly_alu2(ctx, POLY_OP_MUL, x, x));
  PolyUOp *inner = poly_alu2(ctx, POLY_OP_ADD, x,
    poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 0.044715), x3));
  PolyUOp *scaled = poly_alu2(ctx, POLY_OP_MUL,
    cf(ctx, x, sqrt(2.0 / M_PI)), inner);
  return poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 0.5),
    poly_alu2(ctx, POLY_OP_MUL, x,
      poly_alu2(ctx, POLY_OP_ADD, cf(ctx, x, 1.0), poly_tanh_act(ctx, scaled))));
}

PolyUOp *poly_quick_gelu(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL, x,
    poly_sigmoid(ctx, poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 1.702), x)));
}

PolyUOp *poly_silu(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL, x, poly_sigmoid(ctx, x));
}

PolyUOp *poly_elu(PolyCtx *ctx, PolyUOp *x, double alpha) {
  return poly_alu2(ctx, POLY_OP_SUB,
    poly_relu(ctx, x),
    poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, alpha),
      poly_relu(ctx,
        poly_alu2(ctx, POLY_OP_SUB, cf(ctx, x, 1.0), poly_exp(ctx, x)))));
}

PolyUOp *poly_softplus(PolyCtx *ctx, PolyUOp *x, double beta) {
  PolyUOp *bx = poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, beta), x);
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *m = poly_alu2(ctx, POLY_OP_MAX, bx, zero);
  PolyUOp *ea = poly_exp(ctx, poly_alu2(ctx, POLY_OP_SUB, bx, m));
  PolyUOp *eb = poly_exp(ctx, poly_alu2(ctx, POLY_OP_SUB, zero, m));
  PolyUOp *lae = poly_alu2(ctx, POLY_OP_ADD, m,
    poly_log(ctx, poly_alu2(ctx, POLY_OP_ADD, ea, eb)));
  return poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 1.0 / beta), lae);
}

PolyUOp *poly_mish(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL, x,
    poly_tanh_act(ctx, poly_softplus(ctx, x, 1.0)));
}

PolyUOp *poly_hardtanh(PolyCtx *ctx, PolyUOp *x, double min_val, double max_val) {
  return poly_clamp(ctx, x, min_val, max_val);
}

PolyUOp *poly_hardswish(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL,
    poly_alu2(ctx, POLY_OP_MUL, x,
      poly_relu6(ctx, poly_alu2(ctx, POLY_OP_ADD, x, cf(ctx, x, 3.0)))),
    cf(ctx, x, 1.0 / 6.0));
}

PolyUOp *poly_hardsigmoid(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *t = poly_alu2(ctx, POLY_OP_ADD,
    poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 1.0/6.0), x), cf(ctx, x, 0.5));
  return poly_alu2(ctx, POLY_OP_SUB,
    poly_relu(ctx, t),
    poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, t, cf(ctx, x, 1.0))));
}

/* ── Comparisons (broadcasting) ──────────────────────────────────────── */

PolyUOp *poly_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS]; int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  PolyUOp *ne = poly_alu2(ctx, POLY_OP_CMPNE, a, b);
  return poly_alu3(ctx, POLY_OP_WHERE, ne, cf(ctx, a, 0.0), cf(ctx, a, 1.0));
}

PolyUOp *poly_ne(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS]; int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  return poly_alu2(ctx, POLY_OP_CMPNE, a, b);
}

PolyUOp *poly_gt(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS]; int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  return poly_alu2(ctx, POLY_OP_CMPLT, b, a);
}

PolyUOp *poly_ge(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS]; int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  PolyUOp *lt = poly_alu2(ctx, POLY_OP_CMPLT, a, b);
  return poly_alu3(ctx, POLY_OP_WHERE, lt, cf(ctx, a, 0.0), cf(ctx, a, 1.0));
}

PolyUOp *poly_le(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS]; int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  PolyUOp *gt_val = poly_alu2(ctx, POLY_OP_CMPLT, b, a);
  return poly_alu3(ctx, POLY_OP_WHERE, gt_val, cf(ctx, a, 0.0), cf(ctx, a, 1.0));
}

PolyUOp *poly_cast(PolyCtx *ctx, PolyUOp *x, PolyDType target) {
  return poly_uop1(ctx, POLY_OP_CAST, target, x, poly_arg_none());
}

PolyUOp *poly_cast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id) {
  if (dtype_id < 0 || dtype_id >= N_DTYPE_FFI) return NULL;
  return poly_cast(ctx, x, *_dtype_table_ffi[dtype_id]);
}

PolyUOp *poly_where_op(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y) {
  int64_t s[POLY_MAX_DIMS]; int nd;
  poly_broadcast_pair(ctx, &x, &y, s, &nd);
  poly_broadcast_pair(ctx, &cond, &x, s, &nd);
  return poly_alu3(ctx, POLY_OP_WHERE, cond, x, y);
}

PolyUOp *poly_maximum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS]; int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  return poly_alu2(ctx, POLY_OP_MAX, a, b);
}

PolyUOp *poly_minimum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS]; int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  return poly_alu1(ctx, POLY_OP_NEG,
    poly_alu2(ctx, POLY_OP_MAX,
      poly_alu1(ctx, POLY_OP_NEG, a),
      poly_alu1(ctx, POLY_OP_NEG, b)));
}

PolyUOp *poly_clamp(PolyCtx *ctx, PolyUOp *x, double lo, double hi) {
  PolyUOp *lo_c = cf(ctx, x, lo);
  PolyUOp *hi_c = cf(ctx, x, hi);
  PolyUOp *lt_lo = poly_alu2(ctx, POLY_OP_CMPLT, x, lo_c);
  PolyUOp *clamped_lo = poly_alu3(ctx, POLY_OP_WHERE, lt_lo, lo_c, x);
  PolyUOp *gt_hi = poly_alu2(ctx, POLY_OP_CMPLT, hi_c, clamped_lo);
  return poly_alu3(ctx, POLY_OP_WHERE, gt_hi, hi_c, clamped_lo);
}

PolyUOp *poly_detach(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop1(ctx, POLY_OP_DETACH, x->dtype, x, poly_arg_none());
}

/* ── Creation ────────────────────────────────────────────────────────── */

PolyUOp *poly_full(PolyCtx *ctx, const int64_t *shape, int ndim, double fill_value) {
  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel < 0) return NULL;
  if (numel == 0) return poly_buffer(ctx, POLY_FLOAT32, 0);
  float *data = malloc((size_t)numel * sizeof(float));
  if (!data) return NULL;
  for (int64_t i = 0; i < numel; i++) data[i] = (float)fill_value;
  PolyUOp *u = make_const_f32_tensor(ctx, data, shape, ndim);
  free(data);
  return u;
}

PolyUOp *poly_arange(PolyCtx *ctx, double start, double stop, double step) {
  if (step == 0.0) {
    fprintf(stderr, "polygrad: arange: step must be non-zero\n");
    return NULL;
  }
  int64_t n = 0;
  if ((step > 0.0 && start < stop) || (step < 0.0 && start > stop)) {
    double span = (stop - start) / step;
    n = (int64_t)ceil(span - 1e-12);
    if (n < 0) n = 0;
  }
  if (n == 0) return poly_buffer(ctx, POLY_FLOAT32, 0);
  float *data = malloc((size_t)n * sizeof(float));
  if (!data) return NULL;
  for (int64_t i = 0; i < n; i++) data[i] = (float)(start + (double)i * step);
  int64_t shape[1] = { n };
  PolyUOp *u = make_const_f32_tensor(ctx, data, shape, 1);
  free(data);
  return u;
}

PolyUOp *poly_linspace(PolyCtx *ctx, double start, double stop, int64_t steps) {
  if (steps <= 0) return poly_buffer(ctx, POLY_FLOAT32, 0);
  float *data = malloc((size_t)steps * sizeof(float));
  if (!data) return NULL;
  if (steps == 1) data[0] = (float)start;
  else {
    for (int64_t i = 0; i < steps; i++)
      data[i] = (float)(start + (stop - start) * (double)i / (double)(steps - 1));
  }
  int64_t shape[1] = { steps };
  PolyUOp *u = make_const_f32_tensor(ctx, data, shape, 1);
  free(data);
  return u;
}

PolyUOp *poly_eye(PolyCtx *ctx, int64_t n) {
  if (n <= 0) return poly_buffer(ctx, POLY_FLOAT32, 0);
  if ((size_t)n > SIZE_MAX / (size_t)n) return NULL;
  size_t numel = (size_t)n * (size_t)n;
  float *data = calloc(numel, sizeof(float));
  if (!data) return NULL;
  for (int64_t i = 0; i < n; i++) data[(size_t)i * (size_t)n + (size_t)i] = 1.0f;
  int64_t shape[2] = { n, n };
  PolyUOp *u = make_const_f32_tensor(ctx, data, shape, 2);
  free(data);
  return u;
}

PolyUOp *poly_tril(PolyCtx *ctx, PolyUOp *x, int diagonal) {
  if (!x) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim != 2) {
    fprintf(stderr, "polygrad: tril: only 2D tensors are supported\n");
    return NULL;
  }
  PolyUOp *mask = tri_mask(ctx, shape, diagonal, false);
  if (!mask) return NULL;
  return poly_alu2(ctx, POLY_OP_MUL, x, mask);
}

PolyUOp *poly_triu(PolyCtx *ctx, PolyUOp *x, int diagonal) {
  if (!x) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim != 2) {
    fprintf(stderr, "polygrad: triu: only 2D tensors are supported\n");
    return NULL;
  }
  PolyUOp *mask = tri_mask(ctx, shape, diagonal, true);
  if (!mask) return NULL;
  return poly_alu2(ctx, POLY_OP_MUL, x, mask);
}

PolyUOp *poly_rand(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed) {
  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel <= 0) return NULL;
  if ((size_t)numel > SIZE_MAX / sizeof(uint32_t)) return NULL;
  uint32_t *counter = malloc((size_t)numel * sizeof(uint32_t));
  if (!counter) {
    return NULL;
  }

  uint32_t key_lo = (uint32_t)(seed & 0xffffffffu);
  uint32_t key_hi = (uint32_t)(seed >> 32);
  uint32_t mixed_key = key_lo ^ ((key_hi << 16) | (key_hi >> 16));
  for (int64_t i = 0; i < numel; i++) {
    counter[i] = (uint32_t)i;
  }

  PolyUOp *counter_t = make_const_u32_tensor(ctx, counter, shape, ndim);
  free(counter);
  if (!counter_t) return NULL;
  PolyUOp *key_t = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int((int64_t)mixed_key));

  PolyUOp *bits = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, counter_t, key_t, poly_arg_none());
  PolyUOp *sh = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(8));
  PolyUOp *hi24 = poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, bits, sh, poly_arg_none());
  PolyUOp *as_f = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, hi24, poly_arg_none());
  return poly_alu2(ctx, POLY_OP_MUL, as_f, cf(ctx, as_f, 1.0 / 16777216.0));
}

PolyUOp *poly_randn(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed) {
  PolyUOp *u1 = poly_rand(ctx, shape, ndim, seed);
  PolyUOp *u2 = poly_rand(ctx, shape, ndim, seed ^ 0x9E3779B97F4A7C15ull);
  if (!u1 || !u2) return NULL;
  PolyUOp *u1_safe = poly_maximum(ctx, u1, cf(ctx, u1, 1e-7));
  PolyUOp *r = poly_alu1(ctx, POLY_OP_SQRT,
              poly_alu2(ctx, POLY_OP_MUL, cf(ctx, u1, -2.0), poly_log(ctx, u1_safe)));
  PolyUOp *theta = poly_alu2(ctx, POLY_OP_MUL, cf(ctx, u2, 2.0 * M_PI), u2);
  return poly_alu2(ctx, POLY_OP_MUL, r, poly_cos(ctx, theta));
}

PolyUOp *poly_cholesky(PolyCtx *ctx, PolyUOp *x, int upper) {
  (void)ctx; (void)x; (void)upper;
  fprintf(stderr, "polygrad: cholesky: Track C fallback not yet implemented in frontend graph builder\n");
  return NULL;
}

PolyUOp *poly_triangular_solve(PolyCtx *ctx, PolyUOp *a, PolyUOp *b,
                               int upper, int transpose_a, int unit_diagonal) {
  (void)ctx; (void)a; (void)b;
  (void)upper; (void)transpose_a; (void)unit_diagonal;
  fprintf(stderr, "polygrad: triangular_solve: Track C fallback not yet implemented in frontend graph builder\n");
  return NULL;
}

/* ── Reductions ──────────────────────────────────────────────────────── */

PolyUOp *poly_sum_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t shape[POLY_MAX_DIMS], out_shape[POLY_MAX_DIMS]; int ndim, out_ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  return do_reduce(ctx, POLY_OP_ADD, x, shape, ndim, axis, keepdim, out_shape, &out_ndim);
}

PolyUOp *poly_max_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t shape[POLY_MAX_DIMS], out_shape[POLY_MAX_DIMS]; int ndim, out_ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  return do_reduce(ctx, POLY_OP_MAX, x, shape, ndim, axis, keepdim, out_shape, &out_ndim);
}

PolyUOp *poly_mean_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t shape[POLY_MAX_DIMS], out_shape[POLY_MAX_DIMS]; int ndim, out_ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  int64_t count = shape[axis];
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, x, shape, ndim, axis, keepdim,
                          out_shape, &out_ndim);
  return poly_alu2(ctx, POLY_OP_FDIV, s, cf(ctx, x, (double)count));
}

PolyUOp *poly_var_reduce(PolyCtx *ctx, PolyUOp *x,
                         int axis, int keepdim, int correction) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  int64_t count = shape[axis];
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;
  /* var(x) = mean((x - mean(x))^2) * count / (count - correction) */
  /* First get mean with keepdim=1 for broadcast */
  PolyUOp *m = poly_mean_reduce(ctx, x_view, axis, 1);
  /* Expand mean back to full shape for subtraction */
  PolyUOp *m_expanded = poly_expand(ctx, m, (int64_t *)shape, ndim);
  /* (x - mean)^2 */
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_SUB, x_view, m_expanded);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
  /* sum of squares / (count - correction) */
  int64_t out_shape[POLY_MAX_DIMS]; int out_ndim;
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, sq, shape, ndim, axis, keepdim,
                          out_shape, &out_ndim);
  double divisor = (double)(count - correction);
  if (divisor <= 0.0) divisor = 1.0;
  return poly_alu2(ctx, POLY_OP_FDIV, s, cf(ctx, x, divisor));
}

PolyUOp *poly_logsumexp(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;
  int64_t keep_shape[POLY_MAX_DIMS];
  int keep_ndim = 0;
  PolyUOp *m = do_reduce(ctx, POLY_OP_MAX, x_view, shape, ndim, axis, 1, keep_shape, &keep_ndim);
  PolyUOp *shifted = poly_sub(ctx, x_view, m);
  PolyUOp *e = poly_exp(ctx, shifted);
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, e, shape, ndim, axis, 1, keep_shape, &keep_ndim);
  PolyUOp *lse_keep = poly_add(ctx, poly_log(ctx, s), m);
  if (keepdim) return lse_keep;

  int64_t final_shape[POLY_MAX_DIMS];
  int fn = 0;
  for (int i = 0; i < ndim; i++) {
    if (i == axis) continue;
    final_shape[fn++] = shape[i];
  }
  if (fn == 0) return poly_reshape(ctx, lse_keep, NULL, 0);
  return poly_reshape(ctx, lse_keep, final_shape, fn);
}

/* ── Matmul ──────────────────────────────────────────────────────────── */

PolyUOp *poly_dot(PolyCtx *ctx, PolyUOp *x, PolyUOp *w) {
  if (!ctx || !x || !w) return NULL;
  int64_t x_shape[POLY_MAX_DIMS], w_shape[POLY_MAX_DIMS];
  int x_ndim = uop_shape(ctx, x, x_shape);
  int w_ndim = uop_shape(ctx, w, w_shape);
  if (x_ndim < 0 || w_ndim < 0) return NULL;
  int64_t out_shape[POLY_MAX_DIMS]; int out_ndim = 0;
  if (x_ndim < 1 || w_ndim < 1 || x_ndim > POLY_MAX_DIMS || w_ndim > POLY_MAX_DIMS) return NULL;

  int64_t K = x_shape[x_ndim - 1];
  int axis_w = w_ndim - (w_ndim >= 2 ? 2 : 1);
  if (K != w_shape[axis_w]) return NULL;

  int64_t xs[POLY_MAX_DIMS]; int xn = 0;
  for (int i = 0; i < x_ndim - 1; i++) xs[xn++] = x_shape[i];
  int n_ones_x;
  {
    int a = x_ndim - 1, b = w_ndim - 1;
    n_ones_x = a < b ? a : b;
    if (n_ones_x > 1) n_ones_x = 1;
  }
  for (int i = 0; i < n_ones_x; i++) xs[xn++] = 1;
  xs[xn++] = K;
  PolyUOp *xr = poly_reshape(ctx, x, xs, xn);

  int64_t ws[POLY_MAX_DIMS]; int wn = 0;
  for (int i = 0; i < w_ndim - 2; i++) ws[wn++] = w_shape[i];
  for (int i = 0; i < n_ones_x; i++) ws[wn++] = 1;
  for (int i = axis_w; i < w_ndim; i++) ws[wn++] = w_shape[i];
  PolyUOp *wr = poly_reshape(ctx, w, ws, wn);

  int new_axis_w = wn - 2;
  if (new_axis_w < 0) new_axis_w = 0;
  int64_t perm[POLY_MAX_DIMS];
  for (int i = 0; i < wn; i++) perm[i] = i;
  perm[wn - 1] = new_axis_w;
  perm[new_axis_w] = wn - 1;
  PolyUOp *wt = poly_permute(ctx, wr, perm, wn);
  int64_t wt_shape[POLY_MAX_DIMS];
  for (int i = 0; i < wn; i++) wt_shape[i] = ws[perm[i]];

  int max_ndim = xn > wn ? xn : wn;
  int64_t bc_shape[POLY_MAX_DIMS];
  for (int i = 0; i < max_ndim; i++) {
    int xi = i - (max_ndim - xn);
    int wi = i - (max_ndim - wn);
    int64_t xd = (xi >= 0) ? xs[xi] : 1;
    int64_t wd = (wi >= 0) ? wt_shape[wi] : 1;
    if (xd != wd && xd != 1 && wd != 1) return NULL;
    bc_shape[i] = xd > wd ? xd : wd;
  }

  PolyUOp *x_exp, *w_exp;
  if (xn < max_ndim) {
    int64_t padded[POLY_MAX_DIMS];
    int pad = max_ndim - xn;
    for (int i = 0; i < pad; i++) padded[i] = 1;
    for (int i = 0; i < xn; i++) padded[pad + i] = xs[i];
    x_exp = poly_reshape(ctx, xr, padded, max_ndim);
  } else {
    x_exp = xr;
  }
  if (wn < max_ndim) {
    int64_t padded[POLY_MAX_DIMS];
    int pad = max_ndim - wn;
    for (int i = 0; i < pad; i++) padded[i] = 1;
    for (int i = 0; i < wn; i++) padded[pad + i] = wt_shape[i];
    w_exp = poly_reshape(ctx, wt, padded, max_ndim);
  } else {
    w_exp = wt;
  }
  x_exp = poly_expand(ctx, x_exp, bc_shape, max_ndim);
  w_exp = poly_expand(ctx, w_exp, bc_shape, max_ndim);

  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, x_exp, w_exp);
  int64_t sum_axis[] = { max_ndim - 1 };
  PolyUOp *summed = poly_reduce_axis(ctx, POLY_OP_ADD, mul, sum_axis, 1);

  int on = 0;
  for (int i = 0; i < max_ndim - 1; i++) {
    out_shape[on++] = bc_shape[i];
  }
  if (on == 0) { out_shape[0] = 1; on = 1; }
  out_ndim = on;

  return poly_reshape(ctx, summed, out_shape, on);
}

/* ── Softmax ─────────────────────────────────────────────────────────── */

PolyUOp *poly_softmax(PolyCtx *ctx, PolyUOp *x, int axis) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;

  int64_t max_shape[8]; int max_ndim;
  PolyUOp *m = do_reduce(ctx, POLY_OP_MAX, x_view, shape, ndim, axis, 1,
                          max_shape, &max_ndim);
  PolyUOp *m_exp = poly_expand(ctx, m, (int64_t *)shape, ndim);
  PolyUOp *shifted = poly_alu2(ctx, POLY_OP_SUB, x_view, m_exp);
  PolyUOp *e = poly_exp(ctx, shifted);

  int64_t sum_shape[8]; int sum_ndim;
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, e, shape, ndim, axis, 1,
                          sum_shape, &sum_ndim);
  PolyUOp *s_exp = poly_expand(ctx, s, (int64_t *)shape, ndim);

  return poly_alu2(ctx, POLY_OP_FDIV, e, s_exp);
}

PolyUOp *poly_log_softmax(PolyCtx *ctx, PolyUOp *x, int axis) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;

  int64_t max_shape[8]; int max_ndim;
  PolyUOp *m = do_reduce(ctx, POLY_OP_MAX, x_view, shape, ndim, axis, 1,
                          max_shape, &max_ndim);
  PolyUOp *m_exp = poly_expand(ctx, m, (int64_t *)shape, ndim);
  PolyUOp *shifted = poly_alu2(ctx, POLY_OP_SUB, x_view, m_exp);
  PolyUOp *e = poly_exp(ctx, shifted);

  int64_t sum_shape[8]; int sum_ndim;
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, e, shape, ndim, axis, 1,
                          sum_shape, &sum_ndim);
  PolyUOp *log_s = poly_log(ctx, s);
  PolyUOp *log_s_exp = poly_expand(ctx, log_s, (int64_t *)shape, ndim);

  return poly_alu2(ctx, POLY_OP_SUB, shifted, log_s_exp);
}

PolyUOp *poly_cross_entropy(PolyCtx *ctx,
                            PolyUOp *logits, PolyUOp *target, int axis) {
  if (!ctx || !logits || !target) return NULL;
  int64_t logits_shape[POLY_MAX_DIMS], target_shape[POLY_MAX_DIMS];
  int logits_ndim = uop_shape(ctx, logits, logits_shape);
  int target_ndim = uop_shape(ctx, target, target_shape);
  if (logits_ndim < 1 || logits_ndim > POLY_MAX_DIMS || target_ndim < 0 || target_ndim > POLY_MAX_DIMS) return NULL;

  if (axis < 0) axis += logits_ndim;
  if (axis < 0 || axis >= logits_ndim) return NULL;

  const bool dense_targets = poly_shape_equal(logits_shape, logits_ndim, target_shape, target_ndim);
  const bool sparse_targets = shape_equal_except_axis(logits_shape, logits_ndim, target_shape, target_ndim, axis);
  if (!dense_targets && !sparse_targets) return NULL;

  PolyUOp *weights = dense_targets
    ? poly_reshape(ctx, target, (int64_t *)target_shape, target_ndim)
    : target;
  if (sparse_targets) {
    const int64_t classes = logits_shape[axis];
    int64_t target_us_shape[POLY_MAX_DIMS];
    for (int i = 0; i < axis; i++) target_us_shape[i] = target_shape[i];
    target_us_shape[axis] = 1;
    for (int i = axis; i < target_ndim; i++) target_us_shape[i + 1] = target_shape[i];

    PolyUOp *target_us = poly_reshape(ctx, target, target_us_shape, target_ndim + 1);
    PolyUOp *target_exp = poly_expand(ctx, target_us, (int64_t *)logits_shape, logits_ndim);

    PolyUOp *classes_uop = poly_arange(ctx, 0.0, (double)classes, 1.0);
    int64_t classes_shape[POLY_MAX_DIMS];
    for (int i = 0; i < logits_ndim; i++) classes_shape[i] = 1;
    classes_shape[axis] = classes;
    PolyUOp *classes_r = poly_reshape(ctx, classes_uop, classes_shape, logits_ndim);
    PolyUOp *classes_exp = poly_expand(ctx, classes_r, (int64_t *)logits_shape, logits_ndim);

    weights = poly_eq(ctx, target_exp, classes_exp);
  }

  PolyUOp *log_probs = poly_log_softmax(ctx, logits, axis);
  PolyUOp *weighted = poly_alu2(ctx, POLY_OP_MUL, log_probs, weights);

  int64_t per_sample_shape[POLY_MAX_DIMS];
  int per_sample_ndim = 0;
  PolyUOp *per_sample = do_reduce(ctx, POLY_OP_ADD, weighted, logits_shape, logits_ndim, axis, 0,
                                  per_sample_shape, &per_sample_ndim);

  PolyUOp *total = per_sample;
  if (per_sample_ndim > 0) {
    int64_t axes[POLY_MAX_DIMS];
    for (int i = 0; i < per_sample_ndim; i++) axes[i] = i;
    total = poly_reduce_axis(ctx, POLY_OP_ADD, per_sample, axes, per_sample_ndim);
  }

  int64_t denom = poly_shape_numel_checked(per_sample_shape, per_sample_ndim);
  if (denom <= 0) return NULL;

  PolyUOp *mean = poly_alu2(ctx, POLY_OP_FDIV, total, cf(ctx, log_probs, (double)denom));
  PolyUOp *loss = poly_alu1(ctx, POLY_OP_NEG, mean);
  return poly_reshape(ctx, loss, NULL, 0);
}

/* ── Einsum ──────────────────────────────────────────────────────────── */

#define MAX_EINSUM_TENSORS 8

PolyUOp *poly_einsum(PolyCtx *ctx, const char *formula,
                     PolyUOp **tensors, int n_tensors) {
  if (!formula || n_tensors <= 0 || n_tensors > MAX_EINSUM_TENSORS) return NULL;

  /* Read shapes from UOps */
  int64_t shape_store[MAX_EINSUM_TENSORS][POLY_MAX_DIMS];
  const int64_t *shapes[MAX_EINSUM_TENSORS];
  int ndims[MAX_EINSUM_TENSORS];
  for (int t = 0; t < n_tensors; t++) {
    ndims[t] = uop_shape(ctx, tensors[t], shape_store[t]);
    shapes[t] = shape_store[t];
    if (ndims[t] < 0) return NULL;
  }

  char clean[256];
  int ci = 0;
  for (const char *p = formula; *p && ci < 255; p++)
    if (*p != ' ') clean[ci++] = *p;
  clean[ci] = '\0';

  char lhs_buf[256], rhs_buf[64];
  char *arrow = strstr(clean, "->");
  if (arrow) {
    int lhs_len = (int)(arrow - clean);
    memcpy(lhs_buf, clean, lhs_len);
    lhs_buf[lhs_len] = '\0';
    strcpy(rhs_buf, arrow + 2);
  } else {
    strcpy(lhs_buf, clean);
    int count[26] = {0};
    for (char *p2 = lhs_buf; *p2; p2++)
      if (*p2 >= 'a' && *p2 <= 'z') count[*p2 - 'a']++;
    int ri = 0;
    for (int i = 0; i < 26; i++)
      if (count[i] == 1) rhs_buf[ri++] = (char)('a' + i);
    rhs_buf[ri] = '\0';
  }

  char *input_specs[MAX_EINSUM_TENSORS];
  int n_inputs = 0;
  char *pp = lhs_buf;
  while (*pp && n_inputs < MAX_EINSUM_TENSORS) {
    input_specs[n_inputs++] = pp;
    while (*pp && *pp != ',') pp++;
    if (*pp == ',') *pp++ = '\0';
  }
  if (n_inputs != n_tensors) return NULL;

  int64_t sz[26];
  bool has_letter[26];
  memset(has_letter, 0, sizeof(has_letter));
  for (int t = 0; t < n_tensors; t++) {
    const char *spec = input_specs[t];
    int spec_len = (int)strlen(spec);
    if (spec_len != ndims[t]) return NULL;
    for (int d = 0; d < spec_len; d++) {
      int li = spec[d] - 'a';
      if (li < 0 || li >= 26) return NULL;
      if (has_letter[li]) {
        if (sz[li] != shapes[t][d]) return NULL;
      } else {
        sz[li] = shapes[t][d];
        has_letter[li] = true;
      }
    }
  }

  /* Trace: extract diagonal when a letter repeats in a single input. */
  char trace_specs[MAX_EINSUM_TENSORS][64];
  PolyUOp *trace_tensors[MAX_EINSUM_TENSORS];
  int trace_ndims[MAX_EINSUM_TENSORS];
  int64_t trace_shapes[MAX_EINSUM_TENSORS][POLY_MAX_DIMS];

  for (int t = 0; t < n_tensors; t++) {
    strcpy(trace_specs[t], input_specs[t]);
    trace_tensors[t] = tensors[t];
    trace_ndims[t] = ndims[t];
    for (int d = 0; d < ndims[t]; d++)
      trace_shapes[t][d] = shapes[t][d];
  }

  for (int t = 0; t < n_tensors; t++) {
    char *s = trace_specs[t];
    int slen = (int)strlen(s);
    PolyUOp *x = trace_tensors[t];
    int x_ndim = trace_ndims[t];
    int64_t *x_shape = trace_shapes[t];

    for (int ci2 = 0; ci2 < slen; ci2++) {
      char c = s[ci2];
      int ki = -1;
      for (int k = ci2 + 1; k < slen; k++)
        if (s[k] == c) { ki = k; break; }
      if (ki < 0) continue;

      int64_t n = x_shape[ci2];

      int64_t perm[POLY_MAX_DIMS];
      int pi = 0;
      for (int d = 0; d < x_ndim; d++)
        if (d != ci2 && d != ki) perm[pi++] = d;
      perm[pi++] = ci2;
      perm[pi++] = ki;
      x = poly_permute(ctx, x, perm, x_ndim);

      int64_t pshape[POLY_MAX_DIMS];
      for (int d = 0; d < x_ndim; d++)
        pshape[d] = x_shape[perm[d]];
      memcpy(x_shape, pshape, x_ndim * sizeof(int64_t));

      int64_t flat_shape[POLY_MAX_DIMS];
      int flat_ndim = x_ndim - 1;
      for (int d = 0; d < flat_ndim - 1; d++)
        flat_shape[d] = x_shape[d];
      flat_shape[flat_ndim - 1] = n * n;
      x = poly_reshape(ctx, x, flat_shape, flat_ndim);

      int64_t pad_pairs[POLY_MAX_DIMS][2];
      for (int d = 0; d < flat_ndim; d++) {
        pad_pairs[d][0] = 0;
        pad_pairs[d][1] = 0;
      }
      pad_pairs[flat_ndim - 1][1] = n;
      x = poly_pad(ctx, x, pad_pairs, flat_ndim);

      int64_t uf_shape[POLY_MAX_DIMS];
      int uf_ndim = flat_ndim + 1;
      for (int d = 0; d < flat_ndim - 1; d++)
        uf_shape[d] = flat_shape[d];
      uf_shape[flat_ndim - 1] = n;
      uf_shape[flat_ndim] = n + 1;
      x = poly_reshape(ctx, x, uf_shape, uf_ndim);

      int64_t shrink_pairs[POLY_MAX_DIMS][2];
      for (int d = 0; d < uf_ndim; d++) {
        shrink_pairs[d][0] = 0;
        shrink_pairs[d][1] = uf_shape[d];
      }
      shrink_pairs[uf_ndim - 1][0] = 0;
      shrink_pairs[uf_ndim - 1][1] = 1;
      x = poly_shrink(ctx, x, shrink_pairs, uf_ndim);

      int64_t final_shape[POLY_MAX_DIMS];
      int final_ndim = uf_ndim - 1;
      for (int d = 0; d < final_ndim; d++)
        final_shape[d] = uf_shape[d];
      x = poly_reshape(ctx, x, final_shape, final_ndim);

      for (int k = ki; k < slen - 1; k++)
        s[k] = s[k + 1];
      s[slen - 1] = '\0';
      slen--;

      x_ndim = final_ndim;
      memcpy(x_shape, final_shape, final_ndim * sizeof(int64_t));

      ci2--;
    }

    trace_tensors[t] = x;
    trace_ndims[t] = x_ndim;
    input_specs[t] = trace_specs[t];
  }

  /* Rebuild size dict after trace reduction */
  memset(has_letter, 0, sizeof(has_letter));
  for (int t = 0; t < n_tensors; t++) {
    const char *spec = input_specs[t];
    int spec_len = (int)strlen(spec);
    for (int d = 0; d < spec_len; d++) {
      int li = spec[d] - 'a';
      if (!has_letter[li]) {
        sz[li] = trace_shapes[t][d];
        has_letter[li] = true;
      }
    }
  }

  char alpha[26];
  int n_alpha = 0;
  for (int i = 0; i < 26; i++)
    if (has_letter[i]) alpha[n_alpha++] = (char)('a' + i);

  PolyUOp *aligned[MAX_EINSUM_TENSORS];
  for (int t = 0; t < n_tensors; t++) {
    const char *spec = input_specs[t];
    int spec_len = (int)strlen(spec);
    PolyUOp *x = trace_tensors[t];
    if (spec_len == 0) { aligned[t] = x; continue; }

    char sorted_spec[27];
    memcpy(sorted_spec, spec, spec_len);
    sorted_spec[spec_len] = '\0';
    for (int i = 0; i < spec_len - 1; i++)
      for (int j = i + 1; j < spec_len; j++)
        if (sorted_spec[i] > sorted_spec[j]) {
          char tmp = sorted_spec[i];
          sorted_spec[i] = sorted_spec[j];
          sorted_spec[j] = tmp;
        }

    int64_t perm[POLY_MAX_DIMS];
    bool needs_perm = false;
    for (int i = 0; i < spec_len; i++) {
      for (int j = 0; j < spec_len; j++)
        if (spec[j] == sorted_spec[i]) { perm[i] = j; break; }
      if (perm[i] != i) needs_perm = true;
    }
    if (needs_perm)
      x = poly_permute(ctx, x, perm, spec_len);

    int64_t rshape[POLY_MAX_DIMS];
    for (int i = 0; i < n_alpha; i++) {
      bool found = false;
      for (int j = 0; j < spec_len; j++)
        if (sorted_spec[j] == alpha[i]) { found = true; break; }
      rshape[i] = found ? sz[(int)(alpha[i] - 'a')] : 1;
    }
    x = poly_reshape(ctx, x, rshape, n_alpha);

    int64_t full[POLY_MAX_DIMS];
    for (int i = 0; i < n_alpha; i++)
      full[i] = sz[(int)(alpha[i] - 'a')];
    x = poly_expand(ctx, x, full, n_alpha);

    aligned[t] = x;
  }

  PolyUOp *result = aligned[0];
  for (int t = 1; t < n_tensors; t++)
    result = poly_alu2(ctx, POLY_OP_MUL, result, aligned[t]);

  int64_t sum_axes[POLY_MAX_DIMS];
  int n_sum = 0;
  for (int i = 0; i < n_alpha; i++) {
    bool in_rhs = false;
    for (const char *r = rhs_buf; *r; r++)
      if (*r == alpha[i]) { in_rhs = true; break; }
    if (!in_rhs)
      sum_axes[n_sum++] = i;
  }
  if (n_sum > 0)
    result = poly_reduce_axis(ctx, POLY_OP_ADD, result, sum_axes, n_sum);

  char remaining[26];
  int n_remaining = 0;
  for (int i = 0; i < n_alpha; i++) {
    bool summed = false;
    for (int j = 0; j < n_sum; j++)
      if (sum_axes[j] == i) { summed = true; break; }
    if (!summed)
      remaining[n_remaining++] = alpha[i];
  }

  int rhs_len = (int)strlen(rhs_buf);
  if (rhs_len != n_remaining) return NULL;

  int64_t out_perm[POLY_MAX_DIMS];
  bool needs_final_perm = false;
  for (int i = 0; i < rhs_len; i++) {
    for (int j = 0; j < n_remaining; j++)
      if (remaining[j] == rhs_buf[i]) {
        out_perm[i] = j;
        if (j != i) needs_final_perm = true;
        break;
      }
  }
  if (needs_final_perm)
    result = poly_permute(ctx, result, out_perm, rhs_len);

  return result;
}

/* ── Rearrange (einops) ──────────────────────────────────────────────── */

#define MAX_REARRANGE_TOKENS 32

static int parse_rearrange_side(const char *s,
                                 char tokens[][32], int *n_tokens,
                                 int groups[][2], int *n_groups) {
  *n_tokens = 0;
  *n_groups = 0;
  int paren_start = -1;

  const char *p = s;
  while (*p) {
    while (*p == ' ' || *p == '\t') p++;
    if (!*p) break;

    if (*p == '(') {
      paren_start = *n_tokens;
      p++;
      continue;
    }
    if (*p == ')') {
      if (paren_start >= 0) {
        groups[*n_groups][0] = paren_start;
        groups[*n_groups][1] = *n_tokens;
        (*n_groups)++;
      }
      paren_start = -1;
      p++;
      continue;
    }

    int ti = 0;
    while (*p && *p != ' ' && *p != '\t' && *p != '(' && *p != ')' && ti < 31)
      tokens[*n_tokens][ti++] = *p++;
    tokens[*n_tokens][ti] = '\0';
    (*n_tokens)++;
    if (*n_tokens >= MAX_REARRANGE_TOKENS) break;
  }
  return *n_tokens;
}

static int64_t find_axis_size(const char *name,
                               const char *axis_names, const int64_t *axis_values,
                               int n_axis_sizes) {
  if (!axis_names || n_axis_sizes <= 0) return -1;
  const char *p = axis_names;
  int idx = 0;
  while (*p && idx < n_axis_sizes) {
    while (*p == ' ') p++;
    if (!*p) break;
    const char *start = p;
    while (*p && *p != ' ') p++;
    int len = (int)(p - start);
    if ((int)strlen(name) == len && memcmp(start, name, len) == 0)
      return axis_values[idx];
    idx++;
  }
  return -1;
}

PolyUOp *poly_rearrange(PolyCtx *ctx, const char *formula,
                        PolyUOp *x, const char *axis_names,
                        const int64_t *axis_values, int n_axis_sizes) {
  if (!formula || !x) return NULL;
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;

  const char *arrow_pos = strstr(formula, "->");
  if (!arrow_pos) return NULL;

  char lhs_str[256], rhs_str[256];
  int lhs_len = (int)(arrow_pos - formula);
  memcpy(lhs_str, formula, lhs_len);
  lhs_str[lhs_len] = '\0';
  strcpy(rhs_str, arrow_pos + 2);

  char lhs_tok[MAX_REARRANGE_TOKENS][32], rhs_tok[MAX_REARRANGE_TOKENS][32];
  int lhs_grp[8][2], rhs_grp[8][2];
  int n_lt = 0, n_rt = 0, n_lg = 0, n_rg = 0;

  parse_rearrange_side(lhs_str, lhs_tok, &n_lt, lhs_grp, &n_lg);
  parse_rearrange_side(rhs_str, rhs_tok, &n_rt, rhs_grp, &n_rg);

  PolyUOp *result = x;
  int64_t cur_shape[POLY_MAX_DIMS];
  int cur_ndim = ndim;
  memcpy(cur_shape, shape, ndim * sizeof(int64_t));

  /* Phase 1: Unflatten (lhs groups) */
  if (n_lg > 0) {
    bool in_group[MAX_REARRANGE_TOKENS];
    int g_id[MAX_REARRANGE_TOKENS];
    memset(in_group, 0, sizeof(in_group));
    for (int i = 0; i < MAX_REARRANGE_TOKENS; i++) g_id[i] = -1;
    for (int g = 0; g < n_lg; g++)
      for (int i = lhs_grp[g][0]; i < lhs_grp[g][1]; i++) {
        in_group[i] = true;
        g_id[i] = g;
      }

    int64_t new_shape[POLY_MAX_DIMS];
    int new_ndim = 0, input_dim = 0, ti = 0;

    while (ti < n_lt) {
      if (in_group[ti]) {
        int g = g_id[ti];
        int gs = lhs_grp[g][0], ge = lhs_grp[g][1];
        int gc = ge - gs;
        int64_t sub[POLY_MAX_DIMS];
        int64_t known = 1;
        int unk = -1;
        for (int i = 0; i < gc; i++) {
          const char *nm = lhs_tok[gs + i];
          if (strcmp(nm, "1") == 0) { sub[i] = 1; }
          else {
            int64_t v = find_axis_size(nm, axis_names, axis_values, n_axis_sizes);
            if (v > 0) sub[i] = v;
            else { if (unk >= 0) return NULL; unk = i; sub[i] = -1; }
          }
          if (sub[i] > 0) known *= sub[i];
        }
        if (unk >= 0) {
          if (input_dim >= cur_ndim) return NULL;
          sub[unk] = cur_shape[input_dim] / known;
        }
        for (int i = 0; i < gc; i++) new_shape[new_ndim++] = sub[i];
        input_dim++;
        ti = ge;
      } else {
        if (strcmp(lhs_tok[ti], "1") == 0) new_shape[new_ndim++] = 1;
        else {
          if (input_dim >= cur_ndim) return NULL;
          new_shape[new_ndim++] = cur_shape[input_dim];
        }
        input_dim++;
        ti++;
      }
    }

    if (new_ndim != cur_ndim || memcmp(new_shape, cur_shape, cur_ndim * sizeof(int64_t)) != 0) {
      result = poly_reshape(ctx, result, new_shape, new_ndim);
      memcpy(cur_shape, new_shape, new_ndim * sizeof(int64_t));
      cur_ndim = new_ndim;
    }
  }

  /* Phase 2: Permute (lhs order -> rhs order) */
  int64_t perm[POLY_MAX_DIMS];
  bool need_perm = false;
  for (int i = 0; i < n_rt; i++) {
    perm[i] = -1;
    for (int j = 0; j < n_lt; j++)
      if (strcmp(rhs_tok[i], lhs_tok[j]) == 0) { perm[i] = j; break; }
    if (perm[i] < 0) return NULL;
    if (perm[i] != i) need_perm = true;
  }
  if (need_perm) {
    result = poly_permute(ctx, result, perm, n_rt);
    int64_t ps[POLY_MAX_DIMS];
    for (int i = 0; i < n_rt; i++) ps[i] = cur_shape[perm[i]];
    memcpy(cur_shape, ps, n_rt * sizeof(int64_t));
    cur_ndim = n_rt;
  }

  /* Phase 3: Flatten (rhs groups, process right to left) */
  for (int g = n_rg - 1; g >= 0; g--) {
    int gs = rhs_grp[g][0], ge = rhs_grp[g][1];
    if (ge - gs <= 1) continue;
    int64_t flat = 1;
    for (int i = gs; i < ge && i < cur_ndim; i++) flat *= cur_shape[i];
    int64_t ns[POLY_MAX_DIMS];
    int nn = 0;
    for (int i = 0; i < gs; i++) ns[nn++] = cur_shape[i];
    ns[nn++] = flat;
    for (int i = ge; i < cur_ndim; i++) ns[nn++] = cur_shape[i];
    result = poly_reshape(ctx, result, ns, nn);
    memcpy(cur_shape, ns, nn * sizeof(int64_t));
    cur_ndim = nn;
  }

  return result;
}

/* ── Gather (embedding lookup) ───────────────────────────────────────── */

PolyUOp *poly_gather(PolyCtx *ctx, PolyUOp *table, PolyUOp *indices) {
  if (!ctx || !table || !indices) return NULL;
  int64_t table_shape[POLY_MAX_DIMS], idx_shape[POLY_MAX_DIMS];
  int table_ndim = uop_shape(ctx, table, table_shape);
  int idx_ndim = uop_shape(ctx, indices, idx_shape);
  if (table_ndim != 2 || idx_ndim < 1) return NULL;

  int64_t V = table_shape[0];
  int64_t D = table_shape[1];

  PolyUOp *arange_buf = poly_arange(ctx, 0.0, (double)V, 1.0);

  int64_t idx_us_shape[POLY_MAX_DIMS];
  int idx_us_ndim = idx_ndim + 1;
  for (int i = 0; i < idx_ndim; i++) idx_us_shape[i] = idx_shape[i];
  idx_us_shape[idx_ndim] = 1;
  PolyUOp *idx_us = poly_reshape(ctx, indices, idx_us_shape, idx_us_ndim);

  int64_t idx_bcast[POLY_MAX_DIMS];
  for (int i = 0; i < idx_ndim; i++) idx_bcast[i] = idx_shape[i];
  idx_bcast[idx_ndim] = V;
  PolyUOp *idx_exp = poly_expand(ctx, idx_us, idx_bcast, idx_us_ndim);

  int64_t arange_shape[POLY_MAX_DIMS];
  int arange_ndim = idx_us_ndim;
  for (int i = 0; i < idx_ndim; i++) arange_shape[i] = 1;
  arange_shape[idx_ndim] = V;
  PolyUOp *arange_r = poly_reshape(ctx, arange_buf, arange_shape, arange_ndim);
  PolyUOp *arange_exp = poly_expand(ctx, arange_r, idx_bcast, arange_ndim);

  PolyUOp *mask = poly_eq(ctx, idx_exp, arange_exp);

  int64_t mask_us_shape[POLY_MAX_DIMS];
  int mask_us_ndim = idx_us_ndim + 1;
  for (int i = 0; i < idx_us_ndim; i++) mask_us_shape[i] = idx_bcast[i];
  mask_us_shape[idx_us_ndim] = 1;
  PolyUOp *mask_us = poly_reshape(ctx, mask, mask_us_shape, mask_us_ndim);

  int64_t mask_bcast[POLY_MAX_DIMS];
  for (int i = 0; i < idx_us_ndim; i++) mask_bcast[i] = idx_bcast[i];
  mask_bcast[idx_us_ndim] = D;
  PolyUOp *mask_exp = poly_expand(ctx, mask_us, mask_bcast, mask_us_ndim);

  int64_t tbl_shape[POLY_MAX_DIMS];
  int tbl_ndim = mask_us_ndim;
  for (int i = 0; i < idx_ndim; i++) tbl_shape[i] = 1;
  tbl_shape[idx_ndim] = V;
  tbl_shape[idx_ndim + 1] = D;
  PolyUOp *tbl_r = poly_reshape(ctx, table, tbl_shape, tbl_ndim);
  PolyUOp *tbl_exp = poly_expand(ctx, tbl_r, mask_bcast, tbl_ndim);

  PolyUOp *zero = cf(ctx, tbl_exp, 0.0);
  PolyUOp *selected = poly_where_op(ctx, mask_exp, tbl_exp, zero);

  int64_t reduce_axes[] = { idx_ndim };
  PolyUOp *gathered = poly_reduce_axis(ctx, POLY_OP_ADD, selected,
                                        reduce_axes, 1);

  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = idx_ndim + 1;
  for (int i = 0; i < idx_ndim; i++) out_shape[i] = idx_shape[i];
  out_shape[idx_ndim] = D;

  return poly_reshape(ctx, gathered, out_shape, out_ndim);
}

/* ── Additional composed ops ─────────────────────────────────────────── */

PolyUOp *poly_sdpa(PolyCtx *ctx, PolyUOp *q, PolyUOp *k, PolyUOp *v,
                       PolyUOp *mask, int is_causal) {
  int64_t q_shape[POLY_MAX_DIMS], k_shape[POLY_MAX_DIMS], v_shape[POLY_MAX_DIMS];
  int q_ndim, k_ndim, v_ndim;
  q_ndim = uop_shape(ctx, q, q_shape);
  k_ndim = uop_shape(ctx, k, k_shape);
  v_ndim = uop_shape(ctx, v, v_shape);
  if (q_ndim < 2 || k_ndim < 2 || v_ndim < 2) return NULL;

  int64_t d_k = q_shape[q_ndim - 1];
  double scale = 1.0 / sqrt((double)d_k);

  int64_t k_perm[POLY_MAX_DIMS];
  for (int i = 0; i < k_ndim; i++) k_perm[i] = i;
  k_perm[k_ndim - 2] = k_ndim - 1;
  k_perm[k_ndim - 1] = k_ndim - 2;
  PolyUOp *k_t = poly_permute(ctx, k, k_perm, k_ndim);

  PolyUOp *scores = poly_dot(ctx, q, k_t);
  scores = poly_alu2(ctx, POLY_OP_MUL, scores, poly_const_float(ctx, scale));

  if (is_causal) {
    int64_t seq_q = q_shape[q_ndim - 2];
    int64_t seq_k = k_shape[k_ndim - 2];
    PolyUOp *ones = poly_full(ctx, (int64_t[]){seq_q, seq_k}, 2, 1.0);
    PolyUOp *tril_m = poly_tril(ctx, ones, 0);
    PolyUOp *zero = poly_const_float(ctx, 0.0);
    PolyUOp *neg_inf = poly_const_float(ctx, -1e9);
    PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, tril_m, poly_const_float(ctx, 0.5));
    PolyUOp *cmask = poly_alu3(ctx, POLY_OP_WHERE, cond, neg_inf, zero);
    scores = poly_add(ctx, scores, cmask);
  }

  if (mask) {
    scores = poly_add(ctx, scores, mask);
  }

  PolyUOp *attn = poly_softmax(ctx, scores, -1);

  return poly_dot(ctx, attn, v);
}

PolyUOp *poly_rope(PolyCtx *ctx, PolyUOp *x,
                       PolyUOp *freqs_cos, PolyUOp *freqs_sin) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 1) return NULL;
  int64_t half_dim = shape[ndim - 1] / 2;
  if (half_dim <= 0) return NULL;

  int64_t pairs1[POLY_MAX_DIMS][2], pairs2[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim - 1; i++) {
    pairs1[i][0] = 0; pairs1[i][1] = shape[i];
    pairs2[i][0] = 0; pairs2[i][1] = shape[i];
  }
  pairs1[ndim - 1][0] = 0;         pairs1[ndim - 1][1] = half_dim;
  pairs2[ndim - 1][0] = half_dim;  pairs2[ndim - 1][1] = shape[ndim - 1];

  PolyUOp *x1 = poly_shrink(ctx, x, pairs1, ndim);
  PolyUOp *x2 = poly_shrink(ctx, x, pairs2, ndim);

  PolyUOp *r1 = poly_alu2(ctx, POLY_OP_SUB,
    poly_alu2(ctx, POLY_OP_MUL, x1, freqs_cos),
    poly_alu2(ctx, POLY_OP_MUL, x2, freqs_sin));
  PolyUOp *r2 = poly_alu2(ctx, POLY_OP_ADD,
    poly_alu2(ctx, POLY_OP_MUL, x2, freqs_cos),
    poly_alu2(ctx, POLY_OP_MUL, x1, freqs_sin));

  int64_t pad1[POLY_MAX_DIMS][2], pad2[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    pad1[i][0] = 0; pad1[i][1] = 0;
    pad2[i][0] = 0; pad2[i][1] = 0;
  }
  pad1[ndim - 1][1] = half_dim;
  pad2[ndim - 1][0] = half_dim;

  return poly_alu2(ctx, POLY_OP_ADD,
    poly_pad(ctx, r1, pad1, ndim),
    poly_pad(ctx, r2, pad2, ndim));
}

PolyUOp *poly_repeat_interleave(PolyCtx *ctx, PolyUOp *x, int repeats, int dim) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 1 || repeats <= 0) return NULL;
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) return NULL;

  int64_t ins[POLY_MAX_DIMS];
  int ins_ndim = ndim + 1;
  for (int i = 0; i <= dim; i++) ins[i] = shape[i];
  ins[dim + 1] = 1;
  for (int i = dim + 1; i < ndim; i++) ins[i + 1] = shape[i];
  PolyUOp *r = poly_reshape(ctx, x, ins, ins_ndim);

  int64_t exp[POLY_MAX_DIMS];
  memcpy(exp, ins, ins_ndim * sizeof(int64_t));
  exp[dim + 1] = repeats;
  r = poly_expand(ctx, r, exp, ins_ndim);

  int64_t flat[POLY_MAX_DIMS];
  for (int i = 0; i < dim; i++) flat[i] = shape[i];
  flat[dim] = shape[dim] * repeats;
  for (int i = dim + 1; i < ndim; i++) flat[i] = shape[i];
  return poly_reshape(ctx, r, flat, ndim);
}

PolyUOp *poly_argmax(PolyCtx *ctx, PolyUOp *x, int axis) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 1) return NULL;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) return NULL;

  int64_t N = shape[axis];

  int64_t max_shape[POLY_MAX_DIMS]; int max_ndim;
  PolyUOp *x_max = do_reduce(ctx, POLY_OP_MAX, x, shape, ndim, axis, 1, max_shape, &max_ndim);
  PolyUOp *x_max_bc = poly_expand(ctx, x_max, shape, ndim);
  PolyUOp *m = poly_eq(ctx, x, x_max_bc);

  PolyUOp *m_f = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, m, poly_arg_none());

  PolyUOp *rng = poly_arange(ctx, 0.0, (double)N, 1.0);
  PolyUOp *desc = poly_alu2(ctx, POLY_OP_SUB, poly_const_float(ctx, (double)N), rng);

  int64_t bc[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) bc[i] = 1;
  bc[axis] = N;
  desc = poly_reshape(ctx, desc, bc, ndim);
  desc = poly_expand(ctx, desc, shape, ndim);

  PolyUOp *idx = poly_alu2(ctx, POLY_OP_MUL, m_f, desc);

  int64_t idx_max_shape[POLY_MAX_DIMS]; int idx_max_ndim;
  PolyUOp *idx_max = do_reduce(ctx, POLY_OP_MAX, idx, shape, ndim, axis, 0,
                                idx_max_shape, &idx_max_ndim);
  PolyUOp *result = poly_alu2(ctx, POLY_OP_SUB, poly_const_float(ctx, (double)N), idx_max);
  return poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, result, poly_arg_none());
}

PolyUOp *poly_mse_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  ndim = uop_shape(ctx, pred, shape);
  if (ndim < 0) return NULL;
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_SUB, pred, target);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
  int64_t out_shape[POLY_MAX_DIMS]; int out_ndim;
  PolyUOp *r = sq;
  for (int i = ndim - 1; i >= 0; i--) {
    int64_t s[POLY_MAX_DIMS]; int sn;
    sn = uop_shape(ctx, r, s);
    r = do_reduce(ctx, POLY_OP_ADD, r, s, sn, i, 0, out_shape, &out_ndim);
  }
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) numel *= shape[i];
  return poly_alu2(ctx, POLY_OP_FDIV, r, poly_const_float(ctx, (double)numel));
}

PolyUOp *poly_mae_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target) {
  int64_t shape[POLY_MAX_DIMS]; int ndim;
  ndim = uop_shape(ctx, pred, shape);
  if (ndim < 0) return NULL;
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_SUB, pred, target);
  PolyUOp *absdiff = poly_abs(ctx, diff);
  int64_t out_shape[POLY_MAX_DIMS]; int out_ndim;
  PolyUOp *r = absdiff;
  for (int i = ndim - 1; i >= 0; i--) {
    int64_t s[POLY_MAX_DIMS]; int sn;
    sn = uop_shape(ctx, r, s);
    r = do_reduce(ctx, POLY_OP_ADD, r, s, sn, i, 0, out_shape, &out_ndim);
  }
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) numel *= shape[i];
  return poly_alu2(ctx, POLY_OP_FDIV, r, poly_const_float(ctx, (double)numel));
}

const void *poly_const_buffer_data(PolyCtx *ctx, PolyUOp *buf) {
  return poly_const_registry_lookup(ctx, buf);
}
