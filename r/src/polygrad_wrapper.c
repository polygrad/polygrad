/*
 * polygrad_wrapper.c — .Call() wrappers bridging R SEXP <-> polygrad C API.
 *
 * All opaque C pointers (PolyCtx*, PolyUOp*) are wrapped in R external pointers.
 * Data exchange uses R numeric vectors (REALSXP -> float32).
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <string.h>

/* Include polygrad headers */
#include "../../../src/frontend.h"
#include "../../../src/sched.h"

/* ── Helpers ─────────────────────────────────────────────────────────── */

static void ctx_finalizer(SEXP ptr) {
  PolyCtx *ctx = (PolyCtx *)R_ExternalPtrAddr(ptr);
  if (ctx) {
    poly_ctx_destroy(ctx);
    R_ClearExternalPtr(ptr);
  }
}

static SEXP wrap_ptr(void *p) {
  if (!p) return R_NilValue;
  return R_MakeExternalPtr(p, R_NilValue, R_NilValue);
}

static void *unwrap_ptr(SEXP x) {
  if (x == R_NilValue) return NULL;
  return R_ExternalPtrAddr(x);
}

/* ── Context ─────────────────────────────────────────────────────────── */

SEXP C_poly_ctx_new(void) {
  PolyCtx *ctx = poly_ctx_new();
  SEXP ptr = PROTECT(R_MakeExternalPtr(ctx, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ptr, ctx_finalizer, TRUE);
  UNPROTECT(1);
  return ptr;
}

SEXP C_poly_ctx_destroy(SEXP ctx_ptr) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  if (ctx) {
    poly_ctx_destroy(ctx);
    R_ClearExternalPtr(ctx_ptr);
  }
  return R_NilValue;
}

/* ── Op helpers ──────────────────────────────────────────────────────── */

SEXP C_poly_op_count(void) {
  return ScalarInteger(poly_op_count());
}

SEXP C_poly_op_name(SEXP op) {
  const char *name = poly_op_name(asInteger(op));
  if (!name) return R_NilValue;
  return mkString(name);
}

/* ── Constants ───────────────────────────────────────────────────────── */

SEXP C_poly_const_float(SEXP ctx_ptr, SEXP value) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  return wrap_ptr(poly_const_float(ctx, asReal(value)));
}

/* ── Buffer ──────────────────────────────────────────────────────────── */

SEXP C_poly_buffer_f32(SEXP ctx_ptr, SEXP size) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  return wrap_ptr(poly_buffer_f32(ctx, (int64_t)asReal(size)));
}

/* ── ALU ops ─────────────────────────────────────────────────────────── */

SEXP C_poly_alu1(SEXP ctx_ptr, SEXP op, SEXP src) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  return wrap_ptr(poly_alu1(ctx, asInteger(op), unwrap_ptr(src)));
}

SEXP C_poly_alu2(SEXP ctx_ptr, SEXP op, SEXP a, SEXP b) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  return wrap_ptr(poly_alu2(ctx, asInteger(op), unwrap_ptr(a), unwrap_ptr(b)));
}

/* ── Movement ops ────────────────────────────────────────────────────── */

SEXP C_poly_reshape(SEXP ctx_ptr, SEXP src, SEXP dims) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  int n = length(dims);
  int64_t d[8];
  for (int i = 0; i < n && i < 8; i++) d[i] = (int64_t)REAL(dims)[i];
  return wrap_ptr(poly_reshape(ctx, unwrap_ptr(src), d, n));
}

SEXP C_poly_permute(SEXP ctx_ptr, SEXP src, SEXP perm) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  int n = length(perm);
  int64_t p[8];
  for (int i = 0; i < n && i < 8; i++) p[i] = (int64_t)REAL(perm)[i];
  return wrap_ptr(poly_permute(ctx, unwrap_ptr(src), p, n));
}

SEXP C_poly_flip(SEXP ctx_ptr, SEXP src, SEXP axes) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  int n = length(axes);
  int64_t a[8];
  for (int i = 0; i < n && i < 8; i++) a[i] = (int64_t)REAL(axes)[i];
  return wrap_ptr(poly_flip(ctx, unwrap_ptr(src), a, n));
}

SEXP C_poly_pad(SEXP ctx_ptr, SEXP src, SEXP pairs_flat, SEXP ndim) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  int n = asInteger(ndim);
  int64_t pairs[8][2];
  for (int i = 0; i < n && i < 8; i++) {
    pairs[i][0] = (int64_t)REAL(pairs_flat)[i * 2];
    pairs[i][1] = (int64_t)REAL(pairs_flat)[i * 2 + 1];
  }
  return wrap_ptr(poly_pad(ctx, unwrap_ptr(src), pairs, n));
}

SEXP C_poly_shrink(SEXP ctx_ptr, SEXP src, SEXP pairs_flat, SEXP ndim) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  int n = asInteger(ndim);
  int64_t pairs[8][2];
  for (int i = 0; i < n && i < 8; i++) {
    pairs[i][0] = (int64_t)REAL(pairs_flat)[i * 2];
    pairs[i][1] = (int64_t)REAL(pairs_flat)[i * 2 + 1];
  }
  return wrap_ptr(poly_shrink(ctx, unwrap_ptr(src), pairs, n));
}

SEXP C_poly_expand(SEXP ctx_ptr, SEXP src, SEXP dims) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  int n = length(dims);
  int64_t d[8];
  for (int i = 0; i < n && i < 8; i++) d[i] = (int64_t)REAL(dims)[i];
  return wrap_ptr(poly_expand(ctx, unwrap_ptr(src), d, n));
}

/* ── Reduce ──────────────────────────────────────────────────────────── */

SEXP C_poly_reduce_axis(SEXP ctx_ptr, SEXP op, SEXP src, SEXP axes) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  int n = length(axes);
  int64_t a[8];
  for (int i = 0; i < n && i < 8; i++) a[i] = (int64_t)REAL(axes)[i];
  return wrap_ptr(poly_reduce_axis(ctx, asInteger(op), unwrap_ptr(src), a, n));
}

/* ── Graph construction ──────────────────────────────────────────────── */

SEXP C_poly_store_val(SEXP ctx_ptr, SEXP buf, SEXP value) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  return wrap_ptr(poly_store_val(ctx, unwrap_ptr(buf), unwrap_ptr(value)));
}

SEXP C_poly_sink1(SEXP ctx_ptr, SEXP store) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  return wrap_ptr(poly_sink1(ctx, unwrap_ptr(store)));
}

/* ── Autograd ────────────────────────────────────────────────────────── */

SEXP C_poly_grad(SEXP ctx_ptr, SEXP loss, SEXP wrt_buffer) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  return wrap_ptr(poly_grad(ctx, unwrap_ptr(loss), unwrap_ptr(wrt_buffer)));
}

/* ── Realize ─────────────────────────────────────────────────────────── */

/*
 * C_poly_realize(ctx, sink, buffer_ptrs, data_list)
 *   buffer_ptrs: list of external pointers (BUFFER UOps)
 *   data_list:   list of numeric vectors (float32 data)
 */
SEXP C_poly_realize(SEXP ctx_ptr, SEXP sink, SEXP buffer_ptrs, SEXP data_list) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  int n = length(buffer_ptrs);

  poly_realize_begin(ctx);
  for (int i = 0; i < n; i++) {
    PolyUOp *buf = unwrap_ptr(VECTOR_ELT(buffer_ptrs, i));
    SEXP data_vec = VECTOR_ELT(data_list, i);
    int len = length(data_vec);

    /* Convert R double -> float32 */
    float *fdata = (float *)R_alloc(len, sizeof(float));
    double *rdata = REAL(data_vec);
    for (int j = 0; j < len; j++) fdata[j] = (float)rdata[j];

    poly_realize_bind(ctx, buf, fdata);
  }

  int ret = poly_realize_exec(ctx, unwrap_ptr(sink));
  return ScalarInteger(ret);
}

/*
 * C_poly_realize_read(ctx, sink, buffer_ptrs, data_list, out_buf, out_len)
 *   Realize and read output into pre-allocated R vector.
 *   The last binding (out_buf/out_data) is the output; all others are inputs.
 *   Returns the output as a numeric vector.
 */
SEXP C_poly_realize_read(SEXP ctx_ptr, SEXP sink,
                         SEXP in_bufs, SEXP in_data_list,
                         SEXP out_buf_ptr, SEXP out_len_sexp) {
  PolyCtx *ctx = unwrap_ptr(ctx_ptr);
  int n_in = length(in_bufs);
  int out_len = asInteger(out_len_sexp);

  /* Allocate float32 arrays for inputs */
  poly_realize_begin(ctx);
  for (int i = 0; i < n_in; i++) {
    PolyUOp *buf = unwrap_ptr(VECTOR_ELT(in_bufs, i));
    SEXP data_vec = VECTOR_ELT(in_data_list, i);
    int len = length(data_vec);
    float *fdata = (float *)R_alloc(len, sizeof(float));
    double *rdata = REAL(data_vec);
    for (int j = 0; j < len; j++) fdata[j] = (float)rdata[j];
    poly_realize_bind(ctx, buf, fdata);
  }

  /* Output buffer */
  float *out_data = (float *)R_alloc(out_len, sizeof(float));
  memset(out_data, 0, out_len * sizeof(float));
  poly_realize_bind(ctx, unwrap_ptr(out_buf_ptr), out_data);

  int ret = poly_realize_exec(ctx, unwrap_ptr(sink));
  if (ret != 0) {
    error("poly_realize failed");
  }

  /* Convert float32 -> R double */
  SEXP result = PROTECT(allocVector(REALSXP, out_len));
  double *rout = REAL(result);
  for (int i = 0; i < out_len; i++) rout[i] = (double)out_data[i];
  UNPROTECT(1);
  return result;
}

/* ── Registration ────────────────────────────────────────────────────── */

static const R_CallMethodDef CallEntries[] = {
  {"C_poly_ctx_new",       (DL_FUNC) &C_poly_ctx_new,       0},
  {"C_poly_ctx_destroy",   (DL_FUNC) &C_poly_ctx_destroy,   1},
  {"C_poly_op_count",      (DL_FUNC) &C_poly_op_count,      0},
  {"C_poly_op_name",       (DL_FUNC) &C_poly_op_name,       1},
  {"C_poly_const_float",   (DL_FUNC) &C_poly_const_float,   2},
  {"C_poly_buffer_f32",    (DL_FUNC) &C_poly_buffer_f32,    2},
  {"C_poly_alu1",          (DL_FUNC) &C_poly_alu1,          3},
  {"C_poly_alu2",          (DL_FUNC) &C_poly_alu2,          4},
  {"C_poly_reshape",       (DL_FUNC) &C_poly_reshape,       3},
  {"C_poly_permute",       (DL_FUNC) &C_poly_permute,       3},
  {"C_poly_flip",          (DL_FUNC) &C_poly_flip,          3},
  {"C_poly_pad",           (DL_FUNC) &C_poly_pad,           4},
  {"C_poly_shrink",        (DL_FUNC) &C_poly_shrink,        4},
  {"C_poly_expand",        (DL_FUNC) &C_poly_expand,        3},
  {"C_poly_reduce_axis",   (DL_FUNC) &C_poly_reduce_axis,   4},
  {"C_poly_store_val",     (DL_FUNC) &C_poly_store_val,     3},
  {"C_poly_sink1",         (DL_FUNC) &C_poly_sink1,         2},
  {"C_poly_grad",          (DL_FUNC) &C_poly_grad,          3},
  {"C_poly_realize",       (DL_FUNC) &C_poly_realize,       4},
  {"C_poly_realize_read",  (DL_FUNC) &C_poly_realize_read,  6},
  {NULL, NULL, 0}
};

void R_init_polygrad(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
