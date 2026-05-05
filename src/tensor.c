/*
 * tensor.c -- Composed tensor ops (elementwise, reduction, creation, etc.)
 *
 * These are higher-level ops built from the core UOp primitives.
 */

#define _GNU_SOURCE
#include "tensor.h"
#include "ctx.h"
#include "device.h"
#include "engine/schedule.h"
#include "utils.h"
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

/* Dtype table for FFI (used by poly_cast_by_id) */

static const PolyDType *_dtype_table_ffi[] = {
    &POLY_VOID,    &POLY_BOOL,     &POLY_INT8,    &POLY_UINT8,   &POLY_INT16,
    &POLY_UINT16,  &POLY_INT32,    &POLY_UINT32,  &POLY_INT64,   &POLY_UINT64,
    &POLY_FLOAT16, &POLY_BFLOAT16, &POLY_FLOAT32, &POLY_FLOAT64,
};
#define N_DTYPE_FFI ((int)(sizeof(_dtype_table_ffi) / sizeof(_dtype_table_ffi[0])))

static int poly_dyn_buffer_id = 1000000; /* avoid colliding with regular small buffer tags */

PolyUOp *poly_buffer_var(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *batch_var,
    const int64_t *inner_dims,
    int n_inner
) {
  assert(batch_var->op == POLY_OP_DEFINE_VAR || batch_var->op == POLY_OP_BIND);
  assert(n_inner >= 0 && n_inner < POLY_MAX_DIMS);
  PolyUOp *bound = batch_var->op == POLY_OP_BIND && batch_var->n_src >= 1 ? batch_var->src[0] : batch_var;
  assert(bound->op == POLY_OP_DEFINE_VAR);
  int64_t alloc = bound->arg.define_var.max_val;
  /* src[0] = UNIQUE (prevent CSE), src[1] = dynamic bound,
   * src[2..] = fixed inner dimension CONSTs. */
  int n_src = 2 + n_inner;
  PolyUOp *src[POLY_MAX_DIMS + 2];
  src[0] = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(poly_dyn_buffer_id++));
  src[1] = batch_var;
  for (int i = 0; i < n_inner; i++) {
    alloc *= inner_dims[i];
    src[2 + i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(inner_dims[i]));
  }
  return poly_uop(ctx, POLY_OP_BUFFER, dt, src, n_src, poly_arg_int(alloc));
}

PolyUOp *poly_legacy_assign_buffer(PolyCtx *ctx, PolyUOp *target, PolyUOp *value) {
  /* Legacy full-buffer assignment only. Movement views are normalized to their
   * base BUFFER. Do not use this from frontends; poly_tensor_assign is the
   * tinygrad-style AFTER/STORE path. */
  PolyUOp *base = target;
  while (poly_opset_has(POLY_GROUP_MOVEMENT, base->op) && base->n_src > 0)
    base = base->src[0];

  if (base != target && base->op == POLY_OP_BUFFER) {
    int64_t numel = (base->arg.kind == POLY_ARG_INT) ? base->arg.i : 0;
    if (numel > 0) {
      int64_t flat_shape[1] = {numel};
      value = poly_reshape(ctx, value, flat_shape, 1);
    }
    target = base;
  }

  PolyUOp *srcs[2] = {target, value};
  return poly_uop(ctx, POLY_OP_ASSIGN, target->dtype, srcs, 2, poly_arg_none());
}

/* Core PolyTensor handles */

typedef struct {
  PolyTensor **items;
  int n;
  int cap;
} PolyTensorList;

static PolyUOp *tensor_current_uop(PolyTensor *tensor) {
  return tensor ? (tensor->uop_physical ? tensor->uop_physical : tensor->uop_logical) : NULL;
}

static PolyTensorList *tensor_list_for_uop(PolyCtx *ctx, PolyUOp *uop, bool create) {
  if (!ctx || !uop) return NULL;
  PolyTensorList *list =
      poly_map_get(ctx->tensors_by_uop, poly_ptr_hash(uop), uop, poly_ptr_eq);
  if (list || !create) return list;
  list = calloc(1, sizeof(PolyTensorList));
  if (!list) return NULL;
  poly_map_set(ctx->tensors_by_uop, poly_ptr_hash(uop), uop, list, poly_ptr_eq);
  return list;
}

static bool tensor_list_append(PolyTensorList *list, PolyTensor *tensor) {
  if (!list || !tensor) return false;
  for (int i = 0; i < list->n; i++)
    if (list->items[i] == tensor) return true;
  if (list->n >= list->cap) {
    int new_cap = list->cap ? list->cap * 2 : 4;
    PolyTensor **new_items = realloc(list->items, (size_t)new_cap * sizeof(PolyTensor *));
    if (!new_items) return false;
    list->items = new_items;
    list->cap = new_cap;
  }
  list->items[list->n++] = tensor;
  return true;
}

static bool tensor_index_add_current(PolyCtx *ctx, PolyTensor *tensor) {
  PolyUOp *current = tensor_current_uop(tensor);
  PolyTensorList *list = tensor_list_for_uop(ctx, current, true);
  return tensor_list_append(list, tensor);
}

static void tensor_index_remove(PolyCtx *ctx, PolyUOp *current, PolyTensor *tensor) {
  if (!ctx || !current || !tensor) return;
  PolyTensorList *list = tensor_list_for_uop(ctx, current, false);
  if (!list) return;
  for (int i = 0; i < list->n; i++) {
    if (list->items[i] != tensor) continue;
    list->items[i] = list->items[list->n - 1];
    list->n--;
    if (list->n == 0) {
      poly_map_remove(ctx->tensors_by_uop, poly_ptr_hash(current), current, poly_ptr_eq);
      free(list->items);
      free(list);
    }
    return;
  }
}

PolyTensor *poly_tensor_find_current(
    PolyCtx *ctx, PolyUOp *current, PolyDevice device, PolyTensorRole role
) {
  PolyTensorList *list = tensor_list_for_uop(ctx, current, false);
  if (!list) return NULL;
  PolyTensor *best = NULL;
  for (int i = 0; i < list->n; i++) {
    PolyTensor *t = list->items[i];
    if (!t || tensor_current_uop(t) != current) continue;
    if (role != (PolyTensorRole)-1 && t->role != role) continue;
    if (device != POLY_DEVICE_AUTO && !poly_devices_share_storage(t->device, device)) continue;
    if (!best || t->order > best->order) best = t;
  }
  return best;
}

static void free_tensor_list_entry(const void *key, void *value, void *userdata) {
  (void)key;
  (void)userdata;
  PolyTensorList *list = value;
  if (list) {
    free(list->items);
    free(list);
  }
}

void poly_tensor_ctx_cleanup(PolyCtx *ctx) {
  if (!ctx) return;
  if (ctx->tensors_by_uop) poly_map_foreach(ctx->tensors_by_uop, free_tensor_list_entry, NULL);
  if (ctx->tensors) {
    for (int i = 0; i < ctx->n_tensors; i++)
      free(ctx->tensors[i]);
  }
}

PolyTensor *poly_tensor_create_with_roots(
    PolyCtx *ctx,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
) {
  if (!ctx || !uop_logical) return NULL;

  if (ctx->n_tensors >= ctx->tensors_cap) {
    int new_cap = ctx->tensors_cap ? ctx->tensors_cap * 2 : 64;
    PolyTensor **new_tensors = realloc(ctx->tensors, (size_t)new_cap * sizeof(PolyTensor *));
    if (!new_tensors) return NULL;
    ctx->tensors = new_tensors;
    ctx->tensors_cap = new_cap;
  }

  PolyTensor *tensor = calloc(1, sizeof(PolyTensor));
  if (!tensor) return NULL;
  tensor->uop_logical = uop_logical;
  tensor->uop_physical = uop_physical;
  tensor->role = role;
  tensor->device = device;
  tensor->order = ctx->next_tensor_order++;

  if (!tensor_index_add_current(ctx, tensor)) {
    free(tensor);
    return NULL;
  }
  ctx->tensors[ctx->n_tensors++] = tensor;
  return tensor;
}

PolyTensor *poly_tensor_create(
    PolyCtx *ctx, PolyUOp *uop, PolyTensorRole role, PolyDevice device
) {
  return poly_tensor_create_with_roots(ctx, uop, NULL, role, device);
}

int poly_tensor_update(
    PolyCtx *ctx,
    PolyTensor *tensor,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
) {
  if (!ctx || !tensor) return -1;
  PolyUOp *old_current = tensor_current_uop(tensor);
  PolyUOp *new_logical = uop_logical ? uop_logical : tensor->uop_logical;
  PolyUOp *new_physical = uop_logical ? NULL : tensor->uop_physical;
  if (uop_physical) new_physical = uop_physical;
  PolyUOp *new_current = new_physical ? new_physical : new_logical;
  if (!new_current) return -1;

  bool reindex = old_current != new_current;
  if (reindex) {
    PolyTensor tmp = *tensor;
    tmp.uop_logical = new_logical;
    tmp.uop_physical = new_physical;
    if (!tensor_index_add_current(ctx, &tmp)) return -1;
    tensor_index_remove(ctx, new_current, &tmp);
  }

  if (uop_logical) {
    tensor->uop_logical = uop_logical;
    tensor->uop_physical = NULL;
  }
  if (uop_physical) tensor->uop_physical = uop_physical;
  tensor->role = role;
  if (device != POLY_DEVICE_AUTO) tensor->device = device;
  if (role != POLY_TENSOR_PLACE) tensor->source = NULL;

  if (reindex) {
    tensor_index_remove(ctx, old_current, tensor);
    if (!tensor_index_add_current(ctx, tensor)) return -1;
  }
  return 0;
}

PolyTensor *poly_tensor_to_device(PolyCtx *ctx, PolyTensor *tensor, PolyDevice device) {
  if (!ctx || !tensor || !tensor->uop_logical) return NULL;
  if (poly_devices_share_storage(tensor->device, device)) return tensor;
  /* .to(device) preserves the portable logical root for export, but the
   * frontend current root must follow realized sources. Two tensors can share
   * one preserved logical UOp while pointing at different realized buffers, so
   * PLACE facts must be keyed by the source tensor's current root. */
  PolyUOp *current = poly_tensor_uop(tensor);
  PolyUOp *physical = (current && current != tensor->uop_logical) ? current : NULL;
  PolyTensor *placed =
      poly_tensor_create_with_roots(ctx, tensor->uop_logical, physical, POLY_TENSOR_PLACE, device);
  if (placed) placed->source = tensor;
  return placed;
}

PolyTensor *poly_tensor_assign(PolyCtx *ctx, PolyTensor *target, PolyTensor *value) {
  if (!ctx || !target || !value) return NULL;
  PolyUOp *target_uop = poly_tensor_uop(target);
  PolyUOp *value_uop = poly_tensor_uop(value);
  if (!target_uop || !value_uop) return NULL;

  /* Match tinygrad Tensor.assign: non-DISK assigns require same device and
   * dtype before constructing the AFTER/STORE effect graph. Without this,
   * a CPU target could silently physicalize a CUDA value back to CPU, which
   * changes user-visible placement semantics. */
  if (target->device != POLY_DEVICE_AUTO && value->device != POLY_DEVICE_AUTO &&
      !poly_devices_share_storage(target->device, value->device))
    return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(target_uop->dtype), poly_dtype_scalar(value_uop->dtype)))
    return NULL;

  PolyUOp *store = poly_store_val(ctx, target_uop, value_uop);
  if (!store) return NULL;
  PolyUOp *src[2] = {target_uop, store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, target_uop->dtype, src, 2, poly_arg_none());
  if (!after) return NULL;

  if (poly_tensor_update(ctx, target, after, NULL, target->role, target->device) != 0)
    return NULL;
  return target;
}

PolyUOp *poly_tensor_uop(PolyTensor *tensor) {
  return tensor_current_uop(tensor);
}

PolyUOp *poly_tensor_uop_logical(PolyTensor *tensor) {
  return tensor ? tensor->uop_logical : NULL;
}

PolyUOp *poly_tensor_uop_physical(PolyTensor *tensor) {
  return tensor ? tensor->uop_physical : NULL;
}

PolyDevice poly_tensor_device(PolyTensor *tensor) {
  return tensor ? tensor->device : POLY_DEVICE_AUTO;
}

/* Internal helpers */

/* Helper: float constant matching the dtype of a given UOp.
 * For float inputs: creates a constant with the same float dtype.
 * For non-float inputs (comparisons producing bool): defaults to float32. */
static inline PolyUOp *cf(PolyCtx *ctx, PolyUOp *ref, double v) {
  PolyDType dt = poly_dtype_scalar(ref->dtype);
  if (poly_dtype_is_float(dt)) return poly_const_typed(ctx, dt, v);
  return poly_const_float(ctx, v);
}

/* Helper: const with explicit dtype -- use in special-math ops for dtype correctness */
static inline PolyUOp *cdt(PolyCtx *ctx, PolyDType dt, double v) {
  return poly_const_typed(ctx, dt, v);
}

static const PolyDType *ffi_dtype_from_id(int dtype_id) {
  if (dtype_id < 0 || dtype_id >= N_DTYPE_FFI) return NULL;
  return _dtype_table_ffi[dtype_id];
}

static bool ffi_dtype_is_integer_like(int dtype_id, PolyDType *out_dt) {
  const PolyDType *dt = ffi_dtype_from_id(dtype_id);
  if (!dt) return false;
  PolyDType sdt = poly_dtype_scalar(*dt);
  if (!poly_dtype_is_int(sdt) && !poly_dtype_is_bool(sdt)) return false;
  if (out_dt) *out_dt = sdt;
  return true;
}

static bool ffi_dtype_is_float_like(int dtype_id, PolyDType *out_dt) {
  const PolyDType *dt = ffi_dtype_from_id(dtype_id);
  if (!dt) return false;
  PolyDType sdt = poly_dtype_scalar(*dt);
  if (!poly_dtype_is_float(sdt)) return false;
  if (out_dt) *out_dt = sdt;
  return true;
}

static PolyUOp *poly_const_exact_int(PolyCtx *ctx, PolyDType dt, int64_t value) {
  dt = poly_dtype_scalar(dt);
  if (poly_dtype_is_bool(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_bool(value != 0));
  if (!poly_dtype_is_int(dt)) return NULL;
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(value));
}

static PolyUOp *poly_const_exact_float(PolyCtx *ctx, PolyDType dt, double value) {
  dt = poly_dtype_scalar(dt);
  if (!poly_dtype_is_float(dt)) return NULL;
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_float(value));
}

static PolyUOp *poly_const_unique_exact_int(PolyCtx *ctx, PolyDType dt, int64_t value) {
  dt = poly_dtype_scalar(dt);
  if (poly_dtype_is_bool(dt))
    return poly_uop_tagged(ctx, POLY_OP_CONST, dt, NULL, 0, poly_arg_bool(value != 0), ctx->next_buf_tag++);
  if (!poly_dtype_is_int(dt)) return NULL;
  return poly_uop_tagged(ctx, POLY_OP_CONST, dt, NULL, 0, poly_arg_int(value), ctx->next_buf_tag++);
}

static PolyUOp *poly_const_unique_exact_float(PolyCtx *ctx, PolyDType dt, double value) {
  dt = poly_dtype_scalar(dt);
  if (!poly_dtype_is_float(dt)) return NULL;
  return poly_uop_tagged(ctx, POLY_OP_CONST, dt, NULL, 0, poly_arg_float(value), ctx->next_buf_tag++);
}

static PolyUOp *poly_empty_shaped(PolyCtx *ctx, PolyDType dt, const int64_t *shape, int ndim) {
  PolyUOp *buf = poly_buffer(ctx, dt, 0);
  if (!buf || ndim <= 1) return buf;
  return poly_reshape(ctx, buf, (int64_t *)shape, ndim);
}

static PolyUOp *poly_full_from_scalar(PolyCtx *ctx, const int64_t *shape, int ndim, PolyUOp *scalar) {
  if (!scalar || ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  if (ndim == 0) return scalar;
  if (!shape) return NULL;

  bool has_zero = false;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] < 0) return NULL;
    if (shape[i] == 0) has_zero = true;
  }
  if (has_zero) return poly_empty_shaped(ctx, poly_dtype_scalar(scalar->dtype), shape, ndim);
  if (poly_shape_numel_checked(shape, ndim) < 0) return NULL;

  int64_t ones[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    ones[i] = 1;
  PolyUOp *r = poly_reshape(ctx, scalar, ones, ndim);
  return poly_expand(ctx, r, (int64_t *)shape, ndim);
}

static int64_t poly_arange_len(long double start, long double stop, long double step) {
  if (step == 0.0L) return -1;
  if ((step > 0.0L && start < stop) || (step < 0.0L && start > stop)) {
    long double span = (stop - start) / step;
    long double n = ceill(span - 1e-12L);
    if (n < 0.0L) return 0;
    if (n > (long double)INT64_MAX) return -1;
    return (int64_t)n;
  }
  return 0;
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

static bool shape_equal_except_axis(
    const int64_t *full,
    int full_ndim,
    const int64_t *reduced,
    int reduced_ndim,
    int axis
) {
  if (!full || !reduced || full_ndim != reduced_ndim + 1) return false;
  if (axis < 0) axis += full_ndim;
  if (axis < 0 || axis >= full_ndim) return false;
  for (int i = 0, j = 0; i < full_ndim; i++) {
    if (i == axis) continue;
    if (full[i] != reduced[j++]) return false;
  }
  return true;
}

/* Internal: compute output shape for a single-axis reduction */

static void reduce_output_shape(
    const int64_t *shape,
    int ndim,
    int axis,
    int keepdim,
    int64_t *out_shape,
    int *out_ndim
) {
  if (axis < 0) axis += ndim;
  int on = 0;
  for (int i = 0; i < ndim; i++) {
    if (i == axis) {
      if (keepdim) out_shape[on++] = 1;
    } else {
      out_shape[on++] = shape[i];
    }
  }
  if (on == 0) {
    out_shape[0] = 1;
    on = 1;
  }
  *out_ndim = on;
}

/* Internal: do a single-axis reduce and optionally reshape away the axis */
static PolyUOp *do_reduce(
    PolyCtx *ctx,
    PolyOps reduce_op,
    PolyUOp *x,
    const int64_t *shape,
    int ndim,
    int axis,
    int keepdim,
    int64_t *out_shape,
    int *out_ndim
) {
  if (axis < 0) axis += ndim;
  int64_t axes[] = {axis};
  PolyUOp *r = poly_reduce_axis(ctx, reduce_op, x, axes, 1);
  reduce_output_shape(shape, ndim, axis, keepdim, out_shape, out_ndim);
  if (!keepdim) {
    r = poly_reshape(ctx, r, out_shape, *out_ndim);
  }
  return r;
}

static PolyUOp *reshape_logical_input(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim) {
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

/* erf tau helper */

/* A&S 7.1.26: tau(|x|) = t * P(t) * exp(-x^2) where t = 1/(1+p*|x|).
 * erf(x) = sign(x) * (1 - tau(|x|)).
 * erfc(x) = tau(x) for x >= 0, 2 - tau(|x|) for x < 0.
 * Computing tau directly avoids the 1-erf(x) cancellation in erfc. */
static PolyUOp *erf_tau(PolyCtx *ctx, PolyUOp *ax, PolyDType dt) {
  PolyUOp *t = poly_alu1(
      ctx, POLY_OP_RECIPROCAL,
      poly_alu2(
          ctx, POLY_OP_ADD, cdt(ctx, dt, 1.0),
          poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.3275911), ax)
      )
  );
  PolyUOp *p = cdt(ctx, dt, 1.061405429);
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, -1.453152027), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.421413741), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, -0.284496736), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 0.254829592), poly_alu2(ctx, POLY_OP_MUL, t, p));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, ax, ax);
  PolyUOp *e = poly_exp(ctx, poly_alu1(ctx, POLY_OP_NEG, x2));
  return poly_alu2(ctx, POLY_OP_MUL, t, poly_alu2(ctx, POLY_OP_MUL, p, e));
}

/* lgamma Lanczos helper */

static PolyUOp *poly_lgamma_forward_lanczos(PolyCtx *ctx, PolyUOp *x, PolyDType dt) {
  /* Lanczos approximation with reflection. */
  const double g = 7.0;
  const double c0 = 0.99999999999980993;
  const double c[8] = {676.5203681218851,     -1259.1392167224028,  771.32342877765313,
                       -176.61502916214059,   12.507343278686905,   -0.13857109526572012,
                       9.9843695780195716e-6, 1.5056327351493116e-7};

  PolyUOp *xm1 = poly_alu2(ctx, POLY_OP_SUB, x, cdt(ctx, dt, 1.0));
  PolyUOp *a = cdt(ctx, dt, c0);
  for (int i = 0; i < 8; i++) {
    PolyUOp *den = poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, (double)(i + 1)));
    a = poly_alu2(ctx, POLY_OP_ADD, a, poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, c[i]), den));
  }
  PolyUOp *t = poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, g + 0.5));
  PolyUOp *lg_pos = poly_alu2(
      ctx, POLY_OP_ADD, cdt(ctx, dt, 0.91893853320467274178),
      poly_alu2(
          ctx, POLY_OP_ADD,
          poly_alu2(
              ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, 0.5)),
              poly_log(ctx, t)
          ),
          poly_alu2(ctx, POLY_OP_SUB, poly_log(ctx, a), t)
      )
  );

  PolyUOp *one_minus_x = poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 1.0), x);
  PolyUOp *xm1r = poly_alu2(ctx, POLY_OP_SUB, one_minus_x, cdt(ctx, dt, 1.0));
  PolyUOp *ar = cdt(ctx, dt, c0);
  for (int i = 0; i < 8; i++) {
    PolyUOp *den = poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, (double)(i + 1)));
    ar = poly_alu2(ctx, POLY_OP_ADD, ar, poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, c[i]), den));
  }
  PolyUOp *tr = poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, g + 0.5));
  PolyUOp *lg_ref_base = poly_alu2(
      ctx, POLY_OP_ADD, cdt(ctx, dt, 0.91893853320467274178),
      poly_alu2(
          ctx, POLY_OP_ADD,
          poly_alu2(
              ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, 0.5)),
              poly_log(ctx, tr)
          ),
          poly_alu2(ctx, POLY_OP_SUB, poly_log(ctx, ar), tr)
      )
  );
  PolyUOp *sinpix = poly_sin(ctx, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, M_PI), x));
  PolyUOp *lg_ref = poly_alu2(
      ctx, POLY_OP_SUB,
      poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, log(M_PI)), poly_log(ctx, poly_abs(ctx, sinpix))),
      lg_ref_base
  );
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0.5));
  return poly_alu3(ctx, POLY_OP_WHERE, cond, lg_ref, lg_pos);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Public functions                                                     */
/* ══════════════════════════════════════════════════════════════════════ */

/* Broadcasting (tinygrad _broadcasted / _broadcast_to) */

PolyUOp *poly_broadcast_to(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim) {
  if (!ctx || !x || !shape || ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  int64_t x_shape[POLY_MAX_DIMS];
  int x_ndim = uop_shape(ctx, x, x_shape);
  if (x_ndim < 0) return x; /* shapeless (CONST scalar) -- pass through */
  if (ndim < x_ndim) return NULL; /* can't broadcast to fewer dims */

  /* Already matching */
  if (x_ndim == ndim) {
    bool same = true;
    for (int i = 0; i < ndim; i++)
      if (x_shape[i] != shape[i]) {
        same = false;
        break;
      }
    if (same) return x;
  }

  /* Left-pad with 1s to match ndim (tinygrad _align_left) */
  int64_t aligned[POLY_MAX_DIMS];
  int pad = ndim - x_ndim;
  for (int i = 0; i < pad; i++)
    aligned[i] = 1;
  for (int i = 0; i < x_ndim; i++)
    aligned[pad + i] = x_shape[i];

  /* Validate: each aligned dim must be 1 or equal to target */
  for (int i = 0; i < ndim; i++) {
    if (aligned[i] != shape[i] && aligned[i] != 1) {
      fprintf(
          stderr, "poly_broadcast_to: incompatible dim %d: %lld vs %lld\n", i,
          (long long)aligned[i], (long long)shape[i]
      );
      return NULL;
    }
  }

  PolyUOp *r = poly_reshape(ctx, x, aligned, ndim);

  /* Expand where aligned[i]==1 and shape[i]>1 */
  bool need_expand = false;
  for (int i = 0; i < ndim; i++)
    if (aligned[i] != shape[i]) {
      need_expand = true;
      break;
    }
  if (need_expand) r = poly_expand(ctx, r, (int64_t *)shape, ndim);
  return r;
}

bool poly_broadcast_pair(
    PolyCtx *ctx,
    PolyUOp **a,
    PolyUOp **b,
    int64_t *out_shape,
    int *out_ndim
) {
  int64_t sa[POLY_MAX_DIMS], sb[POLY_MAX_DIMS];
  int na = uop_shape(ctx, *a, sa);
  int nb = uop_shape(ctx, *b, sb);

  /* Scalars or shapeless -- no broadcast needed */
  if (na <= 0 && nb <= 0) {
    *out_ndim = 0;
    return true;
  }
  if (na <= 0) {
    na = 0;
  }
  if (nb <= 0) {
    nb = 0;
  }

  /* Compute broadcast shape (tinygrad _broadcast_shape) */
  int nd = na > nb ? na : nb;
  if (nd > POLY_MAX_DIMS) {
    *out_ndim = 0;
    return false;
  }
  int pa = nd - na, pb = nd - nb;
  for (int i = 0; i < nd; i++) {
    int64_t da = (i >= pa) ? sa[i - pa] : 1;
    int64_t db = (i >= pb) ? sb[i - pb] : 1;
    if (da != db && da != 1 && db != 1) {
      fprintf(
          stderr, "poly_broadcast_pair: incompatible shapes at dim %d: %lld vs %lld\n", i,
          (long long)da, (long long)db
      );
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

/* Broadcasting binary ops */

PolyUOp *poly_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  poly_broadcast_pair(ctx, &a, &b, shape, &ndim);
  return poly_alu2(ctx, POLY_OP_ADD, a, b);
}

PolyUOp *poly_sub(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  poly_broadcast_pair(ctx, &a, &b, shape, &ndim);
  return poly_alu2(ctx, POLY_OP_SUB, a, b);
}

PolyUOp *poly_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  poly_broadcast_pair(ctx, &a, &b, shape, &ndim);
  return poly_alu2(ctx, POLY_OP_MUL, a, b);
}

PolyUOp *poly_div(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  poly_broadcast_pair(ctx, &a, &b, shape, &ndim);
  return poly_alu2(ctx, POLY_OP_FDIV, a, b);
}

/* Contiguous (realize barrier) */

PolyUOp *poly_contiguous(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  if (x->op == POLY_OP_CONTIGUOUS) return x;
  if (poly_uop_has_buffer_identity(x)) return x;
  return poly_uop1(ctx, POLY_OP_CONTIGUOUS, x->dtype, x, poly_arg_none());
}

/* Math */

PolyUOp *poly_exp(PolyCtx *ctx, PolyUOp *x) {
  /* exp(x) = exp2(x * (1/ln2)) */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  return poly_alu1(ctx, POLY_OP_EXP2, poly_alu2(ctx, POLY_OP_MUL, x, cdt(ctx, dt, 1.0 / M_LN2)));
}

PolyUOp *poly_log(PolyCtx *ctx, PolyUOp *x) {
  /* log(x) = log2(x) * ln2 */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  return poly_alu2(ctx, POLY_OP_MUL, poly_alu1(ctx, POLY_OP_LOG2, x), cdt(ctx, dt, M_LN2));
}

PolyUOp *poly_log1p(PolyCtx *ctx, PolyUOp *x) {
  /* Stable near zero: log(1+x) ~ x - x^2/2 + x^3/3 */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *small = poly_alu2(ctx, POLY_OP_CMPLT, ax, cdt(ctx, dt, 1e-4));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x2, x);
  PolyUOp *poly = poly_alu2(
      ctx, POLY_OP_ADD, x,
      poly_alu2(
          ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -0.5), x2),
          poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0 / 3.0), x3)
      )
  );
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
  PolyUOp *poly = poly_alu2(
      ctx, POLY_OP_ADD, x,
      poly_alu2(
          ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.5), x2),
          poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0 / 6.0), x3)
      )
  );
  PolyUOp *direct = poly_alu2(ctx, POLY_OP_SUB, poly_exp(ctx, x), cdt(ctx, dt, 1.0));
  return poly_alu3(ctx, POLY_OP_WHERE, small, poly, direct);
}

PolyUOp *poly_sin(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu1(ctx, POLY_OP_SIN, x);
}

PolyUOp *poly_cos(PolyCtx *ctx, PolyUOp *x) {
  /* cos(x) = sin(pi/2 - x) */
  return poly_alu1(ctx, POLY_OP_SIN, poly_alu2(ctx, POLY_OP_SUB, cf(ctx, x, M_PI / 2.0), x));
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
  PolyUOp *term1 = poly_alu2(
      ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, 2.0 / (M_PI * 0.147)), one),
      poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.5), ln)
  );
  PolyUOp *term2 = poly_alu2(ctx, POLY_OP_FDIV, ln, a);
  PolyUOp *inside = poly_alu2(ctx, POLY_OP_SUB, poly_alu2(ctx, POLY_OP_MUL, term1, term1), term2);
  PolyUOp *root = poly_alu1(
      ctx, POLY_OP_SQRT, poly_alu2(ctx, POLY_OP_SUB, poly_alu1(ctx, POLY_OP_SQRT, inside), term1)
  );
  return poly_alu2(ctx, POLY_OP_MUL, poly_sign(ctx, x), root);
}

PolyUOp *poly_ndtri(PolyCtx *ctx, PolyUOp *x) {
  /* ndtri(p) = sqrt(2) * erfinv(2p-1) */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *arg = poly_alu2(
      ctx, POLY_OP_SUB, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 2.0), x), cdt(ctx, dt, 1.0)
  );
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
    xx =
        poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_ADD, xx, cdt(ctx, dt, 1.0)), xx);
  }
  PolyUOp *inv = poly_alu1(ctx, POLY_OP_RECIPROCAL, xx);
  PolyUOp *inv2 = poly_alu2(ctx, POLY_OP_MUL, inv, inv);
  PolyUOp *inv4 = poly_alu2(ctx, POLY_OP_MUL, inv2, inv2);
  PolyUOp *inv6 = poly_alu2(ctx, POLY_OP_MUL, inv4, inv2);
  PolyUOp *asym = poly_alu2(
      ctx, POLY_OP_ADD, poly_log(ctx, xx),
      poly_alu2(
          ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -0.5), inv),
          poly_alu2(
              ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -1.0 / 12.0), inv2),
              poly_alu2(
                  ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0 / 120.0), inv4),
                  poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -1.0 / 252.0), inv6)
              )
          )
      )
  );
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
  return poly_alu1(ctx, POLY_OP_RECIPROCAL, poly_alu2(ctx, POLY_OP_ADD, cf(ctx, x, 1.0), e));
}

PolyUOp *poly_tanh_act(PolyCtx *ctx, PolyUOp *x) {
  /* tanh(x) = 2*sigmoid(2x) - 1 */
  PolyUOp *two_x = poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 2.0), x);
  return poly_alu2(
      ctx, POLY_OP_SUB, poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 2.0), poly_sigmoid(ctx, two_x)),
      cf(ctx, x, 1.0)
  );
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
  return poly_alu2(ctx, POLY_OP_ADD, result, poly_alu2(ctx, POLY_OP_MUL, x, zero));
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
  return poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_ADD, b, cf(ctx, x, 1.0)), b);
}

PolyUOp *poly_floor(PolyCtx *ctx, PolyUOp *x) {
  /* floor(x) = (x < (b=trunc(x))).where(b-1, b) */
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, b); /* x < b */
  return poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_SUB, b, cf(ctx, x, 1.0)), b);
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
  return poly_alu3(
      ctx, POLY_OP_WHERE, cond, poly_ceil(ctx, poly_alu2(ctx, POLY_OP_SUB, x, half)),
      poly_floor(ctx, poly_alu2(ctx, POLY_OP_ADD, x, half))
  );
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

/* Activations */

PolyUOp *poly_relu(PolyCtx *ctx, PolyUOp *x) {
  /* relu(x) = where(0 < x, x, 0) */
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, zero, x);
  return poly_alu3(ctx, POLY_OP_WHERE, cond, x, zero);
}

PolyUOp *poly_relu6(PolyCtx *ctx, PolyUOp *x) {
  /* relu6(x) = relu(x) - relu(x - 6) */
  return poly_alu2(
      ctx, POLY_OP_SUB, poly_relu(ctx, x),
      poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, x, cf(ctx, x, 6.0)))
  );
}

PolyUOp *poly_leaky_relu(PolyCtx *ctx, PolyUOp *x, double neg_slope) {
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, cf(ctx, x, 0.0));
  return poly_alu3(
      ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, neg_slope), x), x
  );
}

PolyUOp *poly_gelu(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x, poly_alu2(ctx, POLY_OP_MUL, x, x));
  PolyUOp *inner =
      poly_alu2(ctx, POLY_OP_ADD, x, poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 0.044715), x3));
  PolyUOp *scaled = poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, sqrt(2.0 / M_PI)), inner);
  return poly_alu2(
      ctx, POLY_OP_MUL, cf(ctx, x, 0.5),
      poly_alu2(
          ctx, POLY_OP_MUL, x,
          poly_alu2(ctx, POLY_OP_ADD, cf(ctx, x, 1.0), poly_tanh_act(ctx, scaled))
      )
  );
}

PolyUOp *poly_quick_gelu(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(
      ctx, POLY_OP_MUL, x, poly_sigmoid(ctx, poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 1.702), x))
  );
}

PolyUOp *poly_silu(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL, x, poly_sigmoid(ctx, x));
}

PolyUOp *poly_elu(PolyCtx *ctx, PolyUOp *x, double alpha) {
  return poly_alu2(
      ctx, POLY_OP_SUB, poly_relu(ctx, x),
      poly_alu2(
          ctx, POLY_OP_MUL, cf(ctx, x, alpha),
          poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, cf(ctx, x, 1.0), poly_exp(ctx, x)))
      )
  );
}

PolyUOp *poly_softplus(PolyCtx *ctx, PolyUOp *x, double beta) {
  PolyUOp *bx = poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, beta), x);
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *m = poly_alu2(ctx, POLY_OP_MAX, bx, zero);
  PolyUOp *ea = poly_exp(ctx, poly_alu2(ctx, POLY_OP_SUB, bx, m));
  PolyUOp *eb = poly_exp(ctx, poly_alu2(ctx, POLY_OP_SUB, zero, m));
  PolyUOp *lae = poly_alu2(ctx, POLY_OP_ADD, m, poly_log(ctx, poly_alu2(ctx, POLY_OP_ADD, ea, eb)));
  return poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 1.0 / beta), lae);
}

PolyUOp *poly_mish(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL, x, poly_tanh_act(ctx, poly_softplus(ctx, x, 1.0)));
}

PolyUOp *poly_hardtanh(PolyCtx *ctx, PolyUOp *x, double min_val, double max_val) {
  return poly_clamp(ctx, x, min_val, max_val);
}

PolyUOp *poly_hardswish(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(
      ctx, POLY_OP_MUL,
      poly_alu2(
          ctx, POLY_OP_MUL, x, poly_relu6(ctx, poly_alu2(ctx, POLY_OP_ADD, x, cf(ctx, x, 3.0)))
      ),
      cf(ctx, x, 1.0 / 6.0)
  );
}

PolyUOp *poly_hardsigmoid(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *t = poly_alu2(
      ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 1.0 / 6.0), x), cf(ctx, x, 0.5)
  );
  return poly_alu2(
      ctx, POLY_OP_SUB, poly_relu(ctx, t),
      poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, t, cf(ctx, x, 1.0)))
  );
}

/* Comparisons (broadcasting) */

/* Logical NOT — polygrad's canonical form is `CMPNE(x, CONST(true))` for
 * bool inputs, matching tinygrad's `logical_not()` after CAST elision
 * (mixin/elementwise.py:25-33 + symbolic.py:126).
 *
 * Polygrad ALSO supports `NEG(bool_uop)` at the kernel level (alu.c:169
 * maps NEG-on-bool to `!a`). Both forms are accepted by codegen, but
 * tensor-level helpers should produce the CMPNE form for tinygrad parity
 * so Phase D's reduce_collapse rules can match the IR shape verbatim. */
PolyUOp *poly_logical_not(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *t = poly_const_typed(ctx, POLY_BOOL, 1);
  return poly_alu2(ctx, POLY_OP_CMPNE, x, t);
}

/* All comparison helpers return BOOL, mirroring tinygrad's
 * mixin/elementwise.py:218-247:
 *   eq(a,b) = (a != b).logical_not()
 *   ne(a,b) = CMPNE(a,b)
 *   gt(a,b) = CMPLT(b,a)         (operand swap)
 *   lt(a,b) = CMPLT(a,b)
 *   ge(a,b) = (a < b).logical_not()
 *   le(a,b) = (a > b).logical_not() = (b < a).logical_not()
 *
 * Polygrad previously had ge/le returning float WHERE(0,1); fixed in P5
 * for tinygrad parity and to let Phase D's reduce_collapse Rule 4 match
 * polygrad's tril/triu masks. */
PolyUOp *poly_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  PolyUOp *ne = poly_alu2(ctx, POLY_OP_CMPNE, a, b);
  return poly_logical_not(ctx, ne);
}

PolyUOp *poly_ne(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  return poly_alu2(ctx, POLY_OP_CMPNE, a, b);
}

PolyUOp *poly_gt(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  return poly_alu2(ctx, POLY_OP_CMPLT, b, a);
}

PolyUOp *poly_ge(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  PolyUOp *lt = poly_alu2(ctx, POLY_OP_CMPLT, a, b);
  return poly_logical_not(ctx, lt);
}

PolyUOp *poly_le(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  PolyUOp *gt = poly_alu2(ctx, POLY_OP_CMPLT, b, a);
  return poly_logical_not(ctx, gt);
}

PolyUOp *poly_cast(PolyCtx *ctx, PolyUOp *x, PolyDType target) {
  return poly_uop1(ctx, POLY_OP_CAST, target, x, poly_arg_none());
}

PolyUOp *poly_cast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id) {
  if (dtype_id < 0 || dtype_id >= N_DTYPE_FFI) return NULL;
  return poly_cast(ctx, x, *_dtype_table_ffi[dtype_id]);
}

PolyUOp *poly_where_op(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  poly_broadcast_pair(ctx, &x, &y, s, &nd);
  /* tinygrad Tensor.where casts non-bool conditions to bool before building
   * Ops.WHERE. Keeping that in the core constructor preserves the expected
   * CMPNE(cond, 0) node in helper graphs such as nonzero-value padding. */
  if (!poly_dtype_is_bool(poly_dtype_scalar(cond->dtype))) cond = poly_cast(ctx, cond, POLY_BOOL);
  poly_broadcast_pair(ctx, &cond, &x, s, &nd);
  return poly_alu3(ctx, POLY_OP_WHERE, cond, x, y);
}

PolyUOp *poly_maximum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  return poly_alu2(ctx, POLY_OP_MAX, a, b);
}

PolyUOp *poly_minimum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  poly_broadcast_pair(ctx, &a, &b, s, &nd);
  return poly_alu1(
      ctx, POLY_OP_NEG,
      poly_alu2(ctx, POLY_OP_MAX, poly_alu1(ctx, POLY_OP_NEG, a), poly_alu1(ctx, POLY_OP_NEG, b))
  );
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

/* ══════════════════════════════════════════════════════════════════════ */
/*  Movement-op helpers (port of tinygrad mixin/movement.py)             */
/*                                                                       */
/*  Pure tensor-level helpers built from existing primitives. Static     */
/*  int64_t shapes only -- polygrad does not yet match tinygrad's        */
/*  symbolic-shape movement args.                                        */
/* ══════════════════════════════════════════════════════════════════════ */

/* Tensor.repeat -- movement.py:465 */
PolyUOp *poly_repeat(PolyCtx *ctx, PolyUOp *x, const int64_t *repeats, int n_repeats) {
  int64_t in_shape[POLY_MAX_DIMS];
  int in_ndim = uop_shape(ctx, x, in_shape);
  if (in_ndim < 0 || in_ndim > n_repeats || n_repeats > POLY_MAX_DIMS) return NULL;

  /* _align_left: pad input shape with leading 1s to match n_repeats */
  int64_t base[POLY_MAX_DIMS];
  int pad = n_repeats - in_ndim;
  for (int i = 0; i < pad; i++)
    base[i] = 1;
  for (int i = 0; i < in_ndim; i++)
    base[pad + i] = in_shape[i];

  /* unsqueezed = flatten([[s] if r==1 else [1,s] for r,s in zip(repeats, base)])
   * expanded   = flatten([[s] if r==1 else [r,s] for r,s in zip(repeats, base)])
   * final      = [r*s for r,s in zip(repeats, base)] */
  int64_t unsq[POLY_MAX_DIMS * 2], exp[POLY_MAX_DIMS * 2], final_sh[POLY_MAX_DIMS];
  int n = 0;
  for (int i = 0; i < n_repeats; i++) {
    int64_t r = repeats[i], s = base[i];
    if (r == 1) {
      unsq[n] = s;
      exp[n] = s;
      n++;
    } else {
      unsq[n] = 1;
      exp[n] = r;
      n++;
      unsq[n] = s;
      exp[n] = s;
      n++;
    }
    final_sh[i] = r * s;
  }

  return poly_reshape(
      ctx, poly_expand(ctx, poly_reshape(ctx, x, unsq, n), exp, n), final_sh, n_repeats
  );
}

/* Tensor.shrink_to -- movement.py:168. ends[i] == -1 means no-op (keep dim). */
PolyUOp *poly_shrink_to(PolyCtx *ctx, PolyUOp *x, const int64_t *ends, int n_ends) {
  int64_t in_shape[POLY_MAX_DIMS];
  int in_ndim = uop_shape(ctx, x, in_shape);
  if (in_ndim != n_ends) return NULL;

  int64_t pairs[POLY_MAX_DIMS][2];
  bool any = false;
  for (int i = 0; i < n_ends; i++) {
    int64_t e = (ends[i] == -1) ? in_shape[i] : ends[i];
    pairs[i][0] = 0;
    pairs[i][1] = e;
    if (e != in_shape[i]) any = true;
  }
  return any ? poly_shrink(ctx, x, pairs, n_ends) : x;
}

/* Tensor._pool -- movement.py:487. General N-d pool via repeat/shrink/reshape/permute. */
PolyUOp *poly_pool(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *k_,
    int nk,
    const int64_t *stride_,
    const int64_t *dilation_
) {
  int64_t sh[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, sh);
  if (ndim < nk) return NULL;
  int noop = ndim - nk;

  int64_t s_[POLY_MAX_DIMS], d_[POLY_MAX_DIMS], i_[POLY_MAX_DIMS];
  int64_t o_[POLY_MAX_DIMS], f_[POLY_MAX_DIMS];
  for (int j = 0; j < nk; j++) {
    s_[j] = stride_ ? stride_[j] : 1;
    d_[j] = dilation_ ? dilation_[j] : 1;
    i_[j] = sh[noop + j];
    if (d_[j] * (k_[j] - 1) + 1 > i_[j]) return NULL;
    o_[j] = (i_[j] - d_[j] * (k_[j] - 1) + s_[j] - 1) / s_[j]; /* ceildiv */
    int64_t fn = o_[j] * s_[j] - d_[j];
    int64_t fv = fn <= 0 ? 1 : (fn + i_[j] - 1) / i_[j];
    f_[j] = fv < 1 ? 1 : fv;
  }
  if (noop + 3 * nk > POLY_MAX_DIMS) return NULL;

  /* x = repeat([1]*noop + [ceildiv(k*(i*f+d), i) for ...]) */
  int64_t rep[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++)
    rep[j] = 1;
  for (int j = 0; j < nk; j++) {
    int64_t num = k_[j] * (i_[j] * f_[j] + d_[j]);
    rep[noop + j] = (num + i_[j] - 1) / i_[j];
  }
  PolyUOp *r = poly_repeat(ctx, x, rep, ndim);

  /* shrink_to(noop + [k*(i*f+d) for ...]) */
  int64_t e1[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++)
    e1[j] = -1;
  for (int j = 0; j < nk; j++)
    e1[noop + j] = k_[j] * (i_[j] * f_[j] + d_[j]);
  r = poly_shrink_to(ctx, r, e1, ndim);

  /* reshape(noop + flatten((k, i*f+d) for ...)) */
  int64_t s1[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++)
    s1[j] = sh[j];
  for (int j = 0; j < nk; j++) {
    s1[noop + 2 * j] = k_[j];
    s1[noop + 2 * j + 1] = i_[j] * f_[j] + d_[j];
  }
  r = poly_reshape(ctx, r, s1, noop + 2 * nk);

  /* shrink_to(noop + flatten((k, o*s) for ...)).reshape(noop + flatten((k, o, s) for ...)) */
  int64_t e2[POLY_MAX_DIMS], s2[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++) {
    e2[j] = -1;
    s2[j] = sh[j];
  }
  for (int j = 0; j < nk; j++) {
    e2[noop + 2 * j] = k_[j];
    e2[noop + 2 * j + 1] = o_[j] * s_[j];
    s2[noop + 3 * j] = k_[j];
    s2[noop + 3 * j + 1] = o_[j];
    s2[noop + 3 * j + 2] = s_[j];
  }
  r = poly_reshape(ctx, poly_shrink_to(ctx, r, e2, noop + 2 * nk), s2, noop + 3 * nk);

  /* shrink_to(noop + flatten((k, o, 1) for ...)).reshape(noop + flatten((k, o) for ...)) */
  int64_t e3[POLY_MAX_DIMS], s3[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++) {
    e3[j] = -1;
    s3[j] = sh[j];
  }
  for (int j = 0; j < nk; j++) {
    e3[noop + 3 * j] = k_[j];
    e3[noop + 3 * j + 1] = o_[j];
    e3[noop + 3 * j + 2] = 1;
    s3[noop + 2 * j] = k_[j];
    s3[noop + 2 * j + 1] = o_[j];
  }
  r = poly_reshape(ctx, poly_shrink_to(ctx, r, e3, noop + 3 * nk), s3, noop + 2 * nk);

  /* permute(*range(noop), *[noop + i*2 + 1 for i in range(nk)],
   *[noop + i*2     for i in range(nk)]) */
  int64_t perm[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++)
    perm[j] = j;
  for (int j = 0; j < nk; j++)
    perm[noop + j] = noop + 2 * j + 1;
  for (int j = 0; j < nk; j++)
    perm[noop + nk + j] = noop + 2 * j;
  return poly_permute(ctx, r, perm, noop + 2 * nk);
}

/* Pad with arbitrary value / circular / reflect / replicate */

/* ones_like(x): same shape and dtype as x, filled with 1.
 * Pure UOp (no const-registry): scalar CONST -> reshape to (1,)*ndim -> expand. */
static PolyUOp *poly_ones_like(PolyCtx *ctx, PolyUOp *x) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *one = poly_const_typed(ctx, dt, 1.0);
  int64_t ones[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    ones[i] = 1;
  PolyUOp *r = poly_reshape(ctx, one, ones, ndim);
  return poly_expand(ctx, r, shape, ndim);
}

/* Tensor.cat -- tensor.py:1364
 *   dim_cumsum = accumulate([t.shape[dim] for t in tensors], initial=0)
 *   for i, t in enumerate(tensors):
 *     tensors[i] = t.pad([(dim_cumsum[i], dim_cumsum[-1]-dim_cumsum[i+1])
 *                          if j==dim else None for j in range(t.ndim)])
 *   return reduce(add, tensors) */
PolyUOp *poly_cat(PolyCtx *ctx, PolyUOp **tensors, int n_tensors, int dim) {
  if (!ctx || !tensors || n_tensors <= 0) return NULL;
  if (n_tensors == 1) return tensors[0];

  int64_t sh0[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, tensors[0], sh0);
  if (ndim < 0) return NULL;
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) return NULL;

  /* All tensors must match shape except along dim */
  int64_t shi[POLY_MAX_DIMS];
  int64_t cum[POLY_MAX_DIMS + 1];
  cum[0] = 0;
  for (int i = 0; i < n_tensors; i++) {
    int ni = uop_shape(ctx, tensors[i], shi);
    if (ni != ndim) {
      fprintf(stderr, "poly_cat: ndim mismatch at tensor %d: %d vs %d\n", i, ni, ndim);
      return NULL;
    }
    for (int j = 0; j < ndim; j++) {
      if (j == dim) continue;
      if (shi[j] != sh0[j]) {
        fprintf(
            stderr, "poly_cat: shape mismatch at tensor %d dim %d: %lld vs %lld\n", i, j,
            (long long)shi[j], (long long)sh0[j]
        );
        return NULL;
      }
    }
    cum[i + 1] = cum[i] + shi[dim];
  }
  int64_t total = cum[n_tensors];

  PolyUOp *acc = NULL;
  for (int i = 0; i < n_tensors; i++) {
    int64_t pads[POLY_MAX_DIMS][2];
    for (int j = 0; j < ndim; j++) {
      pads[j][0] = 0;
      pads[j][1] = 0;
    }
    pads[dim][0] = cum[i];
    pads[dim][1] = total - cum[i + 1];
    PolyUOp *padded = poly_pad(ctx, tensors[i], pads, ndim);
    if (!padded) return NULL;
    acc = (i == 0) ? padded : poly_alu2(ctx, POLY_OP_ADD, acc, padded);
  }
  return acc;
}

/* Tensor._pad_constant -- tensor.py:1067 */
PolyUOp *poly_pad_value(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim, double value) {
  if (!ctx || !x || !pads || ndim <= 0) return NULL;

  int64_t sh[POLY_MAX_DIMS];
  int xnd = uop_shape(ctx, x, sh);
  if (xnd != ndim) {
    fprintf(stderr, "poly_pad_value: ndim mismatch %d vs %d\n", xnd, ndim);
    return NULL;
  }

  /* has_neg = not all(p >= 0) */
  bool has_neg = false;
  for (int i = 0; i < ndim; i++)
    if (pads[i][0] < 0 || pads[i][1] < 0) {
      has_neg = true;
      break;
    }

  PolyUOp *X = x;
  int64_t nn_pads[POLY_MAX_DIMS][2];
  if (has_neg) {
    /* Shrink first for the negative parts:
     * shrink_pair = (-min(pB,0), min(pA + s, s)) */
    int64_t shr[POLY_MAX_DIMS][2];
    for (int i = 0; i < ndim; i++) {
      int64_t pB = pads[i][0], pA = pads[i][1], s = sh[i];
      shr[i][0] = -(pB < 0 ? pB : 0); /* -min(pB, 0) */
      int64_t end = pA + s;
      shr[i][1] = end < s ? end : s; /*  min(pA + s, s) */
    }
    X = poly_shrink(ctx, X, shr, ndim);
    /* Then pad with only the non-negative parts */
    for (int i = 0; i < ndim; i++) {
      nn_pads[i][0] = pads[i][0] > 0 ? pads[i][0] : 0;
      nn_pads[i][1] = pads[i][1] > 0 ? pads[i][1] : 0;
    }
  } else {
    for (int i = 0; i < ndim; i++) {
      nn_pads[i][0] = pads[i][0];
      nn_pads[i][1] = pads[i][1];
    }
  }

  /* Fast path: zero pad */
  PolyUOp *padded_X = poly_pad(ctx, X, nn_pads, ndim);
  if (value == 0.0) return padded_X;

  /* Pad ones_like(X) with the same pads, where==0 -> value, else 0; then add */
  PolyUOp *ones = poly_ones_like(ctx, X);
  PolyUOp *padded_ones = poly_pad(ctx, ones, nn_pads, ndim);
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *zero_c = poly_const_typed(ctx, dt, 0.0);
  PolyUOp *value_c = poly_const_typed(ctx, dt, value);
  /* where(padded_ones, 0, value): tinygrad's .where(0, value) means
   * "where padded_ones is true (=1, the data region) -> 0, else -> value" */
  PolyUOp *fill = poly_where_op(ctx, padded_ones, zero_c, value_c);
  return poly_alu2(ctx, POLY_OP_ADD, padded_X, fill);
}

/* Tensor._pad_circular -- tensor.py:1075
 *   X = self.repeat(tuple(1 + bool(pB) + bool(pA) for pB,pA in pX))
 *   return X.shrink(tuple((0 if pB == 0 else osh-pB,
 *                          xsh if pA == 0 else xsh-osh+pA)
 *                          for (pB,pA),osh,xsh in zip(pX, orig_shape, X.shape))) */
PolyUOp *poly_pad_circular(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim) {
  if (!ctx || !x || !pads || ndim <= 0) return NULL;

  int64_t sh[POLY_MAX_DIMS];
  int xnd = uop_shape(ctx, x, sh);
  if (xnd != ndim) {
    fprintf(stderr, "poly_pad_circular: ndim mismatch %d vs %d\n", xnd, ndim);
    return NULL;
  }

  for (int i = 0; i < ndim; i++) {
    if (pads[i][0] < 0 || pads[i][1] < 0) {
      fprintf(stderr, "poly_pad_circular: negative pads not supported\n");
      return NULL;
    }
    if (pads[i][0] > sh[i] || pads[i][1] > sh[i]) {
      fprintf(
          stderr, "poly_pad_circular: pad %lld/%lld exceeds dim %lld (would wrap >1x)\n",
          (long long)pads[i][0], (long long)pads[i][1], (long long)sh[i]
      );
      return NULL;
    }
  }

  /* repeats = [1 + (pB!=0) + (pA!=0) for ...] */
  int64_t reps[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    reps[i] = 1 + (pads[i][0] != 0 ? 1 : 0) + (pads[i][1] != 0 ? 1 : 0);
  PolyUOp *X = poly_repeat(ctx, x, reps, ndim);
  if (!X) return NULL;

  /* Compute the new (post-repeat) shape so we can shrink correctly */
  int64_t xsh[POLY_MAX_DIMS];
  if (uop_shape(ctx, X, xsh) < 0) return NULL;

  int64_t shr[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    int64_t pB = pads[i][0], pA = pads[i][1];
    int64_t osh = sh[i], xs = xsh[i];
    shr[i][0] = (pB == 0) ? 0 : (osh - pB);
    shr[i][1] = (pA == 0) ? xs : (xs - osh + pA);
  }
  return poly_shrink(ctx, X, shr, ndim);
}

/* Common impl for reflect/replicate. mode_reflect=true means "reflect"
 * (skip the boundary element), false means "replicate" (repeat the boundary).
 * tensor.py:1081. */
static PolyUOp *poly_pad_reflect_replicate(
    PolyCtx *ctx,
    PolyUOp *x,
    int64_t (*pads)[2],
    int ndim,
    bool mode_reflect
) {
  if (!ctx || !x || !pads || ndim <= 0) return NULL;
  int64_t sh[POLY_MAX_DIMS];
  int xnd = uop_shape(ctx, x, sh);
  if (xnd != ndim) {
    fprintf(
        stderr, "poly_pad_%s: ndim mismatch %d vs %d\n", mode_reflect ? "reflect" : "replicate",
        xnd, ndim
    );
    return NULL;
  }

  /* tinygrad: pads = ((max(pB,0), max(pA,0)) for (pB,pA) in pX) -- positive only first */
  int64_t pos_pads[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    pos_pads[i][0] = pads[i][0] > 0 ? pads[i][0] : 0;
    pos_pads[i][1] = pads[i][1] > 0 ? pads[i][1] : 0;
  }

  PolyUOp *X = x;
  for (int d = 0; d < ndim; d++) {
    int64_t pB = pos_pads[d][0], pA = pos_pads[d][1];
    if (pB == 0 && pA == 0) continue;

    int64_t cur_sh[POLY_MAX_DIMS];
    int cur_nd = uop_shape(ctx, X, cur_sh);
    if (cur_nd < 0) return NULL;
    int64_t s = cur_sh[d];

    if (mode_reflect && (pB >= s || pA >= s)) {
      fprintf(
          stderr, "poly_pad_reflect: pad (%lld,%lld) >= dim size %lld at dim %d\n", (long long)pB,
          (long long)pA, (long long)s, d
      );
      return NULL;
    }

    PolyUOp *xB = NULL, *xA = NULL;
    if (mode_reflect) {
      /* slcB = slice(pB, 0, -1) -> indices [pB, pB-1, ..., 1]
       * That's elements at index 1..pB+1, then flipped. */
      if (pB > 0) {
        int64_t shr[POLY_MAX_DIMS][2];
        for (int j = 0; j < ndim; j++) {
          shr[j][0] = 0;
          shr[j][1] = cur_sh[j];
        }
        shr[d][0] = 1;
        shr[d][1] = pB + 1;
        PolyUOp *sl = poly_shrink(ctx, X, shr, ndim);
        int64_t flip_axes[1] = {d};
        xB = poly_flip(ctx, sl, flip_axes, 1);
      }
      /* slcA = slice(s-2, s-2-pA, -1) -> indices [s-2, s-3, ..., s-1-pA]
       * That's elements at index s-1-pA..s-1 (exclusive s-1), then flipped. */
      if (pA > 0) {
        int64_t shr[POLY_MAX_DIMS][2];
        for (int j = 0; j < ndim; j++) {
          shr[j][0] = 0;
          shr[j][1] = cur_sh[j];
        }
        shr[d][0] = s - 1 - pA;
        shr[d][1] = s - 1;
        PolyUOp *sl = poly_shrink(ctx, X, shr, ndim);
        int64_t flip_axes[1] = {d};
        xA = poly_flip(ctx, sl, flip_axes, 1);
      }
    } else {
      /* replicate: shrink to (0,1) and expand to (pB,) on dim d */
      if (pB > 0) {
        int64_t shr[POLY_MAX_DIMS][2];
        for (int j = 0; j < ndim; j++) {
          shr[j][0] = 0;
          shr[j][1] = cur_sh[j];
        }
        shr[d][0] = 0;
        shr[d][1] = 1;
        int64_t exp_sh[POLY_MAX_DIMS];
        for (int j = 0; j < ndim; j++)
          exp_sh[j] = cur_sh[j];
        exp_sh[d] = pB;
        xB = poly_expand(ctx, poly_shrink(ctx, X, shr, ndim), exp_sh, ndim);
      }
      if (pA > 0) {
        int64_t shr[POLY_MAX_DIMS][2];
        for (int j = 0; j < ndim; j++) {
          shr[j][0] = 0;
          shr[j][1] = cur_sh[j];
        }
        shr[d][0] = s - 1;
        shr[d][1] = s;
        int64_t exp_sh[POLY_MAX_DIMS];
        for (int j = 0; j < ndim; j++)
          exp_sh[j] = cur_sh[j];
        exp_sh[d] = pA;
        xA = poly_expand(ctx, poly_shrink(ctx, X, shr, ndim), exp_sh, ndim);
      }
    }

    /* cat([xB, X, xA] for those that exist) */
    PolyUOp *parts[3];
    int n_parts = 0;
    if (xB) parts[n_parts++] = xB;
    parts[n_parts++] = X;
    if (xA) parts[n_parts++] = xA;
    X = poly_cat(ctx, parts, n_parts, d);
    if (!X) return NULL;
  }

  /* shrink after for negative pads (reflect/replicate must see full data first):
   * shrink = ((-min(pB,0), min(pA+s, s)) for ((pB,pA), s) in zip(pX, X.shape)) */
  bool has_neg = false;
  for (int i = 0; i < ndim; i++)
    if (pads[i][0] < 0 || pads[i][1] < 0) {
      has_neg = true;
      break;
    }
  if (has_neg) {
    int64_t cur_sh[POLY_MAX_DIMS];
    if (uop_shape(ctx, X, cur_sh) < 0) return NULL;
    int64_t shr[POLY_MAX_DIMS][2];
    for (int i = 0; i < ndim; i++) {
      int64_t pB = pads[i][0], pA = pads[i][1], s = cur_sh[i];
      shr[i][0] = -(pB < 0 ? pB : 0);
      int64_t end = pA + s;
      shr[i][1] = end < s ? end : s;
    }
    X = poly_shrink(ctx, X, shr, ndim);
  }
  return X;
}

PolyUOp *poly_pad_reflect(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim) {
  return poly_pad_reflect_replicate(ctx, x, pads, ndim, true);
}

PolyUOp *poly_pad_replicate(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim) {
  return poly_pad_reflect_replicate(ctx, x, pads, ndim, false);
}

/* Tensor._cumalu -- tensor.py:2048
 *   pl_sz = shape[axis] - int(not _include_initial)
 *   pooled = transpose(axis,-1).pad((pl_sz, -int(_include_initial)),
 *                                    value=identity_element(op,dtype))._pool((shape[axis],))
 *   return pooled.sum(-1).transpose(axis,-1)
 *
 * Supports ADD/MAX/MUL via poly_pad_value with the operator's identity element. */
PolyUOp *poly_cumalu(PolyCtx *ctx, PolyUOp *x, int axis, PolyOps op, bool include_initial) {
  if (op != POLY_OP_ADD && op != POLY_OP_MAX && op != POLY_OP_MUL) {
    fprintf(stderr, "poly_cumalu: op must be ADD, MAX, or MUL\n");
    return NULL;
  }

  int64_t sh[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, sh);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim || sh[axis] == 0) return NULL;

  int64_t len = sh[axis];
  int64_t pl_sz = len - (include_initial ? 0 : 1);

  /* identity element per op */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  double identity;
  if (op == POLY_OP_ADD)
    identity = 0.0;
  else if (op == POLY_OP_MUL)
    identity = 1.0;
  else /* MAX */ {
    if (poly_dtype_is_float(dt))
      identity = -INFINITY;
    else
      identity = (double)INT64_MIN;
  }

  /* transpose(axis, -1): swap axis with last */
  int64_t perm[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    perm[i] = i;
  perm[axis] = ndim - 1;
  perm[ndim - 1] = axis;
  bool need_transpose = (axis != ndim - 1);
  PolyUOp *r = need_transpose ? poly_permute(ctx, x, perm, ndim) : x;

  /* pad((pl_sz, -int(include_initial))) on the LAST axis only, value=identity. */
  int64_t pads[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    pads[i][0] = 0;
    pads[i][1] = 0;
  }
  pads[ndim - 1][0] = pl_sz;
  pads[ndim - 1][1] = include_initial ? -1 : 0;
  r = poly_pad_value(ctx, r, pads, ndim, identity);

  /* _pool((len,)) on last axis. After _pool, last two dims are (windows, kernel). */
  int64_t k_arr[1] = {len};
  r = poly_pool(ctx, r, k_arr, 1, NULL, NULL);

  /* Reduce on last (kernel) axis with the requested op */
  int64_t axes[1] = {-1};
  /* poly_reduce_axis takes the actual op; output keeps dims, we reshape away the last */
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = 0;
  int64_t r_sh[POLY_MAX_DIMS];
  int r_nd = uop_shape(ctx, r, r_sh);
  if (r_nd < 0) return NULL;
  axes[0] = r_nd - 1; /* concrete axis */
  r = poly_reduce_axis(ctx, op, r, axes, 1);
  /* Drop the reduced axis */
  for (int i = 0; i < r_nd - 1; i++)
    out_shape[out_ndim++] = r_sh[i];
  if (out_ndim == 0) {
    out_shape[0] = 1;
    out_ndim = 1;
  }
  r = poly_reshape(ctx, r, out_shape, out_ndim);

  /* transpose(axis, -1): swap back */
  if (need_transpose) r = poly_permute(ctx, r, perm, ndim);
  return r;
}

/* Creation */

/* tinygrad Tensor.full -- tensor.py:660:
 *   Tensor(fill_value, _force_unique=True).reshape((1,)*ndim).expand(shape)
 *
 * Pure UOp graph -- no const-registry, no host malloc. The scalar seed must
 * stay unique per call, matching tinygrad's `_force_unique=True`, otherwise
 * separate creation ops like `_tri(arange, arange)` collapse together under
 * CSE and create extra realized helper kernels. */
PolyUOp *poly_const_int_by_id(PolyCtx *ctx, int64_t value, int dtype_id) {
  PolyDType dt;
  if (!ffi_dtype_is_integer_like(dtype_id, &dt)) return NULL;
  return poly_const_exact_int(ctx, dt, value);
}

PolyUOp *poly_const_float_by_id(PolyCtx *ctx, double value, int dtype_id) {
  PolyDType dt;
  if (!ffi_dtype_is_float_like(dtype_id, &dt)) return NULL;
  return poly_const_exact_float(ctx, dt, value);
}

PolyUOp *poly_full_int_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    int64_t fill_value,
    int dtype_id
) {
  PolyDType dt;
  if (!ffi_dtype_is_integer_like(dtype_id, &dt)) return NULL;
  PolyUOp *scalar = poly_const_unique_exact_int(ctx, dt, fill_value);
  return poly_full_from_scalar(ctx, shape, ndim, scalar);
}

PolyUOp *poly_full_float_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    double fill_value,
    int dtype_id
) {
  PolyDType dt;
  if (!ffi_dtype_is_float_like(dtype_id, &dt)) return NULL;
  PolyUOp *scalar = poly_const_unique_exact_float(ctx, dt, fill_value);
  return poly_full_from_scalar(ctx, shape, ndim, scalar);
}

PolyUOp *poly_full(PolyCtx *ctx, const int64_t *shape, int ndim, double fill_value) {
  return poly_full_float_by_id(ctx, shape, ndim, fill_value, 12);
}

/* tinygrad Tensor.arange -- tensor.py:722:
 *   Tensor.full((output_len,), step)._cumalu(0, Ops.ADD) + (start - step)
 *
 * Pure UOp graph. The cumulative sum is currently O(N^2) until the
 * range-collapse simplify pass lands in Phase D. */
PolyUOp *poly_arange_int_by_id(PolyCtx *ctx, int64_t start, int64_t stop, int64_t step, int dtype_id) {
  PolyDType dt;
  if (!ffi_dtype_is_integer_like(dtype_id, &dt) || poly_dtype_is_bool(dt)) return NULL;
  if (step == 0) {
    fprintf(stderr, "polygrad: arange: step must be non-zero\n");
    return NULL;
  }

  int64_t n = poly_arange_len((long double)start, (long double)stop, (long double)step);
  if (n < 0) return NULL;
  if (n == 0) return poly_buffer(ctx, dt, 0);

  int64_t shape[1] = {n};
  PolyUOp *base = poly_full_int_by_id(ctx, shape, 1, step, dtype_id);
  PolyUOp *cumsum = poly_cumalu(ctx, base, 0, POLY_OP_ADD, false);
  PolyUOp *bias = poly_const_exact_int(ctx, dt, start - step);
  return poly_alu2(ctx, POLY_OP_ADD, cumsum, bias);
}

PolyUOp *poly_arange_float_by_id(PolyCtx *ctx, double start, double stop, double step, int dtype_id) {
  PolyDType dt;
  if (!ffi_dtype_is_float_like(dtype_id, &dt)) return NULL;
  if (step == 0.0) {
    fprintf(stderr, "polygrad: arange: step must be non-zero\n");
    return NULL;
  }

  int64_t n = poly_arange_len((long double)start, (long double)stop, (long double)step);
  if (n < 0) return NULL;
  if (n == 0) return poly_buffer(ctx, dt, 0);

  int64_t shape[1] = {n};
  PolyUOp *base = poly_full_float_by_id(ctx, shape, 1, step, dtype_id);
  PolyUOp *cumsum = poly_cumalu(ctx, base, 0, POLY_OP_ADD, false);
  PolyUOp *bias = poly_const_exact_float(ctx, dt, start - step);
  return poly_alu2(ctx, POLY_OP_ADD, cumsum, bias);
}

PolyUOp *poly_arange(PolyCtx *ctx, double start, double stop, double step) {
  return poly_arange_float_by_id(ctx, start, stop, step, 12);
}

/* tinygrad Tensor.linspace -- tensor.py:754
 *   (start + Tensor.arange(steps) * ((stop - start) / (steps - 1))).cast(dtype) */
PolyUOp *poly_linspace_by_id(PolyCtx *ctx, double start, double stop, int64_t steps, int dtype_id) {
  const PolyDType *out_ptr = ffi_dtype_from_id(dtype_id);
  if (!out_ptr) return NULL;
  PolyDType out_dt = poly_dtype_scalar(*out_ptr);
  if (poly_dtype_is_bool(out_dt) || poly_dtype_eq(out_dt, POLY_VOID)) {
    fprintf(stderr, "polygrad: linspace: bool/void dtypes are not supported\n");
    return NULL;
  }
  if (steps < 0) {
    fprintf(stderr, "polygrad: linspace: number of steps must be non-negative\n");
    return NULL;
  }
  if (steps == 0) return poly_buffer(ctx, out_dt, 0);

  PolyDType compute_dt = poly_dtype_eq(out_dt, POLY_FLOAT64) ? POLY_FLOAT64 : POLY_FLOAT32;
  int compute_id = poly_dtype_eq(compute_dt, POLY_FLOAT64) ? 13 : 12;

  if (steps == 1) {
    int64_t shape[1] = {1};
    PolyUOp *one = poly_full_float_by_id(ctx, shape, 1, start, compute_id);
    return poly_dtype_eq(out_dt, compute_dt) ? one : poly_cast(ctx, one, out_dt);
  }

  double scale = (stop - start) / (double)(steps - 1);
  PolyUOp *ar = poly_arange_float_by_id(ctx, 0.0, (double)steps, 1.0, compute_id);
  PolyUOp *s_c = poly_const_exact_float(ctx, compute_dt, scale);
  PolyUOp *start_c = poly_const_exact_float(ctx, compute_dt, start);
  PolyUOp *result =
      poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, ar, s_c), start_c);
  return poly_dtype_eq(out_dt, compute_dt) ? result : poly_cast(ctx, result, out_dt);
}

PolyUOp *poly_linspace(PolyCtx *ctx, double start, double stop, int64_t steps) {
  return poly_linspace_by_id(ctx, start, stop, steps, 12);
}

/* tinygrad Tensor.eye -- tensor.py:774
 *   (arange(n).unsqueeze(-1) == arange(m)).cast(dtype) */
static PolyUOp *poly_iota_i32(PolyCtx *ctx, int64_t n) {
  if (n <= 0) return poly_buffer(ctx, POLY_INT32, 0);
  int64_t shape[1] = {n};
  PolyUOp *scalar = poly_const_unique_exact_int(ctx, POLY_INT32, 1);
  PolyUOp *base = poly_expand(ctx, poly_reshape(ctx, scalar, (int64_t[]){1}, 1), shape, 1);
  PolyUOp *cumsum = poly_cumalu(ctx, base, 0, POLY_OP_ADD, false);
  PolyUOp *bias = poly_const_exact_int(ctx, POLY_INT32, -1);
  return poly_alu2(ctx, POLY_OP_ADD, cumsum, bias);
}

PolyUOp *poly_eye_by_id(PolyCtx *ctx, int64_t n, int64_t m, int dtype_id) {
  const PolyDType *out_ptr = ffi_dtype_from_id(dtype_id);
  if (!out_ptr) return NULL;
  PolyDType out_dt = poly_dtype_scalar(*out_ptr);
  if (poly_dtype_eq(out_dt, POLY_VOID) || n < 0 || m < 0) return NULL;
  if (n == 0 || m == 0) {
    int64_t shape[2] = {n, m};
    return poly_empty_shaped(ctx, out_dt, shape, 2);
  }

  PolyUOp *rows = poly_reshape(ctx, poly_iota_i32(ctx, n), (int64_t[]){n, 1}, 2);
  PolyUOp *cols = poly_reshape(ctx, poly_iota_i32(ctx, m), (int64_t[]){1, m}, 2);
  PolyUOp *eq_bool = poly_eq(ctx, rows, cols);
  return poly_dtype_is_bool(out_dt) ? eq_bool : poly_cast(ctx, eq_bool, out_dt);
}

PolyUOp *poly_eye(PolyCtx *ctx, int64_t n) {
  return poly_eye_by_id(ctx, n, n, 12);
}

/* tinygrad Tensor._tri -- tensor.py:2128
 *   arange(r).unsqueeze(-1) + diagonal <= arange(c)
 *   Returns a bool mask of shape (r, c). */
static PolyUOp *poly_tri_mask(PolyCtx *ctx, int64_t r, int64_t c, int diagonal) {
  PolyUOp *rows = poly_reshape(ctx, poly_iota_i32(ctx, r), (int64_t[]){r, 1}, 2);
  PolyUOp *cols = poly_reshape(ctx, poly_iota_i32(ctx, c), (int64_t[]){1, c}, 2);
  PolyUOp *rows_shifted =
      (diagonal == 0)
          ? rows
          : poly_alu2(ctx, POLY_OP_ADD, rows, poly_const_exact_int(ctx, POLY_INT32, diagonal));
  return poly_le(ctx, rows_shifted, cols);
}

/* tinygrad Tensor.tril -- tensor.py:2154
 *   _tri(rows, cols, diagonal+1).where(zeros_like(self), self) */
PolyUOp *poly_tril(PolyCtx *ctx, PolyUOp *x, int diagonal) {
  if (!x) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim != 2) {
    fprintf(stderr, "polygrad: tril: only 2D tensors are supported\n");
    return NULL;
  }
  PolyUOp *mask = poly_tri_mask(ctx, shape[0], shape[1], diagonal + 1);
  PolyUOp *zero = poly_const_typed(ctx, poly_dtype_scalar(x->dtype), 0.0);
  return poly_where_op(ctx, mask, zero, x);
}

/* tinygrad Tensor.triu -- tensor.py:2131
 *   _tri(rows, cols, diagonal).where(self, zeros_like(self)) */
PolyUOp *poly_triu(PolyCtx *ctx, PolyUOp *x, int diagonal) {
  if (!x) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim != 2) {
    fprintf(stderr, "polygrad: triu: only 2D tensors are supported\n");
    return NULL;
  }
  PolyUOp *mask = poly_tri_mask(ctx, shape[0], shape[1], diagonal);
  PolyUOp *zero = poly_const_typed(ctx, poly_dtype_scalar(x->dtype), 0.0);
  return poly_where_op(ctx, mask, x, zero);
}

static PolyUOp *poly_rand_compute(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed, PolyDType out_dt) {
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  if (ndim > 0 && !shape) return NULL;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] < 0) return NULL;
    if (shape[i] == 0) return poly_empty_shaped(ctx, out_dt, shape, ndim);
  }

  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel < 0) return NULL;
  uint32_t key_lo = (uint32_t)(seed & 0xffffffffu);
  uint32_t key_hi = (uint32_t)(seed >> 32);
  uint32_t mixed_key = key_lo ^ ((key_hi << 16) | (key_hi >> 16));

  PolyUOp *counter_flat = poly_arange_int_by_id(ctx, 0, numel, 1, 7);
  if (!counter_flat) return NULL;
  PolyUOp *counter_t = reshape_logical_input(ctx, counter_flat, shape, ndim);
  PolyUOp *key_t = poly_const_exact_int(ctx, POLY_UINT32, (int64_t)mixed_key);

  if (poly_dtype_eq(out_dt, POLY_FLOAT64)) {
    PolyUOp *bits_hi = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, counter_t, key_t, poly_arg_none());
    PolyUOp *key2_t = poly_const_exact_int(ctx, POLY_UINT32, (int64_t)(mixed_key ^ 0x9E3779B9u));
    PolyUOp *bits_lo = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, counter_t, key2_t, poly_arg_none());
    PolyUOp *hi27 =
        poly_alu2(ctx, POLY_OP_SHR, bits_hi, poly_const_exact_int(ctx, POLY_UINT32, 5));
    PolyUOp *lo26 =
        poly_alu2(ctx, POLY_OP_SHR, bits_lo, poly_const_exact_int(ctx, POLY_UINT32, 6));
    PolyUOp *hi_f = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT64, hi27, poly_arg_none());
    PolyUOp *lo_f = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT64, lo26, poly_arg_none());
    PolyUOp *mant = poly_alu2(
        ctx, POLY_OP_ADD,
        poly_alu2(ctx, POLY_OP_MUL, hi_f, cdt(ctx, POLY_FLOAT64, 67108864.0)),
        lo_f
    );
    return poly_alu2(ctx, POLY_OP_MUL, mant, cdt(ctx, POLY_FLOAT64, 1.0 / 9007199254740992.0));
  }

  PolyUOp *bits = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, counter_t, key_t, poly_arg_none());
  PolyUOp *hi24 = poly_alu2(ctx, POLY_OP_SHR, bits, poly_const_exact_int(ctx, POLY_UINT32, 8));
  PolyUOp *as_f = poly_uop1(ctx, POLY_OP_CAST, out_dt, hi24, poly_arg_none());
  return poly_alu2(ctx, POLY_OP_MUL, as_f, cdt(ctx, out_dt, 1.0 / 16777216.0));
}

PolyUOp *poly_rand_by_id(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed, int dtype_id) {
  PolyDType out_dt;
  if (!ffi_dtype_is_float_like(dtype_id, &out_dt)) return NULL;
  PolyDType compute_dt = poly_dtype_eq(out_dt, POLY_FLOAT64) ? POLY_FLOAT64 : POLY_FLOAT32;
  PolyUOp *base = poly_rand_compute(ctx, shape, ndim, seed, compute_dt);
  if (!base) return NULL;
  return poly_dtype_eq(compute_dt, out_dt) ? base : poly_cast(ctx, base, out_dt);
}

PolyUOp *poly_rand(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed) {
  return poly_rand_by_id(ctx, shape, ndim, seed, 12);
}

PolyUOp *poly_randn_by_id(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed, int dtype_id) {
  PolyDType out_dt;
  if (!ffi_dtype_is_float_like(dtype_id, &out_dt)) return NULL;
  PolyDType compute_dt = poly_dtype_eq(out_dt, POLY_FLOAT64) ? POLY_FLOAT64 : POLY_FLOAT32;

  PolyUOp *u1 = poly_rand_compute(ctx, shape, ndim, seed, compute_dt);
  PolyUOp *u2 = poly_rand_compute(ctx, shape, ndim, seed ^ 0x9E3779B97F4A7C15ull, compute_dt);
  if (!u1 || !u2) return NULL;
  PolyUOp *u1_safe = poly_maximum(ctx, u1, cdt(ctx, compute_dt, 1e-7));
  PolyUOp *r = poly_alu1(
      ctx, POLY_OP_SQRT,
      poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, compute_dt, -2.0), poly_log(ctx, u1_safe))
  );
  PolyUOp *theta = poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, compute_dt, 2.0 * M_PI), u2);
  PolyUOp *result = poly_alu2(ctx, POLY_OP_MUL, r, poly_cos(ctx, theta));
  return poly_dtype_eq(compute_dt, out_dt) ? result : poly_cast(ctx, result, out_dt);
}

PolyUOp *poly_randn(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed) {
  return poly_randn_by_id(ctx, shape, ndim, seed, 12);
}

PolyUOp *poly_cholesky(PolyCtx *ctx, PolyUOp *x, int upper) {
  (void)ctx;
  (void)x;
  (void)upper;
  fprintf(
      stderr, "polygrad: cholesky: Track C fallback not yet implemented in frontend graph builder\n"
  );
  return NULL;
}

PolyUOp *poly_triangular_solve(
    PolyCtx *ctx,
    PolyUOp *a,
    PolyUOp *b,
    int upper,
    int transpose_a,
    int unit_diagonal
) {
  (void)ctx;
  (void)a;
  (void)b;
  (void)upper;
  (void)transpose_a;
  (void)unit_diagonal;
  fprintf(
      stderr,
      "polygrad: triangular_solve: Track C fallback not yet implemented in frontend graph builder\n"
  );
  return NULL;
}

/* Reductions */

PolyUOp *poly_sum_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t shape[POLY_MAX_DIMS], out_shape[POLY_MAX_DIMS];
  int ndim, out_ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  return do_reduce(ctx, POLY_OP_ADD, x, shape, ndim, axis, keepdim, out_shape, &out_ndim);
}

PolyUOp *poly_max_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t shape[POLY_MAX_DIMS], out_shape[POLY_MAX_DIMS];
  int ndim, out_ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  return do_reduce(ctx, POLY_OP_MAX, x, shape, ndim, axis, keepdim, out_shape, &out_ndim);
}

PolyUOp *poly_mean_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t shape[POLY_MAX_DIMS], out_shape[POLY_MAX_DIMS];
  int ndim, out_ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  int64_t count = shape[axis];
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, x, shape, ndim, axis, keepdim, out_shape, &out_ndim);
  return poly_alu2(ctx, POLY_OP_FDIV, s, cf(ctx, x, (double)count));
}

PolyUOp *poly_var_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim, int correction) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
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
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim;
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, sq, shape, ndim, axis, keepdim, out_shape, &out_ndim);
  double divisor = (double)(count - correction);
  if (divisor <= 0.0) divisor = 1.0;
  return poly_alu2(ctx, POLY_OP_FDIV, s, cf(ctx, x, divisor));
}

PolyUOp *poly_logsumexp(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
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

/* Matmul */

PolyUOp *poly_dot(PolyCtx *ctx, PolyUOp *x, PolyUOp *w) {
  if (!ctx || !x || !w) return NULL;
  int64_t x_shape[POLY_MAX_DIMS], w_shape[POLY_MAX_DIMS];
  int x_ndim = uop_shape(ctx, x, x_shape);
  int w_ndim = uop_shape(ctx, w, w_shape);
  if (x_ndim < 0 || w_ndim < 0) return NULL;
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = 0;
  if (x_ndim < 1 || w_ndim < 1 || x_ndim > POLY_MAX_DIMS || w_ndim > POLY_MAX_DIMS) return NULL;

  int64_t K = x_shape[x_ndim - 1];
  int axis_w = w_ndim - (w_ndim >= 2 ? 2 : 1);
  if (K != w_shape[axis_w]) return NULL;

  int64_t xs[POLY_MAX_DIMS];
  int xn = 0;
  for (int i = 0; i < x_ndim - 1; i++)
    xs[xn++] = x_shape[i];
  int n_ones_x;
  {
    int a = x_ndim - 1, b = w_ndim - 1;
    n_ones_x = a < b ? a : b;
    if (n_ones_x > 1) n_ones_x = 1;
  }
  for (int i = 0; i < n_ones_x; i++)
    xs[xn++] = 1;
  xs[xn++] = K;
  PolyUOp *xr = poly_reshape(ctx, x, xs, xn);

  int64_t ws[POLY_MAX_DIMS];
  int wn = 0;
  for (int i = 0; i < w_ndim - 2; i++)
    ws[wn++] = w_shape[i];
  for (int i = 0; i < n_ones_x; i++)
    ws[wn++] = 1;
  for (int i = axis_w; i < w_ndim; i++)
    ws[wn++] = w_shape[i];
  PolyUOp *wr = poly_reshape(ctx, w, ws, wn);

  int new_axis_w = wn - 2;
  if (new_axis_w < 0) new_axis_w = 0;
  int64_t perm[POLY_MAX_DIMS];
  for (int i = 0; i < wn; i++)
    perm[i] = i;
  perm[wn - 1] = new_axis_w;
  perm[new_axis_w] = wn - 1;
  PolyUOp *wt = poly_permute(ctx, wr, perm, wn);
  int64_t wt_shape[POLY_MAX_DIMS];
  for (int i = 0; i < wn; i++)
    wt_shape[i] = ws[perm[i]];

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
    for (int i = 0; i < pad; i++)
      padded[i] = 1;
    for (int i = 0; i < xn; i++)
      padded[pad + i] = xs[i];
    x_exp = poly_reshape(ctx, xr, padded, max_ndim);
  } else {
    x_exp = xr;
  }
  if (wn < max_ndim) {
    int64_t padded[POLY_MAX_DIMS];
    int pad = max_ndim - wn;
    for (int i = 0; i < pad; i++)
      padded[i] = 1;
    for (int i = 0; i < wn; i++)
      padded[pad + i] = wt_shape[i];
    w_exp = poly_reshape(ctx, wt, padded, max_ndim);
  } else {
    w_exp = wt;
  }
  x_exp = poly_expand(ctx, x_exp, bc_shape, max_ndim);
  w_exp = poly_expand(ctx, w_exp, bc_shape, max_ndim);

  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, x_exp, w_exp);
  int64_t sum_axis[] = {max_ndim - 1};
  PolyUOp *summed = poly_reduce_axis(ctx, POLY_OP_ADD, mul, sum_axis, 1);

  int on = 0;
  for (int i = 0; i < max_ndim - 1; i++) {
    out_shape[on++] = bc_shape[i];
  }
  if (on == 0) {
    out_shape[0] = 1;
    on = 1;
  }
  out_ndim = on;

  return poly_reshape(ctx, summed, out_shape, on);
}

/* Softmax */

PolyUOp *poly_softmax(PolyCtx *ctx, PolyUOp *x, int axis) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;

  int64_t max_shape[8];
  int max_ndim;
  PolyUOp *m = do_reduce(ctx, POLY_OP_MAX, x_view, shape, ndim, axis, 1, max_shape, &max_ndim);
  PolyUOp *m_exp = poly_expand(ctx, m, (int64_t *)shape, ndim);
  /* tinygrad Tensor._softmax builds subtraction through Tensor.sub, which is
   * `a + (b * -1)`. Preserve that high-level graph shape here instead of
   * emitting a direct SUB: vector/broadcast lowering keeps the repeated
   * multiply-by-minus-one form through the final renderer pass. */
  PolyUOp *neg_m =
      poly_alu2(ctx, POLY_OP_MUL, m_exp, poly_uop0(ctx, POLY_OP_CONST, m_exp->dtype, poly_arg_float(-1.0)));
  PolyUOp *shifted = poly_alu2(ctx, POLY_OP_ADD, x_view, neg_m);
  PolyUOp *e = poly_exp(ctx, shifted);

  int64_t sum_shape[8];
  int sum_ndim;
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, e, shape, ndim, axis, 1, sum_shape, &sum_ndim);
  PolyUOp *s_exp = poly_expand(ctx, s, (int64_t *)shape, ndim);

  return poly_alu2(ctx, POLY_OP_FDIV, e, s_exp);
}

PolyUOp *poly_log_softmax(PolyCtx *ctx, PolyUOp *x, int axis) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;

  int64_t max_shape[8];
  int max_ndim;
  PolyUOp *m = do_reduce(ctx, POLY_OP_MAX, x_view, shape, ndim, axis, 1, max_shape, &max_ndim);
  PolyUOp *m_exp = poly_expand(ctx, m, (int64_t *)shape, ndim);
  PolyUOp *shifted = poly_alu2(ctx, POLY_OP_SUB, x_view, m_exp);
  PolyUOp *e = poly_exp(ctx, shifted);

  int64_t sum_shape[8];
  int sum_ndim;
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, e, shape, ndim, axis, 1, sum_shape, &sum_ndim);
  PolyUOp *log_s = poly_log(ctx, s);
  PolyUOp *log_s_exp = poly_expand(ctx, log_s, (int64_t *)shape, ndim);

  return poly_alu2(ctx, POLY_OP_SUB, shifted, log_s_exp);
}

PolyUOp *poly_cross_entropy(PolyCtx *ctx, PolyUOp *logits, PolyUOp *target, int axis) {
  if (!ctx || !logits || !target) return NULL;
  int64_t logits_shape[POLY_MAX_DIMS], target_shape[POLY_MAX_DIMS];
  int logits_ndim = uop_shape(ctx, logits, logits_shape);
  int target_ndim = uop_shape(ctx, target, target_shape);
  if (logits_ndim < 1 || logits_ndim > POLY_MAX_DIMS || target_ndim < 0 ||
      target_ndim > POLY_MAX_DIMS)
    return NULL;

  if (axis < 0) axis += logits_ndim;
  if (axis < 0 || axis >= logits_ndim) return NULL;

  const bool dense_targets = poly_shape_equal(logits_shape, logits_ndim, target_shape, target_ndim);
  const bool sparse_targets =
      shape_equal_except_axis(logits_shape, logits_ndim, target_shape, target_ndim, axis);
  if (!dense_targets && !sparse_targets) return NULL;

  PolyUOp *weights =
      dense_targets ? poly_reshape(ctx, target, (int64_t *)target_shape, target_ndim) : target;
  if (sparse_targets) {
    const int64_t classes = logits_shape[axis];
    int arange_dtype_id =
        (classes > (int64_t)INT32_MAX) ? poly_dtype_id_by_name("int64")
                                       : poly_dtype_id_by_name("int32");
    const PolyDType *class_dt_ptr = ffi_dtype_from_id(arange_dtype_id);
    if (!class_dt_ptr) return NULL;
    PolyDType class_dt = poly_dtype_scalar(*class_dt_ptr);
    PolyUOp *target_idx = poly_dtype_eq(poly_dtype_scalar(target->dtype), class_dt)
                              ? target
                              : poly_cast(ctx, target, class_dt);

    int64_t target_us_shape[POLY_MAX_DIMS];
    for (int i = 0; i < axis; i++)
      target_us_shape[i] = target_shape[i];
    target_us_shape[axis] = 1;
    for (int i = axis; i < target_ndim; i++)
      target_us_shape[i + 1] = target_shape[i];

    PolyUOp *target_us = poly_reshape(ctx, target_idx, target_us_shape, target_ndim + 1);
    PolyUOp *target_exp = poly_expand(ctx, target_us, (int64_t *)logits_shape, logits_ndim);

    PolyUOp *classes_uop = poly_arange_int_by_id(ctx, 0, classes, 1, arange_dtype_id);
    int64_t classes_shape[POLY_MAX_DIMS];
    for (int i = 0; i < logits_ndim; i++)
      classes_shape[i] = 1;
    classes_shape[axis] = classes;
    PolyUOp *classes_r = poly_reshape(ctx, classes_uop, classes_shape, logits_ndim);
    PolyUOp *classes_exp = poly_expand(ctx, classes_r, (int64_t *)logits_shape, logits_ndim);

    weights = poly_eq(ctx, target_exp, classes_exp);
  }

  PolyUOp *log_probs = poly_log_softmax(ctx, logits, axis);
  /* Cast bool weights to f32 before multiply (tinygrad does this via
   * Tensor._broadcasted dtype promotion; polygrad's C-level ALU doesn't
   * auto-promote, so explicit CAST is needed). */
  if (poly_dtype_is_bool(weights->dtype))
    weights = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, weights, poly_arg_none());
  PolyUOp *weighted = poly_alu2(ctx, POLY_OP_MUL, log_probs, weights);

  int64_t per_sample_shape[POLY_MAX_DIMS];
  int per_sample_ndim = 0;
  PolyUOp *per_sample = do_reduce(
      ctx, POLY_OP_ADD, weighted, logits_shape, logits_ndim, axis, 0, per_sample_shape,
      &per_sample_ndim
  );

  PolyUOp *total = per_sample;
  if (per_sample_ndim > 0) {
    int64_t axes[POLY_MAX_DIMS];
    for (int i = 0; i < per_sample_ndim; i++)
      axes[i] = i;
    total = poly_reduce_axis(ctx, POLY_OP_ADD, per_sample, axes, per_sample_ndim);
  }

  int64_t denom = poly_shape_numel_checked(per_sample_shape, per_sample_ndim);
  if (denom <= 0) return NULL;

  /* tinygrad Tensor.div is multiply by reciprocal, and cross_entropy applies
   * the leading negation outside the mean reduction. For a static denominator
   * this reaches linear IR as MUL(total, -1/denom), not FDIV followed by NEG. */
  PolyUOp *scale = cf(ctx, log_probs, -1.0 / (double)denom);
  PolyUOp *loss = poly_alu2(ctx, POLY_OP_MUL, total, scale);
  return poly_reshape(ctx, loss, NULL, 0);
}

/* Einsum */

#define MAX_EINSUM_TENSORS 8

PolyUOp *poly_einsum(PolyCtx *ctx, const char *formula, PolyUOp **tensors, int n_tensors) {
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
    while (*pp && *pp != ',')
      pp++;
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
        if (s[k] == c) {
          ki = k;
          break;
        }
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
    if (spec_len == 0) {
      aligned[t] = x;
      continue;
    }

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
        if (spec[j] == sorted_spec[i]) {
          perm[i] = j;
          break;
        }
      if (perm[i] != i) needs_perm = true;
    }
    if (needs_perm) x = poly_permute(ctx, x, perm, spec_len);

    int64_t rshape[POLY_MAX_DIMS];
    for (int i = 0; i < n_alpha; i++) {
      bool found = false;
      for (int j = 0; j < spec_len; j++)
        if (sorted_spec[j] == alpha[i]) {
          found = true;
          break;
        }
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
      if (*r == alpha[i]) {
        in_rhs = true;
        break;
      }
    if (!in_rhs) sum_axes[n_sum++] = i;
  }
  if (n_sum > 0) result = poly_reduce_axis(ctx, POLY_OP_ADD, result, sum_axes, n_sum);

  char remaining[26];
  int n_remaining = 0;
  for (int i = 0; i < n_alpha; i++) {
    bool summed = false;
    for (int j = 0; j < n_sum; j++)
      if (sum_axes[j] == i) {
        summed = true;
        break;
      }
    if (!summed) remaining[n_remaining++] = alpha[i];
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
  if (needs_final_perm) result = poly_permute(ctx, result, out_perm, rhs_len);

  return result;
}

/* Rearrange (einops) */

#define MAX_REARRANGE_TOKENS 32

static int parse_rearrange_side(
    const char *s,
    char tokens[][32],
    int *n_tokens,
    int groups[][2],
    int *n_groups
) {
  *n_tokens = 0;
  *n_groups = 0;
  int paren_start = -1;

  const char *p = s;
  while (*p) {
    while (*p == ' ' || *p == '\t')
      p++;
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

static int64_t find_axis_size(
    const char *name,
    const char *axis_names,
    const int64_t *axis_values,
    int n_axis_sizes
) {
  if (!axis_names || n_axis_sizes <= 0) return -1;
  const char *p = axis_names;
  int idx = 0;
  while (*p && idx < n_axis_sizes) {
    while (*p == ' ')
      p++;
    if (!*p) break;
    const char *start = p;
    while (*p && *p != ' ')
      p++;
    int len = (int)(p - start);
    if ((int)strlen(name) == len && memcmp(start, name, len) == 0) return axis_values[idx];
    idx++;
  }
  return -1;
}

PolyUOp *poly_rearrange(
    PolyCtx *ctx,
    const char *formula,
    PolyUOp *x,
    const char *axis_names,
    const int64_t *axis_values,
    int n_axis_sizes
) {
  if (!formula || !x) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
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
    for (int i = 0; i < MAX_REARRANGE_TOKENS; i++)
      g_id[i] = -1;
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
          if (strcmp(nm, "1") == 0) {
            sub[i] = 1;
          } else {
            int64_t v = find_axis_size(nm, axis_names, axis_values, n_axis_sizes);
            if (v > 0)
              sub[i] = v;
            else {
              if (unk >= 0) return NULL;
              unk = i;
              sub[i] = -1;
            }
          }
          if (sub[i] > 0) known *= sub[i];
        }
        if (unk >= 0) {
          if (input_dim >= cur_ndim) return NULL;
          sub[unk] = cur_shape[input_dim] / known;
        }
        for (int i = 0; i < gc; i++)
          new_shape[new_ndim++] = sub[i];
        input_dim++;
        ti = ge;
      } else {
        if (strcmp(lhs_tok[ti], "1") == 0)
          new_shape[new_ndim++] = 1;
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
      if (strcmp(rhs_tok[i], lhs_tok[j]) == 0) {
        perm[i] = j;
        break;
      }
    if (perm[i] < 0) return NULL;
    if (perm[i] != i) need_perm = true;
  }
  if (need_perm) {
    result = poly_permute(ctx, result, perm, n_rt);
    int64_t ps[POLY_MAX_DIMS];
    for (int i = 0; i < n_rt; i++)
      ps[i] = cur_shape[perm[i]];
    memcpy(cur_shape, ps, n_rt * sizeof(int64_t));
    cur_ndim = n_rt;
  }

  /* Phase 3: Flatten (rhs groups, process right to left) */
  for (int g = n_rg - 1; g >= 0; g--) {
    int gs = rhs_grp[g][0], ge = rhs_grp[g][1];
    if (ge - gs <= 1) continue;
    int64_t flat = 1;
    for (int i = gs; i < ge && i < cur_ndim; i++)
      flat *= cur_shape[i];
    int64_t ns[POLY_MAX_DIMS];
    int nn = 0;
    for (int i = 0; i < gs; i++)
      ns[nn++] = cur_shape[i];
    ns[nn++] = flat;
    for (int i = ge; i < cur_ndim; i++)
      ns[nn++] = cur_shape[i];
    result = poly_reshape(ctx, result, ns, nn);
    memcpy(cur_shape, ns, nn * sizeof(int64_t));
    cur_ndim = nn;
  }

  return result;
}

/* Gather (embedding lookup) */

PolyUOp *poly_gather(PolyCtx *ctx, PolyUOp *table, PolyUOp *indices) {
  if (!ctx || !table || !indices) return NULL;
  int64_t table_shape[POLY_MAX_DIMS], idx_shape[POLY_MAX_DIMS];
  int table_ndim = uop_shape(ctx, table, table_shape);
  int idx_ndim = uop_shape(ctx, indices, idx_shape);
  if (table_ndim != 2 || idx_ndim < 1) return NULL;

  int64_t V = table_shape[0];
  int64_t D = table_shape[1];

  /* tinygrad tensor.py:_one_hot_along_dim chooses int32 unless num_classes
   * overflows int32, then int64. Keep gather on the same integer class range
   * so later collapse stages can see the same one-hot compare shape. */
  int arange_dtype_id =
      (V > (int64_t)INT32_MAX) ? poly_dtype_id_by_name("int64") : poly_dtype_id_by_name("int32");
  PolyUOp *arange_buf = poly_arange_int_by_id(ctx, 0, V, 1, arange_dtype_id);
  if (!arange_buf) return NULL;

  int64_t idx_us_shape[POLY_MAX_DIMS];
  int idx_us_ndim = idx_ndim + 1;
  for (int i = 0; i < idx_ndim; i++)
    idx_us_shape[i] = idx_shape[i];
  idx_us_shape[idx_ndim] = 1;
  PolyUOp *idx_us = poly_reshape(ctx, indices, idx_us_shape, idx_us_ndim);

  int64_t idx_bcast[POLY_MAX_DIMS];
  for (int i = 0; i < idx_ndim; i++)
    idx_bcast[i] = idx_shape[i];
  idx_bcast[idx_ndim] = V;
  PolyUOp *idx_exp = poly_expand(ctx, idx_us, idx_bcast, idx_us_ndim);

  int64_t arange_shape[POLY_MAX_DIMS];
  int arange_ndim = idx_us_ndim;
  for (int i = 0; i < idx_ndim; i++)
    arange_shape[i] = 1;
  arange_shape[idx_ndim] = V;
  PolyUOp *arange_r = poly_reshape(ctx, arange_buf, arange_shape, arange_ndim);
  PolyUOp *arange_exp = poly_expand(ctx, arange_r, idx_bcast, arange_ndim);

  PolyUOp *mask = poly_eq(ctx, idx_exp, arange_exp);

  int64_t mask_us_shape[POLY_MAX_DIMS];
  int mask_us_ndim = idx_us_ndim + 1;
  for (int i = 0; i < idx_us_ndim; i++)
    mask_us_shape[i] = idx_bcast[i];
  mask_us_shape[idx_us_ndim] = 1;
  PolyUOp *mask_us = poly_reshape(ctx, mask, mask_us_shape, mask_us_ndim);

  int64_t mask_bcast[POLY_MAX_DIMS];
  for (int i = 0; i < idx_us_ndim; i++)
    mask_bcast[i] = idx_bcast[i];
  mask_bcast[idx_us_ndim] = D;
  PolyUOp *mask_exp = poly_expand(ctx, mask_us, mask_bcast, mask_us_ndim);

  int64_t tbl_shape[POLY_MAX_DIMS];
  int tbl_ndim = mask_us_ndim;
  for (int i = 0; i < idx_ndim; i++)
    tbl_shape[i] = 1;
  tbl_shape[idx_ndim] = V;
  tbl_shape[idx_ndim + 1] = D;
  PolyUOp *tbl_r = poly_reshape(ctx, table, tbl_shape, tbl_ndim);
  PolyUOp *tbl_exp = poly_expand(ctx, tbl_r, mask_bcast, tbl_ndim);

  PolyUOp *zero = cf(ctx, tbl_exp, 0.0);
  PolyUOp *selected = poly_where_op(ctx, mask_exp, tbl_exp, zero);

  int64_t reduce_axes[] = {idx_ndim};
  PolyUOp *gathered = poly_reduce_axis(ctx, POLY_OP_ADD, selected, reduce_axes, 1);

  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = idx_ndim + 1;
  for (int i = 0; i < idx_ndim; i++)
    out_shape[i] = idx_shape[i];
  out_shape[idx_ndim] = D;

  return poly_reshape(ctx, gathered, out_shape, out_ndim);
}

/* Additional composed ops */

PolyUOp *poly_rope(PolyCtx *ctx, PolyUOp *x, PolyUOp *freqs_cos, PolyUOp *freqs_sin) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 1) return NULL;
  int64_t half_dim = shape[ndim - 1] / 2;
  if (half_dim <= 0) return NULL;

  int64_t pairs1[POLY_MAX_DIMS][2], pairs2[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim - 1; i++) {
    pairs1[i][0] = 0;
    pairs1[i][1] = shape[i];
    pairs2[i][0] = 0;
    pairs2[i][1] = shape[i];
  }
  pairs1[ndim - 1][0] = 0;
  pairs1[ndim - 1][1] = half_dim;
  pairs2[ndim - 1][0] = half_dim;
  pairs2[ndim - 1][1] = shape[ndim - 1];

  PolyUOp *x1 = poly_shrink(ctx, x, pairs1, ndim);
  PolyUOp *x2 = poly_shrink(ctx, x, pairs2, ndim);

  PolyUOp *r1 = poly_alu2(
      ctx, POLY_OP_SUB, poly_alu2(ctx, POLY_OP_MUL, x1, freqs_cos),
      poly_alu2(ctx, POLY_OP_MUL, x2, freqs_sin)
  );
  PolyUOp *r2 = poly_alu2(
      ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, x2, freqs_cos),
      poly_alu2(ctx, POLY_OP_MUL, x1, freqs_sin)
  );

  int64_t pad1[POLY_MAX_DIMS][2], pad2[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    pad1[i][0] = 0;
    pad1[i][1] = 0;
    pad2[i][0] = 0;
    pad2[i][1] = 0;
  }
  pad1[ndim - 1][1] = half_dim;
  pad2[ndim - 1][0] = half_dim;

  return poly_alu2(ctx, POLY_OP_ADD, poly_pad(ctx, r1, pad1, ndim), poly_pad(ctx, r2, pad2, ndim));
}

PolyUOp *poly_repeat_interleave(PolyCtx *ctx, PolyUOp *x, int repeats, int dim) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 1 || repeats <= 0) return NULL;
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) return NULL;

  int64_t ins[POLY_MAX_DIMS];
  int ins_ndim = ndim + 1;
  for (int i = 0; i <= dim; i++)
    ins[i] = shape[i];
  ins[dim + 1] = 1;
  for (int i = dim + 1; i < ndim; i++)
    ins[i + 1] = shape[i];
  PolyUOp *r = poly_reshape(ctx, x, ins, ins_ndim);

  int64_t exp[POLY_MAX_DIMS];
  memcpy(exp, ins, ins_ndim * sizeof(int64_t));
  exp[dim + 1] = repeats;
  r = poly_expand(ctx, r, exp, ins_ndim);

  int64_t flat[POLY_MAX_DIMS];
  for (int i = 0; i < dim; i++)
    flat[i] = shape[i];
  flat[dim] = shape[dim] * repeats;
  for (int i = dim + 1; i < ndim; i++)
    flat[i] = shape[i];
  return poly_reshape(ctx, r, flat, ndim);
}

PolyUOp *poly_argmax(PolyCtx *ctx, PolyUOp *x, int axis) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 1) return NULL;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) return NULL;

  int64_t N = shape[axis];

  int64_t max_shape[POLY_MAX_DIMS];
  int max_ndim;
  PolyUOp *x_max = do_reduce(ctx, POLY_OP_MAX, x, shape, ndim, axis, 1, max_shape, &max_ndim);
  PolyUOp *x_max_bc = poly_expand(ctx, x_max, shape, ndim);
  PolyUOp *m = poly_eq(ctx, x, x_max_bc);

  PolyUOp *m_f = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, m, poly_arg_none());

  PolyUOp *rng = poly_arange(ctx, 0.0, (double)N, 1.0);
  PolyUOp *desc = poly_alu2(ctx, POLY_OP_SUB, poly_const_float(ctx, (double)N), rng);

  int64_t bc[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    bc[i] = 1;
  bc[axis] = N;
  desc = poly_reshape(ctx, desc, bc, ndim);
  desc = poly_expand(ctx, desc, shape, ndim);

  PolyUOp *idx = poly_alu2(ctx, POLY_OP_MUL, m_f, desc);

  int64_t idx_max_shape[POLY_MAX_DIMS];
  int idx_max_ndim;
  PolyUOp *idx_max =
      do_reduce(ctx, POLY_OP_MAX, idx, shape, ndim, axis, 0, idx_max_shape, &idx_max_ndim);
  PolyUOp *result = poly_alu2(ctx, POLY_OP_SUB, poly_const_float(ctx, (double)N), idx_max);
  return poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, result, poly_arg_none());
}

PolyUOp *poly_mse_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, pred, shape);
  if (ndim < 0) return NULL;
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_SUB, pred, target);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim;
  PolyUOp *r = sq;
  for (int i = ndim - 1; i >= 0; i--) {
    int64_t s[POLY_MAX_DIMS];
    int sn;
    sn = uop_shape(ctx, r, s);
    r = do_reduce(ctx, POLY_OP_ADD, r, s, sn, i, 0, out_shape, &out_ndim);
  }
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++)
    numel *= shape[i];
  return poly_alu2(ctx, POLY_OP_FDIV, r, poly_const_float(ctx, (double)numel));
}

PolyUOp *poly_mae_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, pred, shape);
  if (ndim < 0) return NULL;
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_SUB, pred, target);
  PolyUOp *absdiff = poly_abs(ctx, diff);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim;
  PolyUOp *r = absdiff;
  for (int i = ndim - 1; i >= 0; i--) {
    int64_t s[POLY_MAX_DIMS];
    int sn;
    sn = uop_shape(ctx, r, s);
    r = do_reduce(ctx, POLY_OP_ADD, r, s, sn, i, 0, out_shape, &out_ndim);
  }
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++)
    numel *= shape[i];
  return poly_alu2(ctx, POLY_OP_FDIV, r, poly_const_float(ctx, (double)numel));
}
