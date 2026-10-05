#include "kernels/kernels.h"
#include "ctx.h"
#include "mixin/creation.h"
#include "mixin/elementwise.h"
#include "mixin/movement.h"
#include <limits.h>
#include <string.h>

static bool shape3(PolyCtx *ctx, PolyUOp *u, int64_t *dims) {
  if (poly_uop_ndim(ctx, u) != 3) return false;
  for (int i = 0; i < 3; i++) {
    PolyUOp *d = poly_uop_shape_dim(ctx, u, i);
    if (!d || d->op != POLY_OP_CONST || d->arg.kind != POLY_ARG_INT || d->arg.i < 1) return false;
    dims[i] = d->arg.i;
  }
  return true;
}

/* Recognize ordinary Tensor operations, not a name/annotation supplied by a
 * producer: sum(where((arange(N)>=lo) & (arange(N)<hi), values, 0), axis=1).
 * Broadcasting is limited to [S,1,1] bounds and [1,N,W] values. */
bool poly_kernel_match_segment(PolyCtx *ctx, PolyUOp *u, PolySegmentDesc *d) {
  if (u->op != POLY_OP_REDUCE || u->arg.kind != POLY_ARG_REDUCE ||
      u->arg.reduce.op != POLY_OP_ADD || u->arg.reduce.num_axes != 1 || u->n_src != 1 ||
      !poly_dtype_eq(u->dtype, POLY_FLOAT32))
    return false;
  PolyUOp *p = u->src[0];
  if (p->op != POLY_OP_PERMUTE || p->arg.kind != POLY_ARG_INT_TUPLE || p->arg.int_tuple.n != 3 ||
      p->arg.int_tuple.vals[0] != 1 || p->arg.int_tuple.vals[1] != 0 ||
      p->arg.int_tuple.vals[2] != 2)
    return false;
  PolyUOp *w = p->src[0];
  if (w->op != POLY_OP_WHERE || w->n_src != 3 || w->src[2]->op != POLY_OP_CONST) return false;
  PolyArg z = w->src[2]->arg;
  if (!((z.kind == POLY_ARG_FLOAT && z.f == 0) || (z.kind == POLY_ARG_INT && z.i == 0)))
    return false;
  PolyUOp *mask = w->src[0];
  if (mask->op != POLY_OP_AND || mask->n_src != 2) return false;
  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *lower = mask->src[swap], *upper = mask->src[1 - swap];
    if (lower->op != POLY_OP_CMPNE || lower->n_src != 2) continue;
    PolyUOp *one = lower->src[1];
    while (one->op == POLY_OP_RESHAPE || one->op == POLY_OP_EXPAND)
      one = one->src[0];
    if (one->op != POLY_OP_CONST || one->arg.kind != POLY_ARG_BOOL || !one->arg.b) continue;
    lower = lower->src[0];
    if (lower->op != POLY_OP_CMPLT || upper->op != POLY_OP_CMPLT || lower->src[0] != upper->src[0])
      continue;
    PolyUOp *idx = lower->src[0], *lo = lower->src[1], *hi = upper->src[1], *v = w->src[1];
    int64_t vs[3], ls[3], hs[3];
    if (!poly_dtype_eq(idx->dtype, POLY_INT32) || !poly_dtype_eq(lo->dtype, POLY_INT32) ||
        !poly_dtype_eq(hi->dtype, POLY_INT32) || !poly_dtype_eq(v->dtype, POLY_FLOAT32) ||
        !shape3(ctx, v, vs) || !shape3(ctx, lo, ls) || !shape3(ctx, hi, hs) || vs[0] != 1 ||
        ls[1] != 1 || ls[2] != 1 || memcmp(ls, hs, sizeof(ls)))
      continue;
    if (vs[1] > INT_MAX / 4 / vs[2] || ls[0] > INT_MAX / 4 / vs[2]) continue;
    PolyUOp *expected = poly_uop_reshape(
        ctx, poly_uop_arange_int_dtype(ctx, 0, vs[1], 1, POLY_INT32), (int64_t[]){1, vs[1], 1}, 3
    );
    if (idx != expected) continue;
    const char *device = poly_uop_device_name(ctx, v);
    const char *ld = poly_uop_device_name(ctx, lo), *hd = poly_uop_device_name(ctx, hi);
    if (!device || !ld || !hd || strcmp(device, ld) || strcmp(device, hd)) continue;
    *d = (PolySegmentDesc){v, lo, hi, vs[1], ls[0], vs[2], device};
    return true;
  }
  return false;
}

static PolyUOp *ci(PolyCtx *ctx, int64_t n) {
  return poly_uop_const_int(ctx, n);
}
static PolyUOp *at(PolyCtx *ctx, PolyUOp *p, PolyUOp *i) {
  return poly_uop_index(ctx, p, &i, 1);
}
static PolyUOp *ld(PolyCtx *ctx, PolyUOp *p, PolyUOp *i) {
  return poly_uop_load(ctx, at(ctx, p, i));
}
static PolyUOp *param(PolyCtx *ctx, int64_t size, PolyDType dtype, int slot) {
  return poly_uop_placeholder(ctx, &size, 1, dtype, slot, POLY_ADDR_GLOBAL, NULL, false);
}
static PolyUOp *clamp(PolyCtx *ctx, PolyUOp *x, int64_t n) {
  return poly_uop_minimum(ctx, poly_uop_alu2(ctx, POLY_OP_MAX, x, ci(ctx, 0)), ci(ctx, n));
}

PolyUOp *poly_kernel_segment_lower(PolyCtx *ctx, const PolySegmentDesc *d) {
  const bool gpu = !strcmp(d->device, "CUDA") || !strcmp(d->device, "WEBGPU");
  int64_t n = d->rows, s = d->segments, width = d->width;
  PolyUOp *out = param(ctx, s * width, POLY_FLOAT32, 0),
          *v = param(ctx, n * width, POLY_FLOAT32, 1);
  PolyUOp *lo = param(ctx, s, POLY_INT32, 2), *hi = param(ctx, s, POLY_INT32, 3);
  PolyUOp *row = poly_uop_range(ctx, s, 0, gpu ? POLY_AXIS_GLOBAL : POLY_AXIS_LOOP);
  PolyUOp *col = poly_uop_range(ctx, width, 1, POLY_AXIS_LOOP);
  PolyUOp *idx = poly_uop_add(ctx, poly_uop_mul(ctx, row, ci(ctx, width)), col);
  /* This is exactly the mask's domain, not an unchecked CSR contract. Negative
   * or oversized bounds clip; reversed/empty intervals sum to zero. */
  PolyUOp *start = clamp(ctx, ld(ctx, lo, row), n), *stop = clamp(ctx, ld(ctx, hi, row), n);
  PolyUOp *length = poly_uop_alu2(ctx, POLY_OP_MAX, poly_uop_sub(ctx, stop, start), ci(ctx, 0));
  PolyUOp *k = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, poly_uop_cast(ctx, length, POLY_WEAKINT),
      poly_arg_range(2, POLY_AXIS_REDUCE)
  );
  PolyUOp *reg =
      poly_uop_placeholder(ctx, (int64_t[]){1}, 1, POLY_FLOAT32, 0, POLY_ADDR_REG, NULL, false);
  PolyUOp *init =
      poly_uop_store_val(ctx, poly_uop_after(ctx, reg, idx), poly_uop_const_float(ctx, 0));
  PolyUOp *acc = poly_uop_after(ctx, reg, init);
  PolyUOp *vi =
      poly_uop_add(ctx, poly_uop_mul(ctx, poly_uop_add(ctx, start, k), ci(ctx, width)), col);
  PolyUOp *sum =
      poly_uop_add(ctx, ld(ctx, poly_uop_after(ctx, acc, k), ci(ctx, 0)), ld(ctx, v, vi));
  PolyUOp *done = poly_uop_end(ctx, poly_uop_store_val(ctx, acc, sum), &k, 1);
  PolyUOp *store =
      poly_uop_store(ctx, at(ctx, out, idx), ld(ctx, poly_uop_after(ctx, acc, done), ci(ctx, 0)));
  PolyUOp *end = poly_uop_end(ctx, store, (PolyUOp *[]){row, col}, 2);
  /* Runtime bounds cannot be estimated symbolically. Use the worst case:
   * intervals may overlap, so a partition-only estimate would be incorrect. */
  int64_t work = n * s * width;
  PolyEstimates estimates = {
      ci(ctx, work), ci(ctx, (work + 3 * s * width) * 4), ci(ctx, (work + 3 * s * width) * 4)};
  PolyKernelInfo info = {.name = "segment_sum", .has_opts_to_apply = true, .estimates = &estimates};
  PolyUOp *body = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_kernel_info(&info));
  PolyUOp *dev = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_str(d->device));
  PolyUOp *buffer =
      poly_uop_new_buffer(ctx, dev, s * width, POLY_FLOAT32, poly_ctx_next_unique_id(ctx));
  PolyUOp *values =
      poly_uop_contiguous(ctx, poly_uop_reshape(ctx, d->values, (int64_t[]){n * width}, 1));
  PolyUOp *starts = poly_uop_contiguous(ctx, poly_uop_reshape(ctx, d->starts, &s, 1));
  PolyUOp *stops = poly_uop_contiguous(ctx, poly_uop_reshape(ctx, d->stops, &s, 1));
  if (!body || !buffer || !values || !starts || !stops) return NULL;
  PolyCallInfo call_info = {0};
  PolyUOp *src[] = {body, buffer, values, starts, stops};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, 5, poly_arg_call_info(&call_info));
  return poly_uop_reshape(ctx, poly_uop_after(ctx, buffer, call), (int64_t[]){s, width}, 2);
}
