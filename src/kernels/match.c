#include "kernels/kernels.h"
#include "mixin/composite.h"
#include <limits.h>
#include <string.h>

static bool fixed_shape(PolyCtx *ctx, PolyUOp *u, int nd, int64_t *shape) {
  for (int i = 0; i < nd; i++) {
    PolyUOp *d = poly_uop_shape_dim(ctx, u, i);
    if (!d || d->op != POLY_OP_CONST || d->arg.kind != POLY_ARG_INT || d->arg.i <= 0) return false;
    shape[i] = d->arg.i;
  }
  return true;
}

static void match_probabilities(PolyCtx *ctx, PolyGemmDesc *d) {
  if (d->a->op != POLY_OP_RESHAPE || d->a->n_src != 2) return;
  PolyUOp *p = d->a->src[0];
  if (!p || p->op != POLY_OP_MUL || p->n_src != 2) return;
  PolyUOp *e = p->src[0], *r = p->src[1];
  if (e->op == POLY_OP_RECIPROCAL) {
    e = p->src[1];
    r = p->src[0];
  }
  if (e->op != POLY_OP_EXP2 || r->op != POLY_OP_RECIPROCAL || r->n_src != 1) return;
  int nd = poly_uop_ndim(ctx, e);
  int64_t shape[POLY_MAX_DIMS], count = 1;
  if (nd < 2 || nd > POLY_MAX_DIMS || !fixed_shape(ctx, e, nd, shape)) return;
  for (int i = 0; i < nd; i++) {
    if (shape[i] > INT64_MAX / count) return;
    count *= shape[i];
  }
  if (r->src[0] != poly_uop_sum_reduce(ctx, e, nd - 1, 1)) return;
  d->probabilities = p;
  d->probability_elements = count;
}

/* Canonical dot: reduce the K prefix of permute(A[...,1,K] * B[...,N,K]).
 * No target, tile or training-mode restrictions belong in recognition. */
static bool match_dot(PolyCtx *ctx, PolyUOp *u, PolyGemmDesc *d) {
  memset(d, 0, sizeof(*d));
  if (u->op != POLY_OP_REDUCE || u->arg.kind != POLY_ARG_REDUCE ||
      u->arg.reduce.op != POLY_OP_ADD || u->arg.reduce.num_axes != 1 || u->n_src != 1 ||
      u->src[0]->op != POLY_OP_PERMUTE)
    return false;
  PolyUOp *order = u->src[0];
  if (order->n_src != 1 || order->arg.kind != POLY_ARG_INT_TUPLE ||
      order->src[0]->op != POLY_OP_MUL || order->src[0]->n_src != 2)
    return false;
  d->root = u;
  d->a = order->src[0]->src[0];
  d->b = order->src[0]->src[1];
  d->ad = poly_uop_ndim(ctx, d->a);
  d->bd = poly_uop_ndim(ctx, d->b);
  if (d->ad < 3 || d->ad > POLY_MAX_DIMS || d->bd < 2 || d->bd > d->ad ||
      order->arg.int_tuple.n != d->ad || order->arg.int_tuple.vals[0] != d->ad - 1)
    return false;
  for (int i = 1; i < d->ad; i++)
    if (order->arg.int_tuple.vals[i] != i - 1) return false;
  if (!fixed_shape(ctx, d->a, d->ad, d->as) || !fixed_shape(ctx, d->b, d->bd, d->bs)) return false;
  d->K = d->as[d->ad - 1];
  d->N = d->bs[d->bd - 2];
  if (d->as[d->ad - 2] != 1 || d->bs[d->bd - 1] != d->K) return false;
  d->M = 1;
  for (int i = 0; i < d->ad - 2; i++) {
    if (d->as[i] > INT64_MAX / d->M) return false;
    d->M *= d->as[i];
  }
  d->device = poly_uop_device_name(ctx, d->a);
  const char *other = poly_uop_device_name(ctx, d->b);
  if (!d->device || !other || strcmp(d->device, other)) return false;
  d->a_base = d->a;
  while (d->a_base->op == POLY_OP_RESHAPE || d->a_base->op == POLY_OP_PERMUTE)
    d->a_base = d->a_base->src[0];
  match_probabilities(ctx, d);
  return true;
}

/* Prove a broadcast axis through movement, not from equal runtime values.
 * Opaque/computed producers conservatively vary along every non-singleton axis. */
static bool broadcast_axis(PolyCtx *ctx, PolyUOp *u, int axis) {
  if (axis < 0) return true;
  PolyUOp *dim = poly_uop_shape_dim(ctx, u, axis);
  if (!dim || dim->op != POLY_OP_CONST || dim->arg.kind != POLY_ARG_INT) return false;
  if (dim->arg.i == 1) return true;
  if (u->op == POLY_OP_PERMUTE && u->arg.kind == POLY_ARG_INT_TUPLE)
    return broadcast_axis(ctx, u->src[0], u->arg.int_tuple.vals[axis]);
  if (u->op == POLY_OP_EXPAND)
    return broadcast_axis(
        ctx, u->src[0], axis - poly_uop_ndim(ctx, u) + poly_uop_ndim(ctx, u->src[0])
    );
  return false;
}

/* Pinned mixin/gradient.py differentiates MUL/broadcast/REDUCE, not a named
 * matmul. Recognize the resulting rank-three contraction without changing
 * autograd: sum_K A[K,M,1] * B[K,1,N]. The reduction axis stays in order. */
static bool match_contraction(PolyCtx *ctx, PolyUOp *u, PolyGemmDesc *d) {
  if (u->op != POLY_OP_REDUCE || u->arg.kind != POLY_ARG_REDUCE ||
      u->arg.reduce.op != POLY_OP_ADD || u->arg.reduce.num_axes != 1 || u->n_src != 1)
    return false;
  PolyUOp *product = u->src[0];
  int64_t order[] = {0, 1, 2}, shape[3];
  if (poly_uop_ndim(ctx, product) != 3 || !fixed_shape(ctx, product, 3, shape)) return false;
  if (product->op == POLY_OP_PERMUTE && product->arg.kind == POLY_ARG_INT_TUPLE &&
      product->arg.int_tuple.n == 3) {
    memcpy(order, product->arg.int_tuple.vals, sizeof(order));
    product = product->src[0];
  }
  if (product->op != POLY_OP_MUL || product->n_src != 2) return false;
  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *a = product->src[swap], *b = product->src[1 - swap];
    int ad = poly_uop_ndim(ctx, a), bd = poly_uop_ndim(ctx, b);
    if (ad < 0 || ad > 3 || bd < 0 || bd > 3 || !broadcast_axis(ctx, a, order[2] - 3 + ad) ||
        !broadcast_axis(ctx, b, order[1] - 3 + bd))
      continue;
    const char *device = poly_uop_device_name(ctx, a), *other = poly_uop_device_name(ctx, b);
    if (!device || !other || strcmp(device, other)) continue;
    int64_t full[3];
    if (!fixed_shape(ctx, product, 3, full)) return false;
    a = poly_uop_expand(ctx, a, full, 3);
    b = poly_uop_expand(ctx, b, full, 3);
    a = a ? poly_uop_permute(ctx, a, order, 3) : NULL;
    b = b ? poly_uop_permute(ctx, b, order, 3) : NULL;
    a = a ? poly_uop_shrink_to(ctx, a, (int64_t[]){shape[0], shape[1], 1}, 3) : NULL;
    b = b ? poly_uop_shrink_to(ctx, b, (int64_t[]){shape[0], 1, shape[2]}, 3) : NULL;
    a = a ? poly_uop_permute(ctx, a, (int64_t[]){1, 2, 0}, 3) : NULL;
    b = b ? poly_uop_permute(ctx, b, (int64_t[]){1, 2, 0}, 3) : NULL;
    b = b ? poly_uop_reshape(ctx, b, (int64_t[]){shape[2], shape[0]}, 2) : NULL;
    if (!a || !b) return false;
    memset(d, 0, sizeof(*d));
    d->root = u;
    d->a = a;
    d->b = b;
    d->device = device;
    d->M = shape[1];
    d->N = shape[2];
    d->K = shape[0];
    d->ad = 3;
    d->bd = 2;
    d->as[0] = d->M;
    d->as[1] = 1;
    d->as[2] = d->K;
    d->bs[0] = d->N;
    d->bs[1] = d->K;
    d->a_base = a->src[0]; /* compact SHRINK; do not materialize its expanded parent */
    return true;
  }
  return false;
}

bool poly_kernel_match_gemm(PolyCtx *ctx, PolyUOp *u, PolyGemmDesc *d) {
  return match_dot(ctx, u, d) || match_contraction(ctx, u, d);
}
