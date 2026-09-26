#include "kernels/kernels.h"
#include "ctx.h"
#include "mixin/elementwise.h"
#include "mixin/movement.h"
#include <string.h>

static PolyUOp *ci(PolyCtx *c, int n) {
  return poly_uop_const_int(c, n);
}
static PolyUOp *add(PolyCtx *c, PolyUOp *a, PolyUOp *b) {
  return poly_uop_add(c, a, b);
}
static PolyUOp *mul(PolyCtx *c, PolyUOp *a, int n) {
  return poly_uop_mul(c, a, ci(c, n));
}
static PolyUOp *at(PolyCtx *c, PolyUOp *p, PolyUOp *i) {
  return poly_uop_index(c, p, &i, 1);
}
static PolyUOp *ld(PolyCtx *c, PolyUOp *p, PolyUOp *i) {
  return poly_uop_load(c, at(c, p, i));
}
static PolyUOp *buffer(PolyCtx *c, int size, int slot, PolyAddrSpace space) {
  return poly_uop_placeholder(c, (int64_t[]){size}, 1, POLY_FLOAT32, slot, space, NULL, false);
}

/* 128 invocations, each owning 4x4 outputs. LOCAL stores -> barrier ->
 * register updates -> barrier mirrors llm/kernels/amd.py:flash_attention.
 * The second barrier prevents a fast invocation overwriting the next tile
 * while a peer still reads this one. Both barriers are workgroup-uniform. */
static PolyUOp *kernel(PolyCtx *c, int M, int N, int K, PolyUOp *a, PolyUOp *b) {
  PolyUOp *out = buffer(c, M * N, 0, POLY_ADDR_GLOBAL);
  PolyUOp *gm = poly_uop_range(c, M / 32, 0, POLY_AXIS_GLOBAL);
  PolyUOp *gn = poly_uop_range(c, N / 64, 1, POLY_AXIS_GLOBAL);
  /* Keep contiguous columns on local_id.x, including after range splitting. */
  PolyUOp *tn = poly_uop_range(c, 16, 2, POLY_AXIS_LOCAL);
  PolyUOp *tm = poly_uop_range(c, 8, 3, POLY_AXIS_LOCAL);
  PolyUOp *tid = add(c, tn, mul(c, tm, 16));
  PolyUOp *ko = poly_uop_range(c, K / 32, 4, POLY_AXIS_REDUCE);
  PolyUOp *ki = poly_uop_range(c, 32, 5, POLY_AXIS_REDUCE);
  /* Pad A's rows so simultaneous broadcasts from different rows do not
   * address the same shared-memory bank. Padding never reaches global data. */
  PolyUOp *sa = buffer(c, 32 * 33, 0, POLY_ADDR_LOCAL);
  PolyUOp *sb = buffer(c, 32 * 64, 1, POLY_ADDR_LOCAL);
  PolyUOp *copies[24];
  for (int i = 0; i < 8; i++) {
    PolyUOp *idx = add(c, tid, ci(c, i * 128));
    PolyUOp *row = add(c, mul(c, gm, 32), poly_uop_alu2(c, POLY_OP_FLOORDIV, idx, ci(c, 32)));
    PolyUOp *k = add(c, mul(c, ko, 32), poly_uop_alu2(c, POLY_OP_FLOORMOD, idx, ci(c, 32)));
    PolyUOp *dst = add(c, idx, poly_uop_alu2(c, POLY_OP_FLOORDIV, idx, ci(c, 32)));
    copies[i] = poly_uop_store(
        c, at(c, poly_uop_after(c, sa, ko), dst), ld(c, a, add(c, mul(c, row, K), k))
    );
  }
  for (int i = 0; i < 16; i++) {
    PolyUOp *idx = add(c, tid, ci(c, i * 128));
    PolyUOp *k = add(c, mul(c, ko, 32), poly_uop_alu2(c, POLY_OP_FLOORDIV, idx, ci(c, 64)));
    PolyUOp *col = add(c, mul(c, gn, 64), poly_uop_alu2(c, POLY_OP_FLOORMOD, idx, ci(c, 64)));
    /* B's descriptor is [N,K]; the carried view maps back to its producer. */
    copies[8 + i] = poly_uop_store(
        c, at(c, poly_uop_after(c, sb, ko), idx), ld(c, b, add(c, mul(c, col, K), k))
    );
  }
  PolyUOp *ready =
      poly_uop1(c, POLY_OP_BARRIER, POLY_VOID, poly_uop_group(c, copies, 24), poly_arg_none());
  sa = poly_uop_after(c, sa, ready);
  sb = poly_uop_after(c, sb, ready);
  PolyUOp *acc[16], *updates[16], *stores[16];
  for (int i = 0; i < 16; i++) {
    PolyUOp *reg = buffer(c, 1, i, POLY_ADDR_REG);
    acc[i] = poly_uop_after(c, reg, poly_uop_store_val(c, reg, poly_uop_const_float(c, 0)));
    PolyUOp *prev = ld(c, poly_uop_after(c, poly_uop_after(c, acc[i], ko), ki), ci(c, 0));
    PolyUOp *av = ld(c, sa, add(c, mul(c, add(c, tm, ci(c, (i / 4) * 8)), 33), ki));
    PolyUOp *bv = ld(c, sb, add(c, mul(c, ki, 64), add(c, tn, ci(c, (i % 4) * 16))));
    prev = add(c, prev, poly_uop_mul(c, av, bv));
    updates[i] = poly_uop_store_val(c, acc[i], prev);
  }
  PolyUOp *done = poly_uop_end(c, poly_uop_group(c, updates, 16), &ki, 1);
  done = poly_uop1(c, POLY_OP_BARRIER, POLY_VOID, done, poly_arg_none());
  done = poly_uop_end(c, done, &ko, 1);
  for (int i = 0; i < 16; i++) {
    PolyUOp *row = add(c, mul(c, gm, 32), add(c, tm, ci(c, (i / 4) * 8)));
    PolyUOp *col = add(c, mul(c, gn, 64), add(c, tn, ci(c, (i % 4) * 16)));
    stores[i] = poly_uop_store(
        c, at(c, out, add(c, mul(c, row, N), col)), ld(c, poly_uop_after(c, acc[i], done), ci(c, 0))
    );
  }
  PolyUOp *ranges[] = {gm, gn, tn, tm};
  PolyUOp *end = poly_uop_end(c, poly_uop_group(c, stores, 16), ranges, 4);
  PolyKernelInfo info = {.name = "webgpu_gemm", .has_opts_to_apply = true};
  return poly_uop1(c, POLY_OP_SINK, POLY_VOID, end, poly_arg_kernel_info(&info));
}

/* Carry movement indexing into the kernel. Ordinary row-major weights need
 * no transpose/packing dispatch; computed producers still materialize once. */
static PolyUOp *operand(PolyCtx *c, PolyUOp *view, int size, int slot, PolyUOp **input) {
  PolyUOp *base = view;
  while (base->op == POLY_OP_RESHAPE || base->op == POLY_OP_PERMUTE)
    base = base->src[0];
  int nd = poly_uop_ndim(c, base);
  int64_t dims[POLY_MAX_DIMS];
  if (nd < 0 || nd > POLY_MAX_DIMS) return NULL;
  for (int i = 0; i < nd; i++) {
    PolyUOp *dim = poly_uop_shape_dim(c, base, i);
    if (!dim || dim->op != POLY_OP_CONST || dim->arg.kind != POLY_ARG_INT) return NULL;
    dims[i] = dim->arg.i;
  }
  PolyUOp *p = poly_uop_reshape(c, buffer(c, size, slot, POLY_ADDR_GLOBAL), dims, nd);
  p = p ? poly_uop_substitute(c, view, &base, &p, 1) : NULL;
  *input = poly_uop_reshape(c, poly_uop_contiguous(c, base), (int64_t[]){size}, 1);
  return p ? poly_uop_reshape(c, p, (int64_t[]){size}, 1) : NULL;
}

const char *poly_kernel_webgpu_supported(PolyCtx *c, const PolyGemmDesc *d) {
  const char *reason = poly_kernel_gemm_supported(c, d, 32, 64);
  if (reason) return reason;
  return d->K % 32 ? "requires complete K tiles" : NULL;
}

PolyUOp *poly_kernel_webgpu_lower(PolyCtx *c, const PolyGemmDesc *d) {
  PolyUOp *left = NULL, *right = NULL;
  PolyUOp *a = operand(c, d->a, d->M * d->K, 1, &left);
  PolyUOp *b = operand(c, d->b, d->K * d->N, 2, &right);
  if (!a || !b || !left || !right) return NULL;
  PolyUOp *dev = poly_uop0(c, POLY_OP_DEVICE, POLY_VOID, poly_arg_str(d->device));
  PolyUOp *out = poly_uop_new_buffer(c, dev, d->M * d->N, POLY_FLOAT32, poly_ctx_next_unique_id(c));
  PolyUOp *body = kernel(c, d->M, d->N, d->K, a, b);
  PolyUOp *src[] = {body, out, left, right};
  PolyCallInfo info = {0};
  PolyUOp *call = poly_uop(c, POLY_OP_CALL, POLY_VOID, src, 4, poly_arg_call_info(&info));
  int64_t shape[POLY_MAX_DIMS];
  memcpy(shape, d->as, sizeof(shape));
  shape[d->ad - 2] = d->N;
  return poly_uop_reshape(c, poly_uop_after(c, out, call), shape, d->ad - 1);
}
