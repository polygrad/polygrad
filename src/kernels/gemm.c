#include "kernels/kernels.h"
#include "ctx.h"
#include "mixin/elementwise.h"
#include "mixin/movement.h"
#include <string.h>

static PolyUOp *param(PolyCtx *c, int size, int slot) {
  return poly_uop_placeholder(
      c, (int64_t[]){size}, 1, POLY_FLOAT32, slot, POLY_ADDR_GLOBAL, NULL, false
  );
}

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

/* Explicit register state mirrors tinygrad llm/kernels/amd.py:_reg and
 * _quant_linear_wmma. Lanes are independent output columns, never partial sums. */
static PolyUOp *kernel(
    PolyCtx *c,
    int M,
    int K,
    int N,
    PolyUOp *a,
    const PolyGemmTile *tile,
    int workers,
    int thread_axis
) {
  const int rows = tile->rows, lanes = tile->lanes, vectors = tile->vectors;
  const int cols = lanes * vectors, regs = rows * vectors;
  PolyUOp *out = param(c, M * N, 0);
  PolyUOp *b = param(c, K * N, 2);
  PolyUOp *m = poly_uop_range(c, M / rows, 1, POLY_AXIS_LOOP);
  PolyUOp *n = poly_uop_range(c, N / cols, 0, POLY_AXIS_LOOP);
  PolyUOp *ranges[] = {m, n, NULL};
  int n_ranges = 2;
  if (workers > 1) {
    int tiles = thread_axis == 0 ? M / rows : N / cols;
    PolyUOp *worker = poly_uop_range(c, workers, 3, POLY_AXIS_THREAD);
    PolyUOp *local = poly_uop_range(c, tiles / workers, 1 - thread_axis, POLY_AXIS_LOOP);
    PolyUOp *index = add(c, mul(c, worker, tiles / workers), local);
    if (thread_axis == 0)
      m = index;
    else
      n = index;
    ranges[thread_axis] = local;
    ranges[n_ranges++] = worker;
    /* Even, contiguous output-axis slices need no tail mask. Packed B is
     * read-only; workers own disjoint outputs and preserve serial K order. */
  }
  PolyUOp *k = poly_uop_range(c, K, 2, POLY_AXIS_REDUCE);
  PolyUOp *acc[regs], *updates[regs], *av[rows], *bv[vectors], *stores[rows * cols];
  for (int i = 0; i < rows; i++) {
    PolyUOp *v = ld(c, a, add(c, mul(c, add(c, mul(c, m, rows), ci(c, i)), K), k));
    PolyUOp *values[lanes];
    for (int l = 0; l < lanes; l++)
      values[l] = v;
    av[i] = poly_uop_stack(c, values, lanes);
  }
  for (int j = 0; j < vectors; j++) {
    PolyUOp *base = add(c, mul(c, n, K * cols), mul(c, k, cols));
    PolyUOp *values[lanes];
    for (int l = 0; l < lanes; l++)
      values[l] = ld(c, b, add(c, base, ci(c, j * lanes + l)));
    bv[j] = poly_uop_stack(c, values, lanes);
  }
  for (int i = 0; i < regs; i++) {
    PolyUOp *reg =
        poly_uop_placeholder(c, (int64_t[]){lanes}, 1, POLY_FLOAT32, i, POLY_ADDR_REG, NULL, false);
    /* Reset per output tile, not once outside the enclosing loops. */
    reg = poly_uop_after(c, poly_uop_after(c, reg, m), n);
    PolyUOp *zeros[lanes];
    for (int l = 0; l < lanes; l++)
      zeros[l] = poly_uop_const_float(c, 0);
    acc[i] = poly_uop_after(c, reg, poly_uop_store_val(c, reg, poly_uop_stack(c, zeros, lanes)));
    PolyUOp *previous[lanes];
    for (int l = 0; l < lanes; l++)
      previous[l] = ld(c, poly_uop_after(c, acc[i], k), ci(c, l));
    PolyUOp *src[] = {av[i / vectors], bv[i % vectors], poly_uop_stack(c, previous, lanes)};
    PolyUOp *fma = poly_uop(c, POLY_OP_CUSTOM, POLY_FLOAT32, src, 3, poly_arg_str(tile->fma));
    updates[i] = poly_uop_store_val(c, acc[i], fma);
  }
  PolyUOp *done = poly_uop_end(c, poly_uop_group(c, updates, regs), &k, 1);
  for (int i = 0; i < regs; i++)
    for (int l = 0; l < lanes; l++) {
      PolyUOp *col = add(c, mul(c, n, cols), ci(c, (i % vectors) * lanes + l));
      PolyUOp *idx = add(c, mul(c, add(c, mul(c, m, rows), ci(c, i / vectors)), N), col);
      PolyUOp *v = ld(c, poly_uop_after(c, acc[i], done), ci(c, l));
      stores[i * lanes + l] = poly_uop_store(c, at(c, out, idx), v);
    }
  PolyUOp *end = poly_uop_end(c, poly_uop_group(c, stores, rows * cols), ranges, n_ranges);
  PolyKernelInfo info = {.name = tile->name, .has_opts_to_apply = true};
  return poly_uop1(c, POLY_OP_SINK, POLY_VOID, end, poly_arg_kernel_info(&info));
}

PolyUOp *poly_kernel_gemm_lower(
    PolyCtx *ctx,
    const PolyGemmDesc *d,
    const PolyGemmTile *tile,
    int workers,
    int thread_axis
) {
  if (tile->rows < 1 || tile->rows > 8 || tile->lanes < 1 || tile->lanes > 16 ||
      tile->vectors < 1 || tile->vectors > 4 || workers < 1 || thread_axis < 0 || thread_axis > 1)
    return NULL;
  const int cols = tile->lanes * tile->vectors;
  if ((thread_axis == 0 ? d->M / tile->rows : d->N / cols) % workers) return NULL;
  PolyUOp *a = d->a, *b = d->b;
  int64_t M = d->M, K = d->K, N = d->N;
  int64_t as[POLY_MAX_DIMS];
  memcpy(as, d->as, sizeof(as));
  int ad = d->ad;
  const char *device = d->device;
  PolyUOp *base = d->a_base;
  int nd = poly_uop_ndim(ctx, base);
  if (nd < 0 || nd > POLY_MAX_DIMS) return NULL;
  int64_t dims[POLY_MAX_DIMS];
  for (int i = 0; i < nd; i++) {
    PolyUOp *dim = poly_uop_shape_dim(ctx, base, i);
    if (!dim || dim->op != POLY_OP_CONST || dim->arg.kind != POLY_ARG_INT || dim->arg.i <= 0)
      return NULL;
    dims[i] = dim->arg.i;
  }
  /* Keep the producer's iteration order, carrying reshape/permute indexing
   * into the kernel instead of forcing a reordered copy. Materializing a
   * merged head axis changes the upstream attention tile and duplicates EXP2
   * work in both pinned Tinygrad and Polygrad. Other movement ops retain their
   * ordinary materialization boundary. These views preserve M*K elements. */
  PolyUOp *ap = poly_uop_reshape(ctx, param(ctx, M * K, 1), dims, nd);
  ap = ap ? poly_uop_substitute(ctx, a, &base, &ap, 1) : NULL;
  ap = ap ? poly_uop_reshape(ctx, ap, (int64_t[]){M * K}, 1) : NULL;
  PolyUOp *left = poly_uop_contiguous(ctx, base);
  left = left ? poly_uop_reshape(ctx, left, (int64_t[]){M * K}, 1) : NULL;
  /* Panel packing is a normal scheduled copy, repeated on every replay.
   * There is no persistent packed-weight cache to invalidate after mutation. */
  PolyUOp *right = poly_uop_reshape(ctx, b, (int64_t[]){N / cols, cols, K}, 3);
  right = right ? poly_uop_permute(ctx, right, (int64_t[]){0, 2, 1}, 3) : NULL;
  right = right ? poly_uop_contiguous(ctx, right) : NULL;
  right = right ? poly_uop_reshape(ctx, right, (int64_t[]){K * N}, 1) : NULL;
  PolyUOp *dev = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_str(device));
  PolyUOp *out = poly_uop_new_buffer(ctx, dev, M * N, POLY_FLOAT32, poly_ctx_next_unique_id(ctx));
  if (!ap || !left || !right || !out) return NULL;
  PolyUOp *body = kernel(ctx, (int)M, (int)K, (int)N, ap, tile, workers, thread_axis);
  PolyUOp *src[] = {body, out, left, right};
  PolyCallInfo info = {0};
  PolyUOp *call =
      body ? poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, 4, poly_arg_call_info(&info)) : NULL;
  if (!call) return NULL;
  as[ad - 2] = N;
  return poly_uop_reshape(ctx, poly_uop_after(ctx, out, call), as, ad - 1);
}

PolyUOp *poly_kernel_probabilities_lower(PolyCtx *ctx, const PolyGemmDesc *d) {
  PolyUOp *a = d->a, *p = d->probabilities, *dot = d->root;
  /* Pinned softmax @ values can fuse EXP2 into each AV output reduction.
   * Preserve the expression, but store it once before reusing it across
   * columns. Only physical scheduling changes; gradients/exports do not. */
  PolyUOp *stored = poly_uop_contiguous(ctx, p);
  /* Rebuild the matched dot spine; recursive substitution of p -> contiguous(p)
   * would follow the replacement into itself. */
  PolyUOp *left = stored ? poly_uop_replace_src(ctx, a, (PolyUOp *[]){stored, a->src[1]}) : NULL;
  PolyUOp *product = dot->src[0]->src[0];
  product = left ? poly_uop_replace_src(ctx, product, (PolyUOp *[]){left, product->src[1]}) : NULL;
  PolyUOp *order = product ? poly_uop_replace_src(ctx, dot->src[0], &product) : NULL;
  return order ? poly_uop_replace_src(ctx, dot, &order) : NULL;
}
