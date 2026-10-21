#include "kernels/kernels.h"
#include "ctx.h"
#include "mixin/elementwise.h"
#include "mixin/movement.h"
#include <string.h>

static const char *wasm_supported(PolyCtx *ctx, const PolyGemmDesc *d) {
  return poly_kernel_gemm_supported(ctx, d, 1, 1);
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
static PolyUOp *load(PolyCtx *c, PolyUOp *p, PolyUOp *i) {
  return poly_uop_load(c, poly_uop_index(c, p, &i, 1));
}
static PolyUOp *buffer(PolyCtx *c, int size, int slot, PolyAddrSpace space) {
  return poly_uop_placeholder(c, (int64_t[]){size}, 1, POLY_FLOAT32, slot, space, NULL, false);
}
static PolyUOp *vector(PolyCtx *c, PolyUOp *p, PolyUOp *i) {
  PolyUOp *lanes[4];
  for (int l = 0; l < 4; l++)
    lanes[l] = load(c, p, add(c, i, ci(c, l)));
  return poly_uop_stack(c, lanes, 4);
}

/* One SIMD register per output, with K%4 stripes. Each update is strict
 * multiply then add; the final order is (s0+s2)+(s1+s3), PG-DIV-012.
 * Reassociation improves throughput, not every cancellation-sensitive answer.
 * No packed weights or host-dependent relaxed multiply-add are used. */
static PolyUOp *wasm_kernel(PolyCtx *c, int M, int K, int N, PolyUOp *a, PolyUOp *b) {
  const int rows = M % 2 ? 1 : 2, cols = N % 4 ? 1 : 4, regs = rows * cols;
  PolyUOp *out = buffer(c, M * N, 0, POLY_ADDR_GLOBAL);
  PolyUOp *m = poly_uop_range(c, M / rows, 1, POLY_AXIS_LOOP);
  PolyUOp *n = poly_uop_range(c, N / cols, 0, POLY_AXIS_LOOP);
  PolyUOp *k = poly_uop_range(c, K / 16, 2, POLY_AXIS_REDUCE);
  PolyUOp *acc[8], *updates[8], *stores[8];
  for (int i = 0; i < regs; i++) {
    PolyUOp *reg = buffer(c, 4, i, POLY_ADDR_REG), *zeros[4];
    reg = poly_uop_after(c, poly_uop_after(c, reg, m), n);
    for (int l = 0; l < 4; l++)
      zeros[l] = poly_uop_const_float(c, 0);
    acc[i] = poly_uop_after(c, reg, poly_uop_store_val(c, reg, poly_uop_stack(c, zeros, 4)));
  }
  for (int u = 0; K >= 16 && u < 4; u++) {
    PolyUOp *av[2], *bv[4], *ki = add(c, mul(c, k, 16), ci(c, 4 * u));
    for (int r = 0; r < rows; r++)
      av[r] = vector(c, a, add(c, mul(c, add(c, mul(c, m, rows), ci(c, r)), K), ki));
    for (int j = 0; j < cols; j++)
      bv[j] = vector(c, b, add(c, mul(c, add(c, mul(c, n, cols), ci(c, j)), K), ki));
    for (int i = 0; i < regs; i++) {
      PolyUOp *prev = vector(c, poly_uop_after(c, acc[i], k), ci(c, 0));
      updates[i] =
          poly_uop_store_val(c, acc[i], add(c, prev, poly_uop_mul(c, av[i / cols], bv[i % cols])));
      acc[i] = poly_uop_after(c, acc[i], updates[i]);
    }
  }
  PolyUOp *done = K >= 16 ? poly_uop_end(c, poly_uop_group(c, updates, regs), &k, 1) : NULL;
  if (done)
    for (int i = 0; i < regs; i++)
      acc[i] = poly_uop_after(c, acc[i], done);
  for (int i = 0; i < regs; i++) {
    PolyUOp *v[4];
    for (int l = 0; l < 4; l++)
      v[l] = load(c, acc[i], ci(c, l));
    /* Finish fixed tails as expressions: partial register writes would prevent
     * SIMD promotion of the entire loop. No speculative reads or extra +0. */
    for (int t = K / 16 * 16; t < K; t += 4) {
      for (int l = 0; l < 4 && t + l < K; l++) {
        PolyUOp *ai = add(c, mul(c, add(c, mul(c, m, rows), ci(c, i / cols)), K), ci(c, t + l));
        PolyUOp *bi = add(c, mul(c, add(c, mul(c, n, cols), ci(c, i % cols)), K), ci(c, t + l));
        v[l] = add(c, v[l], poly_uop_mul(c, load(c, a, ai), load(c, b, bi)));
      }
    }
    PolyUOp *sum = add(c, add(c, v[0], v[2]), add(c, v[1], v[3]));
    PolyUOp *idx =
        add(c, mul(c, add(c, mul(c, m, rows), ci(c, i / cols)), N),
            add(c, mul(c, n, cols), ci(c, i % cols)));
    stores[i] = poly_uop_store(c, poly_uop_index(c, out, &idx, 1), sum);
  }
  PolyUOp *ranges[] = {m, n};
  PolyUOp *end = poly_uop_end(c, poly_uop_group(c, stores, regs), ranges, 2);
  PolyKernelInfo info = {.name = "wasm_gemm", .has_opts_to_apply = true};
  return poly_uop1(c, POLY_OP_SINK, POLY_VOID, end, poly_arg_kernel_info(&info));
}

/* For KxN storage, adjacent output columns are contiguous. Vectorize those
 * independent outputs instead of gathering four strided K terms. This keeps
 * serial K order and uses the same no-pack operand views and CALL ABI. */
static PolyUOp *wasm_column_kernel(PolyCtx *c, int M, int K, int N, PolyUOp *a, PolyUOp *b) {
  const int cols = N % 16 == 0 ? 16 : N % 4 == 0 ? 4 : 1;
  PolyGemmTile tile = {
      .name = "wasm_gemm",
      .rows = M % 4 == 0 ? 4
              : M % 2    ? 1
                         : 2,
      .lanes = cols == 1 ? 1 : 4,
      .vectors = cols > 4 ? cols / 4 : 1};
  /* Only the parameter's indexing changes; the input buffer is never packed. */
  PolyUOp *view = poly_uop_reshape(c, b, (int64_t[]){N / cols, cols, K}, 3);
  view = poly_uop_permute(c, view, (int64_t[]){0, 2, 1}, 3);
  view = poly_uop_reshape(c, view, (int64_t[]){N * K}, 1);
  return poly_kernel_gemm_program(c, M, K, N, a, view, &tile, 1, 0);
}

static PolyUOp *wasm_lower(PolyCtx *ctx, const PolyGemmDesc *d) {
  PolyUOp *left = NULL, *right = NULL;
  PolyUOp *a = poly_kernel_gemm_operand(ctx, d->a, d->M * d->K, 1, &left);
  PolyUOp *b = poly_kernel_gemm_operand(ctx, d->b, d->N * d->K, 2, &right);
  if (!a || !b || !left || !right) return NULL;
  PolyUOp *dev = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_str(d->device));
  PolyUOp *out =
      poly_uop_new_buffer(ctx, dev, d->M * d->N, POLY_FLOAT32, poly_ctx_next_unique_id(ctx));
  PolyUOp *nk = poly_uop_reshape(ctx, b, (int64_t[]){d->N, d->K}, 2);
  PolyUOp *kn = poly_uop_permute(ctx, nk, (int64_t[]){1, 0}, 2);
  int64_t offset;
  bool columns = poly_uop_contiguous_view_offset(ctx, kn, &offset) == 0;
  if (!out) return NULL;
  int tile = columns ? 16 : 4, full = d->N / tile * tile;
  for (int start = 0; start < d->N;) {
    int width = start == 0 && full ? full : d->N - start;
    PolyUOp *part = poly_uop_shrink(ctx, nk, (int64_t[][2]){{start, start + width}, {0, d->K}}, 2);
    part = poly_uop_reshape(ctx, part, (int64_t[]){width * d->K}, 1);
    PolyUOp *body = columns ? wasm_column_kernel(ctx, d->M, d->K, width, a, part)
                            : wasm_kernel(ctx, d->M, d->K, width, a, part);
    if (!body) return NULL;
    if (width != d->N) {
      PolyUOp *slot = buffer(ctx, d->M * width, 0, POLY_ADDR_GLOBAL);
      PolyUOp *view = poly_uop_reshape(
          ctx, buffer(ctx, d->M * d->N, 0, POLY_ADDR_GLOBAL), (int64_t[]){d->M, d->N}, 2
      );
      view = poly_uop_shrink(ctx, view, (int64_t[][2]){{0, d->M}, {start, start + width}}, 2);
      view = poly_uop_reshape(ctx, view, (int64_t[]){d->M * width}, 1);
      body = poly_uop_substitute(ctx, body, &slot, &view, 1);
    }
    /* Prefix and tail write disjoint columns of one output. The AFTER argument
     * orders the second ordinary CALL without a temporary or a merge kernel. */
    PolyUOp *src[] = {body, out, left, right};
    PolyCallInfo info = {0};
    PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, 4, poly_arg_call_info(&info));
    if (!call) return NULL;
    out = poly_uop_after(ctx, out, call);
    start += width;
  }
  int64_t shape[POLY_MAX_DIMS];
  memcpy(shape, d->as, sizeof(shape));
  shape[d->ad - 2] = d->N;
  return poly_uop_reshape(ctx, out, shape, d->ad - 1);
}

const PolyKernelImpl *poly_portable_kernel_impls(const char *device, int *count) {
  static const PolyKernelImpl wasm[] = {{"gemm_simd128_striped", wasm_supported, wasm_lower}};
  static const PolyKernelImpl webgpu[] = {
      {"gemm_workgroup_32x64", poly_kernel_webgpu_supported, poly_kernel_webgpu_lower}};
  *count = 1;
  if (!strcmp(device, "WASM")) return wasm;
  if (!strcmp(device, "WEBGPU")) return webgpu;
  *count = 0;
  return NULL;
}
