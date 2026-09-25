#include "polygrad.h"
#include "codegen/codegen.h"
#include "device.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Experimental CPU-only kernel. No importer/Model dispatch or export changes. */
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

static PolyUOp *gelu(PolyCtx *c, PolyUOp *v) {
  return poly_uop_mul(
      c, poly_uop_mul(c, poly_uop_const_float(c, .5), v),
      poly_uop_add(
          c, poly_uop_const_float(c, 1),
          poly_uop_erf(c, poly_uop_mul(c, v, poly_uop_const_float(c, 0.70710678118654752)))
      )
  );
}

/* Explicit register state mirrors tinygrad llm/kernels/amd.py:_reg and
 * _quant_linear_wmma. Lanes are independent output columns, never partial sums. */
static PolyUOp *kernel(PolyCtx *c, int M, int K, int N, int packed, int fused) {
  PolyUOp *out = param(c, M * N, 0);
  PolyUOp *bias = param(c, N, 1);
  PolyUOp *a = param(c, M * K, 2);
  PolyUOp *b = param(c, K * N, 3);
  PolyUOp *m = poly_uop_range(c, M / 4, 1, POLY_AXIS_LOOP);
  PolyUOp *n = poly_uop_range(c, N / 24, 0, POLY_AXIS_LOOP);
  PolyUOp *k = poly_uop_range(c, K, 2, POLY_AXIS_REDUCE);
  PolyUOp *acc[12], *updates[12], *av[4], *bv[3], *stores[96];
  for (int i = 0; i < 4; i++) {
    PolyUOp *v = ld(c, a, add(c, mul(c, add(c, mul(c, m, 4), ci(c, i)), K), k));
    PolyUOp *lanes[8];
    for (int l = 0; l < 8; l++)
      lanes[l] = v;
    av[i] = poly_uop_stack(c, lanes, 8);
  }
  for (int j = 0; j < 3; j++) {
    PolyUOp *base =
        packed ? add(c, mul(c, n, K * 24), mul(c, k, 24)) : add(c, mul(c, k, N), mul(c, n, 24));
    PolyUOp *lanes[8];
    for (int l = 0; l < 8; l++)
      lanes[l] = ld(c, b, add(c, base, ci(c, j * 8 + l)));
    bv[j] = poly_uop_stack(c, lanes, 8);
  }
  for (int i = 0; i < 12; i++) {
    PolyUOp *reg =
        poly_uop_placeholder(c, (int64_t[]){8}, 1, POLY_FLOAT32, i, POLY_ADDR_REG, NULL, false);
    /* Reset per output tile, not once outside the enclosing loops. */
    reg = poly_uop_after(c, poly_uop_after(c, reg, m), n);
    PolyUOp *zeros[8];
    for (int l = 0; l < 8; l++)
      zeros[l] = poly_uop_const_float(c, 0);
    acc[i] = poly_uop_after(c, reg, poly_uop_store_val(c, reg, poly_uop_stack(c, zeros, 8)));
    PolyUOp *previous[8];
    for (int l = 0; l < 8; l++)
      previous[l] = ld(c, poly_uop_after(c, acc[i], k), ci(c, l));
    PolyUOp *src[] = {av[i / 3], bv[i % 3], poly_uop_stack(c, previous, 8)};
    PolyUOp *fma = poly_uop(
        c, POLY_OP_CUSTOM, POLY_FLOAT32, src, 3,
        poly_arg_str("__builtin_ia32_vfmaddps256({0}, {1}, {2})")
    );
    updates[i] = poly_uop_store_val(c, acc[i], fma);
  }
  PolyUOp *done = poly_uop_end(c, poly_uop_group(c, updates, 12), &k, 1);
  for (int i = 0; i < 12; i++)
    for (int l = 0; l < 8; l++) {
      PolyUOp *col = add(c, mul(c, n, 24), ci(c, (i % 3) * 8 + l));
      PolyUOp *idx = add(c, mul(c, add(c, mul(c, m, 4), ci(c, i / 3)), N), col);
      PolyUOp *v = add(c, ld(c, poly_uop_after(c, acc[i], done), ci(c, l)), ld(c, bias, col));
      if (fused) v = gelu(c, v);
      stores[i * 8 + l] = poly_uop_store(c, at(c, out, idx), v);
    }
  PolyUOp *ranges[] = {m, n};
  PolyUOp *end = poly_uop_end(c, poly_uop_group(c, stores, 96), ranges, 2);
  PolyKernelInfo info = {.name = "avx2_gemm", .has_opts_to_apply = true};
  return poly_uop1(c, POLY_OP_SINK, POLY_VOID, end, poly_arg_kernel_info(&info));
}

static int eligible(int M, int K, int N) {
  return M > 0 && M <= 512 && K > 0 && K <= 4096 && N > 0 && N <= 4096 && M % 4 == 0 && N % 24 == 0;
}

static PolyUOp *gelu_kernel(PolyCtx *c, int size) {
  PolyUOp *out = param(c, size, 0), *in = param(c, size, 1);
  PolyUOp *r = poly_uop_range(c, size, 0, POLY_AXIS_LOOP);
  PolyUOp *end = poly_uop_end(c, poly_uop_store(c, at(c, out, r), gelu(c, ld(c, in, r))), &r, 1);
  PolyKernelInfo info = {.name = "gelu", .has_opts_to_apply = true};
  return poly_uop1(c, POLY_OP_SINK, POLY_VOID, end, poly_arg_kernel_info(&info));
}

static PolyUOp *pack_kernel(PolyCtx *c, int K, int N) {
  PolyUOp *out = param(c, K * N, 0), *in = param(c, K * N, 1);
  PolyUOp *n = poly_uop_range(c, N / 24, 0, POLY_AXIS_LOOP);
  PolyUOp *k = poly_uop_range(c, K, 1, POLY_AXIS_LOOP), *stores[24];
  for (int i = 0; i < 24; i++) {
    PolyUOp *dst = add(c, add(c, mul(c, n, K * 24), mul(c, k, 24)), ci(c, i));
    PolyUOp *src = add(c, add(c, mul(c, k, N), mul(c, n, 24)), ci(c, i));
    stores[i] = poly_uop_store(c, at(c, out, dst), ld(c, in, src));
  }
  PolyUOp *ranges[] = {n, k}, *end = poly_uop_end(c, poly_uop_group(c, stores, 24), ranges, 2);
  PolyKernelInfo info = {.name = "pack_weights", .has_opts_to_apply = true};
  return poly_uop1(c, POLY_OP_SINK, POLY_VOID, end, poly_arg_kernel_info(&info));
}

/* Exercise actual Tensor.custom_kernel admission/realization, separately from
 * isolated kernel timings. Inputs stay owned through all collection points. */
int bench_custom_run(
    int M,
    int K,
    int N,
    int packed,
    float *out,
    const float *bias,
    const float *a,
    const float *b
) {
  if (!eligible(M, K, N) || (packed != 0 && packed != 1)) return -1;
  PolyCtx *c = poly_ctx_new();
  int sizes[] = {M * N, N, M * K, K * N}, rc = -1;
  const float *data[] = {NULL, bias, a, b};
  PolyTensor *inputs[4] = {0}, *outputs[4] = {0}, *realized = NULL;
  for (int i = 0; i < 4; i++) {
    inputs[i] = poly_tensor_empty(c, POLY_FLOAT32, (int64_t[]){sizes[i]}, 1, POLY_DEVICE_CPU);
    if (!inputs[i] ||
        (data[i] && poly_buffer_write(
                        c, poly_tensor_uop_physical(inputs[i]), data[i], sizes[i] * sizeof(float)
                    )))
      goto done;
  }
  if (poly_tensor_custom_kernel(c, kernel(c, M, K, N, packed, 1), inputs, 4, 0, outputs) ||
      poly_realize_tensors(c, outputs, 1, &realized))
    goto done;
  rc = poly_buffer_read(c, poly_tensor_uop_physical(realized), out, M * N * sizeof(float));
done:
  /* realize returns borrowed aliases of the supplied Tensor handles. */
  for (int i = 0; i < 4; i++) {
    if (outputs[i]) poly_tensor_release(outputs[i]);
    if (inputs[i]) poly_tensor_release(inputs[i]);
  }
  poly_ctx_destroy(c);
  return rc;
}

int main(int argc, char **argv) {
  if (argc != 6) return 2;
  int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
  if (!eligible(M, K, N)) return 2;
  int mode = atoi(argv[4]);
  if (mode < 0 || mode > 5) return 2;
  PolyCtx *c = poly_ctx_new();
  PolyUOp *sink = mode == 2   ? pack_kernel(c, K, N)
                  : mode == 5 ? gelu_kernel(c, M * N)
                              : kernel(c, M, K, N, mode == 1 || mode == 4, mode < 2);
  PolyUOp *lower = poly_full_rewrite_to_sink_ex(
      c, sink,
      (PolyRewriteOpts){.optimize = true, .device = POLY_DEVICE_CPU, .caps = poly_c_renderer_caps()}
  );
  if (!lower) {
    fprintf(stderr, "lower failed\n");
    return 3;
  }
  int count = 0;
  PolyUOp **linear = poly_do_linearize(c, lower, &count);
  if (!linear) return 4;
  char *source = poly_render_c(
      c, linear, count,
      mode == 2   ? "pack_weights"
      : mode == 5 ? "gelu"
                  : "avx2_gemm"
  );
  if (!source) return 5;
  FILE *f = fopen(argv[5], "w");
  if (!f) return 6;
  fputs(source, f);
  fclose(f);
  fprintf(stderr, "uops=%d bytes=%zu\n", count, strlen(source));
  free(source);
  free(linear);
  poly_ctx_destroy(c);
  return 0;
}
