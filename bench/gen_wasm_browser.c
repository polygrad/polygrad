#define _POSIX_C_SOURCE 200809L
/*
 * gen_wasm_browser.c — Generate fixed-size WASM kernels for browser benchmark
 *
 * Output:
 *   bench/browser/wasm/<op>_<N>.wasm
 */

#include "../src/codegen.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
} Kernel;

typedef struct {
  const char *name;
  PolyOps op;
  int is_unary;
} KernelSpec;

static int ensure_dir(const char *path) {
  if (mkdir(path, 0777) == 0) return 1;
  return errno == EEXIST;
}

static Kernel make_binop(PolyOps op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *alu = poly_uop2(ctx, op, POLY_FLOAT32, ld0, ld1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  return (Kernel){ctx, sink};
}

static Kernel make_unop(PolyOps op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *alu = poly_uop1(ctx, op, POLY_FLOAT32, ld0, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  return (Kernel){ctx, sink};
}

static int write_kernel(const char *path, KernelSpec spec, int n) {
  int ok = 0;
  Kernel k = spec.is_unary ? make_unop(spec.op, n) : make_binop(spec.op, n);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n_lin);
  if (!lin) goto cleanup;

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  if (!wasm || wasm_size <= 0) {
    free(wasm);
    goto cleanup;
  }

  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "failed to open %s for writing\n", path);
    free(wasm);
    goto cleanup;
  }
  size_t wrote = fwrite(wasm, 1, (size_t)wasm_size, f);
  fclose(f);
  free(wasm);
  if (wrote != (size_t)wasm_size) {
    fprintf(stderr, "failed to write full wasm to %s\n", path);
    goto cleanup;
  }
  ok = 1;

cleanup:
  free(lin);
  poly_ctx_destroy(k.ctx);
  return ok;
}

int main(void) {
  const int sizes[] = {1024, 16384, 262144, 1048576};
  const KernelSpec specs[] = {
      {"add", POLY_OP_ADD, 0},
      {"mul", POLY_OP_MUL, 0},
      {"sub", POLY_OP_SUB, 0},
      {"neg", POLY_OP_NEG, 1},
      {"sqrt", POLY_OP_SQRT, 1},
      {"exp2", POLY_OP_EXP2, 1},
  };

  if (!ensure_dir("bench/browser")) {
    fprintf(stderr, "failed to create bench/browser\n");
    return 1;
  }
  if (!ensure_dir("bench/browser/wasm")) {
    fprintf(stderr, "failed to create bench/browser/wasm\n");
    return 1;
  }

  int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
  int n_specs = (int)(sizeof(specs) / sizeof(specs[0]));
  int failures = 0;

  for (int si = 0; si < n_sizes; si++) {
    for (int oi = 0; oi < n_specs; oi++) {
      char path[256];
      snprintf(path, sizeof(path), "bench/browser/wasm/%s_%d.wasm", specs[oi].name, sizes[si]);
      if (!write_kernel(path, specs[oi], sizes[si])) {
        fprintf(stderr, "FAIL %s\n", path);
        failures++;
      } else {
        printf("wrote %s\n", path);
      }
    }
  }

  if (failures) {
    fprintf(stderr, "generation failed: %d module(s)\n", failures);
    return 1;
  }
  return 0;
}
