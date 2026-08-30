#ifndef POLY_CODEGEN_OPT_TC_H
#define POLY_CODEGEN_OPT_TC_H

#include "polygrad.h"

/* Current Tinygrad 2026-08-22/a9069c177a9d codegen/opt/tc.py:TensorCore. */
#define POLY_TC_MAX_OPTS 8
#define POLY_TC_MAX_SWIZZLE 8

typedef struct {
  int dims[3];
  int threads;
  int elements_per_thread[3];
  PolyDType dtype_in;
  PolyDType dtype_out;
  struct {
    char type;
    int dim;
  } opts[POLY_TC_MAX_OPTS];
  int n_opts;
  const char *swizzle[2][3][POLY_TC_MAX_SWIZZLE];
  int swizzle_len[2][3];
} PolyTensorCore;

/* Current Tinygrad codegen/opt/tc.py:get_cuda/get_amd. */
const PolyTensorCore *poly_tc_get_cuda(int arch, int *count);
const PolyTensorCore *poly_tc_get_amd(const char *arch, int *count);
int poly_tc_get_reduce_axes(const PolyTensorCore *tc, int out[][2]);
int poly_tc_count_local(const PolyTensorCore *tc);
int poly_tc_count_upcast(const PolyTensorCore *tc);
int poly_tc_base_shape_str(const PolyTensorCore *tc, const char *out[], int max_n);
int poly_tc_base_upcast_axes(const PolyTensorCore *tc, const char *out[], int max_n);
void poly_tc_permute_for_shape_str(
    const PolyTensorCore *tc,
    int swizzle,
    const char *shape_str[],
    int n_shape,
    int perm[],
    int max_n
);

#endif
