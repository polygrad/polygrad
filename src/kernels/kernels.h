#ifndef POLY_KERNELS_H
#define POLY_KERNELS_H

#include "uop/upat.h"

/* Borrowed graph views, valid for this compilation pass. M flattens A's
 * non-reduction axes; as/bs retain batch and broadcast information. Neither
 * a nor a_base implies allocated or immutable storage. */
typedef struct {
  PolyUOp *root, *a, *b, *a_base, *probabilities;
  int ad, bd;
  int64_t as[POLY_MAX_DIMS], bs[POLY_MAX_DIMS];
  int64_t M, N, K, probability_elements;
  const char *device;
} PolyGemmDesc;

typedef struct {
  const char *name, *fma;
  int rows, lanes, vectors;
} PolyGemmTile;

typedef struct {
  const char *name;
  /* NULL means supported; otherwise a static diagnostic reason. Lowering
   * failure is an error, not permission to silently pick another kernel. */
  const char *(*supports)(PolyCtx *, const PolyGemmDesc *);
  PolyUOp *(*lower)(PolyCtx *, const PolyGemmDesc *);
} PolyKernelImpl;

bool poly_kernel_match_gemm(PolyCtx *, PolyUOp *, PolyGemmDesc *);
PolyUOp *poly_kernel_gemm_lower(
    PolyCtx *,
    const PolyGemmDesc *,
    const PolyGemmTile *,
    int workers,
    int thread_axis
);
PolyUOp *poly_kernel_probabilities_lower(PolyCtx *, const PolyGemmDesc *);
const PolyKernelImpl *poly_cpu_kernel_impls(int *count);
PolyUOp *poly_kernel_select(PolyCtx *, PolyUOp *);

#endif
