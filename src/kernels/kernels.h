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

/* Ordinary UOp body; b indexes [N/cols,K,cols]. It may be a parameter view,
 * not packed storage. Caller owns input materialization and CALL construction. */
PolyUOp *poly_kernel_gemm_program(PolyCtx *, int M, int K, int N, PolyUOp *a, PolyUOp *b,
    const PolyGemmTile *, int workers, int thread_axis);

typedef struct {
  const char *name;
  /* NULL means supported; otherwise a static diagnostic reason. Lowering
   * failure is an error, not permission to silently pick another kernel. */
  const char *(*supports)(PolyCtx *, const PolyGemmDesc *);
  PolyUOp *(*lower)(PolyCtx *, const PolyGemmDesc *);
} PolyKernelImpl;

bool poly_kernel_match_gemm(PolyCtx *, PolyUOp *, PolyGemmDesc *);
const char *poly_kernel_gemm_supported(PolyCtx *, const PolyGemmDesc *, int rows, int cols);
PolyUOp *poly_kernel_gemm_lower(
    PolyCtx *,
    const PolyGemmDesc *,
    const PolyGemmTile *,
    int workers,
    int thread_axis
);
PolyUOp *poly_kernel_probabilities_lower(PolyCtx *, const PolyGemmDesc *);
/* Carry reshape/permute indexing into a kernel parameter while retaining the
 * producer's storage order. Other producers keep a normal contiguous boundary. */
PolyUOp *poly_kernel_gemm_operand(PolyCtx *, PolyUOp *view, int size, int slot, PolyUOp **input);
const PolyKernelImpl *poly_cpu_kernel_impls(int *count);
const PolyKernelImpl *poly_portable_kernel_impls(const char *device, int *count);
const char *poly_kernel_webgpu_supported(PolyCtx *, const PolyGemmDesc *);
PolyUOp *poly_kernel_webgpu_lower(PolyCtx *, const PolyGemmDesc *);
typedef struct {
  PolyUOp *values, *starts, *stops;
  int64_t rows, segments, width;
  const char *device;
} PolySegmentDesc;
bool poly_kernel_match_segment(PolyCtx *, PolyUOp *, PolySegmentDesc *);
PolyUOp *poly_kernel_segment_lower(PolyCtx *, const PolySegmentDesc *);
PolyUOp *poly_kernel_select(PolyCtx *, PolyUOp *);
/* One already-rewritten tensor node; NULL is failure, unchanged is no match. */
PolyUOp *poly_kernel_select_node(PolyCtx *, PolyUOp *);

#endif
