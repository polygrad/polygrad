/*
 * bind.h -- Bind index for copying decoded tensors into PolyModel
 *
 * PolyBindIndex maps buffer names to buffer indices for O(1) lookup.
 * poly_import_copy_named_tensor copies F32 data into the named buffer.
 */

#ifndef POLY_BIND_H
#define POLY_BIND_H

#include "../model.h"
#include "decoded.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PolyBindIndex PolyBindIndex;

/* A model adapter records exactly the same binds/checks before payload delivery.
 * The plan borrows decoded metadata until its unpublished Model is completed. */
typedef struct PolyImportPlan {
  void *user;
  int (*bind)(void *, const char *, const PolyDecodedTensor *, int, int);
  int (*equal)(void *, const PolyDecodedTensor *, const PolyDecodedTensor *, bool);
} PolyImportPlan;

#if defined(__GNUC__)
__attribute__((visibility("hidden")))
#endif
PolyBindIndex *
poly_bind_index_create_planned(PolyModel *inst, PolyImportPlan *plan);

#if defined(__GNUC__)
__attribute__((visibility("hidden")))
#endif
int poly_import_tensors_equal(
    PolyBindIndex *idx, const PolyDecodedTensor *a, const PolyDecodedTensor *b, bool as_f32
);

PolyBindIndex *poly_bind_index_create(PolyModel *inst);
void poly_bind_index_destroy(PolyBindIndex *idx);
/* Buffer index, or -1 if absent. Used for duplicate/role validation at import. */
int poly_bind_index_find(const PolyBindIndex *idx, const char *name);

/* Declare checkpoint-typed storage without retaining source bytes. NULL source
 * declares F32 for config-only construction; otherwise name and shape must match.
 * Caller owns the returned Tensor. */
PolyTensor *poly_import_parameter(
    PolyModel *model,
    const PolyDecodedTensor *source,
    int count,
    const char *name,
    const int64_t *shape,
    int ndim
);

/*
 * Copy F32 tensor data into the named buffer of the bound PolyModel.
 * Shapes must match after optional transpose. crop_axis=-1 forbids cropping;
 * crop_axis=0 permits a leading-axis prefix (e.g. a shorter position table).
 * Other crop axes and combined transpose/cropping are not supported.
 *
 * Returns:
 *    1 = success (copied)
 *    0 = skipped (no buffer with that name)
 *   -1 = error (shape/numel mismatch)
 */
int poly_import_copy_named_tensor(
    PolyBindIndex *idx,
    const char *dst_name,
    const float *src_data,
    const int64_t *src_shape,
    int src_ndim,
    int transpose_2d,
    int crop_axis
);

/* Convert/dequantize, validate and upload a decoded binding at most once per
 * index. Distinct checkpoint aliases must not overwrite one destination name.
 * Unknown names return 0 without decoding; adapters own their skip policy.
 * Temporary host
 * storage is released before returning; the model retains only its own state. */
int poly_import_bind_tensor(
    PolyBindIndex *idx,
    const char *name,
    const PolyDecodedTensor *tensor,
    int transpose_2d,
    int crop_axis
);

/*
 * Lookup destination shape for a named buffer.
 * Returns ndim, writes shape to shape_out. Returns 0 if not found.
 */
int poly_bind_index_dst_shape(
    const PolyBindIndex *idx,
    const char *name,
    int64_t *shape_out,
    int max_dims
);

#ifdef __cplusplus
}
#endif

#endif /* POLY_BIND_H */
