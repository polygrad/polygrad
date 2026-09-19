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

PolyBindIndex *poly_bind_index_create(PolyModel *inst);
void poly_bind_index_destroy(PolyBindIndex *idx);
/* Buffer index, or -1 if absent. Used for duplicate/role validation at import. */
int poly_bind_index_find(const PolyBindIndex *idx, const char *name);

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
