/*
 * bind.h -- Bind index for copying decoded tensors into PolyInstance
 *
 * PolyBindIndex maps buffer names to buffer indices for O(1) lookup.
 * poly_import_copy_named_tensor copies F32 data into the named buffer.
 */

#ifndef POLY_BIND_H
#define POLY_BIND_H

#include "../instance.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PolyBindIndex PolyBindIndex;

PolyBindIndex *poly_bind_index_create(PolyInstance *inst);
void           poly_bind_index_destroy(PolyBindIndex *idx);

/*
 * Copy F32 tensor data into the named buffer of the bound PolyInstance.
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
    int transpose_2d);

/*
 * Lookup destination shape for a named buffer.
 * Returns ndim, writes shape to shape_out. Returns 0 if not found.
 */
int poly_bind_index_dst_shape(
    const PolyBindIndex *idx,
    const char *name,
    int64_t *shape_out,
    int max_dims);

#ifdef __cplusplus
}
#endif

#endif /* POLY_BIND_H */
