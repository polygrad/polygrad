/*
 * decoded.c -- poly_decoded_tensor_to_f32 implementation
 *
 * Converts decoded tensors from any supported format dtype to F32.
 * Delegates to safetensors F32 conversion for HF dtypes.
 */

#include "decoded.h"
#include "../safetensors.h"
#include <stdlib.h>
#include <string.h>

float *poly_decoded_tensor_to_f32(const PolyDecodedTensor *t) {
    if (!t || !t->data || t->numel <= 0) return NULL;

    /*
     * For HF sources, dtype matches PolySafetensorDType.
     * Build a temporary PolySafetensorViewEx and delegate.
     */
    PolySafetensorViewEx view;
    memset(&view, 0, sizeof(view));
    view.raw_data = (const uint8_t *)t->data;
    view.numel = t->numel;
    view.ndim = t->ndim;
    view.dtype = (PolySafetensorDType)t->dtype;
    for (int i = 0; i < t->ndim && i < 8; i++)
        view.shape[i] = t->shape[i];

    return poly_safetensors_to_f32(&view);
}
