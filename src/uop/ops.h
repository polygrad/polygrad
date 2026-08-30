#ifndef POLYGRAD_UOP_OPS_H
#define POLYGRAD_UOP_OPS_H

#include "../polygrad.h"

/* Current tinygrad/uop/ops.py:axis_letters and range_str. */
char poly_axis_letter(PolyArg arg);
char *poly_range_str(PolyArg arg);

/* Heap-backed result of current Tinygrad UOp.split_uop. Caller frees it. */
PolyUOp **poly_uop_split(PolyUOp *u, PolyOps sep, int *n_out);

/* Tinygrad helpers.py:is_image_shape on the UOp's encoded shape. */
bool poly_uop_is_image_shape(PolyCtx *ctx, const PolyUOp *u);

/* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:UOp.get_idx/get_valid. */
PolyUOp *poly_uop_get_idx(PolyCtx *ctx, PolyUOp *u);
PolyUOp *poly_uop_get_valid(PolyCtx *ctx, PolyUOp *u);

/* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:UOp.addrspace. */
bool poly_uop_addrspace(const PolyUOp *u, PolyAddrSpace *out);

#endif
