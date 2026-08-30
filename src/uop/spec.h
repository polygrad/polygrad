/* Tinygrad uop/spec.py validation entry points. */
#ifndef POLY_UOP_SPEC_H
#define POLY_UOP_SPEC_H

#include "polygrad.h"

bool poly_type_verify_tensor(PolyCtx *ctx, PolyUOp *root);
bool poly_type_verify_kernel_graph(PolyCtx *ctx, PolyUOp *root);
bool poly_type_verify_program(PolyCtx *ctx, PolyUOp *root);

#endif /* POLY_UOP_SPEC_H */
