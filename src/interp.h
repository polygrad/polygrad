/*
 * interp.h -- Interpreter backend for polygrad
 *
 * Evaluates linearized UOp sequences directly in C without rendering
 * to source code or invoking an external compiler. Serves as the
 * correctness oracle and the portable floor for environments where
 * no compiler/JIT is available.
 */

#ifndef POLY_INTERP_H
#define POLY_INTERP_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Interpret a linearized UOp sequence with buffer pointer arguments.
 *
 * lin/n_lin: linearized UOps (output of poly_linearize, includes
 *            full_rewrite_to_sink decompositions).
 * args:      buffer pointers, indexed by PARAM arg.i, followed by
 *            int* pointers for DEFINE_VAR params.
 * n_args:    total number of args (buffer params + var params).
 *
 * Returns 0 on success, <0 on error.
 */
int poly_interp_eval(PolyUOp **lin, int n_lin, void **args, int n_args);

#ifdef __cplusplus
}
#endif

#endif /* POLY_INTERP_H */
