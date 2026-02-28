/*
 * poly_ir.h -- Binary IR codec for tensor-level UOp graphs
 *
 * Format: poly.ir.uops@1
 * Scope: tensor-level graphs only (pre-scheduling).
 *        No pointer dtypes, no vector dtypes.
 *
 * Used by PolyInstance for portable graph serialization.
 */

#ifndef POLY_IR_H
#define POLY_IR_H

#include "polygrad.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Buffer roles ───────────────────────────────────────────────────── */

#define POLY_IR_ROLE_PARAM   0
#define POLY_IR_ROLE_INPUT   1
#define POLY_IR_ROLE_TARGET  2
#define POLY_IR_ROLE_OUTPUT  3
#define POLY_IR_ROLE_AUX     4

/* ── IR spec: graph + metadata ──────────────────────────────────────── */

/* Named buffer entry (interface table row) */
typedef struct {
  const char *name;          /* e.g. "layers.0.weight", "x", "output" */
  uint8_t role;              /* POLY_IR_ROLE_* */
  PolyUOp *buffer;           /* BUFFER UOp */
  int64_t shape[8];
  int ndim;
} PolyIrBufEntry;

/* Named entrypoint (SINK) */
typedef struct {
  const char *name;          /* e.g. "forward", "loss" */
  PolyUOp *sink;             /* SINK UOp */
} PolyIrEntrypoint;

/* Full IR spec: graph context + named buffers + named entrypoints */
typedef struct {
  PolyCtx *ctx;              /* UOp context (not owned, caller manages) */
  PolyIrBufEntry *bufs;      /* named buffer entries */
  int n_bufs;
  PolyIrEntrypoint *entrypoints;
  int n_entrypoints;
} PolyIrSpec;

/* ── Export ──────────────────────────────────────────────────────────── */

/* Export a tensor-level IR spec to binary format.
 * Caller frees returned bytes.
 * Returns NULL on error (unsupported dtypes, etc). */
uint8_t *poly_ir_export(const PolyIrSpec *spec, int *out_len);

/* ── Import ──────────────────────────────────────────────────────────── */

/* Import binary IR into a fresh PolyIrSpec.
 * Creates a new PolyCtx and reconstructs the UOp graph.
 * Caller must eventually:
 *   - free spec->bufs (and each name string)
 *   - free spec->entrypoints (and each name string)
 *   - poly_ctx_destroy(spec->ctx)
 * Returns 0 on success, -1 on error. */
int poly_ir_import(const uint8_t *data, int len, PolyIrSpec *out);

/* Free an imported PolyIrSpec (frees names, arrays; does NOT destroy ctx). */
void poly_ir_spec_free(PolyIrSpec *spec);

#ifdef __cplusplus
}
#endif

#endif /* POLY_IR_H */
