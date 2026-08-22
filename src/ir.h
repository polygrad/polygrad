/*
 * poly_ir.h -- Binary IR codec for tensor-level UOp graphs
 *
 * Current format: poly.ir.uops@10 (v1-v9 import remains supported).
 * Scope: tensor-level graphs only (pre-scheduling).
 *        Pointer dtypes are rejected. Vector count is serialized explicitly
 *        because pinned tensor movement shape sources are weakint vectors.
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

/* Buffer roles */

#define POLY_IR_ROLE_PARAM 0
#define POLY_IR_ROLE_INPUT 1
#define POLY_IR_ROLE_TARGET 2
#define POLY_IR_ROLE_OUTPUT 3
#define POLY_IR_ROLE_AUX 4
#define POLY_IR_MAX_DIMS 8

/* IR spec: graph + metadata */

/* Named interface/state entry. PARAM/AUX rows may name any exact logical
 * value UOp; INPUT/TARGET/OUTPUT rows normally name their logical storage.
 * The binary table has always stored a node index, so this does not add a new
 * wire representation. */
typedef struct {
  const char *name; /* e.g. "layers.0.weight", "x", "output" */
  uint8_t role; /* POLY_IR_ROLE_* */
  PolyUOp *buffer; /* exact named logical UOp (legacy field name) */
  int64_t shape[POLY_IR_MAX_DIMS];
  int ndim;
  bool trainable; /* PARAM default when absent in old IR payloads */
  bool trainable_set;
} PolyIrBufEntry;

/* Named entrypoint (SINK) */
typedef struct {
  const char *name; /* e.g. "forward", "loss" */
  PolyUOp *sink; /* SINK UOp */
  const char **inputs; /* nullable explicit ABI input binding names */
  int n_inputs;
  const char **outputs; /* nullable explicit ABI output binding names */
  int n_outputs;
  const char *objective; /* nullable; must name one output when present */
  uint32_t flags;
} PolyIrEntrypoint;

/* Exact named logical boundary used by explicit non-uniform placement. The
 * portable program retains the boundary; a device assignment remains a
 * separate policy input. */
typedef struct {
  const char *name;
  PolyUOp **inputs;
  int n_inputs;
  PolyUOp *output;
} PolyIrModule;

/* Full IR spec: graph context + named buffers + named entrypoints */
typedef struct {
  PolyCtx *ctx; /* UOp context (not owned, caller manages) */
  PolyIrBufEntry *bufs; /* named buffer entries */
  int n_bufs;
  PolyIrEntrypoint *entrypoints;
  int n_entrypoints;
  PolyIrModule *modules;
  int n_modules;
} PolyIrSpec;

/* Export */

/* Export a tensor-level IR spec to binary format.
 * Caller frees returned bytes.
 * Returns NULL on error (unsupported dtypes, etc). */
uint8_t *poly_ir_export(const PolyIrSpec *spec, int *out_len);

/* Import */

/* Import binary IR into a fresh PolyIrSpec.
 * Creates a new PolyCtx and reconstructs the UOp graph.
 * Caller must eventually:
 *   - free spec->bufs (and each name string)
 *   - free spec->entrypoints (and each name string)
 *   - free spec->modules (and each name/input array)
 *   - poly_ctx_destroy(spec->ctx)
 * Returns 0 on success, -1 on error. */
int poly_ir_import(const uint8_t *data, int len, PolyIrSpec *out);

/* Free an imported PolyIrSpec (frees names, arrays; does NOT destroy ctx). */
void poly_ir_spec_free(PolyIrSpec *spec);

/* Internal executable-graph codec used by Instance program export.  It shares
 * the graph/interface tables above, but uses a distinct ABI-bound format that
 * permits lowered pointer dtypes, PROGRAM metadata, SOURCE and BINARY nodes.
 * It is not portable PGIR and must never be used as a placement source. */
uint8_t *poly_program_graph_export(const PolyIrSpec *spec, int *out_len);
int poly_program_graph_import(const uint8_t *data, int len, PolyIrSpec *out);

#ifdef __cplusplus
}
#endif

#endif /* POLY_IR_H */
