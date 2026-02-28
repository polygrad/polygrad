/*
 * poly_instance.h -- Runtime for portable tensor-level model instances
 *
 * PolyInstance owns a UOp graph + named buffers + compiled execution steps.
 * Created from IR bytes (poly.ir.uops@1) or family builders (MLP, etc.).
 * Provides: forward pass, training step, weight export/import.
 *
 * This is a "product layer" above the tinygrad-aligned compiler core.
 */

#ifndef POLY_INSTANCE_H
#define POLY_INSTANCE_H

#include "polygrad.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct PolyInstance PolyInstance;

/* Buffer roles (matches poly_ir.h) */
#define POLY_ROLE_PARAM   0
#define POLY_ROLE_INPUT   1
#define POLY_ROLE_TARGET  2
#define POLY_ROLE_OUTPUT  3
#define POLY_ROLE_AUX     4

/* Optimizer kinds */
#define POLY_OPTIM_NONE   0
#define POLY_OPTIM_SGD    1
#define POLY_OPTIM_ADAM   2
#define POLY_OPTIM_ADAMW  3

/* ── Lifecycle ───────────────────────────────────────────────────────── */

/* Create from IR bytes + optional safetensors weights.
 * Pass NULL/0 for weights to skip (params zero-initialized).
 * Returns NULL on error. */
PolyInstance *poly_instance_from_ir(
    const uint8_t *ir_data, int ir_len,
    const uint8_t *weights_data, int weights_len);

void poly_instance_free(PolyInstance *inst);

/* ── Param Enumeration ───────────────────────────────────────────────── */

int         poly_instance_param_count(const PolyInstance *inst);
const char *poly_instance_param_name(const PolyInstance *inst, int i);
int         poly_instance_param_shape(const PolyInstance *inst, int i,
                                      int64_t *shape_out, int max_dims);
float      *poly_instance_param_data(PolyInstance *inst, int i,
                                     int64_t *numel_out);

/* ── Buffer Enumeration ──────────────────────────────────────────────── */

int         poly_instance_buf_count(const PolyInstance *inst);
const char *poly_instance_buf_name(const PolyInstance *inst, int i);
int         poly_instance_buf_role(const PolyInstance *inst, int i);
int         poly_instance_buf_shape(const PolyInstance *inst, int i,
                                    int64_t *shape_out, int max_dims);
float      *poly_instance_buf_data(PolyInstance *inst, int i,
                                   int64_t *numel_out);

/* ── Weight I/O (safetensors) ────────────────────────────────────────── */

/* Export all param buffers as safetensors. Caller frees returned bytes. */
uint8_t *poly_instance_export_weights(PolyInstance *inst, int *out_len);

/* Import weights from safetensors. Matches by name. Returns 0 on success. */
int poly_instance_import_weights(PolyInstance *inst,
                                 const uint8_t *data, int len);

/* ── IR Export ───────────────────────────────────────────────────────── */

uint8_t *poly_instance_export_ir(PolyInstance *inst, int *out_len);

/* ── Execution ───────────────────────────────────────────────────────── */

/* I/O binding for forward/train calls. */
typedef struct {
    const char *name;
    float *data;
} PolyIOBinding;

/* Forward pass. Compiles "forward" entrypoint lazily on first call.
 * Input bindings required for all role=INPUT buffers.
 * Output written to instance-owned output buffers
 * (retrieve via poly_instance_buf_data). Returns 0 on success. */
int poly_instance_forward(PolyInstance *inst,
                          PolyIOBinding *inputs, int n_inputs);

/* Single train step. Requires "loss" entrypoint in IR.
 * Builds backward + update graphs lazily on first call.
 * Returns loss value via *loss_out. Returns 0 on success. */
int poly_instance_train_step(PolyInstance *inst,
                             PolyIOBinding *io, int n_io,
                             float *loss_out);

/* Configure optimizer. Call before first train_step. */
int poly_instance_set_optimizer(PolyInstance *inst, int kind,
                                float lr, float beta1, float beta2,
                                float eps, float weight_decay);

#ifdef __cplusplus
}
#endif

#endif /* POLY_INSTANCE_H */
