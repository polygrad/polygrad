/*
 * poly_instance.h -- Runtime for portable tensor-level model instances
 *
 * PolyInstance owns a UOp graph + named buffers + compiled execution steps.
 * Created from IR bytes (poly.ir.uops@1) or family builders (MLP, etc.).
 * Provides: forward pass, training step, weight export/import.
 *
 * Backend-aware: the same instance can execute on different devices
 * (CPU, interpreter, CUDA, WASM JIT) via poly_instance_set_device().
 * Default device is CPU. Prepared step cache survives device changes;
 * executable step cache retains entries for all previously-used devices.
 *
 * This is a "product layer" above the tinygrad-aligned compiler core.
 */

#ifndef POLY_INSTANCE_H
#define POLY_INSTANCE_H

#include "polygrad.h"
#include "exec_plan.h" /* PolyDeviceId */
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

/* ── Device configuration ────────────────────────────────────────────── */

/* Set the execution device. Default is POLY_DEVICE_CPU.
 * May be called more than once. Changing device:
 *   - prepared step cache survives (backend-neutral)
 *   - executable step cache retains entries for all previously-used devices
 *   - weight materialization transfers data to the new memory domain
 *     (no-op for CPU/INTERP since both use host malloc)
 * POLY_DEVICE_AUTO resolves to CPU on native builds.
 * Returns 0 on success, <0 if device is unsupported. */
int poly_instance_set_device(PolyInstance *inst, PolyDeviceId device);

/* ── Execution ───────────────────────────────────────────────────────── */

/* I/O binding for forward/train calls. */
typedef struct {
    const char *name;
    float *data;
} PolyIOBinding;

/* Generic entrypoint execution. Compiles lazily on first call.
 * I/O bindings match instance buffer names. Output written to
 * instance-owned buffers (retrieve via poly_instance_buf_data).
 * Returns 0 on success. */
int poly_instance_call(PolyInstance *inst, const char *entrypoint,
                       PolyIOBinding *io, int n_io);

/* Forward + backward for a differentiable entrypoint.
 * Builds autograd graph lazily on first call. Computes loss value
 * and per-parameter gradients. Does NOT apply optimizer updates.
 * Returns 0 on success, loss value via *loss_out. */
int poly_instance_value_and_grad(PolyInstance *inst, const char *entrypoint,
                                 PolyIOBinding *io, int n_io,
                                 float *loss_out);

/* ── Convenience wrappers ────────────────────────────────────────────── */

/* forward() = call("forward", ...) */
int poly_instance_forward(PolyInstance *inst,
                          PolyIOBinding *inputs, int n_inputs);

/* train_step() = value_and_grad("loss", ...) + host optimizer update */
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
