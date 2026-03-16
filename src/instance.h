/*
 * poly_instance.h -- Runtime for portable tensor-level model instances
 *
 * PolyInstance is a thin product-layer container over the core:
 *   - Named buffers with roles, shapes, dtypes
 *   - Entrypoints (forward, loss, etc.)
 *   - Optimizer state
 *
 * Execution goes through poly_realize() in the core. The instance
 * builds PolyBufferBinding[] from its named buffers and calls
 * poly_realize(). Device is implicit in buffer handles.
 *
 * set_device() is a bulk rematerialization API that moves all buffer
 * handles to a new memory domain.
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
/* Returns host pointer for host-memory domains (CPU, INTERP, WASM_JIT).
 * Returns NULL for device-memory domains (CUDA, future WEBGPU).
 * Use readback/upload APIs for device-resident data. */
float      *poly_instance_param_data(PolyInstance *inst, int i,
                                     int64_t *numel_out);

/* ── Buffer Enumeration ──────────────────────────────────────────────── */

int         poly_instance_buf_count(const PolyInstance *inst);
const char *poly_instance_buf_name(const PolyInstance *inst, int i);
int         poly_instance_buf_role(const PolyInstance *inst, int i);
int         poly_instance_buf_shape(const PolyInstance *inst, int i,
                                    int64_t *shape_out, int max_dims);
/* Returns host pointer for host-memory domains, NULL for device-memory.
 * Use readback/upload APIs for device-resident data. */
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

/* Bulk rematerialization: moves all buffer handles to the target domain.
 * After set_device(CUDA), all buf_handles[].domain are CUDA.
 * The next poly_instance_call() builds CUDA-domain bindings,
 * poly_realize() infers CUDA, compiles for CUDA, runs on CUDA.
 * Returns 0 on success, <0 if device is unsupported or unavailable. */
int poly_instance_set_device(PolyInstance *inst, PolyDeviceId device);

/* ── Explicit readback/upload for device-resident buffers ────────────── */

int poly_instance_readback_buf(PolyInstance *inst, int i,
                               void *host_dst, size_t dst_len);
int poly_instance_upload_buf(PolyInstance *inst, int i,
                             const void *host_src, size_t src_len);
int poly_instance_readback_param(PolyInstance *inst, int i,
                                 void *host_dst, size_t dst_len);
int poly_instance_upload_param(PolyInstance *inst, int i,
                               const void *host_src, size_t src_len);

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
