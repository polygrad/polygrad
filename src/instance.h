/*
 * poly_instance.h -- Runtime for portable tensor-level model instances
 *
 * PolyInstance is a thin product-layer container over the core:
 *   - Named buffers with roles, shapes, dtypes
 *   - Entrypoints (forward, loss, etc.)
 *   - Optimizer state
 *
 * Execution goes through poly_realize_with_bindings() in the core. The
 * instance
 * builds PolyBufferBinding[] from its named buffers and calls
 * poly_realize_with_bindings(). Device is implicit in buffer handles.
 *
 * set_device() is a bulk rematerialization API that moves all buffer
 * handles to a new memory domain.
 */

#ifndef POLY_INSTANCE_H
#define POLY_INSTANCE_H

#include "polygrad.h"
#include "engine/schedule.h" /* PolyDevice */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct PolyInstance PolyInstance;

typedef enum {
  POLY_STATUS_OK = 0,
  POLY_STATUS_ERROR = -1,
  POLY_STATUS_BAD_STAGE = -2,
  POLY_STATUS_INVALID = -3,
  POLY_STATUS_NOMEM = -4,
} PolyStatus;

typedef enum {
  POLY_INSTANCE_BUILDING = 0,
  POLY_INSTANCE_BUILT = 1,
  POLY_INSTANCE_FAILED = 2,
} PolyInstanceStage;

typedef struct {
  bool own_ctx_on_success;
  bool own_ctx_on_failure;
} PolyInstanceOptions;

typedef struct {
  int code;
  const char *func;
  char message[256];
} PolyInstanceError;

typedef struct {
  const char *objective; /* nullable; must name one output if set */
  uint32_t flags;
} PolyEntrypointOptions;

typedef struct {
  const char *name;
  int role;
  PolyTensor *tensor;
  uint32_t flags;
} PolyBindingSpec;

typedef struct {
  const char *name;
  const char **inputs;
  int n_inputs;
  const char **outputs;
  int n_outputs;
  const char *objective;
  uint32_t flags;
} PolyEntrypointSpec;

/* Buffer roles (matches poly_ir.h) */
#define POLY_ROLE_PARAM 0
#define POLY_ROLE_INPUT 1
#define POLY_ROLE_TARGET 2
#define POLY_ROLE_OUTPUT 3
#define POLY_ROLE_AUX 4

/* Optimizer kinds. Kept here for existing instance callers; optim.h exposes
 * the same constants for custom optimizer graph construction. */
#ifndef POLY_OPTIM_NONE
#define POLY_OPTIM_NONE 0
#define POLY_OPTIM_SGD 1
#define POLY_OPTIM_ADAM 2
#define POLY_OPTIM_ADAMW 3
#endif

/* Lifecycle */

/* Create a mutable build-mode instance. Construction calls append local
 * BindingSpec/EntrypointSpec records. Runtime calls are invalid until
 * poly_instance_build() succeeds. */
PolyInstance *poly_instance_new(PolyCtx *ctx, const PolyInstanceOptions *opts);

PolyInstanceStage poly_instance_stage(const PolyInstance *inst);
const PolyInstanceError *poly_instance_last_error(const PolyInstance *inst);

PolyStatus poly_instance_scope_push(PolyInstance *inst, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
PolyStatus poly_instance_scope_pop(PolyInstance *inst);

PolyTensor *poly_instance_input(
    PolyInstance *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
);
PolyTensor *poly_instance_target(
    PolyInstance *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
);
PolyTensor *poly_instance_param(
    PolyInstance *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
);
PolyStatus poly_instance_state(
    PolyInstance *inst,
    const char *name,
    PolyTensor *tensor,
    uint32_t flags
);
PolyStatus poly_instance_output(PolyInstance *inst, const char *name, PolyTensor *tensor);
PolyStatus poly_instance_aux(
    PolyInstance *inst,
    const char *name,
    PolyTensor *tensor,
    uint32_t flags
);
PolyStatus poly_instance_entrypoint(
    PolyInstance *inst,
    const char *name,
    const char **inputs,
    int n_inputs,
    const char **outputs,
    int n_outputs,
    const PolyEntrypointOptions *opts
);
PolyStatus poly_instance_build(PolyInstance *inst, PolyInstanceError *err);

PolyInstance *poly_instance_from_bindings(
    PolyCtx *ctx,
    const PolyBindingSpec *bindings,
    int n_bindings,
    const PolyEntrypointSpec *entrypoints,
    int n_entrypoints,
    const PolyInstanceOptions *opts,
    PolyInstanceError *err
);

/* FFI-friendly flat-array adapter for languages where C struct marshalling is
 * awkward. Entry inputs/outputs are flat arrays concatenated in entrypoint
 * order; the per-entry counts split them. String and tensor pointers are only
 * borrowed for the duration of this call. */
PolyInstance *poly_instance_from_binding_arrays(
    PolyCtx *ctx,
    const char **binding_names,
    const int *binding_roles,
    PolyTensor **binding_tensors,
    const uint32_t *binding_flags,
    int n_bindings,
    const char **entry_names,
    const char **entry_inputs,
    const int *entry_input_counts,
    const char **entry_outputs,
    const int *entry_output_counts,
    const char **entry_objectives,
    const uint32_t *entry_flags,
    int n_entrypoints,
    const PolyInstanceOptions *opts,
    PolyInstanceError *err
);

/* Create from IR bytes + optional safetensors weights.
 * Pass NULL/0 for weights to skip (params zero-initialized).
 * Returns NULL on error. */
PolyInstance *poly_instance_from_ir(
    const uint8_t *ir_data,
    int ir_len,
    const uint8_t *weights_data,
    int weights_len
);

/* Create from a PolyCtx with named buffer registry + entrypoints.
 * Requires at least one entrypoint registered. The ctx is NOT owned
 * by the instance (caller manages ctx lifetime, must outlive the instance).
 * Returns NULL on error (zero entrypoints, allocation failure). */
PolyInstance *poly_instance_from_ctx(PolyCtx *ctx);

/* Create from selected named sinks instead of every entrypoint registered on
 * the ctx. Frontend export uses this to package one traced lazy graph even if
 * the shared ctx contains old probes or other models. Names are copied by the
 * instance; ctx remains caller-owned. */
PolyInstance *poly_instance_from_sinks(
    PolyCtx *ctx,
    const char **names,
    PolyUOp **sinks,
    int n_sinks
);

void poly_instance_free(PolyInstance *inst);

/* Param Enumeration */

int poly_instance_param_count(const PolyInstance *inst);
const char *poly_instance_param_name(const PolyInstance *inst, int i);
int poly_instance_param_shape(const PolyInstance *inst, int i, int64_t *shape_out, int max_dims);
/* Returns host pointer to param data. For GPU domains, automatically
 * copies device data to the host shadow buffer first (like tinygrad's
 * Tensor.numpy()). Returns NULL only on error. */
float *poly_instance_param_data(PolyInstance *inst, int i, int64_t *numel_out);

/* Buffer Enumeration */

int poly_instance_buf_count(const PolyInstance *inst);
const char *poly_instance_buf_name(const PolyInstance *inst, int i);
int poly_instance_buf_role(const PolyInstance *inst, int i);
bool poly_instance_buf_trainable(const PolyInstance *inst, int i);
bool poly_instance_param_trainable(const PolyInstance *inst, int i);
int poly_instance_set_buf_trainable(PolyInstance *inst, int i, bool trainable);
int poly_instance_set_param_trainable(PolyInstance *inst, int i, bool trainable);
int poly_instance_buf_shape(const PolyInstance *inst, int i, int64_t *shape_out, int max_dims);
/* Returns host pointer to buffer data. For GPU domains, automatically
 * copies device data to the host shadow buffer first. Returns NULL
 * only on error. */
float *poly_instance_buf_data(PolyInstance *inst, int i, int64_t *numel_out);

/* Weight I/O (safetensors) */

/* Export all param buffers as safetensors. Caller frees returned bytes. */
uint8_t *poly_instance_export_weights(PolyInstance *inst, int *out_len);

/* Import weights from safetensors. Matches by name. Returns 0 on success. */
int poly_instance_import_weights(PolyInstance *inst, const uint8_t *data, int len);

/* IR Export */

uint8_t *poly_instance_export_ir(PolyInstance *inst, int *out_len);

/* Device configuration */

/* Bulk rematerialization: moves all buffer handles to the target domain.
 * After set_device(CUDA), all buf_handles[].device are CUDA.
 * The next poly_instance_call() builds CUDA-domain bindings,
 * poly_realize_with_bindings() infers CUDA, compiles for CUDA, runs on CUDA.
 * Returns 0 on success, <0 if device is unsupported or unavailable. */
int poly_instance_set_device(PolyInstance *inst, PolyDevice device);

/* Explicit readback/upload for device-resident buffers */

int poly_instance_readback_buf(PolyInstance *inst, int i, void *host_dst, size_t dst_len);
int poly_instance_upload_buf(PolyInstance *inst, int i, const void *host_src, size_t src_len);
int poly_instance_readback_param(PolyInstance *inst, int i, void *host_dst, size_t dst_len);
int poly_instance_upload_param(PolyInstance *inst, int i, const void *host_src, size_t src_len);

/* Execution */

/* I/O binding for forward/train calls. */
typedef struct {
  const char *name;
  float *data;
} PolyIOBinding;

/* Generic entrypoint execution. Compiles lazily on first call.
 * I/O bindings match instance buffer names. Output written to
 * instance-owned buffers (retrieve via poly_instance_buf_data).
 * Returns 0 on success. */
int poly_instance_call(PolyInstance *inst, const char *entrypoint, PolyIOBinding *io, int n_io);

/* Forward + backward for a differentiable entrypoint.
 * Builds autograd graph lazily on first call. Computes loss value
 * and per-parameter gradients. Does NOT apply optimizer updates.
 * Returns 0 on success, loss value via *loss_out. */
int poly_instance_value_and_grad(
    PolyInstance *inst,
    const char *entrypoint,
    PolyIOBinding *io,
    int n_io,
    float *loss_out
);

/* Convenience wrappers */

/* forward() = call("forward", ...) */
int poly_instance_forward(PolyInstance *inst, PolyIOBinding *inputs, int n_inputs);

/* train_step() = value_and_grad("loss", ...) + scheduled optimizer effects */
int poly_instance_train_step(PolyInstance *inst, PolyIOBinding *io, int n_io, float *loss_out);

/* Configure optimizer. Call before first train_step. */
int poly_instance_set_optimizer(
    PolyInstance *inst,
    int kind,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
);

/* Named accessor helpers */

/* Return the ctx backing this instance. */
PolyCtx *poly_instance_ctx(const PolyInstance *inst);

/* Transfer ctx ownership to the instance. After this call, the instance
 * will destroy the ctx when freed. Use for model builders that create
 * their own ctx internally. */
void poly_instance_own_ctx(PolyInstance *inst);

/* Lookup a BUFFER UOp by named buffer name. Returns NULL if not found. */
PolyUOp *poly_instance_get_buffer(const PolyInstance *inst, const char *name);

/* Lookup a SINK UOp by entrypoint name. Returns NULL if not found. */
PolyUOp *poly_instance_get_sink(const PolyInstance *inst, const char *name);

/* Get host data pointer for a named buffer. Sets *numel_out if non-NULL. */
float *poly_instance_buf_data_named(PolyInstance *inst, const char *name, int64_t *numel_out);

/* Get numel for a named buffer. Returns 0 if not found. */
int64_t poly_instance_buf_numel_named(const PolyInstance *inst, const char *name);

/* Imported-instance composition. Inlines a value entrypoint from `child` into
 * `dst_ctx` under `prefix`. Input/target buffers listed in bindings are
 * substituted with parent-ctx UOps. Child PARAM buffers are recreated in
 * dst_ctx using prefixed names and marked trainable/frozen by `trainable`.
 * Output value UOps are returned by original child output name. */
typedef struct {
  const char *name;
  PolyUOp *uop;
} PolyInstanceInlineBinding;

typedef struct {
  const char *name;
  PolyUOp *uop;
} PolyInstanceInlineOutput;

int poly_instance_inline_entrypoint(
    PolyCtx *dst_ctx,
    const PolyInstance *child,
    const char *entrypoint,
    const char *prefix,
    const PolyInstanceInlineBinding *bindings,
    int n_bindings,
    bool trainable,
    PolyInstanceInlineOutput *outputs,
    int max_outputs,
    int *out_n_outputs
);

/* Copy parameter host values from src into dst using prefix+src_param_name. */
int poly_instance_copy_prefixed_weights(PolyInstance *dst, PolyInstance *src, const char *prefix);

#ifdef __cplusplus
}
#endif

#endif /* POLY_INSTANCE_H */
