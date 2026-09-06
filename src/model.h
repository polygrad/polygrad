/*
 * poly_model.h -- Runtime for portable tensor-level model models
 *
 * PolyModel is a thin product-layer container over the core:
 *   - Named buffers with roles, shapes, dtypes
 *   - Entrypoints (forward, loss, etc.)
 *   - Optimizer state
 *
 * Execution goes through the core schedule runner. The model maps ABI names
 * to logical BUFFER UOps; buffer residency is owned by ctx->buffers.
 *
 * set_device() applies the explicit uniform-device placement policy to the
 * retained logical graph / exact physical template, then migrates bound
 * values through the ctx buffer residency table.
 */

#ifndef POLY_MODEL_H
#define POLY_MODEL_H

#include "polygrad.h"
#include "engine/schedule.h" /* PolyDevice */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct PolyModel PolyModel;

typedef enum {
  POLY_STATUS_OK = 0,
  POLY_STATUS_ERROR = -1,
  POLY_STATUS_BAD_STAGE = -2,
  POLY_STATUS_INVALID = -3,
  POLY_STATUS_NOMEM = -4,
} PolyStatus;

typedef enum {
  POLY_MODEL_BUILDING = 0,
  POLY_MODEL_BUILT = 1,
  POLY_MODEL_FAILED = 2,
} PolyModelStage;

typedef struct {
  bool own_ctx_on_success;
  bool own_ctx_on_failure;
} PolyModelOptions;

typedef struct {
  int code;
  const char *func;
  char message[256];
} PolyModelError;

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

/* Explicit product-layer module cuts for non-uniform placement.  Inputs and
 * output are Tensor handles only at definition time; PolyModel retains the
 * exact logical UOp roots owned by its PolyCtx. */
typedef struct {
  const char *name;
  PolyTensor **inputs;
  int n_inputs;
  PolyTensor *output;
} PolyModelModuleSpec;

/* One exact module-name -> canonical device-name policy row.  String device
 * identities preserve sibling ordinals such as CPU:1. */
typedef struct {
  const char *module;
  const char *device;
} PolyModelDeviceMapEntry;

/* Buffer roles (matches poly_ir.h) */
#define POLY_ROLE_PARAM 0
#define POLY_ROLE_INPUT 1
#define POLY_ROLE_TARGET 2
#define POLY_ROLE_OUTPUT 3
#define POLY_ROLE_AUX 4

/* Binding flags for model-local ABI state. The current IR payload preserves
 * roles/trainability; these flags are runtime/package policy for now. */
#define POLY_BIND_F_NONE 0u
#define POLY_BIND_F_NO_SAVE (1u << 0)
#define POLY_BIND_F_OPTIM (1u << 1)
#define POLY_BIND_F_FROZEN (1u << 2)

/* Model state: parameters and persistent non-optimizer AUX buffers. */
#define POLY_EXPORT_WEIGHTS_PARAMS (1u << 0)
#define POLY_EXPORT_WEIGHTS_OPTIMIZER (1u << 1)
#define POLY_EXPORT_WEIGHTS_DEFAULT (POLY_EXPORT_WEIGHTS_PARAMS | POLY_EXPORT_WEIGHTS_OPTIMIZER)

/* Optimizer kinds. Kept here for existing model callers; optim.h exposes
 * the same constants for custom optimizer graph construction. */
#ifndef POLY_OPTIM_NONE
#define POLY_OPTIM_NONE 0
#define POLY_OPTIM_SGD 1
#define POLY_OPTIM_ADAM 2
#define POLY_OPTIM_ADAMW 3
#endif

/* Lifecycle */

/* Create a mutable build-mode model. Construction calls append local
 * BindingSpec/EntrypointSpec records. Runtime calls are invalid until
 * poly_model_build() succeeds. */
PolyModel *poly_model_new(PolyCtx *ctx, const PolyModelOptions *opts);

PolyModelStage poly_model_stage(const PolyModel *inst);
const PolyModelError *poly_model_last_error(const PolyModel *inst);

PolyStatus poly_model_scope_push(PolyModel *inst, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
PolyStatus poly_model_scope_pop(PolyModel *inst);

PolyTensor *poly_model_input(
    PolyModel *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
);
PolyTensor *poly_model_target(
    PolyModel *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
);
PolyTensor *poly_model_param(
    PolyModel *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
);
PolyStatus poly_model_state(PolyModel *inst, const char *name, PolyTensor *tensor, uint32_t flags);
PolyStatus poly_model_output(PolyModel *inst, const char *name, PolyTensor *tensor);
PolyStatus poly_model_aux(PolyModel *inst, const char *name, PolyTensor *tensor, uint32_t flags);
PolyStatus poly_model_entrypoint(
    PolyModel *inst,
    const char *name,
    const char **inputs,
    int n_inputs,
    const char **outputs,
    int n_outputs,
    const PolyEntrypointOptions *opts
);
PolyStatus poly_model_build(PolyModel *inst, PolyModelError *err);

/* Retain explicit module boundaries on an already built Model.  This does
 * not alter its current physical roots.  Definition is aggregate and atomic. */
int poly_model_define_modules(PolyModel *inst, const PolyModelModuleSpec *modules, int n_modules);

/* FFI-friendly flat-input adapter. Module inputs are concatenated in module
 * order and split by input_counts. */
int poly_model_define_module_arrays(
    PolyModel *inst,
    const char **names,
    PolyTensor **inputs,
    const int *input_counts,
    PolyTensor **outputs,
    int n_modules
);

PolyModel *poly_model_from_bindings(
    PolyCtx *ctx,
    const PolyBindingSpec *bindings,
    int n_bindings,
    const PolyEntrypointSpec *entrypoints,
    int n_entrypoints,
    const PolyModelOptions *opts,
    PolyModelError *err
);

/* FFI-friendly flat-array adapter for languages where C struct marshalling is
 * awkward. Entry inputs/outputs are flat arrays concatenated in entrypoint
 * order; the per-entry counts split them. String and tensor pointers are only
 * borrowed for the duration of this call. */
PolyModel *poly_model_from_binding_arrays(
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
    const PolyModelOptions *opts,
    PolyModelError *err
);

/* Create from IR bytes + optional safetensors weights.
 * Pass NULL/0 for weights to skip (params zero-initialized).
 * Returns NULL on error. */
PolyModel *poly_model_from_ir(
    const uint8_t *ir_data,
    int ir_len,
    const uint8_t *weights_data,
    int weights_len
);

/* Create from a PolyCtx with named buffer registry + entrypoints.
 * Requires at least one entrypoint registered. The ctx is NOT owned
 * by the model (caller manages ctx lifetime, must outlive the model).
 * Returns NULL on error (zero entrypoints, allocation failure).
 *
 * Compatibility API for old ctx-global model construction. New code should
 * use staged PolyModel construction or poly_model_from_bindings(). */
PolyModel *poly_model_from_ctx(PolyCtx *ctx)
    POLY_DEPRECATED("use staged PolyModel construction or poly_model_from_bindings");

/* Create from selected named sinks instead of every entrypoint registered on
 * the ctx. Frontend export uses this to package one traced lazy graph even if
 * the shared ctx contains old probes or other models. Names are copied by the
 * model; ctx remains caller-owned. */
PolyModel *poly_model_from_sinks(PolyCtx *ctx, const char **names, PolyUOp **sinks, int n_sinks);

void poly_model_free(PolyModel *inst);

/* Param Enumeration */

int poly_model_param_count(const PolyModel *inst);
const char *poly_model_param_name(const PolyModel *inst, int i);
int poly_model_param_shape(const PolyModel *inst, int i, int64_t *shape_out, int max_dims);
/* Historical F32-only mutable view. Returns NULL for non-F32 state. */
float *poly_model_param_data(PolyModel *inst, int i, int64_t *numel_out);
/* Exact raw scalar-storage bytes in the declared dtype. */
void *poly_model_param_data_raw(PolyModel *inst, int i, int64_t *numel_out);
int poly_model_param_dtype_id(const PolyModel *inst, int i);
size_t poly_model_param_nbytes(const PolyModel *inst, int i);

/* Buffer Enumeration */

int poly_model_buf_count(const PolyModel *inst);
const char *poly_model_buf_name(const PolyModel *inst, int i);
int poly_model_buf_role(const PolyModel *inst, int i);
bool poly_model_buf_trainable(const PolyModel *inst, int i);
bool poly_model_param_trainable(const PolyModel *inst, int i);
int poly_model_set_buf_trainable(PolyModel *inst, int i, bool trainable);
int poly_model_set_param_trainable(PolyModel *inst, int i, bool trainable);
int poly_model_buf_shape(const PolyModel *inst, int i, int64_t *shape_out, int max_dims);
/* Historical F32-only mutable view. Returns NULL for non-F32 state. */
float *poly_model_buf_data(PolyModel *inst, int i, int64_t *numel_out);
/* Exact raw scalar-storage bytes in the declared dtype. */
void *poly_model_buf_data_raw(PolyModel *inst, int i, int64_t *numel_out);
int poly_model_buf_dtype_id(const PolyModel *inst, int i);
size_t poly_model_buf_nbytes(const PolyModel *inst, int i);

/* Weight I/O (safetensors) */

/* Export selected persistent buffers as safetensors. Caller frees returned bytes.
 * poly_model_export_weights() preserves the historical default: params and
 * optimizer state. */
uint8_t *poly_model_export_weights(PolyModel *inst, int *out_len);
uint8_t *poly_model_export_weights_ex(PolyModel *inst, int *out_len, uint32_t flags);

/* Import weights from safetensors. Matches by name. Returns 0 on success. */
int poly_model_import_weights(PolyModel *inst, const uint8_t *data, int len);

/* IR Export */

uint8_t *poly_model_export_ir(PolyModel *inst, int *out_len);

/* Bound compiled-program export. The artifact contains placed PROGRAM-backed
 * LINEAR entrypoints and their named physical buffer ABI. It is ABI/device
 * bound, is not PGIR, and is never a placement source. Weights remain a
 * separate safetensors artifact. */
uint8_t *poly_model_export_program(PolyModel *inst, int *out_len);
PolyModel *poly_model_from_program(
    const uint8_t *program_data,
    int program_len,
    const uint8_t *weights_data,
    int weights_len
);
uint8_t *poly_model_save_bundle_ex(PolyModel *inst, int *out_len, uint32_t weight_flags);

/* Device configuration */

/* Apply the explicit uniform-device placement policy to retained Model
 * roots and migrate bound values through ctx->buffers. Returns 0 on success,
 * <0 if placement fails or the device is unsupported/unavailable. */
int poly_model_set_device(PolyModel *inst, PolyDevice device);

/* Compile retained logical roots under a complete explicit module/device map,
 * migrate named state, then atomically publish the replacement physical
 * bindings and entrypoint roots. Default Tensor realization never calls it. */
int poly_model_set_device_map(
    PolyModel *inst,
    const PolyModelDeviceMapEntry *entries,
    int n_entries
);

int poly_model_set_device_map_arrays(
    PolyModel *inst,
    const char **modules,
    const char **devices,
    int n_entries
);

#ifdef POLY_TESTING
/* Deterministically fail Model owner-root preparation after N additions. */
void poly_model_test_fail_residency_roots_after(int additions);
bool poly_model_test_has_vag(const PolyModel *inst);
bool poly_model_test_has_train(const PolyModel *inst);
int poly_model_test_optimizer_kind(const PolyModel *inst);
PolyUOp *poly_model_test_execution(const PolyModel *inst, bool training, bool compiled);
#endif

/* Explicit readback/upload for device-resident buffers */

int poly_model_read_buf(PolyModel *inst, int i, void *host_dst, size_t dst_len);
int poly_model_write_buf(PolyModel *inst, int i, const void *host_src, size_t src_len);
int poly_model_read_buf_named(PolyModel *inst, const char *name, void *host_dst, size_t dst_len);
int poly_model_write_buf_named(
    PolyModel *inst,
    const char *name,
    const void *host_src,
    size_t src_len
);

/* Compatibility names for the original copy-style API. */
int poly_model_readback_buf(PolyModel *inst, int i, void *host_dst, size_t dst_len);
int poly_model_upload_buf(PolyModel *inst, int i, const void *host_src, size_t src_len);
int poly_model_readback_param(PolyModel *inst, int i, void *host_dst, size_t dst_len);
int poly_model_upload_param(PolyModel *inst, int i, const void *host_src, size_t src_len);

/* Execution */

/* Typed I/O binding for forward/train calls. The Model input schema owns
 * shape/dtype; each call supplies an exact byte representation and the dtype
 * id is validated before the named BUFFER is mutated. */
typedef struct {
  const char *name;
  const void *data;
  size_t nbytes;
  int dtype_id;
} PolyIOBinding;

#define POLY_IO_BINDING_BYTES(name_, data_, nbytes_, dtype_)                                       \
  ((PolyIOBinding){                                                                                \
      .name = (name_),                                                                             \
      .data = (data_),                                                                             \
      .nbytes = (nbytes_),                                                                         \
      .dtype_id = poly_dtype_id_by_name(poly_dtype_name(dtype_)),                                  \
  })
#define POLY_IO_BINDING_ARRAY(name_, data_, dtype_)                                                \
  POLY_IO_BINDING_BYTES((name_), (data_), sizeof(data_), (dtype_))

/* Generic entrypoint execution. Compiles lazily on first call.
 * I/O bindings match model buffer names. Output written to
 * model-owned buffers (retrieve via poly_model_buf_data).
 * Returns 0 on success. */
int poly_model_call(PolyModel *inst, const char *entrypoint, PolyIOBinding *io, int n_io);

/* Entrypoint signature inspection for language frontends and generic callers. */
int poly_model_entrypoint_count(const PolyModel *inst);
const char *poly_model_entrypoint_name(const PolyModel *inst, int entrypoint_index);
/* Borrowed declared objective name; NULL for an entrypoint without one. */
const char *poly_model_entrypoint_objective(const PolyModel *model, const char *entrypoint);
int poly_model_entrypoint_input_count(const PolyModel *inst, const char *entrypoint);
const char *poly_model_entrypoint_input_name(
    const PolyModel *inst,
    const char *entrypoint,
    int input_index
);
int poly_model_entrypoint_output_count(const PolyModel *inst, const char *entrypoint);
const char *poly_model_entrypoint_output_name(
    const PolyModel *inst,
    const char *entrypoint,
    int output_index
);

/* Forward + backward for a differentiable entrypoint.
 * Builds autograd graph lazily on first call. Computes loss value
 * and per-parameter gradients. Does NOT apply optimizer updates.
 * Returns 0 on success, loss value via *loss_out. */
int poly_model_value_and_grad(
    PolyModel *inst,
    const char *entrypoint,
    PolyIOBinding *io,
    int n_io,
    float *loss_out
);

/* Convenience wrappers */

/* forward() = call("forward", ...) */
int poly_model_forward(PolyModel *inst, PolyIOBinding *inputs, int n_inputs);

/* Differentiate the selected entrypoint's scalar objective and apply scheduled
 * optimizer effects. NULL selects the sole declared objective, or the 'loss'
 * convenience entrypoint when low-level IR omits objective metadata. Multiple
 * declared objectives require an explicit entrypoint. State/moments persist
 * across objective switches; changing objectives invalidates dependent graphs. */
int poly_model_train_step(
    PolyModel *inst,
    const char *entrypoint,
    PolyIOBinding *io,
    int n_io,
    float *loss_out
);

/* Configure optimizer. Call before first train_step. */
int poly_model_set_optimizer(
    PolyModel *inst,
    int kind,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
);

int poly_model_set_optimizer_ex(
    PolyModel *inst,
    int kind,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float momentum,
    bool nesterov,
    bool classic
);

/* Named accessor helpers */

/* Return the ctx backing this model. */
PolyCtx *poly_model_ctx(const PolyModel *inst);

/* Transfer ctx ownership to the model. After this call, the model
 * will destroy the ctx when freed. Use for model builders that create
 * their own ctx internally. */
void poly_model_own_ctx(PolyModel *inst);

/* Lookup a BUFFER UOp by named buffer name. Returns NULL if not found. */
PolyUOp *poly_model_get_buffer(const PolyModel *inst, const char *name);

/* Lookup a SINK UOp by entrypoint name. Returns NULL if not found. */
PolyUOp *poly_model_get_sink(const PolyModel *inst, const char *name);

/* Get host data pointer for a named buffer. Sets *numel_out if non-NULL. */
float *poly_model_buf_data_named(PolyModel *inst, const char *name, int64_t *numel_out);

/* Get numel for a named buffer. Returns 0 if not found. */
int64_t poly_model_buf_numel_named(const PolyModel *inst, const char *name);

/* Imported-model composition. Inlines a value entrypoint from `child` into
 * a building parent under `prefix`. Input/target buffers are replaced by
 * exact parent Tensor occurrences; child state and returned values preserve
 * both the portable logical graph and stored physical template. */
typedef struct {
  const char *name;
  PolyTensor *tensor;
} PolyModelInlineBinding;

typedef struct {
  const char *name;
  PolyTensor *tensor;
} PolyModelInlineOutput;

int poly_model_inline_entrypoint(
    PolyModel *parent,
    const PolyModel *child,
    const char *entrypoint,
    const char *prefix,
    const PolyModelInlineBinding *bindings,
    int n_bindings,
    bool trainable,
    PolyModelInlineOutput *outputs,
    int max_outputs,
    int *out_n_outputs
);

/* Copy parameter host values from src into dst using prefix+src_param_name. */
int poly_model_copy_prefixed_weights(PolyModel *dst, PolyModel *src, const char *prefix);

#ifdef __cplusplus
}
#endif

#endif /* POLY_MODEL_H */
