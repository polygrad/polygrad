/*
 * poly_instance.c -- Runtime for portable tensor-level model instances
 *
 * Product layer above the tinygrad-aligned compiler core.
 * Owns ABI names, binding metadata, entrypoints, and optimizer state.
 * Buffer residency is owned by PolyCtx through ctx->buffers.
 */

#define _POSIX_C_SOURCE 200809L
#include "instance.h"
#include "ctx.h"
#include "utils.h"
#include "ir.h"
#include "safetensors.h"
#include "tensor.h"
#include "optim.h"
#include "engine/realize.h"
#include "engine/schedule.h"
#include "codegen.h" /* poly_cuda_available (POLY_HAS_CUDA) */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* Internal types */

typedef struct {
  char *name;
  uint8_t role;
  uint32_t flags;
  PolyUOp *buffer;
  int64_t shape[8];
  int ndim;
  void *data; /* non-owning cached host-root pointer */
  int64_t numel;
  bool owns_data; /* false for aliases sharing another entry's allocation */
  bool trainable; /* PARAMs can be frozen while still saved as weights */
} NamedBuf;

static size_t named_buf_nbytes(const NamedBuf *b) {
  if (!b || !b->buffer || b->numel <= 0) return 0;
  size_t itemsize = poly_dtype_itemsize(poly_dtype_scalar(b->buffer->dtype));
  return (size_t)b->numel * itemsize;
}

typedef struct {
  int kind;
  float lr, beta1, beta2, eps, weight_decay;
  float momentum;
  bool nesterov;
  bool classic;
  int step;
} OptimState;

/* Value-and-grad metadata (built lazily on first train call) */

typedef struct {
  PolyUOp *combined_sink; /* combined fwd+bwd SINK */
  PolyUOp *loss_out_buf; /* BUFFER UOp for loss output */
  PolyUOp **grad_out_bufs; /* [n_params] gradient BUFFER UOps */
  float **grad_datas; /* [n_params] gradient host data */
  PolyUOp **grad_uops; /* [n_params] raw gradient UOp expressions */
  PolyUOp *loss_value; /* loss value UOp (pre-store) */
  float loss_data; /* scalar loss value */
} VagState;

/* Training state (optimizer graph, built lazily) */

typedef struct {
  PolyUOp *combined_sink; /* fwd+bwd+optimizer SINK */
  PolySchedule *schedule; /* cached lowering for the stable train graph */
  PolyUOp *loss_out_buf; /* BUFFER UOp for loss scalar output */
  float loss_data; /* scalar loss value after step */

  /* Momentum/first-moment buffers. For SGD this is tinygrad's b[] state; for
   * Adam/AdamW this is m[]. */
  PolyUOp **m_bufs; /* [n_params] momentum/first moment BUFFER UOps */
  PolyUOp **v_bufs; /* [n_params] second moment BUFFER UOps */
  int n_moment_bufs;

  /* Adam beta-power scalar buffers (tinygrad b1_t/b2_t). */
  PolyUOp *bc1_buf;
  PolyUOp *bc2_buf;
  float bc1_data;
  float bc2_data;
} TrainState;

typedef struct {
  char *name;
  uint8_t role;
  uint32_t flags;
  PolyTensor *tensor;
  PolyUOp *buffer;
  int64_t shape[8];
  int ndim;
} BuildBinding;

typedef struct {
  char *name;
  char **inputs;
  int n_inputs;
  char **outputs;
  int n_outputs;
  char *objective;
  uint32_t flags;
} BuildEntrypoint;

typedef struct {
  char *name;
  PolyUOp *sink;
  char **inputs;
  int n_inputs;
  char **outputs;
  int n_outputs;
  char *objective;
  uint32_t flags;
} RuntimeEntrypoint;

typedef struct {
  PolyInstanceOptions opts;

  BuildBinding *bindings;
  int n_bindings;
  int bindings_cap;

  BuildEntrypoint *entrypoints;
  int n_entrypoints;
  int entrypoints_cap;

  char **scopes;
  int n_scopes;
  int scopes_cap;
} PolyInstanceBuildState;

struct PolyInstance {
  PolyCtx *ctx;
  bool owns_ctx; /* true: poly_instance_free destroys ctx */
  PolyInstanceStage stage;
  PolyInstanceBuildState *build;
  PolyInstanceError last_error;

  NamedBuf *bufs;
  int n_bufs;

  /* Param subset (indices into bufs[]) */
  int *param_indices;
  int n_params;
  int *trainable_param_indices;
  int n_trainable_params;

  /* Entrypoints */
  RuntimeEntrypoint *entrypoints;
  int n_entrypoints;
  PolySchedule **entry_schedules; /* lazy, one per generic entrypoint */

  /* Value-and-grad state (lazy, per-entrypoint -- currently only "loss") */
  VagState *vag; /* NULL until first value_and_grad call */

  /* Training state (optimizer graph, built lazily) */
  TrainState *train; /* NULL until first train_step call */

  /* Optimizer state */
  OptimState optim;
};

/* Helpers */

static int64_t compute_numel(const int64_t *shape, int ndim) {
  int64_t n = 1;
  for (int i = 0; i < ndim; i++)
    n *= shape[i];
  return n;
}

static int find_entrypoint(const PolyInstance *inst, const char *name) {
  for (int i = 0; i < inst->n_entrypoints; i++)
    if (strcmp(inst->entrypoints[i].name, name) == 0) return i;
  return -1;
}

static int find_buf_by_name(const PolyInstance *inst, const char *name) {
  for (int i = 0; i < inst->n_bufs; i++)
    if (strcmp(inst->bufs[i].name, name) == 0) return i;
  return -1;
}

static bool is_optimizer_binding_name(const char *name) {
  return name && strncmp(name, "optim.", 6) == 0;
}

static bool is_optimizer_binding(const NamedBuf *b) {
  return b && ((b->flags & POLY_BIND_F_OPTIM) || is_optimizer_binding_name(b->name));
}

static bool should_export_weight_buf_flags(const NamedBuf *b, uint32_t flags) {
  if (!b || (b->flags & POLY_BIND_F_NO_SAVE)) return false;
  if ((flags & POLY_EXPORT_WEIGHTS_PARAMS) && b->role == POLY_ROLE_PARAM) return true;
  if ((flags & POLY_EXPORT_WEIGHTS_OPTIMIZER) && is_optimizer_binding(b)) return true;
  return false;
}

static PolyInstance *instance_from_spec(PolyIrSpec *spec, bool owns_ctx, bool free_spec);

static const char *instance_stage_name(PolyInstanceStage stage) {
  switch (stage) {
  case POLY_INSTANCE_BUILDING:
    return "BUILDING";
  case POLY_INSTANCE_BUILT:
    return "BUILT";
  case POLY_INSTANCE_FAILED:
    return "FAILED";
  }
  return "UNKNOWN";
}

static void poly_instance_set_error(
    PolyInstance *inst,
    int code,
    const char *func,
    const char *fmt,
    ...
) {
  if (!inst) return;
  inst->last_error.code = code;
  inst->last_error.func = func;
  if (!fmt) {
    inst->last_error.message[0] = '\0';
    return;
  }
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(inst->last_error.message, sizeof(inst->last_error.message), fmt, ap);
  va_end(ap);
}

static void poly_instance_copy_error(PolyInstance *inst, PolyInstanceError *err) {
  if (inst && err) *err = inst->last_error;
}

static PolyStatus require_stage(PolyInstance *inst, PolyInstanceStage want, const char *func) {
  if (!inst) return POLY_STATUS_INVALID;
  if (inst->stage == want) return POLY_STATUS_OK;
  poly_instance_set_error(
      inst, POLY_STATUS_BAD_STAGE, func, "expected %s, got %s", instance_stage_name(want),
      instance_stage_name(inst->stage)
  );
  return POLY_STATUS_BAD_STAGE;
}

static int grow_array(void **ptr, int *cap, int n, size_t elem_size) {
  if (n <= *cap) return 0;
  int new_cap = *cap ? *cap * 2 : 8;
  while (new_cap < n)
    new_cap *= 2;
  void *new_ptr = realloc(*ptr, (size_t)new_cap * elem_size);
  if (!new_ptr) return -1;
  *ptr = new_ptr;
  *cap = new_cap;
  return 0;
}

static void build_entrypoint_free(BuildEntrypoint *ep) {
  if (!ep) return;
  free(ep->name);
  for (int i = 0; i < ep->n_inputs; i++)
    free(ep->inputs[i]);
  free(ep->inputs);
  for (int i = 0; i < ep->n_outputs; i++)
    free(ep->outputs[i]);
  free(ep->outputs);
  free(ep->objective);
}

static void build_state_free(PolyInstanceBuildState *build) {
  if (!build) return;
  for (int i = 0; i < build->n_bindings; i++)
    free(build->bindings[i].name);
  free(build->bindings);
  for (int i = 0; i < build->n_entrypoints; i++)
    build_entrypoint_free(&build->entrypoints[i]);
  free(build->entrypoints);
  for (int i = 0; i < build->n_scopes; i++)
    free(build->scopes[i]);
  free(build->scopes);
  free(build);
}

static char *dup_cstr(const char *s) {
  if (!s) return NULL;
  size_t len = strlen(s);
  char *out = malloc(len + 1);
  if (!out) return NULL;
  memcpy(out, s, len + 1);
  return out;
}

static char *vformat_cstr(const char *fmt, va_list ap) {
  if (!fmt) return NULL;
  va_list cp;
  va_copy(cp, ap);
  int n = vsnprintf(NULL, 0, fmt, cp);
  va_end(cp);
  if (n < 0) return NULL;
  char *out = malloc((size_t)n + 1);
  if (!out) return NULL;
  vsnprintf(out, (size_t)n + 1, fmt, ap);
  return out;
}

static bool valid_binding_name(const char *name) {
  if (!name || !name[0]) return false;
  if (name[0] == '.') return false;
  char prev = 0;
  for (const char *p = name; *p; p++) {
    unsigned char c = (unsigned char)*p;
    if (c < 32) return false;
    if (*p == '.') {
      if (prev == 0 || prev == '.') return false;
      if (!p[1]) return false;
    }
    prev = *p;
  }
  return true;
}

static char *scoped_name(PolyInstance *inst, const char *name) {
  PolyInstanceBuildState *build = inst ? inst->build : NULL;
  if (!build || build->n_scopes == 0) return dup_cstr(name);
  size_t len = strlen(name);
  for (int i = 0; i < build->n_scopes; i++)
    len += strlen(build->scopes[i]) + 1;
  char *out = malloc(len + 1);
  if (!out) return NULL;
  out[0] = '\0';
  for (int i = 0; i < build->n_scopes; i++) {
    if (i > 0) strcat(out, ".");
    strcat(out, build->scopes[i]);
  }
  strcat(out, ".");
  strcat(out, name);
  return out;
}

static int copy_tensor_shape(PolyCtx *ctx, PolyTensor *tensor, int64_t shape[8], int *ndim) {
  if (!ctx || !tensor || !ndim) return -1;
  PolyUOp *u = poly_tensor_uop(tensor);
  if (!u) return -1;
  PolyShape s = poly_uop_shape_cached(ctx, u);
  if (s.ndim < 0 || s.ndim > 8) return -1;
  *ndim = s.ndim;
  if (s.ndim > 0) memcpy(shape, s.dims, (size_t)s.ndim * sizeof(int64_t));
  return 0;
}

static BuildBinding *find_build_binding(PolyInstanceBuildState *build, const char *name) {
  if (!build || !name) return NULL;
  for (int i = 0; i < build->n_bindings; i++)
    if (strcmp(build->bindings[i].name, name) == 0) return &build->bindings[i];
  return NULL;
}

static PolyStatus append_build_binding(
    PolyInstance *inst,
    const char *name,
    uint8_t role,
    uint32_t flags,
    PolyTensor *tensor,
    PolyUOp *buffer,
    const int64_t *shape,
    int ndim,
    bool default_requires_grad,
    PolyTensorProvenance provenance
) {
  if (!inst || !inst->build || !name || !tensor) return POLY_STATUS_INVALID;
  char *full_name = scoped_name(inst, name);
  if (!full_name) {
    poly_instance_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    return POLY_STATUS_NOMEM;
  }
  if (!valid_binding_name(full_name)) {
    poly_instance_set_error(
        inst, POLY_STATUS_INVALID, __func__, "invalid binding name '%s'", full_name
    );
    free(full_name);
    return POLY_STATUS_INVALID;
  }
  if (find_build_binding(inst->build, full_name)) {
    poly_instance_set_error(
        inst, POLY_STATUS_INVALID, __func__, "duplicate binding '%s'", full_name
    );
    free(full_name);
    return POLY_STATUS_INVALID;
  }
  if (ndim < 0 || ndim > 8 || poly_shape_numel_checked(shape, ndim) < 0) {
    poly_instance_set_error(
        inst, POLY_STATUS_INVALID, __func__, "invalid shape for '%s'", full_name
    );
    free(full_name);
    return POLY_STATUS_INVALID;
  }
  if (grow_array(
          (void **)&inst->build->bindings, &inst->build->bindings_cap, inst->build->n_bindings + 1,
          sizeof(BuildBinding)
      ) != 0) {
    poly_instance_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    free(full_name);
    return POLY_STATUS_NOMEM;
  }
  BuildBinding *b = &inst->build->bindings[inst->build->n_bindings++];
  *b = (BuildBinding){0};
  b->name = full_name;
  b->role = role;
  b->flags = flags;
  b->tensor = tensor;
  b->buffer = buffer;
  b->ndim = ndim;
  if (ndim > 0) memcpy(b->shape, shape, (size_t)ndim * sizeof(int64_t));
  if (role == POLY_ROLE_OUTPUT) {
    if (poly_tensor_provenance(tensor) == POLY_TENSOR_PROVENANCE_UNKNOWN)
      poly_tensor_set_provenance(tensor, provenance);
  } else {
    if (!poly_tensor_requires_grad_is_set(tensor))
      poly_tensor_set_requires_grad(tensor, default_requires_grad);
    poly_tensor_set_provenance(tensor, provenance);
  }
  return POLY_STATUS_OK;
}

static PolyTensor *make_bound_storage_tensor(
    PolyInstance *inst,
    const char *name,
    uint8_t role,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    bool default_requires_grad,
    PolyTensorProvenance provenance
) {
  if (require_stage(inst, POLY_INSTANCE_BUILDING, __func__) != POLY_STATUS_OK) return NULL;
  if (!inst->ctx || (ndim > 0 && !shape) || ndim < 0 || ndim > 8) {
    poly_instance_set_error(inst, POLY_STATUS_INVALID, __func__, "invalid tensor shape");
    return NULL;
  }
  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel < 0) {
    poly_instance_set_error(inst, POLY_STATUS_INVALID, __func__, "invalid tensor shape");
    return NULL;
  }
  PolyDType scalar = poly_dtype_scalar(dt);
  PolyUOp *buf = poly_buffer(inst->ctx, scalar, numel);
  if (!buf) {
    poly_instance_set_error(inst, POLY_STATUS_ERROR, __func__, "failed to create storage buffer");
    return NULL;
  }
  PolyUOp *root = buf;
  if (!(ndim == 1 && shape[0] == numel))
    root = poly_reshape(inst->ctx, buf, (int64_t *)shape, ndim);
  PolyTensor *tensor = poly_tensor_create(inst->ctx, root, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  if (!tensor) {
    poly_instance_set_error(inst, POLY_STATUS_ERROR, __func__, "failed to create tensor");
    return NULL;
  }
  if (append_build_binding(
          inst, name, role, 0, tensor, buf, shape, ndim, default_requires_grad, provenance
      ) != POLY_STATUS_OK)
    return NULL;
  return tensor;
}

static PolyStatus append_existing_tensor_binding(
    PolyInstance *inst,
    const char *name,
    uint8_t role,
    PolyTensor *tensor,
    uint32_t flags,
    bool require_buffer,
    bool default_requires_grad,
    PolyTensorProvenance provenance
) {
  if (require_stage(inst, POLY_INSTANCE_BUILDING, __func__) != POLY_STATUS_OK)
    return POLY_STATUS_BAD_STAGE;
  if (!inst->ctx || !tensor) {
    poly_instance_set_error(inst, POLY_STATUS_INVALID, __func__, "null tensor binding");
    return POLY_STATUS_INVALID;
  }
  int64_t shape[8] = {0};
  int ndim = 0;
  if (copy_tensor_shape(inst->ctx, tensor, shape, &ndim) != 0) {
    poly_instance_set_error(inst, POLY_STATUS_INVALID, __func__, "could not infer binding shape");
    return POLY_STATUS_INVALID;
  }
  PolyUOp *buffer = NULL;
  const PolyUOp *identity = poly_uop_get_buffer_identity(poly_tensor_uop(tensor));
  if (identity) buffer = (PolyUOp *)identity;
  if (require_buffer && !buffer) {
    poly_instance_set_error(
        inst, POLY_STATUS_INVALID, __func__, "binding '%s' has no buffer identity", name
    );
    return POLY_STATUS_INVALID;
  }
  return append_build_binding(
      inst, name, role, flags, tensor, buffer, shape, ndim, default_requires_grad, provenance
  );
}

static char **dup_string_array(const char **items, int n) {
  if (n <= 0) return NULL;
  if (!items) return NULL;
  char **out = calloc((size_t)n, sizeof(char *));
  if (!out) return NULL;
  for (int i = 0; i < n; i++) {
    out[i] = dup_cstr(items[i]);
    if (!out[i]) {
      for (int j = 0; j < i; j++)
        free(out[j]);
      free(out);
      return NULL;
    }
  }
  return out;
}

static int copy_initial_buffer_data(PolyInstance *inst, PolyCtx *ctx) {
  if (!inst || !ctx) return -1;
  for (int i = 0; i < inst->n_bufs; i++) {
    PolyBuffer *src = poly_buffer_get(ctx, inst->bufs[i].buffer);
    size_t nbytes = named_buf_nbytes(&inst->bufs[i]);
    if (!src || !src->ptr || src->nbytes < nbytes || nbytes == 0) continue;
    if (poly_buffer_read(ctx, inst->bufs[i].buffer, inst->bufs[i].data, nbytes) != 0 &&
        src->valid && poly_device_is_host_addressable(src->device))
      memcpy(inst->bufs[i].data, src->ptr, nbytes);
  }
  return 0;
}

PolyInstance *poly_instance_new(PolyCtx *ctx, const PolyInstanceOptions *opts) {
  if (!ctx) return NULL;
  PolyInstance *inst = calloc(1, sizeof(PolyInstance));
  if (!inst) return NULL;
  PolyInstanceBuildState *build = calloc(1, sizeof(PolyInstanceBuildState));
  if (!build) {
    free(inst);
    return NULL;
  }
  if (opts) build->opts = *opts;
  inst->ctx = ctx;
  inst->owns_ctx = false;
  inst->stage = POLY_INSTANCE_BUILDING;
  inst->build = build;
  inst->optim.kind = POLY_OPTIM_NONE;
  return inst;
}

PolyInstanceStage poly_instance_stage(const PolyInstance *inst) {
  return inst ? inst->stage : POLY_INSTANCE_FAILED;
}

const PolyInstanceError *poly_instance_last_error(const PolyInstance *inst) {
  return inst ? &inst->last_error : NULL;
}

PolyStatus poly_instance_scope_push(PolyInstance *inst, const char *fmt, ...) {
  if (require_stage(inst, POLY_INSTANCE_BUILDING, __func__) != POLY_STATUS_OK)
    return POLY_STATUS_BAD_STAGE;
  va_list ap;
  va_start(ap, fmt);
  char *scope = vformat_cstr(fmt, ap);
  va_end(ap);
  if (!scope) {
    poly_instance_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    return POLY_STATUS_NOMEM;
  }
  if (!valid_binding_name(scope)) {
    poly_instance_set_error(inst, POLY_STATUS_INVALID, __func__, "invalid scope '%s'", scope);
    free(scope);
    return POLY_STATUS_INVALID;
  }
  if (grow_array(
          (void **)&inst->build->scopes, &inst->build->scopes_cap, inst->build->n_scopes + 1,
          sizeof(char *)
      ) != 0) {
    poly_instance_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    free(scope);
    return POLY_STATUS_NOMEM;
  }
  inst->build->scopes[inst->build->n_scopes++] = scope;
  return POLY_STATUS_OK;
}

PolyStatus poly_instance_scope_pop(PolyInstance *inst) {
  if (require_stage(inst, POLY_INSTANCE_BUILDING, __func__) != POLY_STATUS_OK)
    return POLY_STATUS_BAD_STAGE;
  if (inst->build->n_scopes <= 0) {
    poly_instance_set_error(inst, POLY_STATUS_INVALID, __func__, "scope stack is empty");
    return POLY_STATUS_INVALID;
  }
  free(inst->build->scopes[--inst->build->n_scopes]);
  inst->build->scopes[inst->build->n_scopes] = NULL;
  return POLY_STATUS_OK;
}

PolyTensor *poly_instance_input(
    PolyInstance *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
) {
  return make_bound_storage_tensor(
      inst, name, POLY_ROLE_INPUT, dt, shape, ndim, false, POLY_TENSOR_PROVENANCE_USER_INPUT
  );
}

PolyTensor *poly_instance_target(
    PolyInstance *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
) {
  return make_bound_storage_tensor(
      inst, name, POLY_ROLE_TARGET, dt, shape, ndim, false, POLY_TENSOR_PROVENANCE_USER_INPUT
  );
}

PolyTensor *poly_instance_param(
    PolyInstance *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
) {
  return make_bound_storage_tensor(
      inst, name, POLY_ROLE_PARAM, dt, shape, ndim, true, POLY_TENSOR_PROVENANCE_PARAM_INIT
  );
}

PolyStatus poly_instance_state(
    PolyInstance *inst,
    const char *name,
    PolyTensor *tensor,
    uint32_t flags
) {
  return append_existing_tensor_binding(
      inst, name, POLY_ROLE_PARAM, tensor, flags, true, true, POLY_TENSOR_PROVENANCE_STATE_LOADED
  );
}

PolyStatus poly_instance_output(PolyInstance *inst, const char *name, PolyTensor *tensor) {
  return append_existing_tensor_binding(
      inst, name, POLY_ROLE_OUTPUT, tensor, 0, false, false, POLY_TENSOR_PROVENANCE_COMPUTED
  );
}

PolyStatus poly_instance_aux(
    PolyInstance *inst,
    const char *name,
    PolyTensor *tensor,
    uint32_t flags
) {
  return append_existing_tensor_binding(
      inst, name, POLY_ROLE_AUX, tensor, flags, true, false, POLY_TENSOR_PROVENANCE_STATE_LOADED
  );
}

PolyStatus poly_instance_entrypoint(
    PolyInstance *inst,
    const char *name,
    const char **inputs,
    int n_inputs,
    const char **outputs,
    int n_outputs,
    const PolyEntrypointOptions *opts
) {
  if (require_stage(inst, POLY_INSTANCE_BUILDING, __func__) != POLY_STATUS_OK)
    return POLY_STATUS_BAD_STAGE;
  if (!valid_binding_name(name) || n_inputs < 0 || n_outputs <= 0 || (n_inputs > 0 && !inputs) ||
      !outputs) {
    poly_instance_set_error(
        inst, POLY_STATUS_INVALID, __func__, "invalid entrypoint '%s'", name ? name : "?"
    );
    return POLY_STATUS_INVALID;
  }
  if (grow_array(
          (void **)&inst->build->entrypoints, &inst->build->entrypoints_cap,
          inst->build->n_entrypoints + 1, sizeof(BuildEntrypoint)
      ) != 0) {
    poly_instance_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    return POLY_STATUS_NOMEM;
  }
  BuildEntrypoint *ep = &inst->build->entrypoints[inst->build->n_entrypoints++];
  *ep = (BuildEntrypoint){0};
  ep->name = dup_cstr(name);
  ep->inputs = dup_string_array(inputs, n_inputs);
  ep->n_inputs = n_inputs;
  ep->outputs = dup_string_array(outputs, n_outputs);
  ep->n_outputs = n_outputs;
  ep->objective = opts && opts->objective ? dup_cstr(opts->objective) : NULL;
  ep->flags = opts ? opts->flags : 0;
  if (!ep->name || (n_inputs > 0 && !ep->inputs) || !ep->outputs ||
      (opts && opts->objective && !ep->objective)) {
    build_entrypoint_free(ep);
    inst->build->n_entrypoints--;
    poly_instance_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    return POLY_STATUS_NOMEM;
  }
  return POLY_STATUS_OK;
}

static BuildBinding *find_build_storage_binding(
    PolyInstanceBuildState *build,
    const PolyUOp *storage
) {
  if (!build || !storage) return NULL;
  for (int i = 0; i < build->n_bindings; i++) {
    BuildBinding *b = &build->bindings[i];
    if (b->role == POLY_ROLE_OUTPUT) continue;
    if (b->buffer == storage) return b;
  }
  return NULL;
}

static PolyStatus validate_build_reachable_storage(PolyInstance *inst) {
  PolyInstanceBuildState *build = inst ? inst->build : NULL;
  if (!inst || !build || !inst->ctx) return POLY_STATUS_INVALID;

  for (int i = 0; i < build->n_bindings; i++) {
    BuildBinding *out = &build->bindings[i];
    if (out->role != POLY_ROLE_OUTPUT) continue;

    PolyUOp *root = poly_tensor_uop(out->tensor);
    if (!root) {
      poly_instance_set_error(
          inst, POLY_STATUS_INVALID, __func__, "output '%s' has no tensor root", out->name
      );
      return POLY_STATUS_INVALID;
    }

    PolyScratchMark scratch = poly_ctx_scratch_mark(inst->ctx);
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_scratch(inst->ctx, root, &n_topo);
    if (!topo && n_topo != 0) {
      poly_ctx_scratch_rewind(inst->ctx, scratch);
      poly_instance_set_error(
          inst, POLY_STATUS_ERROR, __func__, "failed to walk output '%s' graph", out->name
      );
      return POLY_STATUS_ERROR;
    }

    for (int j = 0; j < n_topo; j++) {
      PolyUOp *u = topo[j];
      if (!u || (u->op != POLY_OP_BUFFER && u->op != POLY_OP_BUFFER_VIEW && u->op != POLY_OP_PARAM))
        continue;
      if (find_build_storage_binding(build, u)) continue;
      PolyTensor *leaf_tensor = poly_tensor_find_storage_identity(inst->ctx, u);
      if (leaf_tensor && poly_tensor_requires_grad(leaf_tensor)) {
        poly_ctx_scratch_rewind(inst->ctx, scratch);
        poly_instance_set_error(
            inst, POLY_STATUS_INVALID, __func__,
            "output '%s' references unbound trainable storage %s", out->name, poly_op_name(u->op)
        );
        return POLY_STATUS_INVALID;
      }
      if (leaf_tensor && poly_tensor_provenance(leaf_tensor) != POLY_TENSOR_PROVENANCE_UNKNOWN &&
          poly_tensor_provenance(leaf_tensor) != POLY_TENSOR_PROVENANCE_CONST_INIT) {
        poly_ctx_scratch_rewind(inst->ctx, scratch);
        poly_instance_set_error(
            inst, POLY_STATUS_INVALID, __func__, "output '%s' references unbound %s storage %s",
            out->name,
            poly_tensor_provenance(leaf_tensor) == POLY_TENSOR_PROVENANCE_USER_INPUT ? "input"
                                                                                     : "state",
            poly_op_name(u->op)
        );
        return POLY_STATUS_INVALID;
      }
      poly_ctx_scratch_rewind(inst->ctx, scratch);
      poly_instance_set_error(
          inst, POLY_STATUS_INVALID, __func__, "output '%s' references unbound storage %s",
          out->name, poly_op_name(u->op)
      );
      return POLY_STATUS_INVALID;
    }
    poly_ctx_scratch_rewind(inst->ctx, scratch);
  }
  return POLY_STATUS_OK;
}

static PolyStatus validate_build_entrypoints(PolyInstance *inst) {
  PolyInstanceBuildState *build = inst->build;
  if (build->n_entrypoints <= 0) {
    poly_instance_set_error(inst, POLY_STATUS_INVALID, __func__, "instance has no entrypoints");
    return POLY_STATUS_INVALID;
  }
  for (int i = 0; i < build->n_entrypoints; i++) {
    BuildEntrypoint *ep = &build->entrypoints[i];
    for (int j = 0; j < ep->n_inputs; j++) {
      BuildBinding *b = find_build_binding(build, ep->inputs[j]);
      if (!b || !(b->role == POLY_ROLE_INPUT || b->role == POLY_ROLE_TARGET ||
                  b->role == POLY_ROLE_PARAM || b->role == POLY_ROLE_AUX)) {
        poly_instance_set_error(
            inst, POLY_STATUS_INVALID, __func__, "entrypoint '%s' references unknown input '%s'",
            ep->name, ep->inputs[j]
        );
        return POLY_STATUS_INVALID;
      }
    }
    for (int j = 0; j < ep->n_outputs; j++) {
      BuildBinding *b = find_build_binding(build, ep->outputs[j]);
      if (!b || b->role != POLY_ROLE_OUTPUT) {
        poly_instance_set_error(
            inst, POLY_STATUS_INVALID, __func__, "entrypoint '%s' references unknown output '%s'",
            ep->name, ep->outputs[j]
        );
        return POLY_STATUS_INVALID;
      }
    }
    if (ep->objective) {
      BuildBinding *objective = NULL;
      for (int j = 0; j < ep->n_outputs; j++) {
        if (strcmp(ep->objective, ep->outputs[j]) == 0) {
          objective = find_build_binding(build, ep->outputs[j]);
          break;
        }
      }
      if (!objective) {
        poly_instance_set_error(
            inst, POLY_STATUS_INVALID, __func__, "entrypoint '%s' objective '%s' is not an output",
            ep->name, ep->objective
        );
        return POLY_STATUS_INVALID;
      }
      int64_t numel = poly_shape_numel_checked(objective->shape, objective->ndim);
      if (numel != 1) {
        poly_instance_set_error(
            inst, POLY_STATUS_INVALID, __func__,
            "entrypoint '%s' objective '%s' must be scalar or one element", ep->name, ep->objective
        );
        return POLY_STATUS_INVALID;
      }
    }
  }
  return POLY_STATUS_OK;
}

PolyStatus poly_instance_build(PolyInstance *inst, PolyInstanceError *err) {
  if (require_stage(inst, POLY_INSTANCE_BUILDING, __func__) != POLY_STATUS_OK) {
    poly_instance_copy_error(inst, err);
    return POLY_STATUS_BAD_STAGE;
  }
  PolyInstanceBuildState *build = inst->build;
  PolyStatus st = validate_build_entrypoints(inst);
  if (st != POLY_STATUS_OK) goto fail;
  st = validate_build_reachable_storage(inst);
  if (st != POLY_STATUS_OK) goto fail;

  PolyIrBufEntry *bufs = calloc((size_t)build->n_bindings, sizeof(PolyIrBufEntry));
  PolyIrEntrypoint *eps = calloc((size_t)build->n_entrypoints, sizeof(PolyIrEntrypoint));
  if (!bufs || !eps) {
    free(bufs);
    free(eps);
    poly_instance_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    st = POLY_STATUS_NOMEM;
    goto fail;
  }

  for (int i = 0; i < build->n_bindings; i++) {
    BuildBinding *b = &build->bindings[i];
    if (b->role == POLY_ROLE_OUTPUT) {
      PolyUOp *value = poly_tensor_uop(b->tensor);
      int64_t numel = poly_shape_numel_checked(b->shape, b->ndim);
      b->buffer = poly_buffer(inst->ctx, poly_dtype_scalar(value->dtype), numel);
      if (!b->buffer) {
        poly_instance_set_error(
            inst, POLY_STATUS_ERROR, __func__, "failed to create output buffer '%s'", b->name
        );
        st = POLY_STATUS_ERROR;
        free(bufs);
        free(eps);
        goto fail;
      }
    }
    if (!b->buffer) {
      poly_instance_set_error(
          inst, POLY_STATUS_INVALID, __func__, "binding '%s' has no buffer", b->name
      );
      st = POLY_STATUS_INVALID;
      free(bufs);
      free(eps);
      goto fail;
    }
    bufs[i] = (PolyIrBufEntry){
        .name = b->name,
        .role = b->role,
        .buffer = b->buffer,
        .ndim = b->ndim,
        .trainable = (b->role == POLY_ROLE_PARAM) && poly_tensor_requires_grad(b->tensor),
        .trainable_set = true,
    };
    if (b->ndim > 0) memcpy(bufs[i].shape, b->shape, (size_t)b->ndim * sizeof(int64_t));
  }

  for (int i = 0; i < build->n_entrypoints; i++) {
    BuildEntrypoint *ep = &build->entrypoints[i];
    PolyUOp **stores = calloc((size_t)ep->n_outputs, sizeof(PolyUOp *));
    if (!stores) {
      poly_instance_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
      st = POLY_STATUS_NOMEM;
      free(bufs);
      free(eps);
      goto fail;
    }
    for (int j = 0; j < ep->n_outputs; j++) {
      BuildBinding *out = find_build_binding(build, ep->outputs[j]);
      PolyUOp *value = poly_tensor_uop(out->tensor);
      int64_t numel = poly_shape_numel_checked(out->shape, out->ndim);
      if (!(out->ndim == 1 && out->shape[0] == numel)) {
        int64_t flat[] = {numel};
        value = poly_reshape(inst->ctx, value, flat, 1);
      }
      stores[j] = poly_store_val(inst->ctx, out->buffer, value);
    }
    eps[i].name = ep->name;
    eps[i].sink = ep->n_outputs == 1 ? poly_sink1(inst->ctx, stores[0])
                                     : poly_sink_n(inst->ctx, stores, ep->n_outputs);
    eps[i].inputs = (const char **)ep->inputs;
    eps[i].n_inputs = ep->n_inputs;
    eps[i].outputs = (const char **)ep->outputs;
    eps[i].n_outputs = ep->n_outputs;
    eps[i].objective = ep->objective;
    eps[i].flags = ep->flags;
    free(stores);
    if (!eps[i].sink) {
      poly_instance_set_error(
          inst, POLY_STATUS_ERROR, __func__, "failed to build entrypoint '%s'", ep->name
      );
      st = POLY_STATUS_ERROR;
      free(bufs);
      free(eps);
      goto fail;
    }
  }

  PolyIrSpec spec = {
      .ctx = inst->ctx,
      .bufs = bufs,
      .n_bufs = build->n_bindings,
      .entrypoints = eps,
      .n_entrypoints = build->n_entrypoints,
  };
  PolyInstance *built = instance_from_spec(&spec, false, false);
  free(bufs);
  free(eps);
  if (!built) {
    poly_instance_set_error(inst, POLY_STATUS_ERROR, __func__, "failed to pack runtime instance");
    st = POLY_STATUS_ERROR;
    goto fail;
  }
  copy_initial_buffer_data(built, inst->ctx);

  PolyInstanceOptions opts = build->opts;
  build_state_free(build);
  *inst = *built;
  free(built);
  inst->stage = POLY_INSTANCE_BUILT;
  inst->build = NULL;
  inst->owns_ctx = opts.own_ctx_on_success;
  memset(&inst->last_error, 0, sizeof(inst->last_error));
  if (err) memset(err, 0, sizeof(*err));
  return POLY_STATUS_OK;

fail:
  inst->stage = POLY_INSTANCE_FAILED;
  if (build && build->opts.own_ctx_on_failure) inst->owns_ctx = true;
  poly_instance_copy_error(inst, err);
  return st;
}

PolyInstance *poly_instance_from_bindings(
    PolyCtx *ctx,
    const PolyBindingSpec *bindings,
    int n_bindings,
    const PolyEntrypointSpec *entrypoints,
    int n_entrypoints,
    const PolyInstanceOptions *opts,
    PolyInstanceError *err
) {
  if (!ctx || !bindings || n_bindings <= 0 || !entrypoints || n_entrypoints <= 0) return NULL;
  PolyInstance *inst = poly_instance_new(ctx, opts);
  if (!inst) return NULL;
  for (int i = 0; i < n_bindings; i++) {
    const PolyBindingSpec *b = &bindings[i];
    bool output = b->role == POLY_ROLE_OUTPUT;
    bool trainable = b->role == POLY_ROLE_PARAM;
    PolyTensorProvenance provenance = POLY_TENSOR_PROVENANCE_UNKNOWN;
    switch (b->role) {
    case POLY_ROLE_INPUT:
    case POLY_ROLE_TARGET:
      provenance = POLY_TENSOR_PROVENANCE_USER_INPUT;
      break;
    case POLY_ROLE_PARAM:
    case POLY_ROLE_AUX:
      provenance = POLY_TENSOR_PROVENANCE_STATE_LOADED;
      break;
    case POLY_ROLE_OUTPUT:
      provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
      break;
    default:
      break;
    }
    PolyStatus st = append_existing_tensor_binding(
        inst, b->name, (uint8_t)b->role, b->tensor, b->flags, !output, trainable, provenance
    );
    if (st != POLY_STATUS_OK) {
      poly_instance_copy_error(inst, err);
      poly_instance_free(inst);
      return NULL;
    }
  }
  for (int i = 0; i < n_entrypoints; i++) {
    const PolyEntrypointSpec *ep = &entrypoints[i];
    PolyEntrypointOptions ep_opts = {.objective = ep->objective, .flags = ep->flags};
    PolyStatus st = poly_instance_entrypoint(
        inst, ep->name, ep->inputs, ep->n_inputs, ep->outputs, ep->n_outputs, &ep_opts
    );
    if (st != POLY_STATUS_OK) {
      poly_instance_copy_error(inst, err);
      poly_instance_free(inst);
      return NULL;
    }
  }
  if (poly_instance_build(inst, err) != POLY_STATUS_OK) {
    poly_instance_free(inst);
    return NULL;
  }
  return inst;
}

static void set_plain_instance_error(
    PolyInstanceError *err,
    int code,
    const char *func,
    const char *msg
) {
  if (!err) return;
  err->code = code;
  err->func = func;
  snprintf(err->message, sizeof(err->message), "%s", msg ? msg : "error");
}

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
) {
  if (!ctx || !binding_names || !binding_roles || !binding_tensors || n_bindings <= 0 ||
      !entry_names || !entry_input_counts || !entry_output_counts || n_entrypoints <= 0) {
    set_plain_instance_error(err, POLY_STATUS_INVALID, __func__, "invalid binding array inputs");
    return NULL;
  }

  PolyBindingSpec *bindings = calloc((size_t)n_bindings, sizeof(PolyBindingSpec));
  PolyEntrypointSpec *entrypoints = calloc((size_t)n_entrypoints, sizeof(PolyEntrypointSpec));
  if (!bindings || !entrypoints) {
    free(bindings);
    free(entrypoints);
    set_plain_instance_error(err, POLY_STATUS_NOMEM, __func__, "out of memory");
    return NULL;
  }

  for (int i = 0; i < n_bindings; i++) {
    bindings[i].name = binding_names[i];
    bindings[i].role = binding_roles[i];
    bindings[i].tensor = binding_tensors[i];
    bindings[i].flags = binding_flags ? binding_flags[i] : 0;
  }

  int input_off = 0;
  int output_off = 0;
  for (int i = 0; i < n_entrypoints; i++) {
    int n_inputs = entry_input_counts[i];
    int n_outputs = entry_output_counts[i];
    if (n_inputs < 0 || n_outputs < 0 || (n_inputs > 0 && !entry_inputs) ||
        (n_outputs > 0 && !entry_outputs)) {
      free(bindings);
      free(entrypoints);
      set_plain_instance_error(err, POLY_STATUS_INVALID, __func__, "invalid entrypoint arrays");
      return NULL;
    }
    entrypoints[i].name = entry_names[i];
    entrypoints[i].inputs = n_inputs ? &entry_inputs[input_off] : NULL;
    entrypoints[i].n_inputs = n_inputs;
    entrypoints[i].outputs = n_outputs ? &entry_outputs[output_off] : NULL;
    entrypoints[i].n_outputs = n_outputs;
    entrypoints[i].objective = entry_objectives ? entry_objectives[i] : NULL;
    entrypoints[i].flags = entry_flags ? entry_flags[i] : 0;
    input_off += n_inputs;
    output_off += n_outputs;
  }

  PolyInstance *inst =
      poly_instance_from_bindings(ctx, bindings, n_bindings, entrypoints, n_entrypoints, opts, err);
  free(bindings);
  free(entrypoints);
  return inst;
}

/* Lifecycle */

/* Build a PolyInstance from a PolyIrSpec.
 * owns_ctx: if true, the instance takes ownership of spec->ctx.
 * free_spec: if true, calls poly_ir_spec_free after building.
 * Handles alias data sharing: entries with the same buffer UOp share one allocation. */
static PolyInstance *instance_from_spec(PolyIrSpec *spec, bool owns_ctx, bool free_spec) {
  PolyInstance *inst = calloc(1, sizeof(PolyInstance));
  inst->ctx = spec->ctx;
  inst->owns_ctx = owns_ctx;
  inst->stage = POLY_INSTANCE_BUILT;

  /* Copy buffers with alias data sharing */
  inst->n_bufs = spec->n_bufs;
  inst->bufs = calloc(spec->n_bufs, sizeof(NamedBuf));
  int n_params = 0;
  int n_trainable = 0;
  for (int i = 0; i < spec->n_bufs; i++) {
    inst->bufs[i].name = strdup(spec->bufs[i].name);
    inst->bufs[i].role = spec->bufs[i].role;
    inst->bufs[i].trainable = spec->bufs[i].trainable_set ? spec->bufs[i].trainable
                                                          : (spec->bufs[i].role == POLY_ROLE_PARAM);
    inst->bufs[i].buffer = spec->bufs[i].buffer;
    inst->bufs[i].ndim = spec->bufs[i].ndim;
    memcpy(inst->bufs[i].shape, spec->bufs[i].shape, spec->bufs[i].ndim * sizeof(int64_t));
    inst->bufs[i].numel = compute_numel(spec->bufs[i].shape, spec->bufs[i].ndim);

    /* Check if an earlier entry shares the same buffer UOp (alias) */
    void *shared = NULL;
    for (int j = 0; j < i; j++) {
      if (inst->bufs[j].buffer == spec->bufs[i].buffer) {
        shared = inst->bufs[j].data;
        break;
      }
    }
    if (shared) {
      inst->bufs[i].data = shared;
      inst->bufs[i].owns_data = false;
    } else {
      size_t nbytes = named_buf_nbytes(&inst->bufs[i]);
      PolyBuffer *host = NULL;
      if (nbytes > 0 &&
          poly_buffer_alloc_owned_host(spec->ctx, spec->bufs[i].buffer, nbytes, true, &host) != 0) {
        fprintf(
            stderr, "poly_instance: host buffer allocation FAILED for '%s' (%lld elements = %lld MB)\n",
            spec->bufs[i].name ? spec->bufs[i].name : "?", (long long)inst->bufs[i].numel,
            (long long)(nbytes / 1024 / 1024)
        );
        goto fail;
      }
      inst->bufs[i].data = host ? host->ptr : NULL;
      inst->bufs[i].owns_data = false;
    }
    if (spec->bufs[i].role == POLY_ROLE_PARAM) {
      n_params++;
      if (inst->bufs[i].trainable) n_trainable++;
    }
  }

  /* Build param index table */
  inst->n_params = n_params;
  inst->param_indices = malloc(n_params * sizeof(int));
  inst->n_trainable_params = n_trainable;
  inst->trainable_param_indices = malloc(n_trainable * sizeof(int));
  int pi = 0;
  int tpi = 0;
  for (int i = 0; i < spec->n_bufs; i++) {
    if (spec->bufs[i].role != POLY_ROLE_PARAM) continue;
    inst->param_indices[pi++] = i;
    if (inst->bufs[i].trainable) inst->trainable_param_indices[tpi++] = i;
  }

  /* Copy entrypoints */
  inst->n_entrypoints = spec->n_entrypoints;
  inst->entrypoints = calloc(spec->n_entrypoints, sizeof(*inst->entrypoints));
  inst->entry_schedules = calloc(spec->n_entrypoints, sizeof(*inst->entry_schedules));
  for (int i = 0; i < spec->n_entrypoints; i++) {
    inst->entrypoints[i].name = strdup(spec->entrypoints[i].name);
    inst->entrypoints[i].sink = spec->entrypoints[i].sink;
    inst->entrypoints[i].inputs =
        dup_string_array(spec->entrypoints[i].inputs, spec->entrypoints[i].n_inputs);
    inst->entrypoints[i].n_inputs = spec->entrypoints[i].n_inputs;
    inst->entrypoints[i].outputs =
        dup_string_array(spec->entrypoints[i].outputs, spec->entrypoints[i].n_outputs);
    inst->entrypoints[i].n_outputs = spec->entrypoints[i].n_outputs;
    inst->entrypoints[i].objective = dup_cstr(spec->entrypoints[i].objective);
    inst->entrypoints[i].flags = spec->entrypoints[i].flags;
  }

  if (free_spec) poly_ir_spec_free(spec);

  /* Default optimizer: none */
  inst->optim.kind = POLY_OPTIM_NONE;

  return inst;

fail:
  if (free_spec) poly_ir_spec_free(spec);
  poly_instance_free(inst);
  return NULL;
}

PolyInstance *poly_instance_from_ir(
    const uint8_t *ir_data,
    int ir_len,
    const uint8_t *weights_data,
    int weights_len
) {
  /* Import IR */
  PolyIrSpec spec;
  if (poly_ir_import(ir_data, ir_len, &spec) != 0) {
    fprintf(stderr, "poly_instance_from_ir: IR import failed\n");
    return NULL;
  }

  PolyInstance *inst = instance_from_spec(&spec, true, true);
  if (!inst) return NULL;

  /* Import weights if provided */
  if (weights_data && weights_len > 0) {
    if (poly_instance_import_weights(inst, weights_data, weights_len) != 0) {
      fprintf(stderr, "poly_instance_from_ir: weight import failed\n");
      poly_instance_free(inst);
      return NULL;
    }
  }

  return inst;
}

/* Instance from PolyCtx registry */

#include "frontend_internal.h" /* poly_ptr_hash, poly_ptr_eq */

static PolyInstance *instance_from_named_sinks(
    PolyCtx *ctx,
    const char **names,
    PolyUOp **sinks,
    int n_sinks
) {
  if (!ctx || !names || !sinks || n_sinks <= 0) {
    fprintf(stderr, "poly_instance_from_sinks: zero entrypoints\n");
    return NULL;
  }

  /* Collect reachable BUFFER UOps from all entrypoint SINKs */
  PolyMap *reachable = poly_map_new(32);
  if (!reachable) return NULL;
  for (int i = 0; i < n_sinks; i++) {
    if (!names[i] || !sinks[i]) {
      fprintf(stderr, "poly_instance_from_sinks: null entrypoint at %d\n", i);
      poly_map_destroy(reachable);
      return NULL;
    }
    PolyUOp *sink = sinks[i];
    PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_scratch(ctx, sink, &n_topo);
    if (!topo && n_topo != 0) {
      fprintf(stderr, "poly_instance_from_sinks: failed to walk entrypoint %d\n", i);
      poly_ctx_scratch_rewind(ctx, scratch);
      poly_map_destroy(reachable);
      return NULL;
    }
    for (int j = 0; j < n_topo; j++) {
      if (topo[j]->op == POLY_OP_BUFFER) {
        uint32_t h = poly_ptr_hash(topo[j]);
        if (!poly_map_get(reachable, h, topo[j], poly_ptr_eq))
          poly_map_set(reachable, h, topo[j], topo[j], poly_ptr_eq);
      }
    }
    poly_ctx_scratch_rewind(ctx, scratch);
  }

  /* Build PolyIrBufEntry array from registry entries (reachable only) */
  int n_reg = poly_ctx_named_count(ctx);
  PolyIrBufEntry *bufs = calloc(n_reg, sizeof(PolyIrBufEntry));
  int n_bufs = 0;
  for (int i = 0; i < n_reg; i++) {
    const PolyRegEntry *e = poly_ctx_named_entry(ctx, i);
    uint32_t h = poly_ptr_hash(e->buffer);
    if (!poly_map_get(reachable, h, e->buffer, poly_ptr_eq)) continue;
    bufs[n_bufs] = (PolyIrBufEntry){
        .name = e->name,
        .role = (uint8_t)e->role,
        .buffer = e->buffer,
        .ndim = e->ndim,
        .trainable = e->trainable,
        .trainable_set = true,
    };
    memcpy(bufs[n_bufs].shape, e->shape, e->ndim * sizeof(int64_t));
    n_bufs++;
  }
  poly_map_destroy(reachable);

  /* Build entrypoints array */
  PolyIrEntrypoint *eps = calloc((size_t)n_sinks, sizeof(PolyIrEntrypoint));
  for (int i = 0; i < n_sinks; i++) {
    eps[i].name = names[i];
    eps[i].sink = sinks[i];
  }

  /* Build spec and create instance (does NOT own ctx) */
  PolyIrSpec spec = {
      .ctx = ctx,
      .bufs = bufs,
      .n_bufs = n_bufs,
      .entrypoints = eps,
      .n_entrypoints = n_sinks,
  };
  PolyInstance *inst = instance_from_spec(&spec, false, false);
  if (inst) {
    for (int i = 0; i < inst->n_bufs; i++) {
      PolyBuffer *src = poly_buffer_get(ctx, inst->bufs[i].buffer);
      size_t nbytes = named_buf_nbytes(&inst->bufs[i]);
      if (!src || !src->ptr || src->nbytes < nbytes || nbytes == 0) continue;
      /* Instances own host-side ABI storage, but the source tensor may already
       * live in a backend-specific residency (WASM heap, CUDA, WebGPU, etc.).
       * `valid` only describes the host shadow freshness; backend copyout can
       * still read the current residency, so do not reject invalid host shadows.
       * Fall back to direct copy only for valid legacy host buffers whose
       * allocator cannot service copyout. */
      if (poly_buffer_read(ctx, inst->bufs[i].buffer, inst->bufs[i].data, nbytes) != 0 &&
          src->valid && poly_device_is_host_addressable(src->device))
        memcpy(inst->bufs[i].data, src->ptr, nbytes);
    }
  }

  free(bufs);
  free(eps);
  return inst;
}

PolyInstance *poly_instance_from_ctx(PolyCtx *ctx) {
  if (!ctx) return NULL;
  int n_ep = poly_ctx_entrypoint_count(ctx);
  if (n_ep == 0) {
    fprintf(stderr, "poly_instance_from_ctx: zero entrypoints\n");
    return NULL;
  }

  const char **names = calloc((size_t)n_ep, sizeof(char *));
  PolyUOp **sinks = calloc((size_t)n_ep, sizeof(PolyUOp *));
  if (!names || !sinks) {
    free(names);
    free(sinks);
    return NULL;
  }
  for (int i = 0; i < n_ep; i++) {
    names[i] = poly_ctx_entrypoint_name(ctx, i);
    sinks[i] = poly_ctx_entrypoint_sink(ctx, i);
  }
  PolyInstance *inst = instance_from_named_sinks(ctx, names, sinks, n_ep);
  free(names);
  free(sinks);
  return inst;
}

PolyInstance *poly_instance_from_sinks(
    PolyCtx *ctx,
    const char **names,
    PolyUOp **sinks,
    int n_sinks
) {
  return instance_from_named_sinks(ctx, names, sinks, n_sinks);
}

static void vag_free(VagState *vag, int n_params) {
  if (!vag) return;
  if (vag->grad_datas) {
    for (int i = 0; i < n_params; i++)
      free(vag->grad_datas[i]);
    free(vag->grad_datas);
  }
  free(vag->grad_out_bufs);
  free(vag->grad_uops);
  free(vag);
}

static void train_schedule_clear(TrainState *ts) {
  if (!ts) return;
  poly_schedule_free(ts->schedule);
  ts->schedule = NULL;
}

static void entry_schedules_clear(PolyInstance *inst) {
  if (!inst || !inst->entry_schedules) return;
  for (int i = 0; i < inst->n_entrypoints; i++) {
    poly_schedule_free(inst->entry_schedules[i]);
    inst->entry_schedules[i] = NULL;
  }
}

static void train_free(TrainState *ts, int n_params) {
  if (!ts) return;
  train_schedule_clear(ts);
  free(ts->m_bufs);
  free(ts->v_bufs);
  free(ts);
}

void poly_instance_free(PolyInstance *inst) {
  if (!inst) return;

  build_state_free(inst->build);
  inst->build = NULL;

  /* Free named buffers */
  for (int i = 0; i < inst->n_bufs; i++) {
    free(inst->bufs[i].name);
    if (inst->bufs[i].owns_data) free(inst->bufs[i].data);
  }
  free(inst->bufs);
  free(inst->param_indices);
  free(inst->trainable_param_indices);

  /* Free entrypoints */
  if (inst->entry_schedules) {
    entry_schedules_clear(inst);
    free(inst->entry_schedules);
  }
  for (int i = 0; i < inst->n_entrypoints; i++) {
    free(inst->entrypoints[i].name);
    for (int j = 0; j < inst->entrypoints[i].n_inputs; j++)
      free(inst->entrypoints[i].inputs[j]);
    free(inst->entrypoints[i].inputs);
    for (int j = 0; j < inst->entrypoints[i].n_outputs; j++)
      free(inst->entrypoints[i].outputs[j]);
    free(inst->entrypoints[i].outputs);
    free(inst->entrypoints[i].objective);
  }
  free(inst->entrypoints);

  /* Free execution caches.
   * Order: exec cache first (holds pointers into prepared steps),
   * then context (arena-frees all UOps). */

  /* Free value-and-grad state */
  vag_free(inst->vag, inst->n_params);

  /* Free training state */
  train_free(inst->train, inst->n_params);

  /* Free context (arena-frees all UOps) -- only if we own it */
  if (inst->owns_ctx && inst->ctx) poly_ctx_destroy(inst->ctx);

  free(inst);
}

/* Param Enumeration */

int poly_instance_param_count(const PolyInstance *inst) {
  return inst ? inst->n_params : 0;
}

const char *poly_instance_param_name(const PolyInstance *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_params) return NULL;
  return inst->bufs[inst->param_indices[i]].name;
}

int poly_instance_param_shape(const PolyInstance *inst, int i, int64_t *shape_out, int max_dims) {
  if (!inst || i < 0 || i >= inst->n_params) return 0;
  NamedBuf *b = &inst->bufs[inst->param_indices[i]];
  int n = b->ndim < max_dims ? b->ndim : max_dims;
  memcpy(shape_out, b->shape, n * sizeof(int64_t));
  return b->ndim;
}

/* Sync device buffer to host shadow if on a non-host domain.
 * Returns 0 on success or if already on host; -1 on readback failure. */
static int sync_buf_to_host(PolyInstance *inst, int bi) {
  if (!inst || bi < 0 || bi >= inst->n_bufs) return -1;
  PolyBuffer *host = NULL;
  if (poly_buffer_ensure_host_current(inst->ctx, inst->bufs[bi].buffer, &host) != 0 || !host ||
      !host->ptr)
    return -1;
  inst->bufs[bi].data = host->ptr;
  return 0;
}

static int append_runtime_named_buffer(
    PolyInstance *inst,
    const char *name,
    uint8_t role,
    uint32_t flags,
    PolyUOp *buffer,
    const int64_t *shape,
    int ndim,
    bool trainable,
    bool zero,
    int *out_index
) {
  if (out_index) *out_index = -1;
  if (!inst || !name || !buffer || ndim < 0 || ndim > 8 || (ndim > 0 && !shape))
    return -1;
  if (!valid_binding_name(name) || find_buf_by_name(inst, name) >= 0) return -1;

  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel < 0) return -1;

  NamedBuf *next = realloc(inst->bufs, (size_t)(inst->n_bufs + 1) * sizeof(NamedBuf));
  if (!next) return -1;
  inst->bufs = next;

  NamedBuf nb = {0};
  nb.name = strdup(name);
  if (!nb.name) return -1;
  nb.role = role;
  nb.flags = flags;
  nb.buffer = buffer;
  nb.ndim = ndim;
  if (ndim > 0) memcpy(nb.shape, shape, (size_t)ndim * sizeof(int64_t));
  nb.numel = numel;
  nb.trainable = trainable;

  size_t nbytes = named_buf_nbytes(&nb);
  PolyBuffer *host = NULL;
  if (nbytes > 0 && poly_buffer_alloc_owned_host(inst->ctx, buffer, nbytes, zero, &host) != 0) {
    free(nb.name);
    return -1;
  }
  nb.data = host ? host->ptr : NULL;
  nb.owns_data = false;

  int idx = inst->n_bufs++;
  inst->bufs[idx] = nb;
  if (out_index) *out_index = idx;
  return 0;
}

static char *optimizer_param_state_name(const char *optimizer, const char *kind, const char *param_name) {
  if (!optimizer || !kind || !param_name) return NULL;
  const char *prefix = "optim.";
  size_t lp = strlen(prefix), lo = strlen(optimizer), lk = strlen(kind), ln = strlen(param_name);
  char *out = malloc(lp + lo + 1 + lk + 1 + ln + 1);
  if (!out) return NULL;
  memcpy(out, prefix, lp);
  memcpy(out + lp, optimizer, lo);
  out[lp + lo] = '.';
  memcpy(out + lp + lo + 1, kind, lk);
  out[lp + lo + 1 + lk] = '.';
  memcpy(out + lp + lo + 1 + lk + 1, param_name, ln + 1);
  return out;
}

static int ensure_optimizer_state_buffer(
    PolyInstance *inst,
    const char *name,
    const int64_t *shape,
    int ndim,
    bool init_scalar,
    float scalar_value,
    PolyUOp **out
) {
  if (out) *out = NULL;
  if (!inst || !name || !shape || ndim <= 0 || ndim > 8) return -1;
  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel <= 0) return -1;

  int bi = find_buf_by_name(inst, name);
  if (bi >= 0) {
    NamedBuf *b = &inst->bufs[bi];
    if (!b->buffer || b->role != POLY_ROLE_AUX || !poly_dtype_eq(poly_dtype_scalar(b->buffer->dtype), POLY_FLOAT32) ||
        b->numel != numel)
      return -1;
    b->flags |= POLY_BIND_F_OPTIM;
    if (out) *out = b->buffer;
    return 0;
  }

  PolyUOp *buf = poly_buffer(inst->ctx, POLY_FLOAT32, numel);
  if (!buf) return -1;
  if (append_runtime_named_buffer(
          inst, name, POLY_ROLE_AUX, POLY_BIND_F_OPTIM, buf, shape, ndim, false, true, &bi
      ) != 0)
    return -1;
  if (init_scalar) {
    if (numel != 1 || poly_buffer_write(inst->ctx, buf, &scalar_value, sizeof(float)) != 0)
      return -1;
    if (sync_buf_to_host(inst, bi) != 0) return -1;
  }
  if (out) *out = buf;
  return 0;
}

float *poly_instance_param_data(PolyInstance *inst, int i, int64_t *numel_out) {
  if (!inst || i < 0 || i >= inst->n_params) return NULL;
  int bi = inst->param_indices[i];
  if (numel_out) *numel_out = inst->bufs[bi].numel;
  if (sync_buf_to_host(inst, bi) != 0) return NULL;
  if (poly_buffer_mark_host_written(inst->ctx, inst->bufs[bi].buffer) != 0) return NULL;
  return (float *)inst->bufs[bi].data;
}

/* Buffer Enumeration */

int poly_instance_buf_count(const PolyInstance *inst) {
  return inst ? inst->n_bufs : 0;
}

const char *poly_instance_buf_name(const PolyInstance *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_bufs) return NULL;
  return inst->bufs[i].name;
}

int poly_instance_buf_role(const PolyInstance *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_bufs) return -1;
  return inst->bufs[i].role;
}

bool poly_instance_buf_trainable(const PolyInstance *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_bufs) return false;
  return inst->bufs[i].trainable;
}

bool poly_instance_param_trainable(const PolyInstance *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_params) return false;
  return inst->bufs[inst->param_indices[i]].trainable;
}

static void rebuild_trainable_param_indices(PolyInstance *inst) {
  if (!inst) return;
  int n = 0;
  for (int i = 0; i < inst->n_params; i++) {
    int bi = inst->param_indices[i];
    if (inst->bufs[bi].trainable) n++;
  }
  int *indices = n > 0 ? malloc((size_t)n * sizeof(int)) : NULL;
  int j = 0;
  for (int i = 0; i < inst->n_params; i++) {
    int bi = inst->param_indices[i];
    if (inst->bufs[bi].trainable) indices[j++] = bi;
  }
  free(inst->trainable_param_indices);
  inst->trainable_param_indices = indices;
  inst->n_trainable_params = n;
}

int poly_instance_set_buf_trainable(PolyInstance *inst, int i, bool trainable) {
  if (!inst || i < 0 || i >= inst->n_bufs) return -1;
  PolyUOp *shared = inst->bufs[i].buffer;
  for (int j = 0; j < inst->n_bufs; j++)
    if (inst->bufs[j].buffer == shared) inst->bufs[j].trainable = trainable;
  if (inst->train) {
    train_free(inst->train, inst->n_params);
    inst->train = NULL;
  }
  if (inst->vag) {
    vag_free(inst->vag, inst->n_params);
    inst->vag = NULL;
  }
  rebuild_trainable_param_indices(inst);
  return 0;
}

int poly_instance_set_param_trainable(PolyInstance *inst, int i, bool trainable) {
  if (!inst || i < 0 || i >= inst->n_params) return -1;
  return poly_instance_set_buf_trainable(inst, inst->param_indices[i], trainable);
}

int poly_instance_buf_shape(const PolyInstance *inst, int i, int64_t *shape_out, int max_dims) {
  if (!inst || i < 0 || i >= inst->n_bufs) return 0;
  int n = inst->bufs[i].ndim < max_dims ? inst->bufs[i].ndim : max_dims;
  memcpy(shape_out, inst->bufs[i].shape, n * sizeof(int64_t));
  return inst->bufs[i].ndim;
}

float *poly_instance_buf_data(PolyInstance *inst, int i, int64_t *numel_out) {
  if (!inst || i < 0 || i >= inst->n_bufs) return NULL;
  if (numel_out) *numel_out = inst->bufs[i].numel;
  if (sync_buf_to_host(inst, i) != 0) return NULL;
  if (poly_buffer_mark_host_written(inst->ctx, inst->bufs[i].buffer) != 0) return NULL;
  return (float *)inst->bufs[i].data;
}

/* Readback / Upload */

int poly_instance_readback_buf(PolyInstance *inst, int i, void *host_dst, size_t dst_len) {
  if (!inst || i < 0 || i >= inst->n_bufs) return -1;
  return poly_buffer_read(inst->ctx, inst->bufs[i].buffer, host_dst, dst_len);
}

int poly_instance_upload_buf(PolyInstance *inst, int i, const void *host_src, size_t src_len) {
  if (!inst || i < 0 || i >= inst->n_bufs) return -1;
  int rc = poly_buffer_write(inst->ctx, inst->bufs[i].buffer, host_src, src_len);
  if (rc == 0) sync_buf_to_host(inst, i);
  return rc;
}

int poly_instance_readback_param(PolyInstance *inst, int i, void *host_dst, size_t dst_len) {
  if (!inst || i < 0 || i >= inst->n_params) return -1;
  return poly_instance_readback_buf(inst, inst->param_indices[i], host_dst, dst_len);
}

int poly_instance_upload_param(PolyInstance *inst, int i, const void *host_src, size_t src_len) {
  if (!inst || i < 0 || i >= inst->n_params) return -1;
  return poly_instance_upload_buf(inst, inst->param_indices[i], host_src, src_len);
}

/* Weight I/O */

uint8_t *poly_instance_export_weights_ex(PolyInstance *inst, int *out_len, uint32_t flags) {
  if (out_len) *out_len = 0;
  if (!inst || inst->stage != POLY_INSTANCE_BUILT) {
    return NULL;
  }

  int n_export = 0;
  for (int i = 0; i < inst->n_bufs; i++)
    if (should_export_weight_buf_flags(&inst->bufs[i], flags)) n_export++;
  if (n_export == 0) return NULL;

  PolySafetensorEntry *entries = malloc((size_t)n_export * sizeof(PolySafetensorEntry));
  if (!entries) return NULL;
  int ei = 0;
  for (int i = 0; i < inst->n_bufs; i++) {
    NamedBuf *b = &inst->bufs[i];
    if (!should_export_weight_buf_flags(b, flags)) continue;
    if (sync_buf_to_host(inst, i) != 0) {
      free(entries);
      return NULL;
    }
    entries[ei].name = b->name;
    entries[ei].data = (const float *)b->data;
    entries[ei].shape = b->shape;
    entries[ei].ndim = b->ndim;
    ei++;
  }

  uint8_t *bytes = poly_safetensors_encode(entries, n_export, NULL, out_len);
  free(entries);
  return bytes;
}

uint8_t *poly_instance_export_weights(PolyInstance *inst, int *out_len) {
  return poly_instance_export_weights_ex(inst, out_len, POLY_EXPORT_WEIGHTS_DEFAULT);
}

int poly_instance_import_weights(PolyInstance *inst, const uint8_t *data, int len) {
  if (!inst || inst->stage != POLY_INSTANCE_BUILT) return -1;

  int n_views = 0;
  char *metadata = NULL;
  PolySafetensorView *views = poly_safetensors_decode(data, len, &n_views, &metadata);
  if (!views) return -1;

  /* Match by name */
  for (int i = 0; i < n_views; i++) {
    int bi = find_buf_by_name(inst, views[i].name);
    if (bi < 0) {
      fprintf(stderr, "poly_instance_import_weights: unknown tensor '%s'\n", views[i].name);
      /* Continue - non-fatal */
    } else if (inst->bufs[bi].data && views[i].numel == inst->bufs[bi].numel) {
      size_t nbytes = named_buf_nbytes(&inst->bufs[bi]);
      if (nbytes != (size_t)views[i].numel * sizeof(float)) {
        fprintf(
            stderr,
            "poly_instance_import_weights: dtype mismatch for '%s' "
            "(F32 safetensor into %zu-byte storage)\n",
            views[i].name, nbytes
        );
      } else if (poly_buffer_write(inst->ctx, inst->bufs[bi].buffer, views[i].data, nbytes) != 0)
        fprintf(stderr, "poly_instance_import_weights: write failed for '%s'\n", views[i].name);
      else
        sync_buf_to_host(inst, bi);
    } else if (inst->bufs[bi].data) {
      fprintf(
          stderr,
          "poly_instance_import_weights: shape mismatch for '%s' "
          "(expected %lld, got %lld)\n",
          views[i].name, (long long)inst->bufs[bi].numel, (long long)views[i].numel
      );
    }
    free(views[i].name);
  }
  free(views);
  free(metadata);
  return 0;
}

/* IR Export */

uint8_t *poly_instance_export_ir(PolyInstance *inst, int *out_len) {
  if (!inst || inst->stage != POLY_INSTANCE_BUILT) {
    *out_len = 0;
    return NULL;
  }

  /* Build PolyIrSpec from instance state */
  PolyIrBufEntry *bufs = malloc(inst->n_bufs * sizeof(PolyIrBufEntry));
  for (int i = 0; i < inst->n_bufs; i++) {
    bufs[i].name = inst->bufs[i].name;
    bufs[i].role = inst->bufs[i].role;
    bufs[i].buffer = inst->bufs[i].buffer;
    bufs[i].ndim = inst->bufs[i].ndim;
    bufs[i].trainable = inst->bufs[i].trainable;
    bufs[i].trainable_set = true;
    memcpy(bufs[i].shape, inst->bufs[i].shape, inst->bufs[i].ndim * sizeof(int64_t));
  }

  PolyIrEntrypoint *eps = malloc(inst->n_entrypoints * sizeof(PolyIrEntrypoint));
  for (int i = 0; i < inst->n_entrypoints; i++) {
    eps[i].name = inst->entrypoints[i].name;
    eps[i].sink = inst->entrypoints[i].sink;
    eps[i].inputs = (const char **)inst->entrypoints[i].inputs;
    eps[i].n_inputs = inst->entrypoints[i].n_inputs;
    eps[i].outputs = (const char **)inst->entrypoints[i].outputs;
    eps[i].n_outputs = inst->entrypoints[i].n_outputs;
    eps[i].objective = inst->entrypoints[i].objective;
    eps[i].flags = inst->entrypoints[i].flags;
  }

  PolyIrSpec spec = {inst->ctx, bufs, inst->n_bufs, eps, inst->n_entrypoints};
  uint8_t *bytes = poly_ir_export(&spec, out_len);
  free(bufs);
  free(eps);
  return bytes;
}

/* Device configuration */

int poly_instance_set_device(PolyInstance *inst, PolyDevice device) {
  if (!inst || inst->stage != POLY_INSTANCE_BUILT) return -1;

  /* Resolve AUTO the same way tensor placement does: an explicit environment
   * device wins, otherwise use the platform default. */
  PolyDevice resolved = device;
  if (resolved == POLY_DEVICE_AUTO) {
    const char *dev_env = getenv("POLY_DEVICE");
    if (dev_env && dev_env[0]) {
      resolved = poly_device_by_name(dev_env);
      if (resolved == POLY_DEVICE_AUTO) resolved = poly_device_default();
    } else {
      resolved = poly_device_default();
    }
  }

  /* Validate: backend must exist for this build */
  const PolyBackendDesc *backend = poly_backend_get(resolved);
  if (!backend || !poly_device_can_execute(resolved)) {
    fprintf(stderr, "poly_instance_set_device: unsupported device %d\n", resolved);
    return -1;
  }

#ifdef POLY_HAS_CUDA
  if (resolved == POLY_DEVICE_CUDA && !poly_cuda_available()) {
    fprintf(stderr, "poly_instance_set_device: CUDA not available\n");
    return -1;
  }
#endif
#ifdef POLY_HAS_HIP
  if (resolved == POLY_DEVICE_HIP && !poly_hip_available()) {
    fprintf(stderr, "poly_instance_set_device: HIP not available\n");
    return -1;
  }
#endif

  poly_ctx_set_preferred_device(inst->ctx, resolved);

  /* Prefetch readable ABI buffers. Pure outputs are allocated by the CALL that
   * writes them, matching tinygrad's per-CALL outs/ins execution boundary. */
  for (int i = 0; i < inst->n_bufs; i++) {
    if (inst->bufs[i].role == POLY_ROLE_OUTPUT) continue;
    bool seen = false;
    for (int j = 0; j < i; j++)
      if (inst->bufs[j].buffer == inst->bufs[i].buffer) seen = true;
    if (seen || inst->bufs[i].numel <= 0) continue;
    if (poly_buffer_ensure_device_current(inst->ctx, inst->bufs[i].buffer, resolved) != 0) {
      fprintf(stderr, "poly_instance_set_device: migration failed for buffer %d\n", i);
      return -1;
    }
  }

  entry_schedules_clear(inst);
  train_schedule_clear(inst->train);

  return 0;
}

/* Generic entrypoint execution */

static bool entrypoint_accepts_input(
    const PolyInstance *inst,
    const RuntimeEntrypoint *entry,
    const char *name
) {
  if (!inst || !entry || !name) return false;
  for (int i = 0; i < entry->n_inputs; i++)
    if (entry->inputs[i] && strcmp(entry->inputs[i], name) == 0) return true;
  if (entry->n_inputs > 0) return false;

  /* Legacy ctx-built instances sometimes carry no per-entrypoint input list.
   * Keep the boundary strict by falling back only to binding roles, never to
   * arbitrary name acceptance. */
  int bi = find_buf_by_name(inst, name);
  return bi >= 0 &&
         (inst->bufs[bi].role == POLY_ROLE_INPUT || inst->bufs[bi].role == POLY_ROLE_TARGET);
}

static int prepare_instance_io(
    PolyInstance *inst,
    const RuntimeEntrypoint *entry,
    PolyIOBinding *io,
    int n_io
) {
  if (!inst || !inst->ctx) return -1;

  for (int i = 0; i < n_io; i++) {
    if (!io[i].data) continue;
    if (!io[i].name || !entrypoint_accepts_input(inst, entry, io[i].name)) {
      fprintf(
          stderr, "poly_instance_call: '%s' is not an input of entrypoint '%s'\n",
          io[i].name ? io[i].name : "(null)", entry && entry->name ? entry->name : "(null)"
      );
      return -1;
    }

    int bi = find_buf_by_name(inst, io[i].name);
    if (bi < 0) return -1;
    size_t nbytes = named_buf_nbytes(&inst->bufs[bi]);
    if (poly_buffer_write(inst->ctx, inst->bufs[bi].buffer, io[i].data, nbytes) != 0) return -1;
    if (sync_buf_to_host(inst, bi) != 0) return -1;
  }

  return 0;
}

static int run_instance_sink(
    PolyInstance *inst,
    const RuntimeEntrypoint *entry,
    PolyUOp *sink,
    PolyIOBinding *io,
    int n_io,
    PolySchedule **cached_schedule
) {
  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:instance] enter sink=%p n_io=%d device=%s\n", (void *)sink, n_io,
        poly_device_name(poly_ctx_get_preferred_device(inst->ctx))
    );
    fflush(stderr);
  }
  if (prepare_instance_io(inst, entry, io, n_io) != 0) return -1;
  double t_attach = timing ? poly_now_ms() : 0.0;
  /* Instance entrypoints and train/value-grad combined graphs are already
   * effect sinks. They must skip tensor callify and enter the schedule runner
   * directly, matching tinygrad's separation between tensor realization and
   * schedule execution. */
  PolySchedule *sched = cached_schedule ? *cached_schedule : NULL;
  bool schedule_owned = false;
  if (!sched) {
    sched = poly_complete_create_schedule_with_vars(inst->ctx, sink, POLY_MODE_CALL);
    if (cached_schedule) {
      *cached_schedule = sched;
    } else {
      schedule_owned = true;
    }
  }
  double t_sched = timing ? poly_now_ms() : 0.0;
  int ret = sched ? poly_run_schedule(inst->ctx, sched, NULL, 0) : -1;
  if (schedule_owned) poly_schedule_free(sched);
  if (timing) {
    double t_done = poly_now_ms();
    fprintf(
        stderr,
        "[polygrad:instance] input=%.3fms schedule=%.3fms run=%.3fms total=%.3fms "
        "ret=%d cached=%d\n",
        t_attach - t0, t_sched - t_attach, t_done - t_sched, t_done - t0, ret,
        cached_schedule && *cached_schedule
    );
  }
  return ret;
}

int poly_instance_call(PolyInstance *inst, const char *entrypoint, PolyIOBinding *io, int n_io) {
  if (!inst || inst->stage != POLY_INSTANCE_BUILT || !entrypoint) return -1;

  int ep_idx = find_entrypoint(inst, entrypoint);
  if (ep_idx < 0) {
    fprintf(stderr, "poly_instance_call: no '%s' entrypoint\n", entrypoint);
    return -1;
  }

  PolyUOp *sink = inst->entrypoints[ep_idx].sink;
  PolySchedule **cached = inst->entry_schedules ? &inst->entry_schedules[ep_idx] : NULL;
  return run_instance_sink(inst, &inst->entrypoints[ep_idx], sink, io, n_io, cached);
}

/* Convenience wrapper */

int poly_instance_forward(PolyInstance *inst, PolyIOBinding *inputs, int n_inputs) {
  return poly_instance_call(inst, "forward", inputs, n_inputs);
}

/* Optimizer */

int poly_instance_set_optimizer(
    PolyInstance *inst,
    int kind,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
  return poly_instance_set_optimizer_ex(inst, kind, lr, beta1, beta2, eps, weight_decay, 0.0f, false, false);
}

int poly_instance_set_optimizer_ex(
    PolyInstance *inst,
    int kind,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float momentum,
    bool nesterov,
    bool classic
) {
  if (!inst || inst->stage != POLY_INSTANCE_BUILT) return -1;
  if (momentum < 0.0f) return -1;

  inst->optim.kind = kind;
  inst->optim.lr = lr;
  inst->optim.beta1 = beta1;
  inst->optim.beta2 = beta2;
  inst->optim.eps = eps;
  inst->optim.weight_decay = weight_decay;
  inst->optim.momentum = momentum;
  inst->optim.nesterov = nesterov;
  inst->optim.classic = classic;
  inst->optim.step = 0;

  /* Invalidate training state and slot cache (will be rebuilt lazily) */
  if (inst->train) {
    train_free(inst->train, inst->n_params);
    inst->train = NULL;
  }

  return 0;
}

/* Value and Grad */

/* Compute numel from shape inference. Returns -1 on failure. */
static int64_t uop_numel(PolyCtx *ctx, PolyUOp *u) {
  PolyShape s = poly_uop_shape(ctx, u);
  if (s.ndim < 0) {
    if (s.dims) free(s.dims);
    return -1;
  }
  int64_t n = poly_shape_numel(s);
  if (s.dims) free(s.dims);
  return n;
}

/* Build the combined fwd+bwd SINK for value_and_grad (lazy, once). */
static int ensure_vag_graph(PolyInstance *inst, int loss_ep_idx) {
  if (inst->vag) return 0; /* already built */

  PolyUOp *loss_sink = inst->entrypoints[loss_ep_idx].sink;
  PolyUOp *loss_store = loss_sink->src[0]; /* SINK src[0] = STORE */
  PolyUOp *loss_value = loss_store->src[1]; /* STORE src[1] = value */

  /* Build param target array: use shaped views (RESHAPE) from the loss graph,
   * not raw BUFFERs. Autograd needs the shaped view to produce correct
   * gradient kernels. Raw BUFFER(N) is flat -- differentiating w.r.t. it
   * loses shape context and produces wrong kernel fusion.
   * Mirrors nn.c:490 pattern (param_bufs vs param_uops). */
  PolyUOp **param_bufs = malloc((size_t)inst->n_params * sizeof(PolyUOp *));
  if (!param_bufs) return -1;

  PolyScratchMark scratch = poly_ctx_scratch_mark(inst->ctx);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_scratch(inst->ctx, loss_value, &n_topo);
  if (!topo && n_topo != 0) {
    poly_ctx_scratch_rewind(inst->ctx, scratch);
    free(param_bufs);
    return -1;
  }

  for (int i = 0; i < inst->n_params; i++) {
    NamedBuf *pb = &inst->bufs[inst->param_indices[i]];
    PolyUOp *raw_buf = pb->buffer;
    PolyUOp *shaped = NULL;

    /* Find the RESHAPE in the loss graph whose src[0] is this raw buffer
     * and whose shape matches the declared param shape. */
    for (int j = 0; j < n_topo; j++) {
      if (topo[j]->op != POLY_OP_RESHAPE || topo[j]->n_src < 1 || topo[j]->src[0] != raw_buf)
        continue;
      PolyShape rs = poly_uop_shape(inst->ctx, topo[j]);
      bool match = (rs.ndim == pb->ndim);
      if (match) {
        for (int d = 0; d < rs.ndim; d++) {
          if (rs.dims[d] != pb->shape[d]) {
            match = false;
            break;
          }
        }
      }
      if (rs.ndim > 0 && rs.dims) free(rs.dims);
      if (match) {
        shaped = topo[j];
        break;
      }
    }
    param_bufs[i] = shaped ? shaped : raw_buf;
  }
  poly_ctx_scratch_rewind(inst->ctx, scratch);

  /* Compute gradients */
  PolyUOp **grads = calloc((size_t)inst->n_params, sizeof(PolyUOp *));
  if (!grads) {
    free(param_bufs);
    return -1;
  }
  if (poly_grad_many(inst->ctx, loss_value, NULL, param_bufs, inst->n_params, grads) != 0) {
    fprintf(stderr, "poly_instance: value_and_grad: autograd failed\n");
    free(grads);
    free(param_bufs);
    return -1;
  }

  /* Allocate VagState */
  VagState *vag = calloc(1, sizeof(VagState));
  vag->grad_out_bufs = calloc((size_t)inst->n_params, sizeof(PolyUOp *));
  vag->grad_datas = calloc((size_t)inst->n_params, sizeof(float *));
  vag->grad_uops = calloc((size_t)inst->n_params, sizeof(PolyUOp *));
  vag->loss_value = loss_value;

  /* Build output stores: loss + per-param gradients */
  int n_stores = inst->n_params + 1;
  PolyUOp **stores = calloc((size_t)n_stores, sizeof(PolyUOp *));

  /* Loss output buffer (1 element) */
  PolyDType out_dt = poly_dtype_scalar(loss_value->dtype);
  if (!poly_dtype_is_float(out_dt)) out_dt = POLY_FLOAT32;
  vag->loss_out_buf = poly_buffer(inst->ctx, out_dt, 1);

  PolyUOp *loss_flat = loss_value;
  if (uop_numel(inst->ctx, loss_value) != 1) {
    int64_t one_shape[1] = {1};
    loss_flat = poly_reshape(inst->ctx, loss_value, one_shape, 1);
  }
  stores[0] = poly_store_val(inst->ctx, vag->loss_out_buf, loss_flat);

  /* Save raw gradient UOps for optimizer graph construction */
  for (int i = 0; i < inst->n_params; i++)
    vag->grad_uops[i] = grads[i];

  /* Gradient output buffers */
  for (int i = 0; i < inst->n_params; i++) {
    int64_t numel = uop_numel(inst->ctx, grads[i]);
    if (numel <= 0) {
      fprintf(stderr, "poly_instance: value_and_grad: grad[%d] has unknown shape\n", i);
      free(stores);
      free(grads);
      free(param_bufs);
      vag_free(vag, inst->n_params);
      return -1;
    }
    PolyDType gdt = poly_dtype_scalar(grads[i]->dtype);
    if (!poly_dtype_is_float(gdt)) gdt = POLY_FLOAT32;
    PolyUOp *gbuf = poly_buffer(inst->ctx, gdt, numel);
    vag->grad_out_bufs[i] = gbuf;

    /* Flatten gradient if needed */
    PolyUOp *gflat = grads[i];
    PolyShape gs = poly_uop_shape(inst->ctx, grads[i]);
    if (gs.ndim != 1 || (gs.ndim == 1 && gs.dims[0] != numel)) {
      int64_t flat_shape[1] = {numel};
      gflat = poly_reshape(inst->ctx, grads[i], flat_shape, 1);
    }
    if (gs.dims) free(gs.dims);
    stores[i + 1] = poly_store_val(inst->ctx, gbuf, gflat);

    /* Allocate host storage for gradient data */
    NamedBuf *pb = &inst->bufs[inst->param_indices[i]];
    vag->grad_datas[i] = calloc((size_t)pb->numel, sizeof(float));
  }

  vag->combined_sink = poly_sink_n(inst->ctx, stores, n_stores);

  free(stores);
  free(grads);
  free(param_bufs);

  inst->vag = vag;
  return 0;
}

int poly_instance_value_and_grad(
    PolyInstance *inst,
    const char *entrypoint,
    PolyIOBinding *io,
    int n_io,
    float *loss_out
) {
  if (!inst || inst->stage != POLY_INSTANCE_BUILT || !entrypoint) return -1;

  int ep_idx = find_entrypoint(inst, entrypoint);
  if (ep_idx < 0) {
    fprintf(stderr, "poly_instance_value_and_grad: no '%s' entrypoint\n", entrypoint);
    return -1;
  }

  /* Build combined fwd+bwd graph lazily */
  if (ensure_vag_graph(inst, ep_idx) != 0) return -1;
  VagState *vag = inst->vag;

  int ret = run_instance_sink(inst, &inst->entrypoints[ep_idx], vag->combined_sink, io, n_io, NULL);

  if (ret == 0) {
    if (poly_buffer_read(inst->ctx, vag->loss_out_buf, &vag->loss_data, sizeof(float)) != 0)
      ret = -1;
    for (int i = 0; i < inst->n_params; i++) {
      NamedBuf *pb = &inst->bufs[inst->param_indices[i]];
      if (poly_buffer_read(
              inst->ctx, vag->grad_out_bufs[i], vag->grad_datas[i],
              (size_t)pb->numel * sizeof(float)
          ) != 0)
        ret = -1;
    }
  }

  if (ret != 0) return ret;
  if (loss_out) *loss_out = vag->loss_data;
  return 0;
}

/* Optimizer Graph Builder */

static int param_ordinal_for_buf(const PolyInstance *inst, int buf_idx) {
  if (!inst) return -1;
  for (int i = 0; i < inst->n_params; i++)
    if (inst->param_indices[i] == buf_idx) return i;
  return -1;
}

/* Build optimizer UOp graph (fwd+bwd+optimizer as a single combined SINK).
 * Gradients are consumed directly by AFTER/STORE update effects, not
 * materialized to separate output buffers (D1: no grad stores in optimizer
 * SINK). */
static int ensure_train_graph(PolyInstance *inst, int loss_ep_idx) {
  if (inst->train) return 0; /* already built */

  /* Build fwd+bwd first (gives us loss_value and grad UOps) */
  if (ensure_vag_graph(inst, loss_ep_idx) != 0) return -1;
  VagState *vag = inst->vag;

  PolyCtx *ctx = inst->ctx;
  OptimState *o = &inst->optim;
  int np = inst->n_trainable_params;
  if (np <= 0) {
    fprintf(stderr, "ensure_train_graph: no trainable parameters\n");
    return -1;
  }

  TrainState *ts = calloc(1, sizeof(TrainState));
  if (!ts) return -1;

  /* Loss output buffer (1 scalar, same as vag) */
  PolyDType out_dt = poly_dtype_scalar(vag->loss_value->dtype);
  if (!poly_dtype_is_float(out_dt)) out_dt = POLY_FLOAT32;
  ts->loss_out_buf = poly_buffer(ctx, out_dt, 1);

  if (poly_buffer_write(ctx, ts->loss_out_buf, &ts->loss_data, sizeof(float)) != 0) {
    train_free(ts, np);
    return -1;
  }

  /* Count SINK sources: loss_store + param assigns + optimizer state assigns. */
  bool is_adam = (o->kind == POLY_OPTIM_ADAM || o->kind == POLY_OPTIM_ADAMW);
  bool sgd_momentum = (o->kind == POLY_OPTIM_SGD && o->momentum > 0.0f);
  bool has_m_bufs = is_adam || sgd_momentum;
  int n_sink_srcs = 1 + np; /* loss_store + param assigns */
  if (is_adam) n_sink_srcs += 2 * np + 2; /* + m/v assigns + b1_t/b2_t assigns */
  else if (sgd_momentum) n_sink_srcs += np; /* + momentum assigns */

  PolyUOp **sink_srcs = calloc((size_t)n_sink_srcs, sizeof(PolyUOp *));
  if (!sink_srcs) {
    train_free(ts, np);
    return -1;
  }

  /* Loss store */
  PolyUOp *loss_flat = vag->loss_value;
  if (uop_numel(ctx, vag->loss_value) != 1) {
    int64_t one_shape[1] = {1};
    loss_flat = poly_reshape(ctx, vag->loss_value, one_shape, 1);
  }
  sink_srcs[0] = poly_store_val(ctx, ts->loss_out_buf, loss_flat);

  /* Allocate optimizer state buffers. */
  if (has_m_bufs) {
    ts->m_bufs = calloc((size_t)np, sizeof(PolyUOp *));
    if (is_adam) ts->v_bufs = calloc((size_t)np, sizeof(PolyUOp *));
    if (!ts->m_bufs || (is_adam && !ts->v_bufs)) {
      free(sink_srcs);
      train_free(ts, np);
      return -1;
    }
    ts->n_moment_bufs = np;
  }

  if (is_adam) {
    /* Bias correction scalar buffers */
    int64_t scalar_shape[1] = {1};
    /* tinygrad stores beta powers as state tensors initialized to 1, then
     * schedule_step multiplies them by beta each step before computing
     * 1/(1-beta_t). */
    ts->bc1_data = 1.0f;
    ts->bc2_data = 1.0f;
    if (ensure_optimizer_state_buffer(
            inst, "optim.adam.b1_t", scalar_shape, 1, true, ts->bc1_data, &ts->bc1_buf
        ) != 0 ||
        ensure_optimizer_state_buffer(
            inst, "optim.adam.b2_t", scalar_shape, 1, true, ts->bc2_data, &ts->bc2_buf
        ) != 0) {
      free(sink_srcs);
      train_free(ts, np);
      return -1;
    }

    for (int i = 0; i < np; i++) {
      NamedBuf *pb = &inst->bufs[inst->trainable_param_indices[i]];
      int64_t opt_shape[8] = {0};
      int opt_ndim = pb->ndim;
      if (opt_ndim > 0) memcpy(opt_shape, pb->shape, (size_t)opt_ndim * sizeof(int64_t));
      char *m_name = optimizer_param_state_name("adam", "m", pb->name);
      char *v_name = optimizer_param_state_name("adam", "v", pb->name);
      if (!m_name || !v_name ||
          ensure_optimizer_state_buffer(inst, m_name, opt_shape, opt_ndim, false, 0.0f, &ts->m_bufs[i]) !=
              0 ||
          ensure_optimizer_state_buffer(inst, v_name, opt_shape, opt_ndim, false, 0.0f, &ts->v_bufs[i]) !=
              0) {
        free(m_name);
        free(v_name);
        free(sink_srcs);
        train_free(ts, np);
        return -1;
      }
      free(m_name);
      free(v_name);
    }
  } else if (sgd_momentum) {
    for (int i = 0; i < np; i++) {
      NamedBuf *pb = &inst->bufs[inst->trainable_param_indices[i]];
      int64_t opt_shape[8] = {0};
      int opt_ndim = pb->ndim;
      if (opt_ndim > 0) memcpy(opt_shape, pb->shape, (size_t)opt_ndim * sizeof(int64_t));
      char *b_name = optimizer_param_state_name("sgd", "b", pb->name);
      if (!b_name ||
          ensure_optimizer_state_buffer(inst, b_name, opt_shape, opt_ndim, false, 0.0f, &ts->m_bufs[i]) !=
              0) {
        free(b_name);
        free(sink_srcs);
        train_free(ts, np);
        return -1;
      }
      free(b_name);
    }
  }

  /* Build optimizer update graph for each trainable parameter. */
  int si = 1; /* sink_srcs index (0 = loss_store) */
  PolyUOp *bc1_new = NULL;
  PolyUOp *bc2_new = NULL;

  for (int i = 0; i < np; i++) {
    int buf_idx = inst->trainable_param_indices[i];
    int param_ord = param_ordinal_for_buf(inst, buf_idx);
    if (param_ord < 0) {
      free(sink_srcs);
      train_free(ts, np);
      return -1;
    }
    PolyUOp *param_buf = inst->bufs[buf_idx].buffer;
    NamedBuf *pb_opt = &inst->bufs[buf_idx];

    /* Flatten gradient to 1D to match the flat param buffer.
     * The grad UOp may be shaped (e.g. [3,2]) because autograd now
     * differentiates w.r.t. the shaped view, not the raw buffer. */
    PolyUOp *grad = vag->grad_uops[param_ord];
    {
      PolyShape gs = poly_uop_shape(ctx, grad);
      if (gs.ndim > 1 || (gs.ndim == 1 && gs.dims && gs.dims[0] != pb_opt->numel)) {
        int64_t flat[1] = {pb_opt->numel};
        grad = poly_reshape(ctx, grad, flat, 1);
      }
      if (gs.dims) free(gs.dims);
    }

    PolyOptimConfig cfg = {
        .kind = o->kind,
        .lr = o->lr,
        .beta1 = o->beta1,
        .beta2 = o->beta2,
        .eps = o->eps,
        .weight_decay = o->weight_decay,
        .momentum = o->momentum,
        .nesterov = o->nesterov,
        .classic = o->classic,
    };
    PolyOptimUpdate upd;
    PolyUOp *m_buf = has_m_bufs ? ts->m_bufs[i] : NULL;
    PolyUOp *v_buf = is_adam ? ts->v_bufs[i] : NULL;
    if (poly_optim_build_update(
            ctx, &cfg, param_buf, grad, m_buf, v_buf, ts->bc1_buf, ts->bc2_buf, pb_opt->numel, &upd
        ) != 0) {
      fprintf(stderr, "ensure_train_graph: unsupported optimizer %d\n", o->kind);
      free(sink_srcs);
      train_free(ts, np);
      return -1;
    }
    sink_srcs[si++] = poly_store_buffer_update(ctx, param_buf, upd.param_new);
    if (sgd_momentum) {
      sink_srcs[si++] = poly_store_buffer_update(ctx, m_buf, upd.m_new);
    } else if (is_adam) {
      if (!bc1_new) bc1_new = upd.bc1_new;
      if (!bc2_new) bc2_new = upd.bc2_new;
      sink_srcs[si++] = poly_store_buffer_update(ctx, m_buf, upd.m_new);
      sink_srcs[si++] = poly_store_buffer_update(ctx, v_buf, upd.v_new);
    }
  }

  if (is_adam) {
    sink_srcs[si++] = poly_store_buffer_update(ctx, ts->bc1_buf, bc1_new);
    sink_srcs[si++] = poly_store_buffer_update(ctx, ts->bc2_buf, bc2_new);
  }

  if (si != n_sink_srcs) {
    free(sink_srcs);
    train_free(ts, np);
    return -1;
  }

  ts->combined_sink = poly_sink_n(ctx, sink_srcs, n_sink_srcs);
  free(sink_srcs);

  inst->train = ts;
  return 0;
}

/* Train Step */

int poly_instance_train_step(PolyInstance *inst, PolyIOBinding *io, int n_io, float *loss_out) {
  if (!inst || inst->stage != POLY_INSTANCE_BUILT) return -1;
  if (inst->optim.kind == POLY_OPTIM_NONE) {
    fprintf(stderr, "poly_instance_train_step: no optimizer configured\n");
    return -1;
  }

  /* Find loss entrypoint */
  int ep_idx = find_entrypoint(inst, "loss");
  if (ep_idx < 0) {
    fprintf(stderr, "poly_instance_train_step: no 'loss' entrypoint\n");
    return -1;
  }

  /* Build combined fwd+bwd+optimizer graph lazily */
  if (ensure_train_graph(inst, ep_idx) != 0) return -1;
  TrainState *ts = inst->train;
  OptimState *o = &inst->optim;

  /* tinygrad updates Adam beta-power state inside the scheduled optimizer
   * graph. Keep step only as bookkeeping for public state/checkpoints. */
  o->step++;

  int ret = run_instance_sink(
      inst, &inst->entrypoints[ep_idx], ts->combined_sink, io, n_io, &ts->schedule
  );
  if (ret != 0) {
    o->step--;
    return ret;
  }

  if (poly_buffer_read(inst->ctx, ts->loss_out_buf, &ts->loss_data, sizeof(float)) != 0) {
    o->step--;
    return -1;
  }
  if (loss_out) *loss_out = ts->loss_data;

  /* Update instance "loss" named buffer for consumers */
  int loss_named_idx = find_buf_by_name(inst, "loss");
  if (loss_named_idx >= 0)
    poly_instance_upload_buf(inst, loss_named_idx, &ts->loss_data, sizeof(float));

  return 0;
}

/* Imported-instance composition */

static char *prefixed_name(const char *prefix, const char *name) {
  const char *p = prefix ? prefix : "";
  const char *n = name ? name : "";
  size_t lp = strlen(p), ln = strlen(n);
  char *out = malloc(lp + ln + 1);
  if (!out) return NULL;
  memcpy(out, p, lp);
  memcpy(out + lp, n, ln + 1);
  return out;
}

static const PolyInstanceInlineBinding *find_inline_binding(
    const PolyInstanceInlineBinding *bindings,
    int n_bindings,
    const char *name
) {
  if (!bindings || !name) return NULL;
  for (int i = 0; i < n_bindings; i++)
    if (bindings[i].name && strcmp(bindings[i].name, name) == 0) return &bindings[i];
  return NULL;
}

static const NamedBuf *instance_buf_for_uop(const PolyInstance *inst, PolyUOp *uop) {
  if (!inst || !uop) return NULL;
  for (int i = 0; i < inst->n_bufs; i++)
    if (inst->bufs[i].buffer == uop) return &inst->bufs[i];
  return NULL;
}

typedef struct {
  PolyUOp *child_buf;
  char *parent_name;
} InlineAlias;

static PolyUOp *register_inline_buffer(
    PolyCtx *dst_ctx,
    const NamedBuf *b,
    const char *prefix,
    bool trainable,
    InlineAlias *aliases,
    int *n_aliases,
    int max_aliases
) {
  if (!dst_ctx || !b) return NULL;

  char *full = prefixed_name(prefix, b->name);
  if (!full) return NULL;

  for (int i = 0; i < *n_aliases; i++) {
    if (aliases[i].child_buf != b->buffer) continue;
    int rc = poly_alias(dst_ctx, full, aliases[i].parent_name);
    PolyUOp *aliased = (rc == 0) ? poly_ctx_get(dst_ctx, "%s", full) : NULL;
    free(full);
    return aliased;
  }

  PolyUOp *ret = NULL;
  switch (b->role) {
  case POLY_ROLE_PARAM:
    ret = poly_param(dst_ctx, b->buffer->dtype, b->shape, b->ndim, "%s", full);
    if (ret) poly_ctx_set_trainable(dst_ctx, full, trainable);
    break;
  case POLY_ROLE_INPUT:
    ret = poly_input(dst_ctx, b->buffer->dtype, b->shape, b->ndim, "%s", full);
    break;
  case POLY_ROLE_TARGET:
    ret = poly_target(dst_ctx, b->buffer->dtype, b->shape, b->ndim, "%s", full);
    break;
  case POLY_ROLE_OUTPUT:
    ret = poly_output(dst_ctx, b->buffer->dtype, b->shape, b->ndim, "%s", full);
    break;
  case POLY_ROLE_AUX:
  default:
    ret = poly_aux(dst_ctx, b->buffer->dtype, b->shape, b->ndim, "%s", full);
    break;
  }

  if (ret && *n_aliases < max_aliases) {
    aliases[*n_aliases] = (InlineAlias){b->buffer, full};
    (*n_aliases)++;
  } else {
    free(full);
  }
  return ret;
}

static PolyUOp *clone_uop_into_ctx(PolyCtx *dst_ctx, PolyMap *memo, PolyUOp *u) {
  if (!dst_ctx || !memo || !u) return NULL;

  PolyUOp *cached = poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq);
  if (cached) return cached;

  PolyUOp *stack_src[16];
  PolyUOp **src = u->n_src > (int)(sizeof(stack_src) / sizeof(stack_src[0]))
                      ? malloc((size_t)u->n_src * sizeof(PolyUOp *))
                      : stack_src;
  if (u->n_src > 0 && !src) return NULL;

  for (int i = 0; i < u->n_src; i++) {
    src[i] = clone_uop_into_ctx(dst_ctx, memo, u->src[i]);
    if (!src[i]) {
      if (src != stack_src) free(src);
      return NULL;
    }
  }

  /* Preserve nonzero tags when cloning cross-ctx UOps so BUFFER uniqueness and
   * imported UNIQUE-like identities remain stable inside the destination ctx. */
  PolyUOp *cloned = u->tag
                        ? poly_uop_tagged(dst_ctx, u->op, u->dtype, src, u->n_src, u->arg, u->tag)
                        : poly_uop(dst_ctx, u->op, u->dtype, src, u->n_src, u->arg);
  if (src != stack_src) free(src);
  if (cloned) poly_map_set(memo, poly_ptr_hash(u), u, cloned, poly_ptr_eq);
  return cloned;
}

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
) {
  if (out_n_outputs) *out_n_outputs = 0;
  if (!dst_ctx || !child || !entrypoint || !outputs || max_outputs < 0) return -1;
  int ep_idx = find_entrypoint(child, entrypoint);
  if (ep_idx < 0) return -1;

  PolyMap *memo = poly_map_new(64);
  if (!memo) return -1;

  InlineAlias *aliases = calloc((size_t)child->n_bufs, sizeof(InlineAlias));
  int n_aliases = 0;
  int rc = -1;

  for (int i = 0; i < child->n_bufs; i++) {
    const NamedBuf *b = &child->bufs[i];
    const PolyInstanceInlineBinding *binding = find_inline_binding(bindings, n_bindings, b->name);
    PolyUOp *replacement = binding ? binding->uop : NULL;
    if (!replacement)
      replacement =
          register_inline_buffer(dst_ctx, b, prefix, trainable, aliases, &n_aliases, child->n_bufs);
    if (!replacement) goto done;
    poly_map_set(memo, poly_ptr_hash(b->buffer), b->buffer, replacement, poly_ptr_eq);
  }

  PolyUOp *sink = child->entrypoints[ep_idx].sink;
  if (!sink || sink->op != POLY_OP_SINK) goto done;

  int n_outputs = 0;
  for (int i = 0; i < sink->n_src; i++) {
    PolyUOp *store = sink->src[i];
    if (!store || store->op != POLY_OP_STORE || store->n_src < 2) continue;
    const PolyUOp *identity = poly_uop_get_buffer_identity(store->src[0]);
    const NamedBuf *out_buf = instance_buf_for_uop(child, (PolyUOp *)identity);
    if (!out_buf || out_buf->role != POLY_ROLE_OUTPUT) continue;
    if (n_outputs >= max_outputs) goto done;
    PolyUOp *value = clone_uop_into_ctx(dst_ctx, memo, store->src[1]);
    if (!value) goto done;
    outputs[n_outputs++] = (PolyInstanceInlineOutput){out_buf->name, value};
  }

  if (out_n_outputs) *out_n_outputs = n_outputs;
  rc = 0;

done:
  if (aliases) {
    for (int i = 0; i < n_aliases; i++)
      free(aliases[i].parent_name);
    free(aliases);
  }
  poly_map_destroy(memo);
  return rc;
}

int poly_instance_copy_prefixed_weights(PolyInstance *dst, PolyInstance *src, const char *prefix) {
  if (!dst || !src) return -1;
  for (int i = 0; i < src->n_params; i++) {
    int sbi = src->param_indices[i];
    const NamedBuf *sb = &src->bufs[sbi];
    char *dst_name = prefixed_name(prefix, sb->name);
    if (!dst_name) return -1;
    int dbi = find_buf_by_name(dst, dst_name);
    free(dst_name);
    if (dbi < 0) return -1;
    NamedBuf *db = &dst->bufs[dbi];
    size_t nbytes = named_buf_nbytes(sb);
    if (db->numel != sb->numel || named_buf_nbytes(db) != nbytes || nbytes == 0) return -1;
    uint8_t *tmp = malloc(nbytes);
    if (!tmp) return -1;
    int rc = poly_buffer_read(src->ctx, sb->buffer, tmp, nbytes);
    if (rc == 0) rc = poly_buffer_write(dst->ctx, db->buffer, tmp, nbytes);
    if (rc == 0) rc = sync_buf_to_host(dst, dbi);
    free(tmp);
    if (rc != 0) return -1;
  }
  return 0;
}

/* Named accessor helpers */

PolyCtx *poly_instance_ctx(const PolyInstance *inst) {
  return inst ? inst->ctx : NULL;
}

void poly_instance_own_ctx(PolyInstance *inst) {
  if (inst) inst->owns_ctx = true;
}

PolyUOp *poly_instance_get_buffer(const PolyInstance *inst, const char *name) {
  if (!inst || !name) return NULL;
  int idx = find_buf_by_name(inst, name);
  return (idx >= 0) ? inst->bufs[idx].buffer : NULL;
}

PolyUOp *poly_instance_get_sink(const PolyInstance *inst, const char *name) {
  if (!inst || !name) return NULL;
  int idx = find_entrypoint(inst, name);
  return (idx >= 0) ? inst->entrypoints[idx].sink : NULL;
}

float *poly_instance_buf_data_named(PolyInstance *inst, const char *name, int64_t *numel_out) {
  if (!inst || !name) return NULL;
  int idx = find_buf_by_name(inst, name);
  if (idx < 0) return NULL;
  return poly_instance_buf_data(inst, idx, numel_out);
}

int64_t poly_instance_buf_numel_named(const PolyInstance *inst, const char *name) {
  if (!inst || !name) return 0;
  int idx = find_buf_by_name(inst, name);
  return (idx >= 0) ? inst->bufs[idx].numel : 0;
}
