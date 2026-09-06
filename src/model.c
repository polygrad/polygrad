/*
 * poly_model.c -- Runtime for portable tensor-level models
 *
 * Product layer above the tinygrad-aligned compiler core.
 * Owns ABI names, binding metadata, entrypoints, and optimizer state.
 * Buffer residency is owned by PolyCtx through ctx->buffers.
 */

#define _POSIX_C_SOURCE 200809L
#include "model.h"
#include "ctx.h"
#include "frontend_internal.h"
#include "utils.h"
#include "ir.h"
#include "safetensors.h"
#include "tensor.h"
#include "optim.h"
#include "engine/jit.h"
#include "engine/realize.h"
#include "engine/schedule.h"
#include "device.h"
#include "codegen/codegen.h" /* poly_cuda_available (POLY_HAS_CUDA) */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <limits.h>

/* Internal types */

typedef struct {
  char *name;
  uint8_t role;
  uint32_t flags;
  PolyUOp *buffer; /* mutable current runtime physical identity */
  PolyUOp *logical_value; /* exact named node in the immutable portable program */
  PolyUOp *logical_buffer; /* Model runtime storage identity */
  PolyUOp *capture_buffer; /* immutable pre-policy physical identity; may alias logical */
  int64_t shape[POLY_IR_MAX_DIMS];
  int ndim;
  void *data; /* non-owning cached host-root pointer */
  int64_t numel;
  bool trainable; /* PARAMs can be frozen while still saved as weights */
} NamedBuf;

static bool named_buf_nbytes_checked(const NamedBuf *b, size_t *out) {
  if (out) *out = 0;
  if (!b || !out || !b->buffer || b->numel < 0) return false;
  size_t itemsize = poly_dtype_itemsize(b->buffer->dtype);
  if (itemsize == 0 || (uint64_t)b->numel > (uint64_t)SIZE_MAX / itemsize) return false;
  *out = (size_t)b->numel * itemsize;
  return true;
}

static size_t named_buf_nbytes(const NamedBuf *b) {
  size_t out = 0;
  return named_buf_nbytes_checked(b, &out) ? out : 0;
}

typedef struct {
  int kind;
  float lr, beta1, beta2, eps, weight_decay;
  float momentum;
  bool nesterov;
  bool classic;
  int step;
} OptimState;

/* C-owned form of Tinygrad's `(LINEAR, var_vals)` execution tuple. */
typedef struct {
  PolyUOp *linear;
  PolyVarBinding *var_bindings;
  int n_var_bindings;
} PolyLinearEntry;

/* Value-and-grad metadata (built lazily on first train call) */

typedef struct {
  PolyUOp *combined_sink; /* combined fwd+bwd SINK */
  int entrypoint_index; /* topology is sealed; state writes do not change this key */
  int objective_buffer_index;
  PolyLinearEntry executable;
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
  PolyLinearEntry executable;
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

/* One public Model owns training. The small configuration is inline; expensive
 * graphs are optional and allocated on first use. Candidate Model copies keep
 * replacement transactional without a second public handle or rebind protocol. */
typedef struct {
  VagState *vag;
  TrainState *train;
  OptimState optim;
} PolyTrainState;

typedef struct {
  char *name;
  uint8_t role;
  uint32_t flags;
  PolyTensor *tensor;
  PolyUOp *buffer; /* portable logical binding identity */
  PolyUOp *physical_buffer; /* exact default runtime identity, when constructed */
  PolyUOp *initial_data_buffer; /* exact capture-time residency source */
  bool snapshot_residency_retained; /* owns fresh snapshot bytes until build adopts them */
  PolyUOp *declared_logical_value; /* exact source Tensor root named by this binding */
  PolyUOp *declared_physical_value; /* exact current value to snapshot, never rewritten */
  bool needs_snapshot;
  bool trainable;
  int64_t shape[POLY_IR_MAX_DIMS];
  int ndim;
} BuildBinding;

typedef struct {
  PolyUOp **keys;
  PolyBuffer **heads;
  PolyBuffer *saved_heads;
  int n;
  int n_retained;
} BuildBufferTransaction;

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
  PolyUOp *sink; /* mutable current runtime physical root */
  PolyUOp *logical_sink; /* portable IR/inlining entrypoint */
  PolyUOp *capture_sink; /* immutable pre-policy physical root; may alias logical */
  char **inputs;
  int n_inputs;
  char **outputs;
  int n_outputs;
  char *objective;
  uint32_t flags;
} RuntimeEntrypoint;

typedef struct {
  char *name;
  PolyUOp **logical_inputs;
  int n_inputs;
  PolyUOp *logical_output;
} RuntimeModule;

typedef struct {
  PolyModelOptions opts;

  BuildBinding *bindings;
  int n_bindings;
  int bindings_cap;

  BuildEntrypoint *entrypoints;
  int n_entrypoints;
  int entrypoints_cap;

  char **scopes;
  int n_scopes;
  int scopes_cap;
} PolyModelBuildState;

struct PolyModel {
  PolyCtx *ctx;
  bool owns_ctx; /* true: poly_model_free destroys ctx */
  bool has_portable_source; /* false for bound compiled-program imports */
  PolyModelStage stage;
  PolyModelBuildState *build;
  PolyModelError last_error;

  NamedBuf *bufs;
  int n_bufs;
  bool has_physical_capture;

  /* Param subset (indices into bufs[]) */
  int *param_indices;
  int n_params;
  int *trainable_param_indices;
  int n_trainable_params;

  /* Entrypoints */
  RuntimeEntrypoint *entrypoints;
  int n_entrypoints;
  PolyLinearEntry *entry_executables; /* lazy compiled LINEAR per entrypoint */

  /* Explicit product-layer regions used only by non-uniform place(). */
  RuntimeModule *modules;
  int n_modules;

  PolyTrainState training;

  /* C mechanics for Model-owned UOps. Physical roots also own residency;
   * portable roots own their producer DAG across IR collection. */
  PolyUOp **residency_roots;
  int n_residency_roots;
};

static void runtime_modules_free(RuntimeModule *modules, int n_modules);
static int model_place_uniform_device(PolyModel *inst, PolyDevice device);
static int build_trainable_param_indices(const PolyModel *inst, int **indices_out, int *count_out);
static PolyUOp *entrypoint_store_value(PolyUOp *sink, PolyUOp *buffer);

#ifdef POLY_TESTING
static int test_residency_root_fail_after = -1;

void poly_model_test_fail_residency_roots_after(int additions) {
  test_residency_root_fail_after = additions;
}

bool poly_model_test_has_vag(const PolyModel *inst) {
  return inst && inst->training.vag;
}

bool poly_model_test_has_train(const PolyModel *inst) {
  return inst && inst->training.train;
}

int poly_model_test_optimizer_kind(const PolyModel *inst) {
  return inst ? inst->training.optim.kind : -1;
}

PolyUOp *poly_model_test_execution(const PolyModel *inst, bool training, bool compiled) {
  if (!inst) return NULL;
  if (training) {
    const TrainState *state = inst->training.train;
    return state ? (compiled ? state->executable.linear : state->combined_sink) : NULL;
  }
  const VagState *state = inst->training.vag;
  return state ? (compiled ? state->executable.linear : state->combined_sink) : NULL;
}
#endif

static bool model_append_residency_root(
    PolyUOp ***roots,
    int *count,
    int *capacity,
    PolyUOp *root
) {
  if (!root) return true;
#ifdef POLY_TESTING
  if (test_residency_root_fail_after == 0) {
    test_residency_root_fail_after = -1;
    return false;
  }
  if (test_residency_root_fail_after > 0) test_residency_root_fail_after--;
#endif
  for (int i = 0; i < *count; i++)
    if ((*roots)[i] == root) return true;
  if (*count >= *capacity) {
    int new_capacity = *capacity ? *capacity * 2 : 16;
    PolyUOp **new_roots = realloc(*roots, (size_t)new_capacity * sizeof(*new_roots));
    if (!new_roots) return false;
    *roots = new_roots;
    *capacity = new_capacity;
  }
  (*roots)[(*count)++] = root;
  return true;
}

typedef struct {
  PolyUOp **items;
  int count;
} ModelResidencyRoots;

static bool build_publish_roots(
    const PolyModelBuildState *build,
    PolyUOp ***roots,
    int *count,
    int *capacity
) {
#define PUBLISH_ROOT(root)                                                                         \
  do {                                                                                             \
    if (!model_append_residency_root(roots, count, capacity, (root))) return false;                \
  } while (0)
  if (build) {
    for (int i = 0; i < build->n_bindings; i++) {
      const BuildBinding *binding = &build->bindings[i];
      PUBLISH_ROOT(binding->buffer);
      PUBLISH_ROOT(binding->physical_buffer);
      PUBLISH_ROOT(binding->initial_data_buffer);
      PUBLISH_ROOT(binding->declared_logical_value);
      PUBLISH_ROOT(binding->declared_physical_value);
    }
  }

#undef PUBLISH_ROOT
  return true;
}

static bool training_publish_roots(
    const PolyTrainState *training,
    int n_params,
    PolyUOp ***roots,
    int *count,
    int *capacity
) {
#define PUBLISH_ROOT(root)                                                                         \
  do {                                                                                             \
    if (!model_append_residency_root(roots, count, capacity, (root))) return false;                \
  } while (0)
  if (training->vag) {
    PUBLISH_ROOT(training->vag->combined_sink);
    PUBLISH_ROOT(training->vag->executable.linear);
    PUBLISH_ROOT(training->vag->loss_out_buf);
    PUBLISH_ROOT(training->vag->loss_value);
    for (int i = 0; i < n_params; i++) {
      PUBLISH_ROOT(training->vag->grad_out_bufs[i]);
      PUBLISH_ROOT(training->vag->grad_uops[i]);
    }
  }
  if (training->train) {
    PUBLISH_ROOT(training->train->combined_sink);
    PUBLISH_ROOT(training->train->executable.linear);
    PUBLISH_ROOT(training->train->loss_out_buf);
    PUBLISH_ROOT(training->train->bc1_buf);
    PUBLISH_ROOT(training->train->bc2_buf);
    for (int i = 0; i < training->train->n_moment_bufs; i++) {
      if (training->train->m_bufs) PUBLISH_ROOT(training->train->m_bufs[i]);
      if (training->train->v_bufs) PUBLISH_ROOT(training->train->v_bufs[i]);
    }
  }

#undef PUBLISH_ROOT
  return true;
}

/* PolyModel-only C ownership mechanics: prepare every portable, executable,
 * and residency root before callers publish the state that refers to it. */
static int model_prepare_residency_roots(const PolyModel *state, ModelResidencyRoots *prepared) {
  if (!state || !state->ctx || !prepared) return -1;
  *prepared = (ModelResidencyRoots){0};
  PolyUOp **roots = NULL;
  int count = 0, capacity = 0;
#define ADD_MODEL_ROOT(root)                                                                       \
  do {                                                                                             \
    if (!model_append_residency_root(&roots, &count, &capacity, (root))) goto fail;                \
  } while (0)
  if (!build_publish_roots(state->build, &roots, &count, &capacity)) goto fail;
  for (int i = 0; i < state->n_bufs; i++) {
    ADD_MODEL_ROOT(state->bufs[i].buffer);
    ADD_MODEL_ROOT(state->bufs[i].logical_value);
    ADD_MODEL_ROOT(state->bufs[i].logical_buffer);
    ADD_MODEL_ROOT(state->bufs[i].capture_buffer);
  }
  for (int i = 0; i < state->n_entrypoints; i++) {
    ADD_MODEL_ROOT(state->entrypoints[i].sink);
    ADD_MODEL_ROOT(state->entrypoints[i].logical_sink);
    ADD_MODEL_ROOT(state->entrypoints[i].capture_sink);
    if (state->entry_executables) ADD_MODEL_ROOT(state->entry_executables[i].linear);
  }
  for (int i = 0; i < state->n_modules; i++) {
    ADD_MODEL_ROOT(state->modules[i].logical_output);
    for (int j = 0; j < state->modules[i].n_inputs; j++)
      ADD_MODEL_ROOT(state->modules[i].logical_inputs[j]);
  }
  if (!training_publish_roots(&state->training, state->n_params, &roots, &count, &capacity))
    goto fail;
#undef ADD_MODEL_ROOT
  for (int i = 0; i < count; i++)
    if (poly_uop_retain(state->ctx, roots[i]) != 0) {
      for (int j = 0; j < i; j++)
        poly_uop_release(state->ctx, roots[j]);
      goto fail;
    }
  prepared->items = roots;
  prepared->count = count;
  return 0;

fail:
  free(roots);
  return -1;
}

static void model_discard_prepared_residency_roots(PolyCtx *ctx, ModelResidencyRoots *prepared) {
  if (!prepared) return;
  for (int i = 0; i < prepared->count; i++)
    poly_uop_release(ctx, prepared->items[i]);
  free(prepared->items);
  *prepared = (ModelResidencyRoots){0};
}

static void model_publish_residency_roots(PolyModel *inst, ModelResidencyRoots *prepared) {
  for (int i = 0; i < inst->n_residency_roots; i++)
    poly_uop_release(inst->ctx, inst->residency_roots[i]);
  free(inst->residency_roots);
  inst->residency_roots = prepared->items;
  inst->n_residency_roots = prepared->count;
  *prepared = (ModelResidencyRoots){0};
}

static int model_refresh_residency_roots(PolyModel *inst) {
  ModelResidencyRoots prepared = {0};
  if (model_prepare_residency_roots(inst, &prepared) != 0) return -1;
  model_publish_residency_roots(inst, &prepared);
  return 0;
}

/* Approved logical/physical boundary: portable BUFFERs use deviceless
 * BUFFER(UNIQUE, size); executable Tinygrad-shaped BUFFERs use ParamArg(device). */
static bool model_is_portable_buffer(const PolyUOp *u) {
  return u && u->op == POLY_OP_BUFFER && u->n_src == 1 && u->src[0] &&
         u->src[0]->op == POLY_OP_UNIQUE && u->arg.kind == POLY_ARG_INT;
}

/* Helpers */

static int64_t compute_numel(const int64_t *shape, int ndim) {
  if (ndim < 0 || ndim > POLY_IR_MAX_DIMS || (ndim > 0 && !shape)) return -1;
  int64_t n = 1;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] < 0 || (shape[i] != 0 && n > INT64_MAX / shape[i])) return -1;
    n *= shape[i];
  }
  return n;
}

static int find_entrypoint(const PolyModel *inst, const char *name) {
  for (int i = 0; i < inst->n_entrypoints; i++)
    if (strcmp(inst->entrypoints[i].name, name) == 0) return i;
  return -1;
}

static int find_buf_by_name(const PolyModel *inst, const char *name) {
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
  if (is_optimizer_binding(b)) return (flags & POLY_EXPORT_WEIGHTS_OPTIMIZER) != 0;
  /* Like tinygrad's model state_dict, persistent state includes non-parameters.
   * Optimizer AUX rows belong only to the explicitly selected optimizer set. */
  return (flags & POLY_EXPORT_WEIGHTS_PARAMS) &&
         (b->role == POLY_ROLE_PARAM || b->role == POLY_ROLE_AUX);
}

static PolyModel *model_from_spec(
    PolyIrSpec *spec,
    PolyUOp **physical_buffers,
    PolyUOp **physical_sinks,
    bool owns_ctx,
    bool free_spec
);

static const char *model_stage_name(PolyModelStage stage) {
  switch (stage) {
  case POLY_MODEL_BUILDING:
    return "BUILDING";
  case POLY_MODEL_BUILT:
    return "BUILT";
  case POLY_MODEL_FAILED:
    return "FAILED";
  }
  return "UNKNOWN";
}

static void poly_model_set_error(
    PolyModel *inst,
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

static void poly_model_copy_error(PolyModel *inst, PolyModelError *err) {
  if (inst && err) *err = inst->last_error;
}

static PolyStatus require_stage(PolyModel *inst, PolyModelStage want, const char *func) {
  if (!inst) return POLY_STATUS_INVALID;
  if (inst->stage == want) return POLY_STATUS_OK;
  poly_model_set_error(
      inst, POLY_STATUS_BAD_STAGE, func, "expected %s, got %s", model_stage_name(want),
      model_stage_name(inst->stage)
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

static void build_state_free(PolyModelBuildState *build) {
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

static char *scoped_name(PolyModel *inst, const char *name) {
  PolyModelBuildState *build = inst ? inst->build : NULL;
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

static int copy_tensor_shape(
    PolyCtx *ctx,
    PolyTensor *tensor,
    int64_t shape[POLY_IR_MAX_DIMS],
    int *ndim
) {
  if (!ctx || !tensor || !ndim) return -1;
  PolyUOp *u = poly_tensor_uop_logical(tensor);
  if (!u) return -1;
  PolyShape s = poly_uop_max_shape_cached(ctx, u);
  if (s.ndim < 0 || s.ndim > POLY_IR_MAX_DIMS) return -1;
  *ndim = s.ndim;
  if (s.ndim > 0) memcpy(shape, s.dims, (size_t)s.ndim * sizeof(int64_t));
  return 0;
}

static BuildBinding *find_build_binding(PolyModelBuildState *build, const char *name) {
  if (!build || !name) return NULL;
  for (int i = 0; i < build->n_bindings; i++)
    if (strcmp(build->bindings[i].name, name) == 0) return &build->bindings[i];
  return NULL;
}

static bool set_build_binding_trainable(PolyModel *inst, const char *name, bool trainable) {
  BuildBinding *binding = inst ? find_build_binding(inst->build, name) : NULL;
  if (!binding || binding->role != POLY_ROLE_PARAM) return false;
  binding->trainable = trainable;
  return true;
}

static PolyStatus append_build_binding(
    PolyModel *inst,
    const char *name,
    uint8_t role,
    uint32_t flags,
    PolyTensor *tensor,
    PolyUOp *buffer,
    PolyUOp *physical_buffer,
    PolyUOp *initial_data_buffer,
    const int64_t *shape,
    int ndim,
    bool trainable,
    PolyTensorProvenance provenance
) {
  if (!inst || !inst->build || !name || !tensor) return POLY_STATUS_INVALID;
  char *full_name = scoped_name(inst, name);
  if (!full_name) {
    poly_model_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    return POLY_STATUS_NOMEM;
  }
  if (!valid_binding_name(full_name)) {
    poly_model_set_error(
        inst, POLY_STATUS_INVALID, __func__, "invalid binding name '%s'", full_name
    );
    free(full_name);
    return POLY_STATUS_INVALID;
  }
  if (find_build_binding(inst->build, full_name)) {
    poly_model_set_error(inst, POLY_STATUS_INVALID, __func__, "duplicate binding '%s'", full_name);
    free(full_name);
    return POLY_STATUS_INVALID;
  }
  if (ndim < 0 || ndim > POLY_IR_MAX_DIMS || poly_shape_numel_checked(shape, ndim) < 0) {
    poly_model_set_error(inst, POLY_STATUS_INVALID, __func__, "invalid shape for '%s'", full_name);
    free(full_name);
    return POLY_STATUS_INVALID;
  }
  if (grow_array(
          (void **)&inst->build->bindings, &inst->build->bindings_cap, inst->build->n_bindings + 1,
          sizeof(BuildBinding)
      ) != 0) {
    poly_model_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
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
  b->physical_buffer = physical_buffer;
  b->initial_data_buffer = initial_data_buffer;
  b->declared_logical_value = poly_tensor_uop_logical(tensor);
  b->declared_physical_value = poly_tensor_uop_physical(tensor);
  b->trainable = role == POLY_ROLE_PARAM && trainable;
  b->ndim = ndim;
  if (ndim > 0) memcpy(b->shape, shape, (size_t)ndim * sizeof(int64_t));
  /* Model capture owns immutable build-stage roots immediately. The source
   * Tensor may later realize and retire its wrapper-local logical producer. */
  if (model_refresh_residency_roots(inst) != 0) {
    free(b->name);
    *b = (BuildBinding){0};
    inst->build->n_bindings--;
    poly_model_set_error(inst, POLY_STATUS_NOMEM, __func__, "failed to retain binding roots");
    return POLY_STATUS_NOMEM;
  }
  if (role == POLY_ROLE_OUTPUT) {
    if (poly_tensor_provenance(tensor) == POLY_TENSOR_PROVENANCE_UNKNOWN)
      poly_tensor_set_provenance(tensor, provenance);
  } else {
    poly_tensor_set_provenance(tensor, provenance);
  }
  return POLY_STATUS_OK;
}

static PolyTensor *make_bound_storage_tensor(
    PolyModel *inst,
    const char *name,
    uint8_t role,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    bool trainable,
    PolyTensorProvenance provenance
) {
  if (require_stage(inst, POLY_MODEL_BUILDING, __func__) != POLY_STATUS_OK) return NULL;
  if (!inst->ctx || (ndim > 0 && !shape) || ndim < 0 || ndim > POLY_IR_MAX_DIMS) {
    poly_model_set_error(inst, POLY_STATUS_INVALID, __func__, "invalid tensor shape");
    return NULL;
  }
  /* Model currently builds a portable package. Physical-only tensors can
   * execute and JIT, but cannot declare portable named storage. */
  if (poly_ctx_get_logical_policy(inst->ctx) == POLY_LOGICAL_NEVER) {
    poly_model_set_error(
        inst, POLY_STATUS_INVALID, __func__, "portable Model requires logical construction"
    );
    return NULL;
  }
  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel < 0) {
    poly_model_set_error(inst, POLY_STATUS_INVALID, __func__, "invalid tensor shape");
    return NULL;
  }
  PolyDType scalar = dt;
  PolyDevice device = poly_ctx_get_preferred_device(inst->ctx);
  if (!poly_device_can_execute(device)) device = poly_device_default();
  /* Pinned UOp.empty/new_buffer constructs BUFFER(shape, ParamArg(device))
   * before ordinary Tensor composition (uop/ops.py:733-746). Model also
   * retains the paired device-free BUFFER for portable IR; poly_tensor_empty
   * gives both roots one numeric slot without placement or correspondence. */
  PolyTensor *tensor = poly_tensor_empty(inst->ctx, scalar, shape, ndim, device);
  if (!tensor) {
    poly_model_set_error(inst, POLY_STATUS_ERROR, __func__, "failed to create tensor");
    return NULL;
  }
  PolyUOp *buf = (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_logical(tensor));
  PolyUOp *physical_buf = (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(tensor));
  if (!buf || !physical_buf) {
    poly_model_set_error(inst, POLY_STATUS_ERROR, __func__, "failed to create storage buffer");
    return NULL;
  }
  if (append_build_binding(
          inst, name, role, 0, tensor, buf, physical_buf, buf, shape, ndim, trainable, provenance
      ) != POLY_STATUS_OK)
    return NULL;
  return tensor;
}

static PolyStatus append_existing_tensor_binding(
    PolyModel *inst,
    const char *name,
    uint8_t role,
    PolyTensor *tensor,
    uint32_t flags,
    bool require_buffer,
    bool trainable,
    PolyTensorProvenance provenance
) {
  if (require_stage(inst, POLY_MODEL_BUILDING, __func__) != POLY_STATUS_OK)
    return POLY_STATUS_BAD_STAGE;
  if (!inst->ctx || !tensor) {
    poly_model_set_error(inst, POLY_STATUS_INVALID, __func__, "null tensor binding");
    return POLY_STATUS_INVALID;
  }
  PolyUOp *logical_value = poly_tensor_uop_logical(tensor);
  if (!logical_value) {
    poly_model_set_error(
        inst, POLY_STATUS_INVALID, __func__, "binding '%s' has no logical source", name
    );
    return POLY_STATUS_INVALID;
  }
  int64_t shape[POLY_IR_MAX_DIMS] = {0};
  int ndim = 0;
  if (copy_tensor_shape(inst->ctx, tensor, shape, &ndim) != 0) {
    poly_model_set_error(inst, POLY_STATUS_INVALID, __func__, "could not infer binding shape");
    return POLY_STATUS_INVALID;
  }
  PolyUOp *buffer = NULL;
  PolyUOp *physical_value = poly_tensor_uop_physical(tensor);
  const PolyUOp *identity = poly_uop_get_buffer_identity(logical_value);
  if (identity) buffer = (PolyUOp *)identity;
  bool may_snapshot = role == POLY_ROLE_PARAM || role == POLY_ROLE_AUX;
  if (require_buffer && !buffer &&
      (!may_snapshot || !logical_value || !physical_value ||
       poly_tensor_root_has_unplaced_buffer(inst->ctx, physical_value))) {
    poly_model_set_error(
        inst, POLY_STATUS_INVALID, __func__, "binding '%s' has no buffer identity", name
    );
    return POLY_STATUS_INVALID;
  }
  PolyUOp *initial_data_buffer = NULL;
  PolyUOp *physical_buffer = NULL;
  if (role != POLY_ROLE_OUTPUT) {
    const PolyUOp *current_identity = poly_uop_get_buffer_identity(poly_tensor_uop(tensor));
    if (require_buffer && buffer && !current_identity) {
      if (!may_snapshot || !physical_value ||
          poly_tensor_root_has_unplaced_buffer(inst->ctx, physical_value)) {
        poly_model_set_error(
            inst, POLY_STATUS_INVALID, __func__, "binding '%s' has no current buffer identity", name
        );
        return POLY_STATUS_INVALID;
      }
    }
    physical_buffer = initial_data_buffer = (PolyUOp *)current_identity;
  }
  PolyStatus st = append_build_binding(
      inst, name, role, flags, tensor, buffer, physical_buffer, initial_data_buffer, shape, ndim,
      trainable, provenance
  );
  if (st == POLY_STATUS_OK && require_buffer && (!buffer || !initial_data_buffer))
    inst->build->bindings[inst->build->n_bindings - 1].needs_snapshot = true;
  return st;
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

static int copy_initial_buffer_data(
    PolyModel *inst,
    PolyCtx *ctx,
    const PolyModelBuildState *build
) {
  if (!inst || !ctx || !build || build->n_bindings != inst->n_bufs) return -1;
  for (int i = 0; i < inst->n_bufs; i++) {
    const BuildBinding *binding = &build->bindings[i];
    PolyUOp *source = binding->initial_data_buffer;
    PolyBuffer *physical =
        binding->physical_buffer ? poly_buffer_get(ctx, binding->physical_buffer) : NULL;
    if (physical && (physical->valid || (physical->src && physical->src->valid)))
      source = binding->physical_buffer;
    if (!source) continue;
    PolyBuffer *src = poly_buffer_get(ctx, source);
    size_t nbytes = named_buf_nbytes(&inst->bufs[i]);
    if (!src || !src->ptr || src->nbytes < nbytes || nbytes == 0) continue;
    if (poly_buffer_read(ctx, source, inst->bufs[i].data, nbytes) != 0 && src->valid &&
        poly_device_is_host_addressable(src->device))
      memcpy(inst->bufs[i].data, src->ptr, nbytes);
  }
  return 0;
}

PolyModel *poly_model_new(PolyCtx *ctx, const PolyModelOptions *opts) {
  if (!ctx) return NULL;
  PolyModel *inst = calloc(1, sizeof(PolyModel));
  if (!inst) return NULL;
  PolyModelBuildState *build = calloc(1, sizeof(PolyModelBuildState));
  if (!build) {
    free(inst);
    return NULL;
  }
  if (opts) build->opts = *opts;
  inst->ctx = ctx;
  inst->owns_ctx = false;
  inst->stage = POLY_MODEL_BUILDING;
  inst->build = build;
  inst->training.optim.kind = POLY_OPTIM_NONE;
  return inst;
}

PolyModelStage poly_model_stage(const PolyModel *inst) {
  return inst ? inst->stage : POLY_MODEL_FAILED;
}

const PolyModelError *poly_model_last_error(const PolyModel *inst) {
  return inst ? &inst->last_error : NULL;
}

PolyStatus poly_model_scope_push(PolyModel *inst, const char *fmt, ...) {
  if (require_stage(inst, POLY_MODEL_BUILDING, __func__) != POLY_STATUS_OK)
    return POLY_STATUS_BAD_STAGE;
  va_list ap;
  va_start(ap, fmt);
  char *scope = vformat_cstr(fmt, ap);
  va_end(ap);
  if (!scope) {
    poly_model_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    return POLY_STATUS_NOMEM;
  }
  if (!valid_binding_name(scope)) {
    poly_model_set_error(inst, POLY_STATUS_INVALID, __func__, "invalid scope '%s'", scope);
    free(scope);
    return POLY_STATUS_INVALID;
  }
  if (grow_array(
          (void **)&inst->build->scopes, &inst->build->scopes_cap, inst->build->n_scopes + 1,
          sizeof(char *)
      ) != 0) {
    poly_model_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    free(scope);
    return POLY_STATUS_NOMEM;
  }
  inst->build->scopes[inst->build->n_scopes++] = scope;
  return POLY_STATUS_OK;
}

PolyStatus poly_model_scope_pop(PolyModel *inst) {
  if (require_stage(inst, POLY_MODEL_BUILDING, __func__) != POLY_STATUS_OK)
    return POLY_STATUS_BAD_STAGE;
  if (inst->build->n_scopes <= 0) {
    poly_model_set_error(inst, POLY_STATUS_INVALID, __func__, "scope stack is empty");
    return POLY_STATUS_INVALID;
  }
  free(inst->build->scopes[--inst->build->n_scopes]);
  inst->build->scopes[inst->build->n_scopes] = NULL;
  return POLY_STATUS_OK;
}

PolyTensor *poly_model_input(
    PolyModel *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
) {
  return make_bound_storage_tensor(
      inst, name, POLY_ROLE_INPUT, dt, shape, ndim, false, POLY_TENSOR_PROVENANCE_USER_INPUT
  );
}

PolyTensor *poly_model_target(
    PolyModel *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
) {
  return make_bound_storage_tensor(
      inst, name, POLY_ROLE_TARGET, dt, shape, ndim, false, POLY_TENSOR_PROVENANCE_USER_INPUT
  );
}

PolyTensor *poly_model_param(
    PolyModel *inst,
    const char *name,
    PolyDType dt,
    const int64_t *shape,
    int ndim
) {
  return make_bound_storage_tensor(
      inst, name, POLY_ROLE_PARAM, dt, shape, ndim, true, POLY_TENSOR_PROVENANCE_PARAM_INIT
  );
}

PolyStatus poly_model_state(PolyModel *inst, const char *name, PolyTensor *tensor, uint32_t flags) {
  return append_existing_tensor_binding(
      inst, name, POLY_ROLE_PARAM, tensor, flags, true, true, POLY_TENSOR_PROVENANCE_STATE_LOADED
  );
}

PolyStatus poly_model_output(PolyModel *inst, const char *name, PolyTensor *tensor) {
  return append_existing_tensor_binding(
      inst, name, POLY_ROLE_OUTPUT, tensor, 0, false, false, POLY_TENSOR_PROVENANCE_COMPUTED
  );
}

PolyStatus poly_model_aux(PolyModel *inst, const char *name, PolyTensor *tensor, uint32_t flags) {
  return append_existing_tensor_binding(
      inst, name, POLY_ROLE_AUX, tensor, flags, true, false, POLY_TENSOR_PROVENANCE_STATE_LOADED
  );
}

PolyStatus poly_model_entrypoint(
    PolyModel *inst,
    const char *name,
    const char **inputs,
    int n_inputs,
    const char **outputs,
    int n_outputs,
    const PolyEntrypointOptions *opts
) {
  if (require_stage(inst, POLY_MODEL_BUILDING, __func__) != POLY_STATUS_OK)
    return POLY_STATUS_BAD_STAGE;
  if (!valid_binding_name(name) || n_inputs < 0 || n_outputs <= 0 || (n_inputs > 0 && !inputs) ||
      !outputs) {
    poly_model_set_error(
        inst, POLY_STATUS_INVALID, __func__, "invalid entrypoint '%s'", name ? name : "?"
    );
    return POLY_STATUS_INVALID;
  }
  if (grow_array(
          (void **)&inst->build->entrypoints, &inst->build->entrypoints_cap,
          inst->build->n_entrypoints + 1, sizeof(BuildEntrypoint)
      ) != 0) {
    poly_model_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
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
    poly_model_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    return POLY_STATUS_NOMEM;
  }
  return POLY_STATUS_OK;
}

static BuildBinding *find_build_storage_binding(
    PolyModelBuildState *build,
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

static bool build_snapshot_bindings_are_one_alias(const BuildBinding *a, const BuildBinding *b) {
  if (!a || !b) return false;
  if (a->tensor == b->tensor) return true;
  const PolyUOp *a_storage = poly_uop_get_buffer_identity(a->declared_physical_value);
  const PolyUOp *b_storage = poly_uop_get_buffer_identity(b->declared_physical_value);
  return a_storage && a_storage == b_storage;
}

/* Tinygrad's UOp.base follows movement src[0] to the underlying storage
 * lineage. Model does not yet serialize offset/stride-aware named view
 * resources, so two named state values related only by such a view must fail
 * closed instead of being snapshotted as independent parameters. */
static bool model_value_is_movement_view_of(const PolyUOp *view, const PolyUOp *base) {
  if (!view || !base || view == base) return false;
  while (view && view != base && view->n_src > 0 && poly_opset_has(POLY_GROUP_MOVEMENT, view->op))
    view = view->src[0];
  return view == base;
}

static bool model_role_is_state(uint8_t role) {
  return role == POLY_ROLE_PARAM || role == POLY_ROLE_AUX;
}

static bool model_role_is_abi_input(uint8_t role) {
  return role == POLY_ROLE_INPUT || role == POLY_ROLE_TARGET;
}

static bool build_snapshot_depends_on_abi_input(PolyModel *inst, const BuildBinding *state) {
  if (!inst || !inst->ctx || !inst->build || !state || !state->declared_logical_value) return true;
  for (int i = 0; i < inst->build->n_bindings; i++) {
    const BuildBinding *abi = &inst->build->bindings[i];
    if (!model_role_is_abi_input(abi->role)) continue;
    if ((abi->declared_logical_value &&
         poly_uop_reachable(inst->ctx, state->declared_logical_value, abi->declared_logical_value)
        ) ||
        (abi->buffer && poly_uop_reachable(inst->ctx, state->declared_logical_value, abi->buffer)))
      return true;
  }
  return false;
}

static bool make_build_snapshot_buffers(
    PolyCtx *ctx,
    PolyDType dtype,
    int64_t numel,
    PolyDevice device,
    PolyUOp **logical_out,
    PolyUOp **physical_out
) {
  if (logical_out) *logical_out = NULL;
  if (physical_out) *physical_out = NULL;
  if (!ctx || !logical_out || !physical_out || numel < 0 || !poly_device_can_execute(device))
    return false;

  /* Polygrad's portable resource identity and current UOp.new_buffer physical
   * form share one numeric slot. Snapshot plumbing is not a live Tensor. */
  int64_t slot = poly_ctx_next_unique_id(ctx);
  PolyUOp *device_uop = poly_device_uop(ctx, device);
  PolyUOp *logical = poly_uop_new_logical_buffer_with_slot(ctx, dtype, numel, slot);
  PolyUOp *physical =
      logical && device_uop ? poly_uop_new_buffer(ctx, device_uop, numel, dtype, slot) : NULL;
  if (!logical || !physical) return false;
  *logical_out = logical;
  *physical_out = physical;
  return true;
}

/* C argument adaptation for current Tinygrad UOp.new_buffer. */
static PolyUOp *model_new_buffer(PolyCtx *ctx, PolyDType dtype, int64_t numel) {
  PolyDevice device = poly_ctx_get_preferred_device(ctx);
  if (!poly_device_can_execute(device)) device = poly_device_default();
  PolyUOp *device_uop = poly_device_uop(ctx, device);
  return device_uop
             ? poly_uop_new_buffer(ctx, device_uop, numel, dtype, poly_ctx_next_unique_id(ctx))
             : NULL;
}

typedef struct {
  BuildBufferTransaction *transaction;
  int cap;
} BuildBufferTransactionCapture;

static void capture_build_buffer_binding(const void *key, void *value, void *userdata) {
  BuildBufferTransactionCapture *capture = userdata;
  if (!capture || !capture->transaction || capture->transaction->n >= capture->cap) return;
  int i = capture->transaction->n++;
  capture->transaction->keys[i] = (PolyUOp *)key;
  capture->transaction->heads[i] = (PolyBuffer *)value;
  capture->transaction->saved_heads[i] = *(PolyBuffer *)value;
}

static void build_buffer_transaction_discard(PolyCtx *ctx, BuildBufferTransaction *transaction);

static bool build_buffer_transaction_begin(PolyCtx *ctx, BuildBufferTransaction *transaction) {
  if (!ctx || !transaction) return false;
  memset(transaction, 0, sizeof(*transaction));
  size_t cap_size = poly_map_len(ctx->buffers);
  if (cap_size > INT_MAX) return false;
  int cap = (int)cap_size;
  if (cap == 0) return true;
  transaction->keys = calloc((size_t)cap, sizeof(*transaction->keys));
  transaction->heads = calloc((size_t)cap, sizeof(*transaction->heads));
  transaction->saved_heads = calloc((size_t)cap, sizeof(*transaction->saved_heads));
  if (!transaction->keys || !transaction->heads || !transaction->saved_heads) {
    free(transaction->keys);
    free(transaction->heads);
    free(transaction->saved_heads);
    memset(transaction, 0, sizeof(*transaction));
    return false;
  }
  BuildBufferTransactionCapture capture = {.transaction = transaction, .cap = cap};
  poly_map_foreach(ctx->buffers, capture_build_buffer_binding, &capture);
  if (transaction->n != cap) {
    build_buffer_transaction_discard(ctx, transaction);
    return false;
  }
  /* Polygrad's C-only Model transaction replaces Tinygrad's live Tensor
   * ownership (2026-08-22/a9069c177a9d tensor.py:247-262).  Keep captured
   * residency alive across allocation safe points until commit or rollback. */
  for (int i = 0; i < transaction->n; i++) {
    if (poly_uop_retain(ctx, transaction->keys[i]) != 0) {
      build_buffer_transaction_discard(ctx, transaction);
      return false;
    }
    transaction->n_retained++;
  }
  return true;
}

static void build_buffer_transaction_discard(PolyCtx *ctx, BuildBufferTransaction *transaction) {
  if (!transaction) return;
  for (int i = 0; i < transaction->n_retained; i++)
    poly_uop_release(ctx, transaction->keys[i]);
  free(transaction->keys);
  free(transaction->heads);
  free(transaction->saved_heads);
  memset(transaction, 0, sizeof(*transaction));
}

static int build_buffer_transaction_find(
    const BuildBufferTransaction *transaction,
    const PolyUOp *key
) {
  if (!transaction || !key) return -1;
  for (int i = 0; i < transaction->n; i++)
    if (transaction->keys[i] == key) return i;
  return -1;
}

typedef struct {
  PolyUOp **keys;
  int cap;
  int n;
} BuildBufferCurrentKeys;

static void capture_current_buffer_key(const void *key, void *value, void *userdata) {
  (void)value;
  BuildBufferCurrentKeys *capture = userdata;
  if (!capture || capture->n >= capture->cap) return;
  capture->keys[capture->n++] = (PolyUOp *)key;
}

static bool build_buffer_transaction_rollback(
    PolyCtx *ctx,
    const BuildBufferTransaction *transaction
) {
  if (!ctx || !transaction) return false;
  size_t current_cap_size = poly_map_len(ctx->buffers);
  if (current_cap_size > INT_MAX) return false;
  int current_cap = (int)current_cap_size;
  PolyUOp **current_keys =
      current_cap > 0 ? calloc((size_t)current_cap, sizeof(*current_keys)) : NULL;
  if (current_cap > 0 && !current_keys) return false;
  BuildBufferCurrentKeys current = {
      .keys = current_keys,
      .cap = current_cap,
      .n = 0,
  };
  poly_map_foreach(ctx->buffers, capture_current_buffer_key, &current);
  if (current.n != current_cap) {
    free(current_keys);
    return false;
  }

  bool ok = true;
  for (int i = 0; i < current.n; i++) {
    PolyUOp *key = current.keys[i];
    int prior = build_buffer_transaction_find(transaction, key);
    if (prior < 0) {
      poly_buffer_remove(ctx, key);
      continue;
    }
    PolyBuffer *saved_head = transaction->heads[prior];
    PolyBuffer *head = poly_buffer_get(ctx, key);
    if (head != saved_head) {
      PolyBuffer *cursor = head;
      while (cursor && cursor != saved_head)
        cursor = cursor->src;
      if (cursor != saved_head) {
        ok = false;
        continue;
      }
      cursor = head;
      while (cursor && cursor != saved_head) {
        PolyBuffer *next = cursor->src;
        cursor->src = NULL;
        poly_buffer_free_chain(ctx, cursor);
        cursor = next;
      }
      poly_map_set(ctx->buffers, poly_ptr_hash(key), key, saved_head, poly_ptr_eq);
    }
    *saved_head = transaction->saved_heads[prior];
  }
  free(current_keys);

  for (int i = 0; i < transaction->n; i++) {
    PolyUOp *key = transaction->keys[i];
    if (poly_buffer_get(ctx, key)) continue;
    *transaction->heads[i] = transaction->saved_heads[i];
    poly_map_set(ctx->buffers, poly_ptr_hash(key), key, transaction->heads[i], poly_ptr_eq);
  }
  return ok;
}

static void release_build_named_value_snapshots(PolyCtx *ctx, PolyModelBuildState *build) {
  if (!ctx || !build) return;
  for (int i = 0; i < build->n_bindings; i++) {
    BuildBinding *binding = &build->bindings[i];
    if (!binding->needs_snapshot || !binding->physical_buffer) continue;
    bool first = true;
    for (int j = 0; j < i; j++) {
      BuildBinding *prior = &build->bindings[j];
      if (prior->needs_snapshot && prior->physical_buffer == binding->physical_buffer) {
        first = false;
        break;
      }
    }
    if (first) {
      if (binding->snapshot_residency_retained) poly_uop_release(ctx, binding->physical_buffer);
      poly_buffer_remove(ctx, binding->physical_buffer);
    }
  }
  for (int i = 0; i < build->n_bindings; i++) {
    BuildBinding *binding = &build->bindings[i];
    if (!binding->needs_snapshot) continue;
    binding->buffer = NULL;
    binding->physical_buffer = NULL;
    binding->initial_data_buffer = NULL;
    binding->snapshot_residency_retained = false;
  }
}

static PolyStatus snapshot_build_named_value(PolyModel *inst, BuildBinding *binding) {
  if (!inst || !inst->ctx || !binding || !binding->declared_logical_value ||
      !binding->declared_physical_value)
    return POLY_STATUS_INVALID;
  int64_t numel = poly_shape_numel_checked(binding->shape, binding->ndim);
  PolyDevice device = poly_uop_device(binding->declared_physical_value);
  if (!poly_device_can_execute(device)) device = poly_ctx_get_preferred_device(inst->ctx);
  if (!poly_device_can_execute(device)) device = poly_device_default();
  if (numel < 0 || !poly_device_can_execute(device)) return POLY_STATUS_INVALID;

  PolyUOp *logical_buffer = NULL;
  PolyUOp *physical_buffer = NULL;
  /* Tinygrad 2026-08-22/a9069c177a9d UOp.new_buffer forbids weak storage;
   * UOp.empty_like commits inferred values with strong_dtype
   * (uop/ops.py:814-827). A named initializer crosses the same persistent
   * storage boundary while its preserved logical value remains weak. */
  PolyDType storage_dtype = poly_dtype_strong(binding->declared_logical_value->dtype);
  if (!make_build_snapshot_buffers(
          inst->ctx, storage_dtype, numel, device, &logical_buffer, &physical_buffer
      ))
    return POLY_STATUS_ERROR;

  binding->buffer = logical_buffer;
  binding->physical_buffer = physical_buffer;
  /* Tinygrad keeps a realized named Tensor's UOp live until its bytes are
   * consumed. Model's C-only build adapter needs the equivalent explicit
   * owner across allocation safe points before the built Model exists. */
  if (poly_uop_retain(inst->ctx, physical_buffer) != 0) return POLY_STATUS_ERROR;
  binding->snapshot_residency_retained = true;

  /* A current BUFFER/view already owns the requested bytes.  Otherwise
   * evaluate the exact eager physical value into the fresh resource without
   * publishing callify's becomes-map to any live Tensor. */
  if (!binding->initial_data_buffer && numel > 0) {
    int64_t flat[] = {numel};
    PolyUOp *flat_value = poly_reshape(inst->ctx, binding->declared_physical_value, flat, 1);
    PolyUOp *store = flat_value ? poly_store_val(inst->ctx, physical_buffer, flat_value) : NULL;
    PolyUOp *sink = store ? poly_sink1(inst->ctx, store) : NULL;
    if (!sink || poly_realize_sink(inst->ctx, sink) != 0) return POLY_STATUS_ERROR;
    binding->initial_data_buffer = physical_buffer;
  }
  return POLY_STATUS_OK;
}

static PolyStatus prepare_build_named_value_snapshots(PolyModel *inst) {
  PolyModelBuildState *build = inst ? inst->build : NULL;
  if (!inst || !build || !inst->ctx) return POLY_STATUS_INVALID;

  /* A named initializer is evaluated only while packaging. If it depends on
   * an ABI input/target, substituting the named value with persistent storage
   * would erase a per-call dependency/effect. Reject before evaluating it. */
  for (int i = 0; i < build->n_bindings; i++) {
    BuildBinding *binding = &build->bindings[i];
    if (!binding->needs_snapshot || !model_role_is_state(binding->role)) continue;
    if (build_snapshot_depends_on_abi_input(inst, binding)) {
      poly_model_set_error(
          inst, POLY_STATUS_INVALID, __func__, "named state '%s' depends on an input or target",
          binding->name
      );
      return POLY_STATUS_INVALID;
    }
  }

  /* Full storage aliases already share one logical buffer identity. A named
   * partial/lazy movement view has different value identity but the same
   * physical lineage; until named view resources carry offsets/strides, it
   * must not silently become independent storage. */
  for (int i = 0; i < build->n_bindings; i++) {
    BuildBinding *a = &build->bindings[i];
    if (!model_role_is_state(a->role)) continue;
    for (int j = 0; j < i; j++) {
      BuildBinding *b = &build->bindings[j];
      if (!model_role_is_state(b->role) || a->declared_logical_value == b->declared_logical_value ||
          (a->buffer && a->buffer == b->buffer))
        continue;
      if (model_value_is_movement_view_of(a->declared_physical_value, b->declared_physical_value) ||
          model_value_is_movement_view_of(b->declared_physical_value, a->declared_physical_value)) {
        poly_model_set_error(
            inst, POLY_STATUS_INVALID, __func__, "unsupported named view state for '%s' and '%s'",
            b->name, a->name
        );
        return POLY_STATUS_INVALID;
      }
    }
  }

  /* Validate occurrence injectivity before evaluating any initializer. */
  for (int i = 0; i < build->n_bindings; i++) {
    BuildBinding *a = &build->bindings[i];
    if (!a->needs_snapshot) continue;
    for (int j = 0; j < i; j++) {
      BuildBinding *b = &build->bindings[j];
      if (!b->needs_snapshot) continue;
      const bool same_logical = a->declared_logical_value == b->declared_logical_value;
      const bool same_alias = build_snapshot_bindings_are_one_alias(a, b);
      if (same_logical && !same_alias) {
        poly_model_set_error(
            inst, POLY_STATUS_INVALID, __func__, "ambiguous state occurrence for '%s' and '%s'",
            b->name, a->name
        );
        return POLY_STATUS_INVALID;
      }
      if (!same_logical && same_alias) {
        poly_model_set_error(
            inst, POLY_STATUS_INVALID, __func__, "unsupported named view state for '%s' and '%s'",
            b->name, a->name
        );
        return POLY_STATUS_INVALID;
      }
    }
  }

  for (int i = 0; i < build->n_bindings; i++) {
    BuildBinding *binding = &build->bindings[i];
    if (!binding->needs_snapshot) continue;
    int alias = -1;
    for (int j = 0; j < i; j++) {
      BuildBinding *prior = &build->bindings[j];
      if (prior->needs_snapshot &&
          prior->declared_logical_value == binding->declared_logical_value &&
          build_snapshot_bindings_are_one_alias(prior, binding)) {
        alias = j;
        break;
      }
    }
    if (alias >= 0) {
      BuildBinding *prior = &build->bindings[alias];
      binding->buffer = prior->buffer;
      binding->physical_buffer = prior->physical_buffer;
      binding->initial_data_buffer = prior->initial_data_buffer;
      continue;
    }
    PolyStatus st = snapshot_build_named_value(inst, binding);
    if (st != POLY_STATUS_OK) {
      release_build_named_value_snapshots(inst->ctx, build);
      poly_model_set_error(
          inst, st, __func__, "failed to snapshot named state '%s'", binding->name
      );
      return st;
    }
  }
  return POLY_STATUS_OK;
}

typedef struct {
  const PolyModelBuildState *build;
} BuildNamedValueGate;

static bool build_named_value_gate(PolyUOp *u, void *user_data) {
  const BuildNamedValueGate *gate = user_data;
  if (!u || !gate || !gate->build) return false;
  for (int i = 0; i < gate->build->n_bindings; i++) {
    const BuildBinding *binding = &gate->build->bindings[i];
    if (binding->role != POLY_ROLE_OUTPUT && binding->declared_logical_value == u) return false;
  }
  return true;
}

static PolyStatus validate_build_reachable_storage(PolyModel *inst) {
  PolyModelBuildState *build = inst ? inst->build : NULL;
  if (!inst || !build || !inst->ctx) return POLY_STATUS_INVALID;

  for (int i = 0; i < build->n_bindings; i++) {
    BuildBinding *out = &build->bindings[i];
    if (out->role != POLY_ROLE_OUTPUT) continue;

    PolyUOp *root = out->declared_logical_value;
    if (!root) {
      poly_model_set_error(
          inst, POLY_STATUS_INVALID, __func__, "output '%s' has no tensor root", out->name
      );
      return POLY_STATUS_INVALID;
    }

    BuildNamedValueGate gate = {build};
    int n_topo = 0;
    PolyUOp **topo =
        poly_toposort_ex_user_alloc(inst->ctx, root, &n_topo, build_named_value_gate, &gate, false);
    if (!topo && n_topo != 0) {
      poly_model_set_error(
          inst, POLY_STATUS_ERROR, __func__, "failed to walk output '%s' graph", out->name
      );
      return POLY_STATUS_ERROR;
    }

    for (int j = 0; j < n_topo; j++) {
      PolyUOp *u = topo[j];
      if (!u || (u->op != POLY_OP_BUFFER && u->op != POLY_OP_PARAM)) continue;
      if (find_build_storage_binding(build, u)) continue;
      PolyTensor *leaf_tensor = poly_tensor_find_storage_identity(inst->ctx, u);
      if (leaf_tensor && poly_tensor_provenance(leaf_tensor) != POLY_TENSOR_PROVENANCE_UNKNOWN &&
          poly_tensor_provenance(leaf_tensor) != POLY_TENSOR_PROVENANCE_CONST_INIT) {
        poly_toposort_free(topo);
        poly_model_set_error(
            inst, POLY_STATUS_INVALID, __func__, "output '%s' references unbound %s storage %s",
            out->name,
            poly_tensor_provenance(leaf_tensor) == POLY_TENSOR_PROVENANCE_USER_INPUT ? "input"
                                                                                     : "state",
            poly_op_name(u->op)
        );
        return POLY_STATUS_INVALID;
      }
      poly_toposort_free(topo);
      poly_model_set_error(
          inst, POLY_STATUS_INVALID, __func__, "output '%s' references unbound storage %s",
          out->name, poly_op_name(u->op)
      );
      return POLY_STATUS_INVALID;
    }
    poly_toposort_free(topo);
  }
  return POLY_STATUS_OK;
}

static PolyStatus validate_build_entrypoints(PolyModel *inst) {
  PolyModelBuildState *build = inst->build;
  if (build->n_entrypoints <= 0) {
    poly_model_set_error(inst, POLY_STATUS_INVALID, __func__, "model has no entrypoints");
    return POLY_STATUS_INVALID;
  }
  for (int i = 0; i < build->n_entrypoints; i++) {
    BuildEntrypoint *ep = &build->entrypoints[i];
    for (int j = 0; j < ep->n_inputs; j++) {
      BuildBinding *b = find_build_binding(build, ep->inputs[j]);
      if (!b || !(b->role == POLY_ROLE_INPUT || b->role == POLY_ROLE_TARGET ||
                  b->role == POLY_ROLE_PARAM || b->role == POLY_ROLE_AUX)) {
        poly_model_set_error(
            inst, POLY_STATUS_INVALID, __func__, "entrypoint '%s' references unknown input '%s'",
            ep->name, ep->inputs[j]
        );
        return POLY_STATUS_INVALID;
      }
    }
    for (int j = 0; j < ep->n_outputs; j++) {
      BuildBinding *b = find_build_binding(build, ep->outputs[j]);
      if (!b || b->role != POLY_ROLE_OUTPUT) {
        poly_model_set_error(
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
        poly_model_set_error(
            inst, POLY_STATUS_INVALID, __func__, "entrypoint '%s' objective '%s' is not an output",
            ep->name, ep->objective
        );
        return POLY_STATUS_INVALID;
      }
      int64_t numel = poly_shape_numel_checked(objective->shape, objective->ndim);
      if (numel != 1) {
        poly_model_set_error(
            inst, POLY_STATUS_INVALID, __func__,
            "entrypoint '%s' objective '%s' must be scalar or one element", ep->name, ep->objective
        );
        return POLY_STATUS_INVALID;
      }
    }
  }
  return POLY_STATUS_OK;
}

PolyStatus poly_model_build(PolyModel *inst, PolyModelError *err) {
  if (require_stage(inst, POLY_MODEL_BUILDING, __func__) != POLY_STATUS_OK) {
    poly_model_copy_error(inst, err);
    return POLY_STATUS_BAD_STAGE;
  }
  PolyModelBuildState *build = inst->build;
  BuildBufferTransaction buffer_transaction = {0};
  bool buffer_transaction_active = false;
  PolyStatus st = validate_build_entrypoints(inst);
  if (st != POLY_STATUS_OK) goto fail;
  st = validate_build_reachable_storage(inst);
  if (st != POLY_STATUS_OK) goto fail;
  /* All graph/name/role checks must precede evaluation of a lazy named value.
   * A rejected from_bindings adapter must not realize into its caller's ctx. */
  if (!build_buffer_transaction_begin(inst->ctx, &buffer_transaction)) {
    poly_model_set_error(
        inst, POLY_STATUS_NOMEM, __func__, "failed to snapshot caller buffer residency"
    );
    st = POLY_STATUS_NOMEM;
    goto fail;
  }
  buffer_transaction_active = true;
  st = prepare_build_named_value_snapshots(inst);
  if (st != POLY_STATUS_OK) goto fail;
  if (build->n_bindings <= 0) {
    poly_model_set_error(inst, POLY_STATUS_INVALID, __func__, "model has no bindings");
    st = POLY_STATUS_INVALID;
    goto fail;
  }

  PolyIrBufEntry *bufs = calloc((size_t)build->n_bindings, sizeof(*bufs));
  PolyIrEntrypoint *eps = calloc((size_t)build->n_entrypoints, sizeof(*eps));
  if (!bufs || !eps) {
    poly_model_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    st = POLY_STATUS_NOMEM;
    goto pack_fail;
  }

  for (int i = 0; i < build->n_bindings; i++) {
    BuildBinding *b = &build->bindings[i];
    if (b->role == POLY_ROLE_OUTPUT) {
      PolyUOp *value = b->declared_logical_value;
      int64_t numel = poly_shape_numel_checked(b->shape, b->ndim);
      b->buffer = poly_uop_new_logical_buffer(inst->ctx, poly_dtype_strong(value->dtype), numel);
      if (!b->buffer) {
        poly_model_set_error(
            inst, POLY_STATUS_ERROR, __func__, "failed to create output buffer '%s'", b->name
        );
        st = POLY_STATUS_ERROR;
        goto pack_fail;
      }
    }
    if (b->role == POLY_ROLE_OUTPUT && !b->buffer) {
      poly_model_set_error(
          inst, POLY_STATUS_INVALID, __func__, "binding '%s' has no buffer", b->name
      );
      st = POLY_STATUS_INVALID;
      goto pack_fail;
    }
    bufs[i] = (PolyIrBufEntry){
        .name = b->name,
        .role = b->role,
        .buffer = b->role == POLY_ROLE_OUTPUT ? b->buffer : b->declared_logical_value,
        .ndim = b->ndim,
        .trainable = b->trainable,
        .trainable_set = true,
    };
    if (b->ndim > 0) memcpy(bufs[i].shape, b->shape, (size_t)b->ndim * sizeof(int64_t));
  }

  for (int i = 0; i < build->n_entrypoints; i++) {
    BuildEntrypoint *ep = &build->entrypoints[i];
    PolyUOp **logical_stores = calloc((size_t)ep->n_outputs, sizeof(*logical_stores));
    if (!logical_stores) {
      free(logical_stores);
      poly_model_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
      st = POLY_STATUS_NOMEM;
      goto pack_fail;
    }
    for (int j = 0; j < ep->n_outputs; j++) {
      BuildBinding *out = find_build_binding(build, ep->outputs[j]);
      PolyUOp *logical_value = out->declared_logical_value;
      int64_t numel = poly_shape_numel_checked(out->shape, out->ndim);
      if (!(out->ndim == 1 && out->shape[0] == numel)) {
        int64_t flat[] = {numel};
        logical_value = poly_reshape(inst->ctx, logical_value, flat, 1);
      }
      logical_stores[j] = poly_store_val(inst->ctx, out->buffer, logical_value);
    }
    eps[i].name = ep->name;
    eps[i].sink = ep->n_outputs == 1 ? poly_sink1(inst->ctx, logical_stores[0])
                                     : poly_sink_n(inst->ctx, logical_stores, ep->n_outputs);
    eps[i].inputs = (const char **)ep->inputs;
    eps[i].n_inputs = ep->n_inputs;
    eps[i].outputs = (const char **)ep->outputs;
    eps[i].n_outputs = ep->n_outputs;
    eps[i].objective = ep->objective;
    eps[i].flags = ep->flags;
    free(logical_stores);
    if (!eps[i].sink) {
      poly_model_set_error(
          inst, POLY_STATUS_ERROR, __func__, "failed to build entrypoint '%s'", ep->name
      );
      st = POLY_STATUS_ERROR;
      goto pack_fail;
    }
  }

  PolyIrSpec spec = {
      .ctx = inst->ctx,
      .bufs = bufs,
      .n_bufs = build->n_bindings,
      .entrypoints = eps,
      .n_entrypoints = build->n_entrypoints,
  };
  PolyModel *built = model_from_spec(&spec, NULL, NULL, false, false);
  free(bufs);
  free(eps);
  if (!built) {
    poly_model_set_error(inst, POLY_STATUS_ERROR, __func__, "failed to pack runtime model");
    st = POLY_STATUS_ERROR;
    goto fail;
  }
  if (copy_initial_buffer_data(built, inst->ctx, build) != 0) {
    poly_model_free(built);
    poly_model_set_error(inst, POLY_STATUS_ERROR, __func__, "failed to snapshot initial state");
    st = POLY_STATUS_ERROR;
    goto fail;
  }
  release_build_named_value_snapshots(inst->ctx, build);
  PolyDevice default_device = poly_ctx_get_preferred_device(inst->ctx);
  if (!poly_device_can_execute(default_device)) default_device = poly_device_default();
  if (poly_model_set_device(built, default_device) != 0) {
    poly_model_free(built);
    poly_model_set_error(inst, POLY_STATUS_ERROR, __func__, "default placement failed");
    st = POLY_STATUS_ERROR;
    goto fail;
  }
  PolyModelOptions opts = build->opts;
  build_buffer_transaction_discard(inst->ctx, &buffer_transaction);
  buffer_transaction_active = false;
  build_state_free(build);
  for (int i = 0; i < inst->n_residency_roots; i++)
    poly_uop_release(inst->ctx, inst->residency_roots[i]);
  free(inst->residency_roots);
  *inst = *built;
  free(built);
  inst->stage = POLY_MODEL_BUILT;
  inst->build = NULL;
  inst->owns_ctx = opts.own_ctx_on_success;
  memset(&inst->last_error, 0, sizeof(inst->last_error));
  if (err) memset(err, 0, sizeof(*err));
  return POLY_STATUS_OK;

pack_fail:
  free(bufs);
  free(eps);

fail:
  if (build) release_build_named_value_snapshots(inst->ctx, build);
  if (buffer_transaction_active) {
    if (!build_buffer_transaction_rollback(inst->ctx, &buffer_transaction)) {
      poly_model_set_error(
          inst, POLY_STATUS_ERROR, __func__, "failed to restore caller buffer residency"
      );
      st = POLY_STATUS_ERROR;
    }
    build_buffer_transaction_discard(inst->ctx, &buffer_transaction);
  }
  inst->stage = POLY_MODEL_FAILED;
  if (build && build->opts.own_ctx_on_failure) inst->owns_ctx = true;
  poly_model_copy_error(inst, err);
  return st;
}

static bool runtime_module_has_input(const RuntimeModule *module, PolyUOp *u) {
  if (!module || !u) return false;
  for (int i = 0; i < module->n_inputs; i++)
    if (module->logical_inputs[i] == u) return true;
  return false;
}

static bool runtime_modules_valid(
    const PolyModel *inst,
    const RuntimeModule *modules,
    int n_modules
) {
  if (!inst || !modules || n_modules <= 0) return false;
  for (int i = 0; i < n_modules; i++) {
    const RuntimeModule *module = &modules[i];
    if (!valid_binding_name(module->name) || module->n_inputs < 0 ||
        (module->n_inputs > 0 && !module->logical_inputs) || !module->logical_output ||
        !poly_ctx_owns_ptr(inst->ctx, module->logical_output))
      return false;
    for (int j = 0; j < i; j++)
      if (strcmp(modules[j].name, module->name) == 0 ||
          modules[j].logical_output == module->logical_output)
        return false;
    for (int j = 0; j < module->n_inputs; j++) {
      PolyUOp *input = module->logical_inputs[j];
      if (!input || !poly_ctx_owns_ptr(inst->ctx, input) || input == module->logical_output ||
          !poly_uop_reachable(inst->ctx, module->logical_output, input))
        return false;
      for (int k = 0; k < j; k++)
        if (module->logical_inputs[k] == input) return false;
    }

    bool entry_reachable = false;
    for (int j = 0; j < inst->n_entrypoints && !entry_reachable; j++)
      entry_reachable =
          poly_uop_reachable(inst->ctx, inst->entrypoints[j].logical_sink, module->logical_output);
    if (!entry_reachable) return false;

    for (int j = 0; j < n_modules; j++) {
      if (i == j ||
          !poly_uop_reachable(inst->ctx, module->logical_output, modules[j].logical_output))
        continue;
      if (j >= i || !runtime_module_has_input(module, modules[j].logical_output)) return false;
    }
  }
  return true;
}

int poly_model_define_modules(PolyModel *inst, const PolyModelModuleSpec *modules, int n_modules) {
  if (!inst || inst->stage != POLY_MODEL_BUILT || !inst->ctx || !modules || n_modules <= 0)
    return -1;

  RuntimeModule *candidate = calloc((size_t)n_modules, sizeof(*candidate));
  if (!candidate) return -1;
  int rc = -1;

  for (int i = 0; i < n_modules; i++) {
    const PolyModelModuleSpec *src = &modules[i];
    PolyUOp *output = src->output ? poly_tensor_uop_logical(src->output) : NULL;
    if (!valid_binding_name(src->name) || src->n_inputs < 0 ||
        (src->n_inputs > 0 && !src->inputs) || !output || !poly_ctx_owns_ptr(inst->ctx, output))
      goto cleanup;
    candidate[i].name = dup_cstr(src->name);
    candidate[i].logical_output = output;
    candidate[i].n_inputs = src->n_inputs;
    candidate[i].logical_inputs =
        src->n_inputs > 0 ? calloc((size_t)src->n_inputs, sizeof(PolyUOp *)) : NULL;
    if (!candidate[i].name || (src->n_inputs > 0 && !candidate[i].logical_inputs)) goto cleanup;
    for (int j = 0; j < src->n_inputs; j++) {
      PolyUOp *input = src->inputs[j] ? poly_tensor_uop_logical(src->inputs[j]) : NULL;
      if (!input || !poly_ctx_owns_ptr(inst->ctx, input) || input == output ||
          !poly_uop_reachable(inst->ctx, output, input))
        goto cleanup;
      candidate[i].logical_inputs[j] = input;
    }
  }

  if (!runtime_modules_valid(inst, candidate, n_modules)) goto cleanup;

  runtime_modules_free(inst->modules, inst->n_modules);
  inst->modules = candidate;
  inst->n_modules = n_modules;
  candidate = NULL;
  rc = 0;

cleanup:
  runtime_modules_free(candidate, n_modules);
  return rc;
}

int poly_model_define_module_arrays(
    PolyModel *inst,
    const char **names,
    PolyTensor **inputs,
    const int *input_counts,
    PolyTensor **outputs,
    int n_modules
) {
  if (!inst || !names || !input_counts || !outputs || n_modules <= 0) return -1;
  PolyModelModuleSpec *modules = calloc((size_t)n_modules, sizeof(PolyModelModuleSpec));
  if (!modules) return -1;
  int input_off = 0;
  int rc = -1;
  for (int i = 0; i < n_modules; i++) {
    if (input_counts[i] < 0 || (input_counts[i] > 0 && !inputs)) goto cleanup;
    modules[i] = (PolyModelModuleSpec){
        .name = names[i],
        .inputs = input_counts[i] > 0 ? &inputs[input_off] : NULL,
        .n_inputs = input_counts[i],
        .output = outputs[i],
    };
    input_off += input_counts[i];
  }
  rc = poly_model_define_modules(inst, modules, n_modules);

cleanup:
  free(modules);
  return rc;
}

PolyModel *poly_model_from_bindings(
    PolyCtx *ctx,
    const PolyBindingSpec *bindings,
    int n_bindings,
    const PolyEntrypointSpec *entrypoints,
    int n_entrypoints,
    const PolyModelOptions *opts,
    PolyModelError *err
) {
  if (!ctx || !bindings || n_bindings <= 0 || !entrypoints || n_entrypoints <= 0) return NULL;
  PolyModel *inst = poly_model_new(ctx, opts);
  if (!inst) return NULL;

  typedef struct {
    PolyTensor *tensor;
    PolyTensorProvenance provenance;
  } TensorMetadataSnapshot;
  TensorMetadataSnapshot *metadata = calloc((size_t)n_bindings, sizeof(*metadata));
  int n_metadata = 0;
  if (!metadata) {
    poly_model_set_error(inst, POLY_STATUS_NOMEM, __func__, "out of memory");
    goto fail;
  }
  /* append_build_binding publishes role metadata immediately for the staged
   * builder API.  The one-shot adapter is transactional: capture every unique
   * caller Tensor before the first append and restore it if any later graph,
   * snapshot, packing or placement step rejects the Model. */
  for (int i = 0; i < n_bindings; i++) {
    PolyTensor *tensor = bindings[i].tensor;
    if (!tensor) continue;
    bool seen = false;
    for (int j = 0; j < n_metadata; j++)
      if (metadata[j].tensor == tensor) {
        seen = true;
        break;
      }
    if (seen) continue;
    metadata[n_metadata++] = (TensorMetadataSnapshot){
        .tensor = tensor,
        .provenance = tensor->provenance,
    };
  }
  for (int i = 0; i < n_bindings; i++) {
    const PolyBindingSpec *b = &bindings[i];
    bool output = b->role == POLY_ROLE_OUTPUT;
    bool trainable = b->role == POLY_ROLE_PARAM && !(b->flags & POLY_BIND_F_FROZEN);
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
      goto fail;
    }
  }
  for (int i = 0; i < n_entrypoints; i++) {
    const PolyEntrypointSpec *ep = &entrypoints[i];
    PolyEntrypointOptions ep_opts = {.objective = ep->objective, .flags = ep->flags};
    PolyStatus st = poly_model_entrypoint(
        inst, ep->name, ep->inputs, ep->n_inputs, ep->outputs, ep->n_outputs, &ep_opts
    );
    if (st != POLY_STATUS_OK) {
      goto fail;
    }
  }
  if (poly_model_build(inst, err) != POLY_STATUS_OK) {
    goto fail;
  }
  free(metadata);
  return inst;

fail:
  for (int i = 0; i < n_metadata; i++) {
    metadata[i].tensor->provenance = metadata[i].provenance;
  }
  free(metadata);
  poly_model_copy_error(inst, err);
  poly_model_free(inst);
  return NULL;
}

static void set_plain_model_error(
    PolyModelError *err,
    int code,
    const char *func,
    const char *msg
) {
  if (!err) return;
  err->code = code;
  err->func = func;
  snprintf(err->message, sizeof(err->message), "%s", msg ? msg : "error");
}

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
) {
  if (!ctx || !binding_names || !binding_roles || !binding_tensors || n_bindings <= 0 ||
      !entry_names || !entry_input_counts || !entry_output_counts || n_entrypoints <= 0) {
    set_plain_model_error(err, POLY_STATUS_INVALID, __func__, "invalid binding array inputs");
    return NULL;
  }

  PolyBindingSpec *bindings = calloc((size_t)n_bindings, sizeof(PolyBindingSpec));
  PolyEntrypointSpec *entrypoints = calloc((size_t)n_entrypoints, sizeof(PolyEntrypointSpec));
  if (!bindings || !entrypoints) {
    free(bindings);
    free(entrypoints);
    set_plain_model_error(err, POLY_STATUS_NOMEM, __func__, "out of memory");
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
      set_plain_model_error(err, POLY_STATUS_INVALID, __func__, "invalid entrypoint arrays");
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

  PolyModel *inst =
      poly_model_from_bindings(ctx, bindings, n_bindings, entrypoints, n_entrypoints, opts, err);
  free(bindings);
  free(entrypoints);
  return inst;
}

/* Lifecycle */

/* Build a PolyModel from a PolyIrSpec.
 * owns_ctx: if true, the model takes ownership of spec->ctx.
 * free_spec: if true, calls poly_ir_spec_free after building.
 * Handles alias data sharing: entries with the same buffer UOp share one allocation. */
static PolyModel *model_from_spec(
    PolyIrSpec *spec,
    PolyUOp **physical_buffers,
    PolyUOp **physical_sinks,
    bool owns_ctx,
    bool free_spec
) {
  PolyModel *inst = calloc(1, sizeof(PolyModel));
  if (!inst) return NULL;
  inst->ctx = spec->ctx;
  inst->owns_ctx = owns_ctx;
  inst->has_portable_source = true;
  inst->stage = POLY_MODEL_BUILT;
  inst->has_physical_capture = physical_buffers && physical_sinks;
  if ((physical_buffers != NULL) != (physical_sinks != NULL)) goto fail;
  if (inst->has_physical_capture) {
    for (int i = 0; i < spec->n_bufs; i++)
      if (!physical_buffers[i]) goto fail;
    for (int i = 0; i < spec->n_entrypoints; i++)
      if (!physical_sinks[i]) goto fail;
  }

  /* Copy buffers with alias data sharing */
  inst->n_bufs = spec->n_bufs;
  inst->bufs = calloc(spec->n_bufs, sizeof(NamedBuf));
  int n_params = 0;
  for (int i = 0; i < spec->n_bufs; i++) {
    PolyUOp *logical_value = spec->bufs[i].buffer;
    int64_t numel = compute_numel(spec->bufs[i].shape, spec->bufs[i].ndim);
    PolyDType value_dtype = logical_value ? logical_value->dtype : POLY_VOID;
    /* Model keeps the exact logical initializer but allocates persistent
     * storage at Tinygrad's UOp.empty_like strong_dtype boundary. */
    PolyDType storage_dtype = poly_dtype_strong(value_dtype);
    if (!logical_value || !poly_ctx_owns_ptr(spec->ctx, logical_value) || numel < 0 ||
        poly_dtype_eq(value_dtype, POLY_VOID))
      goto fail;
    PolyShape actual_shape = poly_uop_max_shape_cached(spec->ctx, logical_value);
    int64_t actual_numel = poly_shape_numel_checked(actual_shape.dims, actual_shape.ndim);
    if (actual_numel < 0 || actual_numel != numel) goto fail;
    const PolyUOp *value_identity = poly_uop_get_buffer_identity(logical_value);
    PolyUOp *resource_key = value_identity ? (PolyUOp *)value_identity : logical_value;

    if (model_role_is_state(spec->bufs[i].role)) {
      for (int j = 0; j < i; j++) {
        if (!model_role_is_state(inst->bufs[j].role) ||
            logical_value == inst->bufs[j].logical_value)
          continue;
        const PolyUOp *prior_identity = poly_uop_get_buffer_identity(inst->bufs[j].logical_value);
        if (value_identity && prior_identity == value_identity) continue;
        if (model_value_is_movement_view_of(logical_value, inst->bufs[j].logical_value) ||
            model_value_is_movement_view_of(inst->bufs[j].logical_value, logical_value))
          goto fail;
      }
    }

    int alias = -1;
    for (int j = 0; j < i; j++) {
      const PolyUOp *prior_identity = poly_uop_get_buffer_identity(inst->bufs[j].logical_value);
      PolyUOp *prior_key = prior_identity ? (PolyUOp *)prior_identity : inst->bufs[j].logical_value;
      if (prior_key == resource_key) {
        alias = j;
        break;
      }
    }
    if (alias >= 0) {
      const bool current_abi = model_role_is_abi_input(spec->bufs[i].role);
      const bool prior_abi = model_role_is_abi_input(inst->bufs[alias].role);
      const bool current_state = model_role_is_state(spec->bufs[i].role);
      const bool prior_state = model_role_is_state(inst->bufs[alias].role);
      if ((current_abi && (prior_abi || prior_state)) || (prior_abi && current_state)) goto fail;
    }
    if (alias >= 0 && (inst->bufs[alias].numel != numel ||
                       !poly_dtype_eq(inst->bufs[alias].logical_value->dtype, value_dtype)))
      goto fail;
    inst->bufs[i].name = strdup(spec->bufs[i].name);
    inst->bufs[i].role = spec->bufs[i].role;
    inst->bufs[i].trainable = spec->bufs[i].trainable_set ? spec->bufs[i].trainable
                                                          : (spec->bufs[i].role == POLY_ROLE_PARAM);
    inst->bufs[i].logical_value = logical_value;
    inst->bufs[i].logical_buffer =
        alias >= 0 ? inst->bufs[alias].logical_buffer
                   : (model_is_portable_buffer(value_identity)
                          ? (PolyUOp *)value_identity
                          : poly_uop_new_logical_buffer(spec->ctx, storage_dtype, numel));
    if (!inst->bufs[i].logical_buffer) goto fail;
    inst->bufs[i].capture_buffer =
        inst->has_physical_capture
            ? physical_buffers[i]
            : (alias >= 0 ? inst->bufs[alias].capture_buffer : inst->bufs[i].logical_buffer);
    inst->bufs[i].buffer = inst->bufs[i].capture_buffer;
    inst->bufs[i].ndim = spec->bufs[i].ndim;
    memcpy(inst->bufs[i].shape, spec->bufs[i].shape, spec->bufs[i].ndim * sizeof(int64_t));
    inst->bufs[i].numel = numel;

    /* Alias names share the same ctx-owned residency. */
    void *shared = NULL;
    if (alias >= 0) shared = inst->bufs[alias].data;
    if (shared) {
      inst->bufs[i].data = shared;
    } else {
      size_t nbytes = 0;
      PolyBuffer *host = NULL;
      if (!named_buf_nbytes_checked(&inst->bufs[i], &nbytes)) goto fail;
      if (nbytes > 0 &&
          poly_buffer_alloc_owned_host(spec->ctx, inst->bufs[i].buffer, nbytes, true, &host) != 0) {
        fprintf(
            stderr,
            "poly_model: host buffer allocation FAILED for '%s' (%lld elements = %lld MB)\n",
            spec->bufs[i].name ? spec->bufs[i].name : "?", (long long)inst->bufs[i].numel,
            (long long)(nbytes / 1024 / 1024)
        );
        goto fail;
      }
      inst->bufs[i].data = host ? host->ptr : NULL;
    }
    if (spec->bufs[i].role == POLY_ROLE_PARAM) {
      n_params++;
    }
  }

  /* A computed named state value is an initializer, not a per-call program
   * effect. Binding it to persistent storage must not erase a dependency on
   * an ABI input or target, even when checkpoint bytes are supplied. */
  for (int i = 0; i < inst->n_bufs; i++) {
    NamedBuf *state = &inst->bufs[i];
    if (!model_role_is_state(state->role) || poly_uop_get_buffer_identity(state->logical_value))
      continue;
    for (int j = 0; j < inst->n_bufs; j++) {
      NamedBuf *abi = &inst->bufs[j];
      if (!model_role_is_abi_input(abi->role)) continue;
      if ((abi->logical_value &&
           poly_uop_reachable(spec->ctx, state->logical_value, abi->logical_value)) ||
          (abi->logical_buffer &&
           poly_uop_reachable(spec->ctx, state->logical_value, abi->logical_buffer)))
        goto fail;
    }
  }

  /* Keep one ABI row per named PARAM, but match tinygrad Optimizer.__init__'s
   * dedup(params) at the optimizer boundary: aliases sharing one Model
   * storage identity receive one update and one optimizer-state family. */
  inst->n_params = n_params;
  inst->param_indices = n_params ? malloc((size_t)n_params * sizeof(int)) : NULL;
  if (n_params && !inst->param_indices) goto fail;
  int pi = 0;
  for (int i = 0; i < spec->n_bufs; i++) {
    if (spec->bufs[i].role != POLY_ROLE_PARAM) continue;
    inst->param_indices[pi++] = i;
  }
  if (build_trainable_param_indices(
          inst, &inst->trainable_param_indices, &inst->n_trainable_params
      ) != 0)
    goto fail;

  /* Copy entrypoints */
  inst->n_entrypoints = spec->n_entrypoints;
  inst->entrypoints = calloc(spec->n_entrypoints, sizeof(*inst->entrypoints));
  inst->entry_executables = calloc(spec->n_entrypoints, sizeof(*inst->entry_executables));
  if (!inst->entry_executables && spec->n_entrypoints > 0) goto fail;
  for (int i = 0; i < spec->n_entrypoints; i++) {
    inst->entrypoints[i].name = strdup(spec->entrypoints[i].name);
    inst->entrypoints[i].logical_sink = spec->entrypoints[i].sink;
    inst->entrypoints[i].capture_sink =
        inst->has_physical_capture ? physical_sinks[i] : spec->entrypoints[i].sink;
    inst->entrypoints[i].sink = inst->entrypoints[i].capture_sink;
    inst->entrypoints[i].inputs =
        dup_string_array(spec->entrypoints[i].inputs, spec->entrypoints[i].n_inputs);
    inst->entrypoints[i].n_inputs = spec->entrypoints[i].n_inputs;
    inst->entrypoints[i].outputs =
        dup_string_array(spec->entrypoints[i].outputs, spec->entrypoints[i].n_outputs);
    inst->entrypoints[i].n_outputs = spec->entrypoints[i].n_outputs;
    inst->entrypoints[i].objective = dup_cstr(spec->entrypoints[i].objective);
    inst->entrypoints[i].flags = spec->entrypoints[i].flags;
  }

  /* Portable PGIR v10 retains the same exact logical module boundaries used
   * by live Model construction. Device assignments remain external policy
   * and are deliberately not restored here. */
  inst->n_modules = spec->n_modules;
  inst->modules =
      spec->n_modules > 0 ? calloc((size_t)spec->n_modules, sizeof(*inst->modules)) : NULL;
  if (spec->n_modules > 0 && !inst->modules) goto fail;
  for (int i = 0; i < spec->n_modules; i++) {
    inst->modules[i].name = dup_cstr(spec->modules[i].name);
    inst->modules[i].n_inputs = spec->modules[i].n_inputs;
    inst->modules[i].logical_output = spec->modules[i].output;
    inst->modules[i].logical_inputs =
        spec->modules[i].n_inputs > 0 ? calloc((size_t)spec->modules[i].n_inputs, sizeof(PolyUOp *))
                                      : NULL;
    if (!inst->modules[i].name ||
        (spec->modules[i].n_inputs > 0 && !inst->modules[i].logical_inputs))
      goto fail;
    if (spec->modules[i].n_inputs > 0)
      memcpy(
          inst->modules[i].logical_inputs, spec->modules[i].inputs,
          (size_t)spec->modules[i].n_inputs * sizeof(PolyUOp *)
      );
  }
  if (spec->n_modules > 0 && !runtime_modules_valid(inst, inst->modules, inst->n_modules))
    goto fail;

  if (free_spec) poly_ir_spec_free(spec);

  /* Default optimizer: none */
  inst->training.optim.kind = POLY_OPTIM_NONE;

  if (model_refresh_residency_roots(inst) != 0) goto fail;
  return inst;

fail:
  if (free_spec) poly_ir_spec_free(spec);
  poly_model_free(inst);
  return NULL;
}

static void model_free_safetensor_views(PolySafetensorViewEx *views, int n_views) {
  if (views)
    for (int i = 0; i < n_views; i++)
      free(views[i].name);
  free(views);
}

static bool model_safetensor_dtype(PolyDType dtype, PolySafetensorDType *out) {
  if (!out) return false;
  dtype = dtype;
#define MAP_DTYPE(poly, safe)                                                                      \
  if (poly_dtype_eq(dtype, poly)) {                                                                \
    *out = safe;                                                                                   \
    return true;                                                                                   \
  }
  MAP_DTYPE(POLY_FLOAT32, POLY_ST_F32)
  MAP_DTYPE(POLY_FLOAT16, POLY_ST_F16)
  MAP_DTYPE(POLY_BFLOAT16, POLY_ST_BF16)
  MAP_DTYPE(POLY_FLOAT64, POLY_ST_F64)
  MAP_DTYPE(POLY_INT64, POLY_ST_I64)
  MAP_DTYPE(POLY_INT32, POLY_ST_I32)
  MAP_DTYPE(POLY_INT16, POLY_ST_I16)
  MAP_DTYPE(POLY_INT8, POLY_ST_I8)
  MAP_DTYPE(POLY_UINT8, POLY_ST_U8)
  MAP_DTYPE(POLY_BOOL, POLY_ST_BOOL)
  MAP_DTYPE(POLY_UINT16, POLY_ST_U16)
  MAP_DTYPE(POLY_UINT32, POLY_ST_U32)
  MAP_DTYPE(POLY_UINT64, POLY_ST_U64)
#undef MAP_DTYPE
  return false;
}

static bool model_checkpoint_row_matches(
    const NamedBuf *binding,
    const PolySafetensorViewEx *view
) {
  if (!binding || !view || !binding->buffer || view->ndim != binding->ndim ||
      view->numel != binding->numel)
    return false;
  PolySafetensorDType expected;
  size_t nbytes = 0;
  if (!model_safetensor_dtype(binding->buffer->dtype, &expected) || expected != view->dtype ||
      !named_buf_nbytes_checked(binding, &nbytes) || poly_safetensor_dtype_size(view->dtype) <= 0 ||
      (uint64_t)view->numel > SIZE_MAX / (size_t)poly_safetensor_dtype_size(view->dtype) ||
      nbytes != (size_t)view->numel * (size_t)poly_safetensor_dtype_size(view->dtype))
    return false;
  for (int d = 0; d < binding->ndim; d++)
    if (view->shape[d] != binding->shape[d]) return false;
  return true;
}

static bool model_checkpoint_binding_required(const NamedBuf *binding) {
  if (!binding) return false;
  if (binding->role == POLY_ROLE_PARAM) return true;
  return binding->role == POLY_ROLE_AUX && !poly_uop_get_buffer_identity(binding->logical_value);
}

static int model_validate_checkpoint_views(
    const PolyModel *inst,
    const PolySafetensorViewEx *views,
    int n_views,
    bool require_state
) {
  if (!inst || n_views < 0 || (n_views > 0 && !views)) return -1;
  for (int i = 0; i < n_views; i++) {
    if (!views[i].name) return -1;
    for (int j = 0; j < i; j++)
      if (strcmp(views[i].name, views[j].name) == 0) return -1;
    int bi = find_buf_by_name(inst, views[i].name);
    if (bi >= 0 && !model_checkpoint_row_matches(&inst->bufs[bi], &views[i])) return -1;
  }
  if (!require_state) return 0;
  for (int i = 0; i < inst->n_bufs; i++) {
    const NamedBuf *binding = &inst->bufs[i];
    if (!model_checkpoint_binding_required(binding)) continue;
    bool found = false;
    for (int j = 0; j < n_views; j++)
      if (strcmp(binding->name, views[j].name) == 0) {
        found = true;
        break;
      }
    if (!found) return -1;
  }
  return 0;
}

static int model_validate_checkpoint(
    const PolyModel *inst,
    const uint8_t *weights_data,
    int weights_len,
    bool require_state
) {
  if (!weights_data || weights_len <= 0) return -1;
  int n_views = 0;
  PolySafetensorViewEx *views =
      poly_safetensors_decode_ex(weights_data, weights_len, &n_views, NULL);
  if (!views) return -1;
  int rc = model_validate_checkpoint_views(inst, views, n_views, require_state);
  model_free_safetensor_views(views, n_views);
  return rc;
}

static bool model_is_computed_state(const NamedBuf *binding) {
  return binding && (binding->role == POLY_ROLE_PARAM || binding->role == POLY_ROLE_AUX) &&
         !poly_uop_get_buffer_identity(binding->logical_value);
}

static bool model_closed_initializer_op(PolyOps op) {
  return op == POLY_OP_CONST || op == POLY_OP_STACK || op == POLY_OP_CONTIGUOUS ||
         op == POLY_OP_DETACH || op == POLY_OP_REDUCE ||
         poly_opset_has(POLY_GROUP_ELEMENTWISE, op) || poly_opset_has(POLY_GROUP_MOVEMENT, op);
}

static bool model_closed_initializer_graph(PolyCtx *ctx, PolyUOp *root) {
  if (!ctx || !root) return false;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_user_alloc(ctx, root, &n_topo, NULL, NULL, false);
  if (!topo || n_topo <= 0) {
    poly_toposort_free(topo);
    return false;
  }
  bool valid = true;
  for (int i = 0; i < n_topo; i++)
    if (!topo[i] || !model_closed_initializer_op(topo[i]->op)) {
      valid = false;
      break;
    }
  poly_toposort_free(topo);
  return valid;
}

/* Fresh import may evaluate a named value exactly once only when the existing
 * aggregate placer proves the entire graph is pure and storage-free.  BUFFER,
 * COPY, STORE/AFTER, CALL/FUNCTION, UNSHARD and lowered graphs are rejected by
 * poly_place_roots; this deliberately excludes RNG and external state. */
static int model_initialize_closed_computed_state(PolyModel *inst) {
  if (!inst || !inst->ctx) return -1;
  size_t cap = (size_t)(unsigned)inst->n_bufs;
  PolyUOp **roots = calloc(cap, sizeof(*roots));
  PolyUOp **placed = calloc(cap, sizeof(*placed));
  PolyUOp **realized = calloc(cap, sizeof(*realized));
  int *binding_indices = calloc(cap, sizeof(*binding_indices));
  int n_init = 0;
  int rc = -1;
  if (cap > 0 && (!roots || !placed || !realized || !binding_indices)) goto cleanup;

  for (int i = 0; i < inst->n_bufs; i++) {
    NamedBuf *binding = &inst->bufs[i];
    if (!model_is_computed_state(binding)) continue;
    bool duplicate = false;
    for (int j = 0; j < n_init; j++)
      if (inst->bufs[binding_indices[j]].logical_buffer == binding->logical_buffer) {
        duplicate = true;
        break;
      }
    if (duplicate) continue;

    if (!model_closed_initializer_graph(inst->ctx, binding->logical_value)) goto cleanup;
    PolyShape actual = poly_uop_max_shape_cached(inst->ctx, binding->logical_value);
    PolyShape declared = {(int64_t *)binding->shape, binding->ndim};
    if (!poly_shape_eq(actual, declared) ||
        !poly_dtype_eq(
            poly_dtype_strong(binding->logical_value->dtype), binding->logical_buffer->dtype
        ))
      goto cleanup;
    roots[n_init] = binding->logical_value;
    binding_indices[n_init++] = i;
  }
  if (n_init == 0) {
    rc = 0;
    goto cleanup;
  }

  /* No binding substitutions are supplied here on purpose: any reachable
   * storage makes the initializer incomplete and poly_place_roots rejects the
   * entire aggregate before execution. */
  if (poly_place_roots(inst->ctx, roots, n_init, NULL, NULL, 0, placed) != 0) goto cleanup;

  /* Pinned UOp.empty_like + STORE: virtual values do not request storage.
   * Like build snapshots, import explicitly materializes persistent state;
   * retain every temporary destination across execution/collection. */
  int n_stores = 0;
  for (int i = 0; i < n_init; i++) {
    NamedBuf *binding = &inst->bufs[binding_indices[i]];
    if (named_buf_nbytes(binding) == 0) continue;
    int64_t flat[] = {poly_shape_numel_checked(binding->shape, binding->ndim)};
    PolyUOp *buffer = model_new_buffer(inst->ctx, binding->logical_buffer->dtype, flat[0]);
    if (!buffer || poly_uop_retain(inst->ctx, buffer) != 0) goto cleanup;
    realized[i] = buffer;
    PolyUOp *value = poly_reshape(inst->ctx, placed[i], flat, 1);
    PolyUOp *store = value ? poly_store_val(inst->ctx, buffer, value) : NULL;
    if (!store) goto cleanup;
    roots[n_stores++] = store;
  }
  if (n_stores > 0) {
    PolyUOp *sink = poly_sink_n(inst->ctx, roots, n_stores);
    if (!sink || poly_realize_sink(inst->ctx, sink) != 0) goto cleanup;
  }

  for (int i = 0; i < n_init; i++) {
    NamedBuf *binding = &inst->bufs[binding_indices[i]];
    size_t nbytes = named_buf_nbytes(binding);
    if (nbytes > 0 &&
        (poly_buffer_read(inst->ctx, realized[i], binding->data, nbytes) != 0 ||
         poly_buffer_write(inst->ctx, binding->logical_buffer, binding->data, nbytes) != 0))
      goto cleanup;
  }
  rc = 0;

cleanup:
  if (realized)
    for (int i = 0; i < n_init; i++)
      if (realized[i]) {
        poly_uop_release(inst->ctx, realized[i]);
        poly_buffer_remove(inst->ctx, realized[i]);
      }
  free(binding_indices);
  free(realized);
  free(placed);
  free(roots);
  return rc;
}

PolyModel *poly_model_from_ir(
    const uint8_t *ir_data,
    int ir_len,
    const uint8_t *weights_data,
    int weights_len
) {
  /* Import IR */
  PolyIrSpec spec;
  if (poly_ir_import(ir_data, ir_len, &spec) != 0) {
    fprintf(stderr, "poly_model_from_ir: IR import failed\n");
    return NULL;
  }

  PolyModel *inst = model_from_spec(&spec, NULL, NULL, true, true);
  if (!inst) return NULL;

  if (weights_data && weights_len > 0) {
    /* Checkpoint import remains strict for computed state. A supplied partial
     * archive never silently falls back to initializers for missing names. */
    if (model_validate_checkpoint(inst, weights_data, weights_len, true) != 0) {
      fprintf(stderr, "poly_model_from_ir: checkpoint does not match required named state\n");
      poly_model_free(inst);
      return NULL;
    }
  } else if (model_initialize_closed_computed_state(inst) != 0) {
    fprintf(stderr, "poly_model_from_ir: computed named state is not a closed initializer\n");
    poly_model_free(inst);
    return NULL;
  }

  /* Cross the explicit named-value binding and placement boundary before
   * exposing a runnable Model; default Tensor realization remains
   * placement-free. */
  if (poly_model_set_device(inst, POLY_DEVICE_AUTO) != 0) {
    fprintf(stderr, "poly_model_from_ir: default placement failed\n");
    poly_model_free(inst);
    return NULL;
  }

  /* Import weights if provided */
  if (weights_data && weights_len > 0) {
    if (poly_model_import_weights(inst, weights_data, weights_len) != 0) {
      fprintf(stderr, "poly_model_from_ir: weight import failed\n");
      poly_model_free(inst);
      return NULL;
    }
  }

  return inst;
}

/* Model from PolyCtx registry */

static PolyModel *model_from_named_sinks(
    PolyCtx *ctx,
    const char **names,
    PolyUOp **sinks,
    int n_sinks
) {
  if (!ctx || !names || !sinks || n_sinks <= 0) {
    fprintf(stderr, "poly_model_from_sinks: zero entrypoints\n");
    return NULL;
  }

  /* Collect reachable BUFFER UOps from all entrypoint SINKs */
  PolyMap *reachable = poly_map_new(32);
  if (!reachable) return NULL;
  for (int i = 0; i < n_sinks; i++) {
    if (!names[i] || !sinks[i]) {
      fprintf(stderr, "poly_model_from_sinks: null entrypoint at %d\n", i);
      poly_map_destroy(reachable);
      return NULL;
    }
    PolyUOp *sink = sinks[i];
    PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_scratch(ctx, sink, &n_topo);
    if (!topo && n_topo != 0) {
      fprintf(stderr, "poly_model_from_sinks: failed to walk entrypoint %d\n", i);
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

  /* Build spec and create model (does NOT own ctx) */
  PolyIrSpec spec = {
      .ctx = ctx,
      .bufs = bufs,
      .n_bufs = n_bufs,
      .entrypoints = eps,
      .n_entrypoints = n_sinks,
  };
  PolyModel *inst = model_from_spec(&spec, NULL, NULL, false, false);
  if (inst) {
    for (int i = 0; i < inst->n_bufs; i++) {
      PolyUOp *source = bufs[i].buffer;
      PolyBuffer *src = poly_buffer_get(ctx, source);
      size_t nbytes = named_buf_nbytes(&inst->bufs[i]);
      if (!src || !src->ptr || src->nbytes < nbytes || nbytes == 0) continue;
      /* Models own host-side ABI storage, but the source tensor may already
       * live in a backend-specific residency (WASM heap, CUDA, WebGPU, etc.).
       * `valid` only describes the host shadow freshness; backend copyout can
       * still read the current residency, so do not reject invalid host shadows.
       * Fall back to direct copy only for valid legacy host buffers whose
       * allocator cannot service copyout. */
      if (poly_buffer_read(ctx, source, inst->bufs[i].data, nbytes) != 0 && src->valid &&
          poly_device_is_host_addressable(src->device))
        memcpy(inst->bufs[i].data, src->ptr, nbytes);
    }
    /* Approved Model boundary: registry sinks are portable logical roots.
     * Activate them before exposing the Model so create_linear_with_vars
     * receives device-bound BUFFER arguments, as in Tinygrad 2026-08-22
     * uop/ops.py:814-816 and schedule/__init__.py:181-209. */
    if (poly_model_set_device(inst, POLY_DEVICE_AUTO) != 0) {
      poly_model_free(inst);
      inst = NULL;
    }
  }

  free(bufs);
  free(eps);
  return inst;
}

PolyModel *poly_model_from_ctx(PolyCtx *ctx) {
  if (!ctx) return NULL;
  int n_ep = poly_ctx_entrypoint_count(ctx);
  if (n_ep == 0) {
    fprintf(stderr, "poly_model_from_ctx: zero entrypoints\n");
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
  PolyModel *inst = model_from_named_sinks(ctx, names, sinks, n_ep);
  free(names);
  free(sinks);
  return inst;
}

PolyModel *poly_model_from_sinks(PolyCtx *ctx, const char **names, PolyUOp **sinks, int n_sinks) {
  return model_from_named_sinks(ctx, names, sinks, n_sinks);
}

static void cached_executable_clear(PolyLinearEntry *executable);

static void vag_free(VagState *vag, int n_params) {
  if (!vag) return;
  cached_executable_clear(&vag->executable);
  if (vag->grad_datas) {
    for (int i = 0; i < n_params; i++)
      free(vag->grad_datas[i]);
    free(vag->grad_datas);
  }
  free(vag->grad_out_bufs);
  free(vag->grad_uops);
  free(vag);
}

static void cached_executable_clear(PolyLinearEntry *executable) {
  if (!executable) return;
  free(executable->var_bindings);
  memset(executable, 0, sizeof(*executable));
}

static void train_plan_clear(TrainState *ts) {
  if (!ts) return;
  cached_executable_clear(&ts->executable);
}

static void entry_executables_clear(PolyModel *inst) {
  if (!inst || !inst->entry_executables) return;
  for (int i = 0; i < inst->n_entrypoints; i++)
    cached_executable_clear(&inst->entry_executables[i]);
}

static void train_free(TrainState *ts, int n_params) {
  if (!ts) return;
  train_plan_clear(ts);
  free(ts->m_bufs);
  free(ts->v_bufs);
  free(ts);
}

static void runtime_modules_free(RuntimeModule *modules, int n_modules) {
  if (!modules) return;
  for (int i = 0; i < n_modules; i++) {
    free(modules[i].name);
    free(modules[i].logical_inputs);
  }
  free(modules);
}

void poly_model_free(PolyModel *inst) {
  if (!inst) return;

  for (int i = 0; i < inst->n_residency_roots; i++)
    poly_uop_release(inst->ctx, inst->residency_roots[i]);
  free(inst->residency_roots);
  inst->residency_roots = NULL;
  inst->n_residency_roots = 0;

  build_state_free(inst->build);
  inst->build = NULL;

  /* Free named buffers */
  for (int i = 0; i < inst->n_bufs; i++) {
    free(inst->bufs[i].name);
  }
  free(inst->bufs);
  free(inst->param_indices);
  free(inst->trainable_param_indices);

  /* Free entrypoints */
  if (inst->entry_executables) {
    entry_executables_clear(inst);
    free(inst->entry_executables);
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

  runtime_modules_free(inst->modules, inst->n_modules);
  inst->modules = NULL;
  inst->n_modules = 0;

  /* Free execution caches.
   * Order: exec cache first (holds pointers into prepared steps),
   * then context (arena-frees all UOps). */

  /* Free value-and-grad state */
  vag_free(inst->training.vag, inst->n_params);

  /* Free training state */
  train_free(inst->training.train, inst->n_params);

  /* Free context (arena-frees all UOps) -- only if we own it */
  if (inst->owns_ctx && inst->ctx) poly_ctx_destroy(inst->ctx);

  free(inst);
}

/* Param Enumeration */

int poly_model_param_count(const PolyModel *inst) {
  return inst ? inst->n_params : 0;
}

const char *poly_model_param_name(const PolyModel *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_params) return NULL;
  return inst->bufs[inst->param_indices[i]].name;
}

int poly_model_param_shape(const PolyModel *inst, int i, int64_t *shape_out, int max_dims) {
  if (!inst || i < 0 || i >= inst->n_params) return 0;
  NamedBuf *b = &inst->bufs[inst->param_indices[i]];
  int n = b->ndim < max_dims ? b->ndim : max_dims;
  memcpy(shape_out, b->shape, n * sizeof(int64_t));
  return b->ndim;
}

/* Sync device buffer to host shadow if on a non-host domain.
 * Returns 0 on success or if already on host; -1 on readback failure. */
static int sync_buf_to_host(PolyModel *inst, int bi) {
  if (!inst || bi < 0 || bi >= inst->n_bufs) return -1;
  PolyBuffer *host = NULL;
  if (poly_buffer_ensure_host_current(inst->ctx, inst->bufs[bi].buffer, &host) != 0 || !host ||
      !host->ptr)
    return -1;
  inst->bufs[bi].data = host->ptr;
  return 0;
}

static int append_runtime_named_buffer(
    PolyModel *inst,
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
  if (!inst || !name || !buffer || ndim < 0 || ndim > POLY_IR_MAX_DIMS || (ndim > 0 && !shape))
    return -1;
  if (!valid_binding_name(name) || find_buf_by_name(inst, name) >= 0) return -1;

  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel < 0) return -1;
  if (buffer->op != POLY_OP_BUFFER || buffer->arg.kind != POLY_ARG_PARAM || !buffer->arg.param)
    return -1;

  NamedBuf *next = realloc(inst->bufs, (size_t)(inst->n_bufs + 1) * sizeof(NamedBuf));
  if (!next) return -1;
  inst->bufs = next;

  NamedBuf nb = {0};
  nb.name = strdup(name);
  if (!nb.name) return -1;
  nb.role = role;
  nb.flags = flags;
  PolyUOp *logical = poly_uop_new_logical_buffer_with_slot(
      inst->ctx, buffer->dtype, numel, buffer->arg.param->slot
  );
  if (!logical) return -1;
  nb.buffer = buffer;
  nb.logical_value = logical;
  nb.logical_buffer = logical;
  nb.capture_buffer = buffer;
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

  int idx = inst->n_bufs++;
  inst->bufs[idx] = nb;
  if (out_index) *out_index = idx;
  return 0;
}

static char *optimizer_param_state_name(
    const char *optimizer,
    const char *kind,
    const char *param_name
) {
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
    PolyModel *inst,
    const char *name,
    const int64_t *shape,
    int ndim,
    bool init_scalar,
    float scalar_value,
    PolyUOp **out
) {
  if (out) *out = NULL;
  if (!inst || !name || !shape || ndim <= 0 || ndim > POLY_IR_MAX_DIMS) return -1;
  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel <= 0) return -1;

  int bi = find_buf_by_name(inst, name);
  if (bi >= 0) {
    NamedBuf *b = &inst->bufs[bi];
    if (!b->buffer || b->role != POLY_ROLE_AUX || !poly_dtype_eq(b->buffer->dtype, POLY_FLOAT32) ||
        b->numel != numel)
      return -1;
    b->flags |= POLY_BIND_F_OPTIM;
    if (out) *out = b->buffer;
    return 0;
  }

  PolyUOp *buf = model_new_buffer(inst->ctx, POLY_FLOAT32, numel);
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

static void discard_appended_named_buffers(PolyCtx *ctx, NamedBuf *bufs, int first, int count) {
  if (!ctx || !bufs) return;
  for (int i = first; i < count; i++) {
    poly_buffer_remove(ctx, bufs[i].buffer);
    free(bufs[i].name);
  }
}

void *poly_model_param_data_raw(PolyModel *inst, int i, int64_t *numel_out) {
  if (!inst || i < 0 || i >= inst->n_params) return NULL;
  int bi = inst->param_indices[i];
  if (numel_out) *numel_out = inst->bufs[bi].numel;
  if (sync_buf_to_host(inst, bi) != 0) return NULL;
  if (poly_buffer_mark_host_written(inst->ctx, inst->bufs[bi].buffer) != 0) return NULL;
  return inst->bufs[bi].data;
}

float *poly_model_param_data(PolyModel *inst, int i, int64_t *numel_out) {
  if (poly_model_param_dtype_id(inst, i) != poly_dtype_id_by_name("float32")) return NULL;
  return poly_model_param_data_raw(inst, i, numel_out);
}

int poly_model_param_dtype_id(const PolyModel *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_params) return -1;
  const NamedBuf *b = &inst->bufs[inst->param_indices[i]];
  return b->buffer ? poly_dtype_id_by_name(poly_dtype_name(b->buffer->dtype)) : -1;
}

size_t poly_model_param_nbytes(const PolyModel *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_params) return 0;
  return named_buf_nbytes(&inst->bufs[inst->param_indices[i]]);
}

/* Buffer Enumeration */

int poly_model_buf_count(const PolyModel *inst) {
  return inst ? inst->n_bufs : 0;
}

const char *poly_model_buf_name(const PolyModel *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_bufs) return NULL;
  return inst->bufs[i].name;
}

int poly_model_buf_role(const PolyModel *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_bufs) return -1;
  return inst->bufs[i].role;
}

bool poly_model_buf_trainable(const PolyModel *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_bufs) return false;
  return inst->bufs[i].trainable;
}

bool poly_model_param_trainable(const PolyModel *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_params) return false;
  return inst->bufs[inst->param_indices[i]].trainable;
}

static int build_trainable_param_indices(const PolyModel *inst, int **indices_out, int *count_out) {
  if (!inst || !indices_out || !count_out) return -1;
  *indices_out = NULL;
  *count_out = 0;
  int *indices = inst->n_params > 0 ? malloc((size_t)inst->n_params * sizeof(int)) : NULL;
  if (inst->n_params > 0 && !indices) return -1;
  int n = 0;
  for (int i = 0; i < inst->n_params; i++) {
    int bi = inst->param_indices[i];
    if (!inst->bufs[bi].trainable) continue;
    /* Tinygrad Optimizer dedup: ABI aliases are names, not extra updates. */
    bool duplicate = false;
    for (int j = 0; j < n; j++)
      if (inst->bufs[indices[j]].logical_buffer == inst->bufs[bi].logical_buffer) duplicate = true;
    if (!duplicate) indices[n++] = bi;
  }
  *indices_out = indices;
  *count_out = n;
  return 0;
}

int poly_model_set_buf_trainable(PolyModel *inst, int i, bool trainable) {
  if (!inst || i < 0 || i >= inst->n_bufs) return -1;
  NamedBuf *candidate_bufs = malloc((size_t)inst->n_bufs * sizeof(*candidate_bufs));
  if (!candidate_bufs) return -1;
  memcpy(candidate_bufs, inst->bufs, (size_t)inst->n_bufs * sizeof(*candidate_bufs));

  PolyUOp *shared = candidate_bufs[i].buffer;
  for (int j = 0; j < inst->n_bufs; j++)
    if (candidate_bufs[j].buffer == shared) candidate_bufs[j].trainable = trainable;
  PolyModel candidate = *inst;
  candidate.bufs = candidate_bufs;
  candidate.training.train = NULL;
  candidate.training.vag = NULL;
  int *candidate_indices = NULL;
  int n_candidate_indices = 0;
  if (build_trainable_param_indices(&candidate, &candidate_indices, &n_candidate_indices) != 0) {
    free(candidate_bufs);
    return -1;
  }
  candidate.trainable_param_indices = candidate_indices;
  candidate.n_trainable_params = n_candidate_indices;
  ModelResidencyRoots prepared_roots = {0};
  if (model_prepare_residency_roots(&candidate, &prepared_roots) != 0) {
    free(candidate_indices);
    free(candidate_bufs);
    return -1;
  }

  NamedBuf *old_bufs = inst->bufs;
  int *old_indices = inst->trainable_param_indices;
  VagState *old_vag = inst->training.vag;
  TrainState *old_train = inst->training.train;
  inst->bufs = candidate_bufs;
  inst->trainable_param_indices = candidate_indices;
  inst->n_trainable_params = n_candidate_indices;
  inst->training.vag = NULL;
  inst->training.train = NULL;
  model_publish_residency_roots(inst, &prepared_roots);
  free(old_bufs);
  free(old_indices);
  vag_free(old_vag, inst->n_params);
  train_free(old_train, inst->n_params);
  return 0;
}

int poly_model_set_param_trainable(PolyModel *inst, int i, bool trainable) {
  if (!inst || i < 0 || i >= inst->n_params) return -1;
  return poly_model_set_buf_trainable(inst, inst->param_indices[i], trainable);
}

int poly_model_buf_shape(const PolyModel *inst, int i, int64_t *shape_out, int max_dims) {
  if (!inst || i < 0 || i >= inst->n_bufs) return 0;
  int n = inst->bufs[i].ndim < max_dims ? inst->bufs[i].ndim : max_dims;
  memcpy(shape_out, inst->bufs[i].shape, n * sizeof(int64_t));
  return inst->bufs[i].ndim;
}

void *poly_model_buf_data_raw(PolyModel *inst, int i, int64_t *numel_out) {
  if (!inst || i < 0 || i >= inst->n_bufs) return NULL;
  if (numel_out) *numel_out = inst->bufs[i].numel;
  if (sync_buf_to_host(inst, i) != 0) return NULL;
  if (poly_buffer_mark_host_written(inst->ctx, inst->bufs[i].buffer) != 0) return NULL;
  return inst->bufs[i].data;
}

float *poly_model_buf_data(PolyModel *inst, int i, int64_t *numel_out) {
  if (poly_model_buf_dtype_id(inst, i) != poly_dtype_id_by_name("float32")) return NULL;
  return poly_model_buf_data_raw(inst, i, numel_out);
}

int poly_model_buf_dtype_id(const PolyModel *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_bufs) return -1;
  const NamedBuf *b = &inst->bufs[i];
  return b->buffer ? poly_dtype_id_by_name(poly_dtype_name(b->buffer->dtype)) : -1;
}

size_t poly_model_buf_nbytes(const PolyModel *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_bufs) return 0;
  return named_buf_nbytes(&inst->bufs[i]);
}

/* Read / Write */

int poly_model_read_buf(PolyModel *inst, int i, void *host_dst, size_t dst_len) {
  if (!inst || i < 0 || i >= inst->n_bufs) return -1;
  return poly_buffer_read(inst->ctx, inst->bufs[i].buffer, host_dst, dst_len);
}

int poly_model_write_buf(PolyModel *inst, int i, const void *host_src, size_t src_len) {
  if (!inst || i < 0 || i >= inst->n_bufs) return -1;
  int rc = poly_buffer_write(inst->ctx, inst->bufs[i].buffer, host_src, src_len);
  if (rc == 0) sync_buf_to_host(inst, i);
  return rc;
}

int poly_model_read_buf_named(PolyModel *inst, const char *name, void *host_dst, size_t dst_len) {
  int idx = find_buf_by_name(inst, name);
  if (idx < 0) return -1;
  return poly_model_read_buf(inst, idx, host_dst, dst_len);
}

int poly_model_write_buf_named(
    PolyModel *inst,
    const char *name,
    const void *host_src,
    size_t src_len
) {
  int idx = find_buf_by_name(inst, name);
  if (idx < 0) return -1;
  return poly_model_write_buf(inst, idx, host_src, src_len);
}

int poly_model_readback_buf(PolyModel *inst, int i, void *host_dst, size_t dst_len) {
  return poly_model_read_buf(inst, i, host_dst, dst_len);
}

int poly_model_upload_buf(PolyModel *inst, int i, const void *host_src, size_t src_len) {
  return poly_model_write_buf(inst, i, host_src, src_len);
}

int poly_model_readback_param(PolyModel *inst, int i, void *host_dst, size_t dst_len) {
  if (!inst || i < 0 || i >= inst->n_params) return -1;
  return poly_model_read_buf(inst, inst->param_indices[i], host_dst, dst_len);
}

int poly_model_upload_param(PolyModel *inst, int i, const void *host_src, size_t src_len) {
  if (!inst || i < 0 || i >= inst->n_params) return -1;
  return poly_model_write_buf(inst, inst->param_indices[i], host_src, src_len);
}

/* Weight I/O */

uint8_t *poly_model_export_weights_ex(PolyModel *inst, int *out_len, uint32_t flags) {
  if (out_len) *out_len = 0;
  if (!inst || inst->stage != POLY_MODEL_BUILT) {
    return NULL;
  }

  int n_export = 0;
  for (int i = 0; i < inst->n_bufs; i++)
    if (should_export_weight_buf_flags(&inst->bufs[i], flags)) n_export++;
  if (n_export == 0) return NULL;

  PolySafetensorEntry *entries = calloc((size_t)n_export, sizeof(PolySafetensorEntry));
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
    PolySafetensorDType dtype;
    if (!b->buffer || !model_safetensor_dtype(b->buffer->dtype, &dtype)) {
      free(entries);
      return NULL;
    }
    entries[ei].data = b->data;
    entries[ei].shape = b->shape;
    entries[ei].ndim = b->ndim;
    entries[ei].dtype = dtype;
    ei++;
  }

  uint8_t *bytes = poly_safetensors_encode(entries, n_export, NULL, out_len);
  free(entries);
  return bytes;
}

uint8_t *poly_model_export_weights(PolyModel *inst, int *out_len) {
  return poly_model_export_weights_ex(inst, out_len, POLY_EXPORT_WEIGHTS_DEFAULT);
}

int poly_model_import_weights(PolyModel *inst, const uint8_t *data, int len) {
  if (!inst || inst->stage != POLY_MODEL_BUILT) return -1;

  int n_views = 0;
  char *metadata = NULL;
  PolySafetensorViewEx *views = poly_safetensors_decode_ex(data, len, &n_views, &metadata);
  if (!views) return -1;
  if (model_validate_checkpoint_views(inst, views, n_views, false) != 0) {
    model_free_safetensor_views(views, n_views);
    free(metadata);
    return -1;
  }

  typedef struct {
    int bi;
    size_t nbytes;
    uint8_t *before;
  } ImportUndo;
  ImportUndo *undo = n_views > 0 ? calloc((size_t)n_views, sizeof(*undo)) : NULL;
  if (n_views > 0 && !undo) {
    model_free_safetensor_views(views, n_views);
    free(metadata);
    return -1;
  }
  int n_undo = 0;

  /* Snapshot each distinct target residency before the first write. Alias
   * names intentionally apply in archive order, matching tinygrad's
   * load_state_dict last-writer behavior, but any failure restores the whole
   * pre-import state. */
  for (int i = 0; i < n_views; i++) {
    int bi = find_buf_by_name(inst, views[i].name);
    if (bi < 0) continue;
    bool seen = false;
    for (int j = 0; j < n_undo; j++)
      if (inst->bufs[undo[j].bi].buffer == inst->bufs[bi].buffer) {
        seen = true;
        break;
      }
    if (seen) continue;
    size_t nbytes = named_buf_nbytes(&inst->bufs[bi]);
    uint8_t *before = nbytes > 0 ? malloc(nbytes) : NULL;
    if ((nbytes > 0 &&
         (!before || poly_buffer_read(inst->ctx, inst->bufs[bi].buffer, before, nbytes) != 0))) {
      free(before);
      for (int j = 0; j < n_undo; j++)
        free(undo[j].before);
      free(undo);
      model_free_safetensor_views(views, n_views);
      free(metadata);
      return -1;
    }
    undo[n_undo++] = (ImportUndo){bi, nbytes, before};
  }

  int rc = 0;
  for (int i = 0; i < n_views; i++) {
    int bi = find_buf_by_name(inst, views[i].name);
    if (bi < 0) {
      fprintf(stderr, "poly_model_import_weights: unknown tensor '%s'\n", views[i].name);
      continue;
    }
    size_t nbytes = named_buf_nbytes(&inst->bufs[bi]);
    if (nbytes > 0 &&
        poly_buffer_write(inst->ctx, inst->bufs[bi].buffer, views[i].raw_data, nbytes) != 0) {
      fprintf(stderr, "poly_model_import_weights: write failed for '%s'\n", views[i].name);
      rc = -1;
      break;
    }
  }

  if (rc != 0) {
    for (int i = 0; i < n_undo; i++)
      if (undo[i].nbytes > 0)
        (void)poly_buffer_write(
            inst->ctx, inst->bufs[undo[i].bi].buffer, undo[i].before, undo[i].nbytes
        );
  }
  for (int i = 0; i < n_undo; i++) {
    (void)sync_buf_to_host(inst, undo[i].bi);
    free(undo[i].before);
  }
  free(undo);
  model_free_safetensor_views(views, n_views);
  free(metadata);
  return rc;
}

/* IR Export */

uint8_t *poly_model_export_ir(PolyModel *inst, int *out_len) {
  if (!out_len) return NULL;
  if (!inst || inst->stage != POLY_MODEL_BUILT || !inst->has_portable_source) {
    *out_len = 0;
    return NULL;
  }

  /* Build PolyIrSpec from model state */
  PolyIrBufEntry *bufs = malloc(inst->n_bufs * sizeof(PolyIrBufEntry));
  for (int i = 0; i < inst->n_bufs; i++) {
    bufs[i].name = inst->bufs[i].name;
    bufs[i].role = inst->bufs[i].role;
    bufs[i].buffer = inst->bufs[i].logical_value;
    bufs[i].ndim = inst->bufs[i].ndim;
    bufs[i].trainable = inst->bufs[i].trainable;
    bufs[i].trainable_set = true;
    memcpy(bufs[i].shape, inst->bufs[i].shape, inst->bufs[i].ndim * sizeof(int64_t));
  }

  PolyIrEntrypoint *eps = malloc(inst->n_entrypoints * sizeof(PolyIrEntrypoint));
  for (int i = 0; i < inst->n_entrypoints; i++) {
    eps[i].name = inst->entrypoints[i].name;
    eps[i].sink = inst->entrypoints[i].logical_sink;
    eps[i].inputs = (const char **)inst->entrypoints[i].inputs;
    eps[i].n_inputs = inst->entrypoints[i].n_inputs;
    eps[i].outputs = (const char **)inst->entrypoints[i].outputs;
    eps[i].n_outputs = inst->entrypoints[i].n_outputs;
    eps[i].objective = inst->entrypoints[i].objective;
    eps[i].flags = inst->entrypoints[i].flags;
  }

  PolyIrModule *modules =
      inst->n_modules > 0 ? calloc((size_t)inst->n_modules, sizeof(*modules)) : NULL;
  if (inst->n_modules > 0 && !modules) {
    free(bufs);
    free(eps);
    *out_len = 0;
    return NULL;
  }
  for (int i = 0; i < inst->n_modules; i++) {
    modules[i].name = inst->modules[i].name;
    modules[i].inputs = inst->modules[i].logical_inputs;
    modules[i].n_inputs = inst->modules[i].n_inputs;
    modules[i].output = inst->modules[i].logical_output;
  }

  PolyIrSpec spec = {
      .ctx = inst->ctx,
      .bufs = bufs,
      .n_bufs = inst->n_bufs,
      .entrypoints = eps,
      .n_entrypoints = inst->n_entrypoints,
      .modules = modules,
      .n_modules = inst->n_modules,
  };
  uint8_t *bytes = poly_ir_export(&spec, out_len);
  free(bufs);
  free(eps);
  free(modules);
  return bytes;
}

static int model_publish_cached_executable(
    PolyModel *inst,
    PolyLinearEntry *slot,
    PolyLinearEntry *built
) {
  if (!inst || !slot || !built || !built->linear) return -1;

  PolyModel candidate = *inst;
  PolyLinearEntry *candidate_entries = NULL;
  VagState candidate_vag = {0};
  TrainState candidate_train = {0};
  bool found = false;
  if (inst->entry_executables) {
    for (int i = 0; i < inst->n_entrypoints; i++) {
      if (slot != &inst->entry_executables[i]) continue;
      candidate_entries = malloc((size_t)inst->n_entrypoints * sizeof(*candidate_entries));
      if (!candidate_entries) return -1;
      memcpy(
          candidate_entries, inst->entry_executables,
          (size_t)inst->n_entrypoints * sizeof(*candidate_entries)
      );
      candidate_entries[i] = *built;
      candidate.entry_executables = candidate_entries;
      found = true;
      break;
    }
  }
  if (!found && inst->training.vag && slot == &inst->training.vag->executable) {
    candidate_vag = *inst->training.vag;
    candidate_vag.executable = *built;
    candidate.training.vag = &candidate_vag;
    found = true;
  }
  if (!found && inst->training.train && slot == &inst->training.train->executable) {
    candidate_train = *inst->training.train;
    candidate_train.executable = *built;
    candidate.training.train = &candidate_train;
    found = true;
  }
  if (!found) {
    free(candidate_entries);
    return -1;
  }

  ModelResidencyRoots prepared_roots = {0};
  int rc = model_prepare_residency_roots(&candidate, &prepared_roots);
  free(candidate_entries);
  if (rc != 0) return -1;
  *slot = *built;
  *built = (PolyLinearEntry){0};
  model_publish_residency_roots(inst, &prepared_roots);
  return 0;
}

static PolyLinearEntry *model_ensure_entry_executable(PolyModel *inst, int entrypoint_index) {
  if (!inst || !inst->ctx || !inst->entry_executables || entrypoint_index < 0 ||
      entrypoint_index >= inst->n_entrypoints)
    return NULL;
  PolyLinearEntry *entry = &inst->entry_executables[entrypoint_index];
  if (entry->linear) return entry;

  PolyUOp *root = inst->entrypoints[entrypoint_index].sink;
  PolyVarBinding *var_bindings = NULL;
  int n_var_bindings = 0;
  PolyUOp *linear = root && root->op == POLY_OP_LINEAR
                        ? root
                        : poly_linear_effect_sink(inst->ctx, root, &var_bindings, &n_var_bindings);
  PolyUOp *compiled = linear ? poly_compile_linear(inst->ctx, linear, -1) : NULL;
  if (!compiled) {
    free(var_bindings);
    return NULL;
  }
  PolyLinearEntry built = {compiled, var_bindings, n_var_bindings};
  if (model_publish_cached_executable(inst, entry, &built) != 0) {
    cached_executable_clear(&built);
    return NULL;
  }
  return entry;
}

uint8_t *poly_model_export_program(PolyModel *inst, int *out_len) {
  if (!out_len) return NULL;
  *out_len = 0;
  if (!inst || inst->stage != POLY_MODEL_BUILT || inst->n_entrypoints <= 0 || !inst->entrypoints ||
      !inst->entry_executables)
    return NULL;

  PolyIrBufEntry *bufs = inst->n_bufs > 0 ? calloc((size_t)inst->n_bufs, sizeof(*bufs)) : NULL;
  PolyIrEntrypoint *eps = calloc((size_t)inst->n_entrypoints, sizeof(*eps));
  if ((inst->n_bufs > 0 && !bufs) || !eps) {
    free(bufs);
    free(eps);
    return NULL;
  }

  for (int i = 0; i < inst->n_bufs; i++) {
    bufs[i].name = inst->bufs[i].name;
    bufs[i].role = inst->bufs[i].role;
    bufs[i].buffer = inst->bufs[i].buffer;
    bufs[i].ndim = inst->bufs[i].ndim;
    bufs[i].trainable = inst->bufs[i].trainable;
    bufs[i].trainable_set = true;
    memcpy(bufs[i].shape, inst->bufs[i].shape, (size_t)inst->bufs[i].ndim * sizeof(*bufs[i].shape));
  }
  for (int i = 0; i < inst->n_entrypoints; i++) {
    PolyLinearEntry *compiled = model_ensure_entry_executable(inst, i);
    if (!compiled || !compiled->linear || compiled->linear->op != POLY_OP_LINEAR) {
      free(bufs);
      free(eps);
      return NULL;
    }
    eps[i] = (PolyIrEntrypoint){
        .name = inst->entrypoints[i].name,
        .sink = compiled->linear,
        .inputs = (const char **)inst->entrypoints[i].inputs,
        .n_inputs = inst->entrypoints[i].n_inputs,
        .outputs = (const char **)inst->entrypoints[i].outputs,
        .n_outputs = inst->entrypoints[i].n_outputs,
        .objective = inst->entrypoints[i].objective,
        .flags = inst->entrypoints[i].flags,
    };
  }

  PolyIrSpec spec = {
      .ctx = inst->ctx,
      .bufs = bufs,
      .n_bufs = inst->n_bufs,
      .entrypoints = eps,
      .n_entrypoints = inst->n_entrypoints,
  };
  uint8_t *bytes = poly_program_graph_export(&spec, out_len);
  free(bufs);
  free(eps);
  return bytes;
}

PolyModel *poly_model_from_program(
    const uint8_t *program_data,
    int program_len,
    const uint8_t *weights_data,
    int weights_len
) {
  PolyIrSpec spec = {0};
  if (poly_program_graph_import(program_data, program_len, &spec) != 0) {
    fprintf(stderr, "poly_model_from_program: program import failed\n");
    return NULL;
  }
  PolyUOp **physical_buffers =
      spec.n_bufs > 0 ? malloc((size_t)spec.n_bufs * sizeof(*physical_buffers)) : NULL;
  PolyUOp **physical_sinks =
      spec.n_entrypoints > 0 ? malloc((size_t)spec.n_entrypoints * sizeof(*physical_sinks)) : NULL;
  if ((spec.n_bufs > 0 && !physical_buffers) || (spec.n_entrypoints > 0 && !physical_sinks)) {
    free(physical_buffers);
    free(physical_sinks);
    poly_ir_spec_free(&spec);
    return NULL;
  }
  for (int i = 0; i < spec.n_bufs; i++)
    physical_buffers[i] = spec.bufs[i].buffer;
  for (int i = 0; i < spec.n_entrypoints; i++)
    physical_sinks[i] = spec.entrypoints[i].sink;
  /* Bound import is Polygrad's approved export boundary. As current Tinygrad
   * PROGRAM replay does, retain the exact LINEAR buffer operands
   * (engine/realize.py:263-319); portable placement never consumes them. */
  PolyModel *inst = model_from_spec(&spec, physical_buffers, physical_sinks, true, true);
  free(physical_buffers);
  free(physical_sinks);
  if (!inst) return NULL;
  inst->has_portable_source = false;

  bool has_state = false;
  for (int i = 0; i < inst->n_bufs; i++)
    if (model_role_is_state(inst->bufs[i].role)) {
      has_state = true;
      break;
    }
  if (has_state && (!weights_data || weights_len <= 0)) {
    fprintf(stderr, "poly_model_from_program: bound state requires weights\n");
    poly_model_free(inst);
    return NULL;
  }
  if (weights_data && weights_len > 0) {
    if (model_validate_checkpoint(inst, weights_data, weights_len, true) != 0 ||
        poly_model_import_weights(inst, weights_data, weights_len) != 0) {
      fprintf(stderr, "poly_model_from_program: weight import failed\n");
      poly_model_free(inst);
      return NULL;
    }
  }

  for (int i = 0; i < inst->n_entrypoints; i++)
    if (!model_ensure_entry_executable(inst, i)) {
      fprintf(stderr, "poly_model_from_program: executable reconstruction failed\n");
      poly_model_free(inst);
      return NULL;
    }
  return inst;
}

/* Device configuration */

static PolyUOp *model_binding_source(const NamedBuf *binding) {
  if (!binding || !binding->logical_value) return NULL;
  const PolyUOp *identity = poly_uop_get_buffer_identity(binding->logical_value);
  return identity ? (PolyUOp *)identity : binding->logical_value;
}

static PolyUOp *model_shaped_storage(PolyCtx *ctx, const NamedBuf *binding, PolyUOp *buffer) {
  if (!ctx || !binding || !buffer) return NULL;
  if (binding->ndim == 1 && binding->shape[0] == binding->numel) return buffer;
  return poly_reshape(ctx, buffer, (int64_t *)binding->shape, binding->ndim);
}

/* Activation is the only boundary that binds a named logical value to
 * persistent storage.  This transient result is then consumed by the ordinary
 * explicit placer; the immutable entrypoint and named value roots are never
 * rewritten or retained as a second logical program. */
static int model_bind_named_values(
    PolyModel *inst,
    PolyUOp **roots,
    int n_roots,
    PolyUOp **out_roots
) {
  if (!inst || !inst->ctx || n_roots < 0 || (n_roots > 0 && (!roots || !out_roots))) return -1;
  PolyUOp **from = calloc((size_t)inst->n_bufs, sizeof(*from));
  PolyUOp **to = calloc((size_t)inst->n_bufs, sizeof(*to));
  if (inst->n_bufs > 0 && (!from || !to)) {
    free(to);
    free(from);
    return -1;
  }
  int n_subs = 0;
  for (int i = 0; i < inst->n_bufs; i++) {
    NamedBuf *binding = &inst->bufs[i];
    if (binding->role == POLY_ROLE_OUTPUT) continue;
    PolyUOp *source = model_binding_source(binding);
    if (!source || source == binding->logical_buffer) continue;
    PolyUOp *storage = model_shaped_storage(inst->ctx, binding, binding->logical_buffer);
    if (!storage) goto fail;
    int duplicate = -1;
    for (int j = 0; j < n_subs; j++)
      if (from[j] == source) {
        duplicate = j;
        break;
      }
    if (duplicate >= 0) {
      if (to[duplicate] != storage) goto fail;
      continue;
    }
    from[n_subs] = source;
    to[n_subs++] = storage;
  }
  int rc = 0;
  if (n_subs == 0) {
    if (n_roots > 0) memcpy(out_roots, roots, (size_t)n_roots * sizeof(*out_roots));
  } else {
    rc = poly_uop_substitute_many(inst->ctx, roots, n_roots, from, to, n_subs, out_roots);
  }
  free(to);
  free(from);
  return rc;

fail:
  free(to);
  free(from);
  return -1;
}

static PolyUOp *model_binding_on_device(PolyCtx *ctx, PolyUOp *logical, PolyDevice device) {
  if (!ctx || !model_is_portable_buffer(logical) || !poly_device_can_execute(device)) return NULL;
  /* Polygrad's portable binding owns the slot. Current UOp.new_buffer creates
   * the physical one-source BUFFER (uop/ops.py:811-817). */
  PolyUOp *device_uop = poly_device_uop(ctx, device);
  int64_t size = logical->arg.kind == POLY_ARG_INT ? logical->arg.i : -1;
  int64_t slot = logical->src[0]->arg.kind == POLY_ARG_INT ? logical->src[0]->arg.i : -1;
  PolyUOp *physical =
      device_uop ? poly_uop_new_buffer(ctx, device_uop, size, logical->dtype, slot) : NULL;
  if (!physical) return NULL;
  return (logical->tag != 0 || logical->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(
                   ctx, POLY_OP_BUFFER, physical->dtype, physical->src, physical->n_src,
                   physical->arg, logical->tag, logical->tag_arg
               )
             : physical;
}

static void model_discard_candidate_residencies(
    PolyModel *inst,
    PolyUOp **targets,
    const uint8_t *existed,
    int n
) {
  if (!inst || !targets || !existed) return;
  for (int i = 0; i < n; i++) {
    if (existed[i] || !targets[i]) continue;
    bool duplicate = false;
    for (int j = 0; j < i; j++)
      if (targets[j] == targets[i]) duplicate = true;
    if (duplicate) continue;
    PolyBuffer *candidate = poly_buffer_get(inst->ctx, targets[i]);
    if (!candidate) continue;
    poly_map_remove(inst->ctx->buffers, poly_ptr_hash(targets[i]), targets[i], poly_ptr_eq);
    poly_buffer_free_chain(inst->ctx, candidate);
  }
}

static bool model_backend_available(PolyDevice device) {
  if (!poly_device_can_execute(device) || !poly_backend_get(device)) return false;
#ifdef POLY_HAS_CUDA
  if (device == POLY_DEVICE_CUDA && !poly_cuda_available()) return false;
#endif
#ifdef POLY_HAS_HIP
  if (device == POLY_DEVICE_HIP && !poly_hip_available()) return false;
#endif
  return true;
}

static int model_publish_placement(
    PolyModel *inst,
    PolyUOp **target_bindings,
    PolyUOp **placed_roots,
    bool set_preferred,
    PolyDevice preferred
) {
  if (!inst || !target_bindings || !placed_roots) return -1;
  size_t nb = (size_t)(unsigned)inst->n_bufs;
  size_t ne = (size_t)(unsigned)inst->n_entrypoints;
  void **target_data = calloc(nb, sizeof(*target_data));
  uint8_t *target_existed = calloc(nb, sizeof(*target_existed));
  NamedBuf *candidate_bufs = malloc(nb * sizeof(*candidate_bufs));
  RuntimeEntrypoint *candidate_entrypoints = malloc(ne * sizeof(*candidate_entrypoints));
  ModelResidencyRoots prepared_roots = {0};
  int rc = -1;
  if (!target_data || !target_existed || !candidate_bufs || !candidate_entrypoints) goto cleanup;

  for (int i = 0; i < inst->n_bufs; i++) {
    PolyUOp *device_uop = poly_uop_device_uop_cached(inst->ctx, target_bindings[i], NULL);
    PolyDevice backend = poly_device_from_device_uop(device_uop);
    if (!device_uop || device_uop->arg.kind != POLY_ARG_STRING || !model_backend_available(backend))
      goto cleanup;
    target_existed[i] = poly_buffer_get(inst->ctx, target_bindings[i]) != NULL;
  }

  memcpy(candidate_bufs, inst->bufs, nb * sizeof(*candidate_bufs));
  memcpy(candidate_entrypoints, inst->entrypoints, ne * sizeof(*candidate_entrypoints));
  for (int i = 0; i < inst->n_bufs; i++)
    candidate_bufs[i].buffer = target_bindings[i];
  for (int i = 0; i < inst->n_entrypoints; i++)
    candidate_entrypoints[i].sink = placed_roots[i];
  PolyModel candidate = *inst;
  candidate.bufs = candidate_bufs;
  candidate.entrypoints = candidate_entrypoints;
  candidate.entry_executables = NULL;
  candidate.training.vag = NULL;
  candidate.training.train = NULL;
  if (model_prepare_residency_roots(&candidate, &prepared_roots) != 0) goto cleanup;

  /* Prepare every target residency before publishing a new binding/root set.
   * Aliases share one target UOp and are migrated exactly once. */
  for (int i = 0; i < inst->n_bufs; i++) {
    int alias = -1;
    for (int j = 0; j < i; j++)
      if (target_bindings[j] == target_bindings[i]) {
        alias = j;
        break;
      }
    if (alias >= 0) {
      target_data[i] = target_data[alias];
      continue;
    }

    PolyUOp *old = inst->bufs[i].buffer;
    PolyUOp *target = target_bindings[i];
    PolyUOp *device_uop = poly_uop_device_uop_cached(inst->ctx, target, NULL);
    PolyDevice backend = poly_device_from_device_uop(device_uop);
    size_t nbytes = named_buf_nbytes(&inst->bufs[i]);
    if (nbytes > 0 && target != old) {
      PolyBuffer *host = NULL;
      if (poly_buffer_ensure_host_current(inst->ctx, old, &host) != 0 || !host || !host->ptr ||
          host->nbytes < nbytes || poly_buffer_write(inst->ctx, target, host->ptr, nbytes) != 0)
        goto cleanup_residencies;
    }
    if (nbytes > 0 && inst->bufs[i].role != POLY_ROLE_OUTPUT &&
        poly_buffer_ensure_device_current(inst->ctx, target, backend) != 0)
      goto cleanup_residencies;
    if (nbytes > 0) {
      PolyBuffer *host = NULL;
      if (poly_buffer_ensure_host_current(inst->ctx, target, &host) != 0 || !host || !host->ptr)
        goto cleanup_residencies;
      target_data[i] = host->ptr;
    }
  }

  for (int i = 0; i < inst->n_bufs; i++) {
    inst->bufs[i].buffer = target_bindings[i];
    inst->bufs[i].data = target_data[i];
  }
  for (int i = 0; i < inst->n_entrypoints; i++)
    inst->entrypoints[i].sink = placed_roots[i];
  if (set_preferred) poly_ctx_set_preferred_device(inst->ctx, preferred);

  entry_executables_clear(inst);
  vag_free(inst->training.vag, inst->n_params);
  inst->training.vag = NULL;
  train_free(inst->training.train, inst->n_params);
  inst->training.train = NULL;
  model_publish_residency_roots(inst, &prepared_roots);
  rc = 0;
  goto cleanup;

cleanup_residencies:
  model_discard_candidate_residencies(inst, target_bindings, target_existed, inst->n_bufs);
cleanup:
  model_discard_prepared_residency_roots(inst->ctx, &prepared_roots);
  free(candidate_entrypoints);
  free(candidate_bufs);
  free(target_existed);
  free(target_data);
  return rc;
}

static int model_place_uniform_device(PolyModel *inst, PolyDevice device) {
  if (!inst || !inst->ctx || inst->n_bufs <= 0 || inst->n_entrypoints <= 0 ||
      !poly_device_can_execute(device))
    return -1;

  size_t nb = (size_t)(unsigned)inst->n_bufs;
  size_t ne = (size_t)(unsigned)inst->n_entrypoints;
  PolyUOp **logical_roots = calloc(ne, sizeof(*logical_roots));
  PolyUOp **bound_roots = calloc(ne, sizeof(*bound_roots));
  PolyUOp **placed_roots = calloc(ne, sizeof(*placed_roots));
  PolyUOp **logical_bindings = calloc(nb, sizeof(*logical_bindings));
  PolyUOp **target_bindings = calloc(nb, sizeof(*target_bindings));
  int rc = -1;
  if (!logical_roots || !bound_roots || !placed_roots || !logical_bindings || !target_bindings)
    goto cleanup;

  for (int i = 0; i < inst->n_entrypoints; i++)
    logical_roots[i] = inst->entrypoints[i].logical_sink;
  if (model_bind_named_values(inst, logical_roots, inst->n_entrypoints, bound_roots) != 0)
    goto cleanup;
  for (int i = 0; i < inst->n_bufs; i++) {
    logical_bindings[i] = inst->bufs[i].logical_buffer;
    target_bindings[i] = model_binding_on_device(inst->ctx, logical_bindings[i], device);
    if (!target_bindings[i]) goto cleanup;
  }

  if (poly_place_roots(
          inst->ctx, bound_roots, inst->n_entrypoints, logical_bindings, target_bindings,
          inst->n_bufs, placed_roots
      ) != 0)
    goto cleanup;

  rc = model_publish_placement(inst, target_bindings, placed_roots, true, device);
cleanup:
  free(target_bindings);
  free(logical_bindings);
  free(placed_roots);
  free(bound_roots);
  free(logical_roots);
  return rc;
}

int poly_model_set_device_map(
    PolyModel *inst,
    const PolyModelDeviceMapEntry *entries,
    int n_entries
) {
  if (!inst || inst->stage != POLY_MODEL_BUILT || !inst->has_portable_source || !inst->ctx ||
      !entries || n_entries <= 0 || n_entries != inst->n_modules || !inst->modules)
    return -1;

  size_t nb = (size_t)(unsigned)inst->n_bufs;
  size_t ne = (size_t)(unsigned)inst->n_entrypoints;
  size_t nm = (size_t)(unsigned)inst->n_modules;
  int n_place_nodes = inst->n_entrypoints;
  for (int i = 0; i < inst->n_modules; i++)
    n_place_nodes += 1 + inst->modules[i].n_inputs;
  PolyUOp **placement_inputs = calloc((size_t)n_place_nodes, sizeof(*placement_inputs));
  PolyUOp **placement_values = calloc((size_t)n_place_nodes, sizeof(*placement_values));
  PolyUOp **logical_roots = calloc(ne, sizeof(*logical_roots));
  PolyUOp **placed_roots = calloc(ne, sizeof(*placed_roots));
  PolyUOp **logical_bindings = calloc(nb, sizeof(*logical_bindings));
  PolyUOp **target_bindings = calloc(nb, sizeof(*target_bindings));
  PolyPlaceModule *modules = calloc(nm, sizeof(*modules));
  uint8_t *entry_used = calloc(nm, sizeof(*entry_used));
  int rc = -1;
  if (!placement_inputs || !placement_values || !logical_roots || !placed_roots ||
      !logical_bindings || !target_bindings || !modules || !entry_used)
    goto cleanup;

  int place_off = 0;
  for (int i = 0; i < inst->n_entrypoints; i++)
    placement_inputs[place_off++] = inst->entrypoints[i].logical_sink;
  for (int i = 0; i < inst->n_modules; i++) {
    placement_inputs[place_off++] = inst->modules[i].logical_output;
    for (int j = 0; j < inst->modules[i].n_inputs; j++)
      placement_inputs[place_off++] = inst->modules[i].logical_inputs[j];
  }
  if (model_bind_named_values(inst, placement_inputs, n_place_nodes, placement_values) != 0)
    goto cleanup;
  place_off = 0;
  for (int i = 0; i < inst->n_entrypoints; i++)
    logical_roots[i] = placement_values[place_off++];
  for (int i = 0; i < inst->n_bufs; i++)
    logical_bindings[i] = inst->bufs[i].logical_buffer;

  for (int i = 0; i < inst->n_modules; i++) {
    int found = -1;
    for (int j = 0; j < n_entries; j++) {
      if (!entries[j].module || !entries[j].device ||
          strcmp(entries[j].module, inst->modules[i].name) != 0)
        continue;
      if (found >= 0 || entry_used[j]) goto cleanup;
      found = j;
    }
    if (found < 0) goto cleanup;
    entry_used[found] = 1;
    PolyUOp *module_output = placement_values[place_off++];
    PolyUOp **module_inputs = &placement_values[place_off];
    place_off += inst->modules[i].n_inputs;
    modules[i] = (PolyPlaceModule){
        .name = inst->modules[i].name,
        .output = module_output,
        .inputs = module_inputs,
        .n_inputs = inst->modules[i].n_inputs,
        .device = poly_device_uop_from_name(inst->ctx, entries[found].device),
    };
    if (!modules[i].device) goto cleanup;
  }
  for (int i = 0; i < n_entries; i++)
    if (!entry_used[i]) goto cleanup;

  if (poly_place_module_map(
          inst->ctx, logical_roots, inst->n_entrypoints, logical_bindings, inst->n_bufs, modules,
          inst->n_modules, target_bindings, placed_roots
      ) != 0)
    goto cleanup;

  rc = model_publish_placement(inst, target_bindings, placed_roots, false, POLY_DEVICE_AUTO);

cleanup:
  free(placement_values);
  free(placement_inputs);
  free(entry_used);
  free(modules);
  free(target_bindings);
  free(logical_bindings);
  free(placed_roots);
  free(logical_roots);
  return rc;
}

int poly_model_set_device_map_arrays(
    PolyModel *inst,
    const char **modules,
    const char **devices,
    int n_entries
) {
  if (!inst || !modules || !devices || n_entries <= 0) return -1;
  PolyModelDeviceMapEntry *entries = calloc((size_t)n_entries, sizeof(PolyModelDeviceMapEntry));
  if (!entries) return -1;
  for (int i = 0; i < n_entries; i++)
    entries[i] = (PolyModelDeviceMapEntry){.module = modules[i], .device = devices[i]};
  int rc = poly_model_set_device_map(inst, entries, n_entries);
  free(entries);
  return rc;
}

int poly_model_set_device(PolyModel *inst, PolyDevice device) {
  if (!inst || inst->stage != POLY_MODEL_BUILT || !inst->has_portable_source) return -1;

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
    fprintf(stderr, "poly_model_set_device: unsupported device %d\n", resolved);
    return -1;
  }

#ifdef POLY_HAS_CUDA
  if (resolved == POLY_DEVICE_CUDA && !poly_cuda_available()) {
    fprintf(stderr, "poly_model_set_device: CUDA not available\n");
    return -1;
  }
#endif
#ifdef POLY_HAS_HIP
  if (resolved == POLY_DEVICE_HIP && !poly_hip_available()) {
    fprintf(stderr, "poly_model_set_device: HIP not available\n");
    return -1;
  }
#endif

  if (model_place_uniform_device(inst, resolved) != 0) {
    fprintf(stderr, "poly_model_set_device: placement failed for device %d\n", resolved);
    return -1;
  }
  return 0;
}

/* Generic entrypoint execution */

static bool entrypoint_accepts_input(
    const PolyModel *inst,
    const RuntimeEntrypoint *entry,
    const char *name
) {
  if (!inst || !entry || !name) return false;
  for (int i = 0; i < entry->n_inputs; i++)
    if (entry->inputs[i] && strcmp(entry->inputs[i], name) == 0) return true;
  if (entry->n_inputs > 0) return false;

  /* Legacy ctx-built models sometimes carry no per-entrypoint input list.
   * Keep the boundary strict by falling back only to binding roles, never to
   * arbitrary name acceptance. */
  int bi = find_buf_by_name(inst, name);
  return bi >= 0 &&
         (inst->bufs[bi].role == POLY_ROLE_INPUT || inst->bufs[bi].role == POLY_ROLE_TARGET);
}

static int prepare_model_io(
    PolyModel *inst,
    const RuntimeEntrypoint *entry,
    PolyIOBinding *io,
    int n_io
) {
  if (!inst || !inst->ctx || !entry || n_io < 0 || (n_io > 0 && !io)) return -1;

  /* Pinned TinyJit records an exact ordered input signature and rejects a
   * replay whose names or input graph/shape differs (engine/jit.py:228-246,
   * 303-309).  Validate the complete Model signature before any write so
   * omitted or duplicate rows cannot reuse stale bytes. */
  if (entry->n_inputs > 0) {
    for (int required = 0; required < entry->n_inputs; required++) {
      int seen = 0;
      for (int i = 0; i < n_io; i++)
        if (io[i].name && strcmp(io[i].name, entry->inputs[required]) == 0 && io[i].data) seen++;
      if (seen != 1) {
        fprintf(
            stderr, "poly_model_call: entrypoint '%s' requires input '%s' exactly once\n",
            entry->name, entry->inputs[required]
        );
        return -1;
      }
    }
  }

  /* Validate the complete call before mutating any Model input. A rejected
   * dtype/length in one row must not leave earlier named inputs partially
   * updated. */
  for (int i = 0; i < n_io; i++) {
    if (!io[i].data) {
      fprintf(stderr, "poly_model_call: input row %d has NULL data\n", i);
      return -1;
    }
    if (!io[i].name || !entrypoint_accepts_input(inst, entry, io[i].name)) {
      fprintf(
          stderr, "poly_model_call: '%s' is not an input of entrypoint '%s'\n",
          io[i].name ? io[i].name : "(null)", entry && entry->name ? entry->name : "(null)"
      );
      return -1;
    }
    for (int prior = 0; prior < i; prior++) {
      if (io[prior].name && strcmp(io[prior].name, io[i].name) == 0) {
        fprintf(stderr, "poly_model_call: duplicate input '%s'\n", io[i].name);
        return -1;
      }
    }

    int bi = find_buf_by_name(inst, io[i].name);
    if (bi < 0) return -1;
    size_t nbytes = named_buf_nbytes(&inst->bufs[bi]);
    PolyDType supplied_dtype;
    PolyDType expected_dtype = inst->bufs[bi].buffer->dtype;
    if (io[i].nbytes != nbytes) {
      fprintf(
          stderr, "poly_model_call: input '%s' has %zu bytes, expected %zu\n", io[i].name,
          io[i].nbytes, nbytes
      );
      return -1;
    }
    if (!poly_dtype_by_id(io[i].dtype_id, &supplied_dtype) ||
        !poly_dtype_eq(supplied_dtype, expected_dtype)) {
      fprintf(
          stderr, "poly_model_call: input '%s' dtype id %d does not match %s\n", io[i].name,
          io[i].dtype_id, poly_dtype_name(expected_dtype)
      );
      return -1;
    }
  }

  for (int i = 0; i < n_io; i++) {
    if (!io[i].data) continue;
    int bi = find_buf_by_name(inst, io[i].name);
    if (bi < 0) return -1;
    size_t nbytes = named_buf_nbytes(&inst->bufs[bi]);
    if (poly_buffer_write(inst->ctx, inst->bufs[bi].buffer, io[i].data, nbytes) != 0) return -1;
    if (sync_buf_to_host(inst, bi) != 0) return -1;
  }

  return 0;
}

static int run_model_sink(
    PolyModel *inst,
    const RuntimeEntrypoint *entry,
    PolyUOp *sink,
    PolyIOBinding *io,
    int n_io,
    PolyLinearEntry *cached_executable
) {
  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:model] enter sink=%p n_io=%d device=%s\n", (void *)sink, n_io,
        poly_device_name(poly_ctx_get_preferred_device(inst->ctx))
    );
    fflush(stderr);
  }
  if (prepare_model_io(inst, entry, io, n_io) != 0) return -1;
  double t_attach = timing ? poly_now_ms() : 0.0;
  /* Model entrypoints and train/value-grad combined graphs already own
   * their output/effect storage. Skip tensor output allocation, but retain the
   * shared tinygrad-style concrete-buffer -> shaped-PARAM call boundary before
   * rangeify. */
  PolyLinearEntry local = {0};
  PolyLinearEntry *executable =
      cached_executable && cached_executable->linear ? cached_executable : &local;
  if (!executable->linear) {
    PolyVarBinding *var_bindings = NULL;
    int n_var_bindings = 0;
    PolyUOp *linear = poly_linear_effect_sink(inst->ctx, sink, &var_bindings, &n_var_bindings);
    PolyUOp *compiled = linear ? poly_compile_linear(inst->ctx, linear, -1) : NULL;
    if (compiled) {
      PolyLinearEntry built = {compiled, var_bindings, n_var_bindings};
      if (cached_executable) {
        if (model_publish_cached_executable(inst, cached_executable, &built) != 0) {
          cached_executable_clear(&built);
          return -1;
        }
        executable = cached_executable;
      } else {
        local = built;
      }
    } else {
      free(var_bindings);
    }
  }
  double t_sched = timing ? poly_now_ms() : 0.0;
  int ret = -1;
  if (executable->linear)
    ret = poly_run_linear(
        inst->ctx, executable->linear, executable->var_bindings, executable->n_var_bindings, NULL,
        0, true, true, false
    );
  if (!cached_executable) cached_executable_clear(&local);
  if (timing) {
    double t_done = poly_now_ms();
    fprintf(
        stderr,
        "[polygrad:model] input=%.3fms schedule=%.3fms run=%.3fms total=%.3fms "
        "ret=%d cached=%d\n",
        t_attach - t0, t_sched - t_attach, t_done - t_sched, t_done - t0, ret,
        cached_executable && cached_executable->linear
    );
  }
  return ret;
}

int poly_model_call(PolyModel *inst, const char *entrypoint, PolyIOBinding *io, int n_io) {
  if (!inst || inst->stage != POLY_MODEL_BUILT || !entrypoint) return -1;

  int ep_idx = find_entrypoint(inst, entrypoint);
  if (ep_idx < 0) {
    fprintf(stderr, "poly_model_call: no '%s' entrypoint\n", entrypoint);
    return -1;
  }

  PolyUOp *sink = inst->entrypoints[ep_idx].sink;
  PolyLinearEntry *cached_executable =
      inst->entry_executables ? &inst->entry_executables[ep_idx] : NULL;
  return run_model_sink(inst, &inst->entrypoints[ep_idx], sink, io, n_io, cached_executable);
}

int poly_model_entrypoint_count(const PolyModel *inst) {
  return inst && inst->stage == POLY_MODEL_BUILT ? inst->n_entrypoints : 0;
}

const char *poly_model_entrypoint_name(const PolyModel *inst, int entrypoint_index) {
  if (!inst || inst->stage != POLY_MODEL_BUILT || entrypoint_index < 0 ||
      entrypoint_index >= inst->n_entrypoints)
    return NULL;
  return inst->entrypoints[entrypoint_index].name;
}

int poly_model_entrypoint_input_count(const PolyModel *inst, const char *entrypoint) {
  int idx = inst && entrypoint ? find_entrypoint(inst, entrypoint) : -1;
  return idx >= 0 ? inst->entrypoints[idx].n_inputs : -1;
}

const char *poly_model_entrypoint_objective(const PolyModel *model, const char *entrypoint) {
  int idx = model && entrypoint ? find_entrypoint(model, entrypoint) : -1;
  return idx >= 0 ? model->entrypoints[idx].objective : NULL;
}

const char *poly_model_entrypoint_input_name(
    const PolyModel *inst,
    const char *entrypoint,
    int input_index
) {
  int idx = inst && entrypoint ? find_entrypoint(inst, entrypoint) : -1;
  if (idx < 0 || input_index < 0 || input_index >= inst->entrypoints[idx].n_inputs) return NULL;
  return inst->entrypoints[idx].inputs[input_index];
}

int poly_model_entrypoint_output_count(const PolyModel *inst, const char *entrypoint) {
  int idx = inst && entrypoint ? find_entrypoint(inst, entrypoint) : -1;
  return idx >= 0 ? inst->entrypoints[idx].n_outputs : -1;
}

const char *poly_model_entrypoint_output_name(
    const PolyModel *inst,
    const char *entrypoint,
    int output_index
) {
  int idx = inst && entrypoint ? find_entrypoint(inst, entrypoint) : -1;
  if (idx < 0 || output_index < 0 || output_index >= inst->entrypoints[idx].n_outputs) return NULL;
  return inst->entrypoints[idx].outputs[output_index];
}

/* Convenience wrapper */

int poly_model_forward(PolyModel *inst, PolyIOBinding *inputs, int n_inputs) {
  return poly_model_call(inst, "forward", inputs, n_inputs);
}

/* Optimizer */

int poly_model_set_optimizer(
    PolyModel *inst,
    int kind,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
  return poly_model_set_optimizer_ex(
      inst, kind, lr, beta1, beta2, eps, weight_decay, 0.0f, false, false
  );
}

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
) {
  if (!inst || inst->stage != POLY_MODEL_BUILT || !inst->has_portable_source) return -1;
  if (momentum < 0.0f) return -1;

  OptimState next = {
      .kind = kind,
      .lr = lr,
      .beta1 = beta1,
      .beta2 = beta2,
      .eps = eps,
      .weight_decay = weight_decay,
      .momentum = momentum,
      .nesterov = nesterov,
      .classic = classic,
      .step = 0,
  };
  PolyModel candidate = *inst;
  candidate.training.optim = next;
  candidate.training.train = NULL;
  ModelResidencyRoots prepared_roots = {0};
  if (model_prepare_residency_roots(&candidate, &prepared_roots) != 0) return -1;

  TrainState *old_train = inst->training.train;
  inst->training.optim = next;
  inst->training.train = NULL;
  model_publish_residency_roots(inst, &prepared_roots);
  train_free(old_train, inst->n_params);

  return 0;
}

/* Value and Grad */

/* Compute numel from shape inference. Returns -1 on failure. */
static int64_t uop_numel(PolyCtx *ctx, PolyUOp *u) {
  PolyShape s = poly_uop_max_shape(ctx, u);
  if (s.ndim < 0) {
    if (s.dims) free(s.dims);
    return -1;
  }
  int64_t n = poly_shape_numel(s);
  if (s.dims) free(s.dims);
  return n;
}

/* Build the combined fwd+bwd SINK for value_and_grad (lazy, once). */
static int ensure_vag_graph(PolyModel *inst, int loss_ep_idx) {
  if (inst->training.vag && inst->training.vag->entrypoint_index == loss_ep_idx) return 0;

  RuntimeEntrypoint *ep = &inst->entrypoints[loss_ep_idx];
  const char *objective = ep->objective;
  if (!objective && ep->n_outputs == 1) objective = ep->outputs[0];
  int objective_idx = objective ? find_buf_by_name(inst, objective) : -1;
  /* Low-level IR entrypoints may omit signature rows. Only a single named
   * output STORE is unambiguous; never select the first of several stores. */
  if (!objective && ep->n_outputs == 0 && ep->sink && ep->sink->n_src == 1) {
    PolyUOp *store = ep->sink->src[0];
    const PolyUOp *buffer = store && store->op == POLY_OP_STORE && store->n_src == 2
                                ? poly_uop_get_buffer_identity(store->src[0])
                                : NULL;
    for (int i = 0; buffer && i < inst->n_bufs; i++)
      if (inst->bufs[i].role == POLY_ROLE_OUTPUT && inst->bufs[i].buffer == buffer)
        objective_idx = i;
  }
  PolyUOp *loss_value = objective_idx >= 0
                            ? entrypoint_store_value(ep->sink, inst->bufs[objective_idx].buffer)
                            : NULL;
  if (!loss_value || uop_numel(inst->ctx, loss_value) != 1 ||
      !poly_dtype_is_float(loss_value->dtype)) {
    fprintf(stderr, "poly_model: entrypoint '%s' needs one declared scalar objective\n", ep->name);
    return -1;
  }

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
      PolyShape rs = poly_uop_max_shape(inst->ctx, topo[j]);
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
    fprintf(stderr, "poly_model: value_and_grad: autograd failed\n");
    free(grads);
    free(param_bufs);
    return -1;
  }

  /* Allocate VagState */
  VagState *vag = calloc(1, sizeof(VagState));
  if (!vag) {
    free(grads);
    free(param_bufs);
    return -1;
  }
  vag->entrypoint_index = loss_ep_idx;
  vag->objective_buffer_index = objective_idx;
  vag->grad_out_bufs = calloc((size_t)inst->n_params, sizeof(PolyUOp *));
  vag->grad_datas = calloc((size_t)inst->n_params, sizeof(float *));
  vag->grad_uops = calloc((size_t)inst->n_params, sizeof(PolyUOp *));
  vag->loss_value = loss_value;

  /* Build output stores: loss + per-param gradients */
  int n_stores = inst->n_params + 1;
  PolyUOp **stores = calloc((size_t)n_stores, sizeof(PolyUOp *));
  if (!stores || (inst->n_params && (!vag->grad_out_bufs || !vag->grad_datas || !vag->grad_uops))) {
    free(stores);
    free(grads);
    free(param_bufs);
    vag_free(vag, inst->n_params);
    return -1;
  }

  /* The C diagnostic ABI returns floats. Cast diagnostics in the core graph;
   * never reinterpret narrower/wider loss or gradient storage as host F32. */
  vag->loss_out_buf = model_new_buffer(inst->ctx, POLY_FLOAT32, 1);

  PolyUOp *loss_flat = loss_value;
  if (uop_numel(inst->ctx, loss_value) != 1) {
    int64_t one_shape[1] = {1};
    loss_flat = poly_reshape(inst->ctx, loss_value, one_shape, 1);
  }
  stores[0] =
      poly_store_val(inst->ctx, vag->loss_out_buf, poly_cast(inst->ctx, loss_flat, POLY_FLOAT32));

  /* Save raw gradient UOps for optimizer graph construction */
  for (int i = 0; i < inst->n_params; i++)
    vag->grad_uops[i] = grads[i];

  /* Gradient output buffers */
  for (int i = 0; i < inst->n_params; i++) {
    int64_t numel = uop_numel(inst->ctx, grads[i]);
    if (numel <= 0) {
      fprintf(stderr, "poly_model: value_and_grad: grad[%d] has unknown shape\n", i);
      free(stores);
      free(grads);
      free(param_bufs);
      vag_free(vag, inst->n_params);
      return -1;
    }
    PolyUOp *gbuf = model_new_buffer(inst->ctx, POLY_FLOAT32, numel);
    vag->grad_out_bufs[i] = gbuf;

    /* Flatten gradient if needed */
    PolyUOp *gflat = grads[i];
    PolyShape gs = poly_uop_max_shape(inst->ctx, grads[i]);
    if (gs.ndim != 1 || (gs.ndim == 1 && gs.dims[0] != numel)) {
      int64_t flat_shape[1] = {numel};
      gflat = poly_reshape(inst->ctx, grads[i], flat_shape, 1);
    }
    if (gs.dims) free(gs.dims);
    stores[i + 1] = poly_store_val(inst->ctx, gbuf, poly_cast(inst->ctx, gflat, POLY_FLOAT32));

    /* Allocate host storage for gradient data */
    NamedBuf *pb = &inst->bufs[inst->param_indices[i]];
    vag->grad_datas[i] = calloc((size_t)pb->numel, sizeof(float));
  }

  vag->combined_sink = poly_sink_n(inst->ctx, stores, n_stores);

  free(stores);
  free(grads);
  free(param_bufs);

  PolyModel candidate = *inst;
  candidate.training.vag = vag;
  candidate.training.train = NULL;
  ModelResidencyRoots prepared_roots = {0};
  if (model_prepare_residency_roots(&candidate, &prepared_roots) != 0) {
    vag_free(vag, inst->n_params);
    return -1;
  }
  VagState *old_vag = inst->training.vag;
  TrainState *old_train = inst->training.train;
  inst->training.vag = vag;
  inst->training.train = NULL;
  model_publish_residency_roots(inst, &prepared_roots);
  vag_free(old_vag, inst->n_params);
  train_free(old_train, inst->n_params);
  return 0;
}

int poly_model_value_and_grad(
    PolyModel *inst,
    const char *entrypoint,
    PolyIOBinding *io,
    int n_io,
    float *loss_out
) {
  if (!inst || inst->stage != POLY_MODEL_BUILT || !inst->has_portable_source || !entrypoint)
    return -1;

  int ep_idx = find_entrypoint(inst, entrypoint);
  if (ep_idx < 0) {
    fprintf(stderr, "poly_model_value_and_grad: no '%s' entrypoint\n", entrypoint);
    return -1;
  }

  /* Build combined fwd+bwd graph lazily */
  if (ensure_vag_graph(inst, ep_idx) != 0) return -1;
  VagState *vag = inst->training.vag;

  int ret = run_model_sink(
      inst, &inst->entrypoints[ep_idx], vag->combined_sink, io, n_io, &vag->executable
  );

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

static int param_ordinal_for_buf(const PolyModel *inst, int buf_idx) {
  if (!inst) return -1;
  for (int i = 0; i < inst->n_params; i++)
    if (inst->param_indices[i] == buf_idx) return i;
  return -1;
}

/* Build optimizer UOp graph (fwd+bwd+optimizer as a single combined SINK).
 * Gradients are consumed directly by AFTER/STORE update effects, not
 * materialized to separate output buffers (D1: no grad stores in optimizer
 * SINK). */
static int build_train_graph(PolyModel *inst, TrainState **out) {
  if (!inst || !out || !inst->training.vag) return -1;
  *out = NULL;
  VagState *vag = inst->training.vag;

  PolyCtx *ctx = inst->ctx;
  OptimState *o = &inst->training.optim;
  int np = inst->n_trainable_params;
  if (np <= 0) {
    fprintf(stderr, "ensure_train_graph: no trainable parameters\n");
    return -1;
  }

  TrainState *ts = calloc(1, sizeof(TrainState));
  if (!ts) return -1;

  /* Publish the exact named objective in the training graph. Only a non-F32
   * objective needs an additional F32 diagnostic buffer for the C return ABI. */
  PolyUOp *named_loss = inst->bufs[vag->objective_buffer_index].buffer;
  bool cast_loss = !poly_dtype_eq(named_loss->dtype, POLY_FLOAT32);
  ts->loss_out_buf = cast_loss ? model_new_buffer(ctx, POLY_FLOAT32, 1) : named_loss;

  if (!ts->loss_out_buf) {
    train_free(ts, np);
    return -1;
  }

  /* Count SINK sources: loss_store + param assigns + optimizer state assigns. */
  bool is_adam = (o->kind == POLY_OPTIM_ADAM || o->kind == POLY_OPTIM_ADAMW);
  bool sgd_momentum = (o->kind == POLY_OPTIM_SGD && o->momentum > 0.0f);
  bool has_m_bufs = is_adam || sgd_momentum;
  int n_sink_srcs = 1 + np; /* loss_store + param assigns */
  if (is_adam)
    n_sink_srcs += 2 * np + 2; /* + m/v assigns + b1_t/b2_t assigns */
  else if (sgd_momentum)
    n_sink_srcs += np; /* + momentum assigns */
  if (cast_loss) n_sink_srcs++;

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
  sink_srcs[0] = poly_store_val(ctx, named_loss, loss_flat);
  if (cast_loss)
    sink_srcs[n_sink_srcs - 1] =
        poly_store_val(ctx, ts->loss_out_buf, poly_cast(ctx, loss_flat, POLY_FLOAT32));

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
      int64_t opt_shape[POLY_IR_MAX_DIMS] = {0};
      int opt_ndim = pb->ndim;
      if (opt_ndim > 0) memcpy(opt_shape, pb->shape, (size_t)opt_ndim * sizeof(int64_t));
      char *m_name = optimizer_param_state_name("adam", "m", pb->name);
      char *v_name = optimizer_param_state_name("adam", "v", pb->name);
      if (!m_name || !v_name ||
          ensure_optimizer_state_buffer(
              inst, m_name, opt_shape, opt_ndim, false, 0.0f, &ts->m_bufs[i]
          ) != 0 ||
          ensure_optimizer_state_buffer(
              inst, v_name, opt_shape, opt_ndim, false, 0.0f, &ts->v_bufs[i]
          ) != 0) {
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
      int64_t opt_shape[POLY_IR_MAX_DIMS] = {0};
      int opt_ndim = pb->ndim;
      if (opt_ndim > 0) memcpy(opt_shape, pb->shape, (size_t)opt_ndim * sizeof(int64_t));
      char *b_name = optimizer_param_state_name("sgd", "b", pb->name);
      if (!b_name || ensure_optimizer_state_buffer(
                         inst, b_name, opt_shape, opt_ndim, false, 0.0f, &ts->m_bufs[i]
                     ) != 0) {
        free(b_name);
        free(sink_srcs);
        train_free(ts, np);
        return -1;
      }
      free(b_name);
    }
  }

  /* Build optimizer update graph for each trainable parameter. Keep the loss
   * output first, then tinygrad's optimizer order: state effects, parameters. */
  int m_base = -1;
  int v_base = -1;
  int param_base = 1;
  if (is_adam) {
    m_base = 3;
    v_base = 3 + np;
    param_base = 3 + 2 * np;
  } else if (sgd_momentum) {
    m_base = 1;
    param_base = 1 + np;
  }
  PolyUOp *lr = poly_const_float(ctx, (double)o->lr);
  if (!lr) {
    free(sink_srcs);
    train_free(ts, np);
    return -1;
  }

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
      PolyShape gs = poly_uop_max_shape(ctx, grad);
      if (gs.ndim > 1 || (gs.ndim == 1 && gs.dims && gs.dims[0] != pb_opt->numel)) {
        int64_t flat[1] = {pb_opt->numel};
        grad = poly_reshape(ctx, grad, flat, 1);
      }
      if (gs.dims) free(gs.dims);
    }

    PolyOptimConfig cfg = {
        .kind = o->kind,
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
            ctx, &cfg, lr, param_buf, grad, m_buf, v_buf, ts->bc1_buf, ts->bc2_buf, pb_opt->numel,
            &upd
        ) != 0) {
      fprintf(stderr, "ensure_train_graph: unsupported optimizer %d\n", o->kind);
      free(sink_srcs);
      train_free(ts, np);
      return -1;
    }
    sink_srcs[param_base + i] = poly_store_buffer_update(ctx, param_buf, upd.param_new);
    if (sgd_momentum) {
      sink_srcs[m_base + i] = upd.m_new;
    } else if (is_adam) {
      if (i == 0) {
        sink_srcs[1] = upd.bc1_new;
        sink_srcs[2] = upd.bc2_new;
      } else if (sink_srcs[1] != upd.bc1_new || sink_srcs[2] != upd.bc2_new) {
        free(sink_srcs);
        train_free(ts, np);
        return -1;
      }
      sink_srcs[m_base + i] = upd.m_new;
      sink_srcs[v_base + i] = upd.v_new;
    }
  }

  for (int i = 0; i < n_sink_srcs; i++)
    if (!sink_srcs[i]) {
      free(sink_srcs);
      train_free(ts, np);
      return -1;
    }

  ts->combined_sink = poly_sink_n(ctx, sink_srcs, n_sink_srcs);
  free(sink_srcs);

  *out = ts;
  return 0;
}

static int ensure_train_graph(PolyModel *inst, int loss_ep_idx) {
  if (ensure_vag_graph(inst, loss_ep_idx) != 0) return -1;
  if (inst->training.train) return 0;

  int old_n_bufs = inst->n_bufs;
  NamedBuf *candidate_bufs =
      old_n_bufs > 0 ? malloc((size_t)old_n_bufs * sizeof(*candidate_bufs)) : NULL;
  if (old_n_bufs > 0 && !candidate_bufs) return -1;
  if (old_n_bufs > 0)
    memcpy(candidate_bufs, inst->bufs, (size_t)old_n_bufs * sizeof(*candidate_bufs));

  PolyModel candidate = *inst;
  candidate.bufs = candidate_bufs;
  candidate.training.train = NULL;
  TrainState *train = NULL;
  ModelResidencyRoots prepared_roots = {0};
  if (build_train_graph(&candidate, &train) != 0) goto fail;
  candidate.training.train = train;
  if (model_prepare_residency_roots(&candidate, &prepared_roots) != 0) goto fail;

  NamedBuf *old_bufs = inst->bufs;
  inst->bufs = candidate.bufs;
  inst->n_bufs = candidate.n_bufs;
  inst->training.train = train;
  model_publish_residency_roots(inst, &prepared_roots);
  free(old_bufs);
  return 0;

fail:
  model_discard_prepared_residency_roots(inst->ctx, &prepared_roots);
  if (candidate.bufs)
    discard_appended_named_buffers(inst->ctx, candidate.bufs, old_n_bufs, candidate.n_bufs);
  free(candidate.bufs);
  train_free(train, inst->n_params);
  return -1;
}

/* Train Step */

int poly_model_train_step(
    PolyModel *inst,
    const char *entrypoint,
    PolyIOBinding *io,
    int n_io,
    float *loss_out
) {
  if (!inst || inst->stage != POLY_MODEL_BUILT || !inst->has_portable_source) return -1;
  if (inst->training.optim.kind == POLY_OPTIM_NONE) {
    fprintf(stderr, "poly_model_train_step: no optimizer configured\n");
    return -1;
  }

  /* Default to the sole declared objective. A conventional 'loss' entrypoint
   * remains usable for low-level IR with omitted signatures, not as priority
   * over two explicit objectives. Multiple objectives require a name. */
  int ep_idx = entrypoint ? find_entrypoint(inst, entrypoint) : -1;
  if (!entrypoint) {
    for (int i = 0; i < inst->n_entrypoints; i++) {
      if (!inst->entrypoints[i].objective) continue;
      if (ep_idx >= 0) {
        fprintf(stderr, "poly_model_train_step: multiple objectives; select an entrypoint\n");
        return -1;
      }
      ep_idx = i;
    }
    if (ep_idx < 0) ep_idx = find_entrypoint(inst, "loss");
  }
  if (ep_idx < 0) {
    fprintf(stderr, "poly_model_train_step: no selected objective entrypoint\n");
    return -1;
  }

  /* Build combined fwd+bwd+optimizer graph lazily */
  if (ensure_train_graph(inst, ep_idx) != 0) return -1;
  TrainState *ts = inst->training.train;
  OptimState *o = &inst->training.optim;

  /* tinygrad updates Adam beta-power state inside the scheduled optimizer
   * graph. Keep step only as bookkeeping for public state/checkpoints. */
  o->step++;

  int ret = run_model_sink(
      inst, &inst->entrypoints[ep_idx], ts->combined_sink, io, n_io, &ts->executable
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

  return 0;
}

/* Imported-model composition */

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

static const PolyModelInlineBinding *find_inline_binding(
    const PolyModelInlineBinding *bindings,
    int n_bindings,
    const char *name
) {
  if (!bindings || !name) return NULL;
  for (int i = 0; i < n_bindings; i++)
    if (bindings[i].name && strcmp(bindings[i].name, name) == 0) return &bindings[i];
  return NULL;
}

static const NamedBuf *model_buf_for_uop(const PolyModel *inst, PolyUOp *uop) {
  if (!inst || !uop) return NULL;
  for (int i = 0; i < inst->n_bufs; i++)
    if (inst->bufs[i].logical_buffer == uop) return &inst->bufs[i];
  return NULL;
}

typedef struct {
  PolyUOp *child_buf;
  PolyTensor *parent_tensor;
} InlineAlias;

static PolyTensor *register_inline_tensor(
    PolyModel *parent,
    const NamedBuf *b,
    const char *prefix,
    bool trainable,
    InlineAlias *aliases,
    int *n_aliases,
    int max_aliases
) {
  if (!parent || !parent->ctx || !b) return NULL;

  char *full = prefixed_name(prefix, b->name);
  if (!full) return NULL;

  for (int i = 0; i < *n_aliases; i++) {
    if (aliases[i].child_buf != b->logical_buffer) continue;
    PolyTensor *aliased = aliases[i].parent_tensor;
    int rc = b->role == POLY_ROLE_PARAM ? poly_model_state(parent, full, aliased, b->flags)
                                        : POLY_STATUS_OK;
    bool ok = rc == POLY_STATUS_OK &&
              (b->role != POLY_ROLE_PARAM || set_build_binding_trainable(parent, full, trainable));
    free(full);
    return ok ? aliased : NULL;
  }

  PolyTensor *ret = NULL;
  PolyDevice device = poly_ctx_get_preferred_device(parent->ctx);
  if (!poly_device_can_execute(device)) device = poly_device_default();
  switch (b->role) {
  case POLY_ROLE_PARAM:
    ret = poly_model_param(parent, full, b->logical_value->dtype, b->shape, b->ndim);
    if (ret && !set_build_binding_trainable(parent, full, trainable)) ret = NULL;
    break;
  case POLY_ROLE_INPUT:
  case POLY_ROLE_TARGET:
    /* Parent inputs/targets must be explicit bindings; silently declaring a
     * second ABI input would make the composed entrypoint incomplete. */
    break;
  case POLY_ROLE_OUTPUT:
    ret = poly_tensor_empty(parent->ctx, b->logical_value->dtype, b->shape, b->ndim, device);
    break;
  case POLY_ROLE_AUX:
  default:
    ret = poly_tensor_empty(parent->ctx, b->logical_value->dtype, b->shape, b->ndim, device);
    if (ret && poly_model_aux(parent, full, ret, b->flags) != POLY_STATUS_OK) ret = NULL;
    break;
  }

  if (ret && *n_aliases < max_aliases) {
    aliases[*n_aliases] = (InlineAlias){b->logical_buffer, ret};
    (*n_aliases)++;
  }
  free(full);
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
  PolyUOp *cloned =
      (u->tag || u->tag_arg.kind != POLY_ARG_NONE)
          ? poly_uop_tagged_arg(dst_ctx, u->op, u->dtype, src, u->n_src, u->arg, u->tag, u->tag_arg)
          : poly_uop(dst_ctx, u->op, u->dtype, src, u->n_src, u->arg);
  if (src != stack_src) free(src);
  if (cloned) poly_map_set(memo, poly_ptr_hash(u), u, cloned, poly_ptr_eq);
  return cloned;
}

static PolyUOp *entrypoint_store_value(PolyUOp *sink, PolyUOp *buffer) {
  if (!sink || sink->op != POLY_OP_SINK || !buffer) return NULL;
  for (int i = 0; i < sink->n_src; i++) {
    PolyUOp *store = sink->src[i];
    if (!store || store->op != POLY_OP_STORE || store->n_src < 2) continue;
    if (poly_uop_get_buffer_identity(store->src[0]) == buffer) return store->src[1];
  }
  return NULL;
}

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
) {
  if (out_n_outputs) *out_n_outputs = 0;
  if (!parent || parent->stage != POLY_MODEL_BUILDING || !parent->ctx || !child || !entrypoint ||
      !outputs || max_outputs < 0)
    return -1;
  PolyCtx *dst_ctx = parent->ctx;
  int ep_idx = find_entrypoint(child, entrypoint);
  if (ep_idx < 0) return -1;
  PolyUOp *logical_sink = child->entrypoints[ep_idx].logical_sink;
  /* A Tensor-built child may retain an exact bound pre-policy snapshot.  A
   * portable IR import has no such product and must inline its current root,
   * which has already crossed explicit placement.  Never substitute the
   * unplaced logical root as executable state merely because capture is absent. */
  bool use_capture = child->has_physical_capture;
  PolyUOp *physical_sink =
      use_capture ? child->entrypoints[ep_idx].capture_sink : child->entrypoints[ep_idx].sink;
  if (!logical_sink || logical_sink->op != POLY_OP_SINK || !physical_sink ||
      physical_sink->op != POLY_OP_SINK)
    return -1;

  PolyMap *logical_memo = poly_map_new(64);
  PolyMap *physical_memo = poly_map_new(64);
  if (!logical_memo || !physical_memo) {
    poly_map_destroy(logical_memo);
    poly_map_destroy(physical_memo);
    return -1;
  }

  InlineAlias *aliases = calloc((size_t)child->n_bufs, sizeof(InlineAlias));
  int n_aliases = 0;
  int rc = -1;

  for (int i = 0; i < child->n_bufs; i++) {
    const NamedBuf *b = &child->bufs[i];
    PolyUOp *physical_buffer = use_capture ? b->capture_buffer : b->buffer;
    PolyUOp *logical_source = model_binding_source(b);
    bool logical_reachable =
        logical_source && poly_uop_reachable(child->ctx, logical_sink, logical_source);
    bool physical_reachable = poly_uop_reachable(child->ctx, physical_sink, physical_buffer);
    if (!logical_reachable && !physical_reachable) continue;
    const PolyModelInlineBinding *binding = find_inline_binding(bindings, n_bindings, b->name);
    PolyTensor *replacement = binding ? binding->tensor : NULL;
    if (!replacement)
      replacement =
          register_inline_tensor(parent, b, prefix, trainable, aliases, &n_aliases, child->n_bufs);
    if (!replacement || !replacement->uop_logical || !replacement->uop_physical) goto done;
    poly_map_set(
        logical_memo, poly_ptr_hash(logical_source), logical_source, replacement->uop_logical,
        poly_ptr_eq
    );
    poly_map_set(
        physical_memo, poly_ptr_hash(physical_buffer), physical_buffer, replacement->uop_physical,
        poly_ptr_eq
    );
  }

  int n_outputs = 0;
  for (int i = 0; i < logical_sink->n_src; i++) {
    PolyUOp *store = logical_sink->src[i];
    if (!store || store->op != POLY_OP_STORE || store->n_src < 2) continue;
    const PolyUOp *identity = poly_uop_get_buffer_identity(store->src[0]);
    const NamedBuf *out_buf = model_buf_for_uop(child, (PolyUOp *)identity);
    if (!out_buf || out_buf->role != POLY_ROLE_OUTPUT) continue;
    if (n_outputs >= max_outputs) goto done;
    PolyUOp *physical_output = use_capture ? out_buf->capture_buffer : out_buf->buffer;
    PolyUOp *physical_value_src = entrypoint_store_value(physical_sink, physical_output);
    PolyUOp *logical_value = clone_uop_into_ctx(dst_ctx, logical_memo, store->src[1]);
    PolyUOp *physical_value = clone_uop_into_ctx(dst_ctx, physical_memo, physical_value_src);
    if (!logical_value || !physical_value) goto done;
    PolyDevice device = poly_ctx_get_preferred_device(dst_ctx);
    if (!poly_device_can_execute(device)) device = poly_device_default();
    PolyTensor *value = poly_tensor_create_with_roots(
        dst_ctx, logical_value, physical_value, POLY_TENSOR_VALUE, device
    );
    if (!value) goto done;
    value->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
    outputs[n_outputs++] = (PolyModelInlineOutput){out_buf->name, value};
  }

  if (out_n_outputs) *out_n_outputs = n_outputs;
  rc = 0;

done:
  free(aliases);
  poly_map_destroy(logical_memo);
  poly_map_destroy(physical_memo);
  return rc;
}

int poly_model_copy_prefixed_weights(PolyModel *dst, PolyModel *src, const char *prefix) {
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

PolyCtx *poly_model_ctx(const PolyModel *inst) {
  return inst ? inst->ctx : NULL;
}

void poly_model_own_ctx(PolyModel *inst) {
  if (inst) inst->owns_ctx = true;
}

PolyUOp *poly_model_get_buffer(const PolyModel *inst, const char *name) {
  if (!inst || !name) return NULL;
  int idx = find_buf_by_name(inst, name);
  return (idx >= 0) ? inst->bufs[idx].buffer : NULL;
}

PolyUOp *poly_model_get_sink(const PolyModel *inst, const char *name) {
  if (!inst || !name) return NULL;
  int idx = find_entrypoint(inst, name);
  return (idx >= 0) ? inst->entrypoints[idx].sink : NULL;
}

float *poly_model_buf_data_named(PolyModel *inst, const char *name, int64_t *numel_out) {
  if (!inst || !name) return NULL;
  int idx = find_buf_by_name(inst, name);
  if (idx < 0) return NULL;
  return poly_model_buf_data(inst, idx, numel_out);
}

int64_t poly_model_buf_numel_named(const PolyModel *inst, const char *name) {
  if (!inst || !name) return 0;
  int idx = find_buf_by_name(inst, name);
  return (idx >= 0) ? inst->bufs[idx].numel : 0;
}
