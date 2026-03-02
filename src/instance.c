/*
 * poly_instance.c -- Runtime for portable tensor-level model instances
 *
 * Product layer above the tinygrad-aligned compiler core.
 * Owns graph + named buffers + compiled steps + optimizer state.
 */

#define _POSIX_C_SOURCE 200809L
#include "instance.h"
#include "ir.h"
#include "safetensors.h"
#include "frontend.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ── Internal types ──────────────────────────────────────────────────── */

typedef struct {
  char *name;
  uint8_t role;
  PolyUOp *buffer;
  int64_t shape[8];
  int ndim;
  float *data;       /* owned for param/output/aux; NULL for input/target */
  int64_t numel;
} NamedBuf;

typedef struct {
  int kind;
  float lr, beta1, beta2, eps, weight_decay;
  int step;
  float *m;         /* concatenated first moments (all params) */
  float *v;         /* concatenated second moments (all params) */
  int64_t total_numel; /* total param elements */
} OptimState;

struct PolyInstance {
  PolyCtx *ctx;

  NamedBuf *bufs;
  int n_bufs;

  /* Param subset (indices into bufs[]) */
  int *param_indices;
  int n_params;

  /* Entrypoints */
  struct { char *name; PolyUOp *sink; } *entrypoints;
  int n_entrypoints;

  /* Compiled steps (lazy) */
  PolyStep *forward_step;
  PolyStep *train_step;

  /* Train-step graph pieces (built lazily) */
  PolyUOp *train_sink;        /* combined fwd+bwd+update SINK */
  PolyUOp *loss_buf;          /* BUFFER UOp for loss output */
  float loss_data;
  int train_loss_buf_idx;     /* loss buffer position in train_step->buf_order */
  int *train_grad_buf_idxs;   /* malloc'd [n_params] grad buffer positions */

  /* Optimizer update buffers (for non-ASSIGN path) */
  PolyUOp **grad_bufs;        /* [n_params] gradient BUFFER UOps */
  float **grad_datas;          /* [n_params] gradient host data */
  PolyUOp **update_bufs;      /* [n_params] updated param BUFFER UOps */
  float **update_datas;        /* [n_params] updated param host data */

  OptimState optim;
};

/* ── Helpers ─────────────────────────────────────────────────────────── */

static int64_t compute_numel(const int64_t *shape, int ndim) {
  int64_t n = 1;
  for (int i = 0; i < ndim; i++) n *= shape[i];
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

/* ── Lifecycle ───────────────────────────────────────────────────────── */

PolyInstance *poly_instance_from_ir(
    const uint8_t *ir_data, int ir_len,
    const uint8_t *weights_data, int weights_len)
{
  /* Import IR */
  PolyIrSpec spec;
  if (poly_ir_import(ir_data, ir_len, &spec) != 0) {
    fprintf(stderr, "poly_instance_from_ir: IR import failed\n");
    return NULL;
  }

  PolyInstance *inst = calloc(1, sizeof(PolyInstance));
  inst->ctx = spec.ctx;

  /* Copy buffers */
  inst->n_bufs = spec.n_bufs;
  inst->bufs = calloc(spec.n_bufs, sizeof(NamedBuf));
  int n_params = 0;
  for (int i = 0; i < spec.n_bufs; i++) {
    inst->bufs[i].name = strdup(spec.bufs[i].name);
    inst->bufs[i].role = spec.bufs[i].role;
    inst->bufs[i].buffer = spec.bufs[i].buffer;
    inst->bufs[i].ndim = spec.bufs[i].ndim;
    memcpy(inst->bufs[i].shape, spec.bufs[i].shape,
           spec.bufs[i].ndim * sizeof(int64_t));
    inst->bufs[i].numel = compute_numel(spec.bufs[i].shape, spec.bufs[i].ndim);

    /* Allocate data for all buffers (instance-owned).
     * INPUT buffers need storage too: callers populate them before forward. */
    inst->bufs[i].data = calloc(inst->bufs[i].numel, sizeof(float));
    if (spec.bufs[i].role == POLY_ROLE_PARAM) n_params++;
  }

  /* Build param index table */
  inst->n_params = n_params;
  inst->param_indices = malloc(n_params * sizeof(int));
  int pi = 0;
  for (int i = 0; i < spec.n_bufs; i++)
    if (spec.bufs[i].role == POLY_ROLE_PARAM)
      inst->param_indices[pi++] = i;

  /* Copy entrypoints */
  inst->n_entrypoints = spec.n_entrypoints;
  inst->entrypoints = calloc(spec.n_entrypoints, sizeof(*inst->entrypoints));
  for (int i = 0; i < spec.n_entrypoints; i++) {
    inst->entrypoints[i].name = strdup(spec.entrypoints[i].name);
    inst->entrypoints[i].sink = spec.entrypoints[i].sink;
  }

  /* Free spec arrays (but NOT ctx, we took ownership) */
  poly_ir_spec_free(&spec);

  /* Import weights if provided */
  if (weights_data && weights_len > 0) {
    if (poly_instance_import_weights(inst, weights_data, weights_len) != 0) {
      fprintf(stderr, "poly_instance_from_ir: weight import failed\n");
      poly_instance_free(inst);
      return NULL;
    }
  }

  /* Default optimizer: none */
  inst->optim.kind = POLY_OPTIM_NONE;

  return inst;
}

void poly_instance_free(PolyInstance *inst) {
  if (!inst) return;

  /* Free named buffers */
  for (int i = 0; i < inst->n_bufs; i++) {
    free(inst->bufs[i].name);
    free(inst->bufs[i].data);  /* safe: NULL for input/target */
  }
  free(inst->bufs);
  free(inst->param_indices);

  /* Free entrypoints */
  for (int i = 0; i < inst->n_entrypoints; i++)
    free(inst->entrypoints[i].name);
  free(inst->entrypoints);

  /* Free compiled steps */
  if (inst->forward_step) poly_step_destroy(inst->forward_step);
  if (inst->train_step) poly_step_destroy(inst->train_step);
  free(inst->train_grad_buf_idxs);
  inst->train_grad_buf_idxs = NULL;

  /* Free gradient/update buffers */
  if (inst->grad_bufs) {
    for (int i = 0; i < inst->n_params; i++) free(inst->grad_datas[i]);
    free(inst->grad_bufs);
    free(inst->grad_datas);
  }
  if (inst->update_bufs) {
    for (int i = 0; i < inst->n_params; i++) free(inst->update_datas[i]);
    free(inst->update_bufs);
    free(inst->update_datas);
  }

  /* Free optimizer state */
  free(inst->optim.m);
  free(inst->optim.v);

  /* Free context (arena-frees all UOps) */
  if (inst->ctx) poly_ctx_destroy(inst->ctx);

  free(inst);
}

/* ── Param Enumeration ───────────────────────────────────────────────── */

int poly_instance_param_count(const PolyInstance *inst) {
  return inst ? inst->n_params : 0;
}

const char *poly_instance_param_name(const PolyInstance *inst, int i) {
  if (!inst || i < 0 || i >= inst->n_params) return NULL;
  return inst->bufs[inst->param_indices[i]].name;
}

int poly_instance_param_shape(const PolyInstance *inst, int i,
                               int64_t *shape_out, int max_dims) {
  if (!inst || i < 0 || i >= inst->n_params) return 0;
  NamedBuf *b = &inst->bufs[inst->param_indices[i]];
  int n = b->ndim < max_dims ? b->ndim : max_dims;
  memcpy(shape_out, b->shape, n * sizeof(int64_t));
  return b->ndim;
}

float *poly_instance_param_data(PolyInstance *inst, int i,
                                 int64_t *numel_out) {
  if (!inst || i < 0 || i >= inst->n_params) return NULL;
  NamedBuf *b = &inst->bufs[inst->param_indices[i]];
  if (numel_out) *numel_out = b->numel;
  return b->data;
}

/* ── Buffer Enumeration ──────────────────────────────────────────────── */

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

int poly_instance_buf_shape(const PolyInstance *inst, int i,
                             int64_t *shape_out, int max_dims) {
  if (!inst || i < 0 || i >= inst->n_bufs) return 0;
  int n = inst->bufs[i].ndim < max_dims ? inst->bufs[i].ndim : max_dims;
  memcpy(shape_out, inst->bufs[i].shape, n * sizeof(int64_t));
  return inst->bufs[i].ndim;
}

float *poly_instance_buf_data(PolyInstance *inst, int i,
                               int64_t *numel_out) {
  if (!inst || i < 0 || i >= inst->n_bufs) return NULL;
  if (numel_out) *numel_out = inst->bufs[i].numel;
  return inst->bufs[i].data;
}

/* ── Weight I/O ──────────────────────────────────────────────────────── */

uint8_t *poly_instance_export_weights(PolyInstance *inst, int *out_len) {
  if (!inst || inst->n_params == 0) { *out_len = 0; return NULL; }

  PolySafetensorEntry *entries = malloc(inst->n_params * sizeof(PolySafetensorEntry));
  for (int i = 0; i < inst->n_params; i++) {
    NamedBuf *b = &inst->bufs[inst->param_indices[i]];
    entries[i].name = b->name;
    entries[i].data = b->data;
    entries[i].shape = b->shape;
    entries[i].ndim = b->ndim;
  }

  uint8_t *bytes = poly_safetensors_encode(entries, inst->n_params, NULL, out_len);
  free(entries);
  return bytes;
}

int poly_instance_import_weights(PolyInstance *inst,
                                  const uint8_t *data, int len) {
  if (!inst) return -1;

  int n_views = 0;
  char *metadata = NULL;
  PolySafetensorView *views = poly_safetensors_decode(data, len, &n_views, &metadata);
  if (!views) return -1;

  /* Match by name */
  for (int i = 0; i < n_views; i++) {
    int bi = find_buf_by_name(inst, views[i].name);
    if (bi < 0) {
      fprintf(stderr, "poly_instance_import_weights: unknown tensor '%s'\n",
              views[i].name);
      /* Continue - non-fatal */
    } else if (inst->bufs[bi].data && views[i].numel == inst->bufs[bi].numel) {
      memcpy(inst->bufs[bi].data, views[i].data, views[i].numel * sizeof(float));
    } else if (inst->bufs[bi].data) {
      fprintf(stderr, "poly_instance_import_weights: shape mismatch for '%s' "
              "(expected %lld, got %lld)\n", views[i].name,
              (long long)inst->bufs[bi].numel, (long long)views[i].numel);
    }
    free(views[i].name);
  }
  free(views);
  free(metadata);
  return 0;
}

/* ── IR Export ───────────────────────────────────────────────────────── */

uint8_t *poly_instance_export_ir(PolyInstance *inst, int *out_len) {
  if (!inst) { *out_len = 0; return NULL; }

  /* Build PolyIrSpec from instance state */
  PolyIrBufEntry *bufs = malloc(inst->n_bufs * sizeof(PolyIrBufEntry));
  for (int i = 0; i < inst->n_bufs; i++) {
    bufs[i].name = inst->bufs[i].name;
    bufs[i].role = inst->bufs[i].role;
    bufs[i].buffer = inst->bufs[i].buffer;
    bufs[i].ndim = inst->bufs[i].ndim;
    memcpy(bufs[i].shape, inst->bufs[i].shape, inst->bufs[i].ndim * sizeof(int64_t));
  }

  PolyIrEntrypoint *eps = malloc(inst->n_entrypoints * sizeof(PolyIrEntrypoint));
  for (int i = 0; i < inst->n_entrypoints; i++) {
    eps[i].name = inst->entrypoints[i].name;
    eps[i].sink = inst->entrypoints[i].sink;
  }

  PolyIrSpec spec = { inst->ctx, bufs, inst->n_bufs, eps, inst->n_entrypoints };
  uint8_t *bytes = poly_ir_export(&spec, out_len);
  free(bufs);
  free(eps);
  return bytes;
}

/* ── Forward Execution ───────────────────────────────────────────────── */

int poly_instance_forward(PolyInstance *inst,
                          PolyIOBinding *inputs, int n_inputs) {
  if (!inst) return -1;

  int ep = find_entrypoint(inst, "forward");
  if (ep < 0) {
    fprintf(stderr, "poly_instance_forward: no 'forward' entrypoint\n");
    return -1;
  }

  /* Build buffer bindings */
  int n_bindings = inst->n_bufs;
  PolyBufferBinding *bindings = malloc(n_bindings * sizeof(PolyBufferBinding));
  for (int i = 0; i < inst->n_bufs; i++) {
    bindings[i].buffer = inst->bufs[i].buffer;
    bindings[i].data = inst->bufs[i].data; /* NULL for input/target */
  }

  /* Override input bindings from caller (skip NULL to keep instance data) */
  for (int i = 0; i < n_inputs; i++) {
    int bi = find_buf_by_name(inst, inputs[i].name);
    if (bi >= 0) {
      if (inputs[i].data)
        bindings[bi].data = inputs[i].data;
    } else {
      fprintf(stderr, "poly_instance_forward: unknown input '%s'\n",
              inputs[i].name);
    }
  }

  /* Lazy compile */
  if (!inst->forward_step) {
    inst->forward_step = poly_compile_step(inst->ctx,
                                            inst->entrypoints[ep].sink);
    if (!inst->forward_step) {
      free(bindings);
      fprintf(stderr, "poly_instance_forward: compilation failed\n");
      return -1;
    }
  }

  int ret = poly_step_run(inst->forward_step, bindings, n_bindings);
  free(bindings);
  return ret;
}

/* ── Optimizer ───────────────────────────────────────────────────────── */

int poly_instance_set_optimizer(PolyInstance *inst, int kind,
                                float lr, float beta1, float beta2,
                                float eps, float weight_decay) {
  if (!inst) return -1;

  inst->optim.kind = kind;
  inst->optim.lr = lr;
  inst->optim.beta1 = beta1;
  inst->optim.beta2 = beta2;
  inst->optim.eps = eps;
  inst->optim.weight_decay = weight_decay;
  inst->optim.step = 0;

  /* Free old state if re-configuring */
  free(inst->optim.m);
  free(inst->optim.v);
  inst->optim.m = NULL;
  inst->optim.v = NULL;

  /* Allocate moment buffers for Adam/AdamW */
  if (kind == POLY_OPTIM_ADAM || kind == POLY_OPTIM_ADAMW) {
    int64_t total = 0;
    for (int i = 0; i < inst->n_params; i++)
      total += inst->bufs[inst->param_indices[i]].numel;
    inst->optim.total_numel = total;
    inst->optim.m = calloc(total, sizeof(float));
    inst->optim.v = calloc(total, sizeof(float));
  }

  return 0;
}

/* Apply optimizer update to param data in-place (CPU host side) */
static void apply_optimizer_update(PolyInstance *inst,
                                    int param_idx,
                                    const float *grad,
                                    int64_t numel,
                                    int64_t moment_offset) {
  NamedBuf *b = &inst->bufs[inst->param_indices[param_idx]];
  float *param = b->data;
  OptimState *o = &inst->optim;

  switch (o->kind) {
  case POLY_OPTIM_SGD:
    for (int64_t j = 0; j < numel; j++)
      param[j] -= o->lr * grad[j];
    break;

  case POLY_OPTIM_ADAM:
  case POLY_OPTIM_ADAMW: {
    float *m = o->m + moment_offset;
    float *v = o->v + moment_offset;
    float bc1 = 1.0f - powf(o->beta1, (float)(o->step));
    float bc2 = 1.0f - powf(o->beta2, (float)(o->step));

    for (int64_t j = 0; j < numel; j++) {
      float g = grad[j];

      /* Weight decay (AdamW decoupled) */
      if (o->kind == POLY_OPTIM_ADAMW && o->weight_decay > 0.0f)
        param[j] *= (1.0f - o->lr * o->weight_decay);

      /* Update moments */
      m[j] = o->beta1 * m[j] + (1.0f - o->beta1) * g;
      v[j] = o->beta2 * v[j] + (1.0f - o->beta2) * g * g;

      /* Bias-corrected update */
      float m_hat = m[j] / bc1;
      float v_hat = v[j] / bc2;
      param[j] -= o->lr * m_hat / (sqrtf(v_hat) + o->eps);
    }
    break;
  }
  default:
    break;
  }
}

/* ── Train Step ──────────────────────────────────────────────────────── */

/* Build the training execution: forward + loss + gradients + optimizer.
 * On first call, compiles a single PolyStep for all fwd+bwd computation
 * via poly_compile_value_and_grad. Subsequent calls reuse the compiled step. */
int poly_instance_train_step(PolyInstance *inst,
                             PolyIOBinding *io, int n_io,
                             float *loss_out) {
  if (!inst) return -1;
  if (inst->optim.kind == POLY_OPTIM_NONE) {
    fprintf(stderr, "poly_instance_train_step: no optimizer configured\n");
    return -1;
  }

  /* Find loss entrypoint */
  int loss_ep = find_entrypoint(inst, "loss");
  if (loss_ep < 0) {
    fprintf(stderr, "poly_instance_train_step: no 'loss' entrypoint\n");
    return -1;
  }

  /* Compile training step on first call */
  if (!inst->train_step) {
    PolyUOp *loss_sink = inst->entrypoints[loss_ep].sink;
    PolyUOp *loss_store = loss_sink->src[0]; /* SINK src[0] = STORE */
    PolyUOp *loss_value = loss_store->src[1]; /* STORE src[1] = value */

    /* Build param buffer UOp array */
    PolyUOp **param_bufs = malloc((size_t)inst->n_params * sizeof(PolyUOp *));
    if (!param_bufs) return -1;
    for (int i = 0; i < inst->n_params; i++)
      param_bufs[i] = inst->bufs[inst->param_indices[i]].buffer;

    /* Allocate grad data storage (grad_bufs sentinel + grad_datas) */
    inst->grad_bufs = calloc((size_t)inst->n_params, sizeof(PolyUOp *));
    inst->grad_datas = calloc((size_t)inst->n_params, sizeof(float *));
    if (!inst->grad_bufs || !inst->grad_datas) {
      free(param_bufs);
      return -1;
    }
    for (int i = 0; i < inst->n_params; i++) {
      NamedBuf *b = &inst->bufs[inst->param_indices[i]];
      inst->grad_datas[i] = calloc((size_t)b->numel, sizeof(float));
      if (!inst->grad_datas[i]) {
        free(param_bufs);
        return -1;
      }
    }

    /* Compile combined fwd+bwd step */
    inst->train_grad_buf_idxs = malloc((size_t)inst->n_params * sizeof(int));
    if (!inst->train_grad_buf_idxs) {
      free(param_bufs);
      return -1;
    }
    inst->train_step = poly_compile_value_and_grad(inst->ctx, loss_value,
                           param_bufs, inst->n_params,
                           &inst->train_loss_buf_idx,
                           inst->train_grad_buf_idxs);
    free(param_bufs);

    if (!inst->train_step) {
      fprintf(stderr, "poly_instance_train_step: compile failed\n");
      free(inst->train_grad_buf_idxs);
      inst->train_grad_buf_idxs = NULL;
      return -1;
    }

    /* Guard: loss and grad output buffers must be float32 so host float*
     * bindings are valid.  poly_compile_value_and_grad preserves the input
     * dtype (float16/bfloat16 would pass through), which would silently
     * corrupt the host-side scalar and optimizer state. */
    PolyUOp *loss_out_uop = poly_step_buf_uop(inst->train_step,
                                               inst->train_loss_buf_idx);
    if (loss_out_uop &&
        !poly_dtype_eq(poly_dtype_scalar(loss_out_uop->dtype), POLY_FLOAT32)) {
      fprintf(stderr,
              "poly_instance_train_step: loss output is %s, expected float32\n",
              poly_dtype_name(loss_out_uop->dtype));
      poly_step_destroy(inst->train_step);
      inst->train_step = NULL;
      free(inst->train_grad_buf_idxs);
      inst->train_grad_buf_idxs = NULL;
      return -1;
    }
  }

  /* Build bindings: instance buffers + loss output + grad outputs */
  int n_bindings = inst->n_bufs + 1 + inst->n_params;
  PolyBufferBinding *bindings = malloc((size_t)n_bindings * sizeof(PolyBufferBinding));
  if (!bindings) return -1;

  for (int i = 0; i < inst->n_bufs; i++) {
    bindings[i].buffer = inst->bufs[i].buffer;
    bindings[i].data = inst->bufs[i].data;
  }

  /* Override I/O from caller */
  for (int i = 0; i < n_io; i++) {
    int bi = find_buf_by_name(inst, io[i].name);
    if (bi >= 0) bindings[bi].data = io[i].data;
  }

  /* Loss output binding */
  bindings[inst->n_bufs] = (PolyBufferBinding){
    poly_step_buf_uop(inst->train_step, inst->train_loss_buf_idx),
    &inst->loss_data
  };

  /* Grad output bindings */
  for (int i = 0; i < inst->n_params; i++) {
    bindings[inst->n_bufs + 1 + i] = (PolyBufferBinding){
      poly_step_buf_uop(inst->train_step, inst->train_grad_buf_idxs[i]),
      inst->grad_datas[i]
    };
  }

  int ret = poly_step_run_ex(inst->train_step, bindings, n_bindings, NULL, 0);
  free(bindings);
  if (ret != 0) return ret;

  /* Update instance "loss" buffer so consumers see the current value */
  int loss_named_idx = find_buf_by_name(inst, "loss");
  if (loss_named_idx >= 0 && inst->bufs[loss_named_idx].data)
    inst->bufs[loss_named_idx].data[0] = inst->loss_data;

  if (loss_out) *loss_out = inst->loss_data;

  /* Apply optimizer updates (host-side) */
  inst->optim.step++;
  int64_t moment_offset = 0;
  for (int i = 0; i < inst->n_params; i++) {
    NamedBuf *pb = &inst->bufs[inst->param_indices[i]];
    apply_optimizer_update(inst, i, inst->grad_datas[i], pb->numel, moment_offset);
    moment_offset += pb->numel;
  }

  return 0;
}
