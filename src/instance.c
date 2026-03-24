/*
 * poly_instance.c -- Runtime for portable tensor-level model instances
 *
 * Product layer above the tinygrad-aligned compiler core.
 * Owns graph + named buffers + execution caches + optimizer state.
 *
 * Backend-aware: uses the exec_plan API (prepare + lower + run) for
 * all execution. Device selection at runtime via set_device().
 * Prepared step cache is backend-neutral and survives device changes.
 * Executable step cache retains entries for all previously-used devices.
 */

#define _POSIX_C_SOURCE 200809L
#include "instance.h"
#include "ir.h"
#include "safetensors.h"
#include "frontend.h"
#include "exec_plan.h"
#include "codegen.h"   /* poly_cuda_available (POLY_HAS_CUDA) */
#include "scheduler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

/* ── Internal types ──────────────────────────────────────────────────── */

typedef struct {
  char *name;
  uint8_t role;
  PolyUOp *buffer;
  int64_t shape[8];
  int ndim;
  float *data;       /* owned, allocated for all roles */
  int64_t numel;
} NamedBuf;

typedef struct {
  int kind;
  float lr, beta1, beta2, eps, weight_decay;
  int step;
} OptimState;

/* ── Value-and-grad metadata (built lazily on first train call) ──────── */

typedef struct {
  PolyUOp *combined_sink;           /* combined fwd+bwd SINK */
  PolyUOp *loss_out_buf;            /* BUFFER UOp for loss output */
  PolyUOp **grad_out_bufs;          /* [n_params] gradient BUFFER UOps */
  float **grad_datas;               /* [n_params] gradient host data */
  PolyUOp **grad_uops;             /* [n_params] raw gradient UOp expressions */
  PolyUOp *loss_value;             /* loss value UOp (pre-store) */
  float loss_data;                  /* scalar loss value */

  /* Cached slot indices (set after first prepare_step) */
  int loss_slot;                    /* buf_slot index for loss output */
  int *grad_slots;                  /* [n_params] buf_slot indices for grads */
  bool slots_resolved;              /* true after first resolution */
} VagState;

/* ── Training state (optimizer graph, built lazily) ──────────────────── */

typedef struct {
  PolyUOp *combined_sink;           /* fwd+bwd+optimizer SINK */
  PolyUOp *loss_out_buf;            /* BUFFER UOp for loss scalar output */
  float loss_data;                  /* scalar loss value after step */

  /* Moment buffers (Adam/AdamW only) */
  PolyUOp **m_bufs;                /* [n_params] first moment BUFFER UOps */
  PolyUOp **v_bufs;                /* [n_params] second moment BUFFER UOps */
  int n_moment_bufs;

  /* Moment host data (for initialization + set_device upload) */
  float **m_datas;                 /* [n_params] first moment host data */
  float **v_datas;                 /* [n_params] second moment host data */

  /* Moment buffer handles (for bindings) */
  PolyBufferHandle *m_handles;     /* [n_params] */
  PolyBufferHandle *v_handles;     /* [n_params] */

  /* Bias correction scalar buffers (Adam/AdamW only) */
  PolyUOp *bc1_buf;               /* 1-element buffer for bc1 */
  PolyUOp *bc2_buf;               /* 1-element buffer for bc2 */
  float bc1_data;                  /* host value for bc1 */
  float bc2_data;                  /* host value for bc2 */
  PolyBufferHandle bc1_handle;
  PolyBufferHandle bc2_handle;
} TrainState;

/* ── Cached slot table (avoids O(n_slots×n_bufs) scan each call) ────── */

typedef struct {
  PolyCompiledPlan *plan;      /* cache-owned, do NOT free */
  void **slot_data;            /* [n_slots], pre-filled */
  int n_slots;

  /* Mapping: slot index -> instance buf index, or -1 */
  int *slot_to_buf;

  /* IO fast-patch: which slots to update from IO bindings */
  int *io_slot_indices;        /* [n_io] */
  int *io_buf_indices;         /* [n_io] instance buf index */
  int n_io;
} SlotCache;

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

  /* ── Buffer handles (one per named buffer, carries domain) ── */
  PolyBufferHandle *buf_handles;    /* [n_bufs], ptr + domain + nbytes */

  /* ── Value-and-grad state (lazy, per-entrypoint -- currently only "loss") ── */
  VagState *vag;                    /* NULL until first value_and_grad call */

  /* ── Training state (optimizer graph, built lazily) ── */
  TrainState *train;                /* NULL until first train_step call */

  /* ── Optimizer state ── */
  OptimState optim;

  /* ── Cached slot tables (one per entrypoint path) ── */
  SlotCache *call_cache;            /* for poly_instance_call */
  SlotCache *train_cache;           /* for poly_instance_train_step */
};

/* ── Slot table cache ────────────────────────────────────────────────── */

static void slot_cache_free(SlotCache *c) {
  if (!c) return;
  free(c->slot_data);
  free(c->slot_to_buf);
  free(c->io_slot_indices);
  free(c->io_buf_indices);
  free(c);
}

/* Build a cached slot table for a given sink + device.
 * Maps each schedule slot to the instance buf index whose buffer UOp matches.
 * extra_bufs/extra_ptrs provide non-instance buffers (loss output, moments, etc.).
 * Returns NULL on error. The plan is cache-owned; the SlotCache does NOT free it. */
static SlotCache *slot_cache_build(
    PolyInstance *inst, PolyUOp *sink, PolyDeviceId device,
    PolyUOp **extra_bufs, void **extra_ptrs, int n_extra) {
  PolyCompiledPlan *plan = poly_get_plan(inst->ctx, sink, device);
  if (!plan) return NULL;
  PolySchedule *sched = plan->schedule;

  SlotCache *c = calloc(1, sizeof(SlotCache));
  c->plan = plan;
  c->n_slots = sched->n_buf_slots;
  c->slot_data = calloc((size_t)(c->n_slots > 0 ? c->n_slots : 1), sizeof(void *));
  c->slot_to_buf = malloc((size_t)(c->n_slots > 0 ? c->n_slots : 1) * sizeof(int));

  /* Build slot -> instance buf mapping and fill initial slot_data */
  for (int s = 0; s < c->n_slots; s++) {
    c->slot_to_buf[s] = -1;
    if (sched->buf_slots[s].is_intermediate) continue;
    PolyUOp *slot_uop = sched->buf_slots[s].buf_uop;

    /* Check instance buffers */
    for (int j = 0; j < inst->n_bufs; j++) {
      if (inst->bufs[j].buffer == slot_uop) {
        c->slot_to_buf[s] = j;
        c->slot_data[s] = inst->buf_handles[j].ptr;
        break;
      }
    }

    /* Check extra buffers */
    if (c->slot_to_buf[s] < 0 && extra_bufs) {
      for (int j = 0; j < n_extra; j++) {
        if (extra_bufs[j] == slot_uop) {
          c->slot_data[s] = extra_ptrs[j];
          break;
        }
      }
    }
  }

  /* Build IO fast-patch table: which slots correspond to INPUT/TARGET */
  int max_io = inst->n_bufs;
  c->io_slot_indices = malloc((size_t)max_io * sizeof(int));
  c->io_buf_indices = malloc((size_t)max_io * sizeof(int));
  c->n_io = 0;
  for (int s = 0; s < c->n_slots; s++) {
    int bi = c->slot_to_buf[s];
    if (bi >= 0 && (inst->bufs[bi].role == POLY_ROLE_INPUT ||
                    inst->bufs[bi].role == POLY_ROLE_TARGET)) {
      c->io_slot_indices[c->n_io] = s;
      c->io_buf_indices[c->n_io] = bi;
      c->n_io++;
    }
  }

  return c;
}

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

  /* Initialize buffer handles (CPU domain, pointing at host data) */
  inst->buf_handles = calloc(spec.n_bufs, sizeof(PolyBufferHandle));
  for (int i = 0; i < spec.n_bufs; i++) {
    inst->buf_handles[i] = (PolyBufferHandle){
      .ptr = inst->bufs[i].data,
      .nbytes = (size_t)inst->bufs[i].numel * sizeof(float),
      .domain = POLY_DEVICE_CPU,
      .owned = false,  /* data owned by NamedBuf, not the handle */
    };
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

static void vag_free(VagState *vag, int n_params) {
  if (!vag) return;
  if (vag->grad_datas) {
    for (int i = 0; i < n_params; i++) free(vag->grad_datas[i]);
    free(vag->grad_datas);
  }
  free(vag->grad_out_bufs);
  free(vag->grad_uops);
  free(vag->grad_slots);
  free(vag);
}

static void train_free(TrainState *ts, int n_params) {
  if (!ts) return;
  if (ts->m_datas) {
    for (int i = 0; i < n_params; i++) free(ts->m_datas[i]);
    free(ts->m_datas);
  }
  if (ts->v_datas) {
    for (int i = 0; i < n_params; i++) free(ts->v_datas[i]);
    free(ts->v_datas);
  }
  free(ts->m_bufs);
  free(ts->v_bufs);
  free(ts->m_handles);
  free(ts->v_handles);
  free(ts);
}

void poly_instance_free(PolyInstance *inst) {
  if (!inst) return;

  /* Free device-owned buffer handles */
  if (inst->buf_handles) {
    for (int i = 0; i < inst->n_bufs; i++) {
      PolyBufferHandle *h = &inst->buf_handles[i];
      if (h->owned && h->ptr) {
        const PolyBackendDesc *be = poly_backend_get(h->domain);
        if (be) be->get_allocator()->free(h->ptr, be->get_allocator()->dev_ctx);
      }
    }
    free(inst->buf_handles);
  }

  /* Free named buffers */
  for (int i = 0; i < inst->n_bufs; i++) {
    free(inst->bufs[i].name);
    free(inst->bufs[i].data);
  }
  free(inst->bufs);
  free(inst->param_indices);

  /* Free entrypoints */
  for (int i = 0; i < inst->n_entrypoints; i++)
    free(inst->entrypoints[i].name);
  free(inst->entrypoints);

  /* Free execution caches.
   * Order: exec cache first (holds pointers into prepared steps),
   * then context (arena-frees all UOps). */

  /* Free value-and-grad state */
  vag_free(inst->vag, inst->n_params);

  /* Free training state */
  train_free(inst->train, inst->n_params);

  /* Free slot caches (plans are cache-owned, not freed here) */
  slot_cache_free(inst->call_cache);
  slot_cache_free(inst->train_cache);

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
  int bi = inst->param_indices[i];
  NamedBuf *b = &inst->bufs[bi];
  if (numel_out) *numel_out = b->numel;
  /* Device-memory backends: host data may be stale */
  if (inst->buf_handles && inst->buf_handles[bi].domain == POLY_DEVICE_CUDA)
    return NULL;
#ifdef POLY_HAS_HIP
  if (inst->buf_handles && inst->buf_handles[bi].domain == POLY_DEVICE_HIP)
    return NULL;
#endif
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
  /* Device-memory backends: host data may be stale */
  if (inst->buf_handles && inst->buf_handles[i].domain == POLY_DEVICE_CUDA)
    return NULL;
#ifdef POLY_HAS_HIP
  if (inst->buf_handles && inst->buf_handles[i].domain == POLY_DEVICE_HIP)
    return NULL;
#endif
  return inst->bufs[i].data;
}

/* ── Readback / Upload ──────────────────────────────────────────────── */

static int readback_handle(const PolyBufferHandle *h, void *dst, size_t len) {
  if (!h || !h->ptr || !dst || len == 0) return -1;
  if (h->domain == POLY_DEVICE_CPU || h->domain == POLY_DEVICE_INTERP) {
    memcpy(dst, h->ptr, len);
    return 0;
  }
  const PolyBackendDesc *be = poly_backend_get(h->domain);
  if (!be) return -1;
  return be->get_allocator()->copy_out(dst, h->ptr, len,
                                        be->get_allocator()->dev_ctx);
}

static int upload_handle(PolyBufferHandle *h, const void *src, size_t len) {
  if (!h || !h->ptr || !src || len == 0) return -1;
  if (h->domain == POLY_DEVICE_CPU || h->domain == POLY_DEVICE_INTERP) {
    memcpy(h->ptr, src, len);
    return 0;
  }
  const PolyBackendDesc *be = poly_backend_get(h->domain);
  if (!be) return -1;
  return be->get_allocator()->copy_in(h->ptr, src, len,
                                       be->get_allocator()->dev_ctx);
}

int poly_instance_readback_buf(PolyInstance *inst, int i,
                               void *host_dst, size_t dst_len) {
  if (!inst || i < 0 || i >= inst->n_bufs || !inst->buf_handles) return -1;
  return readback_handle(&inst->buf_handles[i], host_dst, dst_len);
}

int poly_instance_upload_buf(PolyInstance *inst, int i,
                             const void *host_src, size_t src_len) {
  if (!inst || i < 0 || i >= inst->n_bufs || !inst->buf_handles) return -1;
  return upload_handle(&inst->buf_handles[i], host_src, src_len);
}

int poly_instance_readback_param(PolyInstance *inst, int i,
                                 void *host_dst, size_t dst_len) {
  if (!inst || i < 0 || i >= inst->n_params) return -1;
  return poly_instance_readback_buf(inst, inst->param_indices[i], host_dst, dst_len);
}

int poly_instance_upload_param(PolyInstance *inst, int i,
                               const void *host_src, size_t src_len) {
  if (!inst || i < 0 || i >= inst->n_params) return -1;
  return poly_instance_upload_buf(inst, inst->param_indices[i], host_src, src_len);
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

/* ── Device configuration ────────────────────────────────────────────── */

int poly_instance_set_device(PolyInstance *inst, PolyDeviceId device) {
  if (!inst) return -1;

  /* Invalidate slot caches (buffer handles change domain) */
  slot_cache_free(inst->call_cache);  inst->call_cache = NULL;
  slot_cache_free(inst->train_cache); inst->train_cache = NULL;

  /* Resolve AUTO */
  PolyDeviceId resolved = device;
  if (resolved == POLY_DEVICE_AUTO) {
#ifdef __EMSCRIPTEN__
    resolved = POLY_DEVICE_WASM_JIT;
#else
    resolved = POLY_DEVICE_CPU;
#endif
  }

  /* Validate: backend must exist for this build */
  const PolyBackendDesc *backend = poly_backend_get(resolved);
  if (!backend) {
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

  const PolyAllocator *alloc = backend->get_allocator();

  /* Bulk rematerialization: move all buffer handles to the new domain */
  for (int i = 0; i < inst->n_bufs; i++) {
    PolyBufferHandle *h = &inst->buf_handles[i];
    if (h->domain == resolved) continue;  /* already there */

    size_t nbytes = (size_t)inst->bufs[i].numel * sizeof(float);
    if (nbytes == 0) continue;

    /* For host-memory devices (CPU, INTERP, WASM_JIT), point at existing data */
    if (resolved == POLY_DEVICE_CPU || resolved == POLY_DEVICE_INTERP
#ifdef __EMSCRIPTEN__
        || resolved == POLY_DEVICE_WASM_JIT
#endif
    ) {
      /* Free old device handle if it was device-owned */
      if (h->owned && h->domain != POLY_DEVICE_CPU && h->domain != POLY_DEVICE_INTERP) {
        const PolyBackendDesc *old_be = poly_backend_get(h->domain);
        if (old_be) {
          const PolyAllocator *old_alloc = old_be->get_allocator();
          /* Readback to host before freeing device memory */
          if (inst->bufs[i].data)
            old_alloc->copy_out(inst->bufs[i].data, h->ptr, nbytes, old_alloc->dev_ctx);
          old_alloc->free(h->ptr, old_alloc->dev_ctx);
        }
      }
      *h = (PolyBufferHandle){
        .ptr = inst->bufs[i].data, .nbytes = nbytes,
        .domain = resolved, .owned = false,
      };
      continue;
    }

    /* For device-memory backends (CUDA, future WEBGPU): allocate + upload */
    void *dptr = alloc->alloc(nbytes, alloc->dev_ctx);
    if (!dptr) {
      fprintf(stderr, "poly_instance_set_device: alloc failed for buffer %d\n", i);
      return -1;
    }
    if (inst->bufs[i].data)
      alloc->copy_in(dptr, inst->bufs[i].data, nbytes, alloc->dev_ctx);

    /* Free old device handle if owned and from a different device-memory domain */
    if (h->owned && h->ptr && h->domain != POLY_DEVICE_CPU) {
      const PolyBackendDesc *old_be = poly_backend_get(h->domain);
      if (old_be) old_be->get_allocator()->free(h->ptr,
                    old_be->get_allocator()->dev_ctx);
    }

    *h = (PolyBufferHandle){
      .ptr = dptr, .nbytes = nbytes,
      .domain = resolved, .owned = true,
    };
  }

  /* Migrate training state moment handles if they exist */
  if (inst->train) {
    TrainState *ts = inst->train;
    for (int i = 0; i < ts->n_moment_bufs; i++) {
      /* m handles */
      if (ts->m_handles[i].domain != resolved) {
        void *mp = alloc->alloc(ts->m_handles[i].nbytes, alloc->dev_ctx);
        if (mp) {
          if (ts->m_datas[i])
            alloc->copy_in(mp, ts->m_datas[i], ts->m_handles[i].nbytes, alloc->dev_ctx);
          ts->m_handles[i] = (PolyBufferHandle){
            .ptr = mp, .nbytes = ts->m_handles[i].nbytes,
            .domain = resolved, .owned = true,
          };
        }
      }
      /* v handles */
      if (ts->v_handles[i].domain != resolved) {
        void *vp = alloc->alloc(ts->v_handles[i].nbytes, alloc->dev_ctx);
        if (vp) {
          if (ts->v_datas[i])
            alloc->copy_in(vp, ts->v_datas[i], ts->v_handles[i].nbytes, alloc->dev_ctx);
          ts->v_handles[i] = (PolyBufferHandle){
            .ptr = vp, .nbytes = ts->v_handles[i].nbytes,
            .domain = resolved, .owned = true,
          };
        }
      }
    }
    /* Migrate bc scalar handles */
    if (ts->bc1_handle.domain != resolved) {
      void *bp1 = alloc->alloc(sizeof(float), alloc->dev_ctx);
      if (bp1) {
        alloc->copy_in(bp1, &ts->bc1_data, sizeof(float), alloc->dev_ctx);
        ts->bc1_handle = (PolyBufferHandle){
          .ptr = bp1, .nbytes = sizeof(float), .domain = resolved, .owned = true,
        };
      }
      void *bp2 = alloc->alloc(sizeof(float), alloc->dev_ctx);
      if (bp2) {
        alloc->copy_in(bp2, &ts->bc2_data, sizeof(float), alloc->dev_ctx);
        ts->bc2_handle = (PolyBufferHandle){
          .ptr = bp2, .nbytes = sizeof(float), .domain = resolved, .owned = true,
        };
      }
    }
  }

  return 0;
}

/* ── Generic entrypoint execution ────────────────────────────────────── */

/* Build PolyBufferBinding[] from instance named buffers + IO overrides. */
static PolyBufferBinding *build_bindings_for_realize(
    PolyInstance *inst, int ep_idx, PolyIOBinding *io, int n_io, int *n_out) {
  /* Start with all instance buffers */
  int n = inst->n_bufs;
  PolyBufferBinding *bindings = calloc((size_t)n, sizeof(PolyBufferBinding));
  if (!bindings) { *n_out = 0; return NULL; }

  for (int i = 0; i < n; i++) {
    bindings[i].buffer = inst->bufs[i].buffer;
    bindings[i].handle = inst->buf_handles[i];
  }

  /* Override with IO bindings by name.
   * For device-memory domains, upload host data into existing device buffer.
   * For CPU/INTERP/WASM, point directly at host memory (zero-copy). */
  for (int i = 0; i < n_io; i++) {
    if (!io[i].data) continue;
    int bi = find_buf_by_name(inst, io[i].name);
    if (bi < 0) continue;
    if (bindings[bi].handle.domain != POLY_DEVICE_CPU) {
      /* Upload host IO data into device buffer */
      const PolyBackendDesc *bd = poly_backend_get(bindings[bi].handle.domain);
      const PolyAllocator *a = bd ? bd->get_allocator() : NULL;
      if (a && a->copy_in) {
        size_t nbytes = bindings[bi].handle.nbytes;
        a->copy_in(bindings[bi].handle.ptr, io[i].data, nbytes, a->dev_ctx);
      }
      /* handle.ptr stays as device pointer */
    } else {
      bindings[bi].handle.ptr = io[i].data;
    }
  }

  *n_out = n;
  return bindings;
}

/* Extended version: instance buffers + IO overrides + extra buffer/handle pairs.
 * Used by value_and_grad to include vag output buffers in the bindings. */
static PolyBufferBinding *build_bindings_extended(
    PolyInstance *inst, int ep_idx, PolyIOBinding *io, int n_io,
    PolyUOp **extra_bufs, PolyBufferHandle *extra_handles, int n_extra,
    int *n_out) {
  int n_base = 0;
  PolyBufferBinding *base = build_bindings_for_realize(inst, ep_idx, io, n_io, &n_base);
  if (!base) { *n_out = 0; return NULL; }

  int n_total = n_base + n_extra;
  PolyBufferBinding *all = realloc(base, (size_t)n_total * sizeof(PolyBufferBinding));
  if (!all) { free(base); *n_out = 0; return NULL; }

  for (int i = 0; i < n_extra; i++) {
    all[n_base + i].buffer = extra_bufs[i];
    all[n_base + i].handle = extra_handles[i];
  }

  *n_out = n_total;
  return all;
}

int poly_instance_call(PolyInstance *inst, const char *entrypoint,
                       PolyIOBinding *io, int n_io) {
  if (!inst || !entrypoint) return -1;

  int ep_idx = find_entrypoint(inst, entrypoint);
  if (ep_idx < 0) {
    fprintf(stderr, "poly_instance_call: no '%s' entrypoint\n", entrypoint);
    return -1;
  }

  PolyUOp *sink = inst->entrypoints[ep_idx].sink;
  PolyDeviceId device = inst->buf_handles[0].domain;

  /* Build cached slot table on first call (or after invalidation) */
  SlotCache *c = inst->call_cache;
  if (!c) {
    inst->call_cache = slot_cache_build(inst, sink, device, NULL, NULL, 0);
    c = inst->call_cache;
    if (!c) return -1;
  }

  /* Patch IO slots by name */
  for (int i = 0; i < n_io; i++) {
    if (!io[i].data || !io[i].name) continue;
    int bi = find_buf_by_name(inst, io[i].name);
    if (bi < 0) continue;
    if (inst->buf_handles[bi].domain != POLY_DEVICE_CPU) {
      const PolyBackendDesc *bd = poly_backend_get(inst->buf_handles[bi].domain);
      const PolyAllocator *a = bd ? bd->get_allocator() : NULL;
      if (a && a->copy_in)
        a->copy_in(inst->buf_handles[bi].ptr, io[i].data,
                   inst->buf_handles[bi].nbytes, a->dev_ctx);
    } else {
      for (int j = 0; j < c->n_io; j++) {
        if (c->io_buf_indices[j] == bi) {
          c->slot_data[c->io_slot_indices[j]] = io[i].data;
          break;
        }
      }
    }
  }

  return poly_compiled_plan_run(c->plan, c->slot_data, c->n_slots, NULL, 0);
}

/* ── Convenience wrapper ─────────────────────────────────────────────── */

int poly_instance_forward(PolyInstance *inst,
                          PolyIOBinding *inputs, int n_inputs) {
  return poly_instance_call(inst, "forward", inputs, n_inputs);
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

  /* Invalidate training state and slot cache (will be rebuilt lazily) */
  if (inst->train) {
    train_free(inst->train, inst->n_params);
    inst->train = NULL;
  }
  slot_cache_free(inst->train_cache);
  inst->train_cache = NULL;

  return 0;
}

/* ── Value and Grad ──────────────────────────────────────────────────── */

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
  if (inst->vag) return 0;  /* already built */

  PolyUOp *loss_sink = inst->entrypoints[loss_ep_idx].sink;
  PolyUOp *loss_store = loss_sink->src[0]; /* SINK src[0] = STORE */
  PolyUOp *loss_value = loss_store->src[1]; /* STORE src[1] = value */

  /* Build param buffer array */
  PolyUOp **param_bufs = malloc((size_t)inst->n_params * sizeof(PolyUOp *));
  if (!param_bufs) return -1;
  for (int i = 0; i < inst->n_params; i++)
    param_bufs[i] = inst->bufs[inst->param_indices[i]].buffer;

  /* Compute gradients */
  PolyUOp **grads = calloc((size_t)inst->n_params, sizeof(PolyUOp *));
  if (!grads) { free(param_bufs); return -1; }
  if (poly_grad_many(inst->ctx, loss_value, NULL, param_bufs, inst->n_params, grads) != 0) {
    fprintf(stderr, "poly_instance: value_and_grad: autograd failed\n");
    free(grads); free(param_bufs);
    return -1;
  }

  /* Allocate VagState */
  VagState *vag = calloc(1, sizeof(VagState));
  vag->grad_out_bufs = calloc((size_t)inst->n_params, sizeof(PolyUOp *));
  vag->grad_datas = calloc((size_t)inst->n_params, sizeof(float *));
  vag->grad_slots = calloc((size_t)inst->n_params, sizeof(int));
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
      free(stores); free(grads); free(param_bufs); vag_free(vag, inst->n_params);
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
      int64_t flat_shape[1] = { numel };
      gflat = poly_reshape(inst->ctx, grads[i], flat_shape, 1);
    }
    if (gs.dims) free(gs.dims);
    stores[i + 1] = poly_store_val(inst->ctx, gbuf, gflat);

    /* Allocate host storage for gradient data */
    NamedBuf *pb = &inst->bufs[inst->param_indices[i]];
    vag->grad_datas[i] = calloc((size_t)pb->numel, sizeof(float));
  }

  vag->combined_sink = poly_sink_n(inst->ctx, stores, n_stores);
  vag->slots_resolved = false;

  free(stores);
  free(grads);
  free(param_bufs);

  inst->vag = vag;
  return 0;
}

/* Resolve loss/grad buf_slot indices from a prepared step.
 * Called once after the first prepare_step for this vag graph. */
static int resolve_vag_slots(PolyInstance *inst, const PolySchedule *prep) {
  VagState *vag = inst->vag;
  if (vag->slots_resolved) return 0;

  /* Find loss slot */
  vag->loss_slot = -1;
  for (int s = 0; s < prep->n_buf_slots; s++) {
    if (prep->buf_slots[s].buf_uop == vag->loss_out_buf) {
      vag->loss_slot = s;
      break;
    }
  }
  if (vag->loss_slot < 0) {
    fprintf(stderr, "poly_instance: value_and_grad: loss buffer slot not found\n");
    return -1;
  }

  /* Find grad slots */
  for (int i = 0; i < inst->n_params; i++) {
    vag->grad_slots[i] = -1;
    for (int s = 0; s < prep->n_buf_slots; s++) {
      if (prep->buf_slots[s].buf_uop == vag->grad_out_bufs[i]) {
        vag->grad_slots[i] = s;
        break;
      }
    }
    if (vag->grad_slots[i] < 0) {
      fprintf(stderr, "poly_instance: value_and_grad: grad[%d] buffer slot not found\n", i);
      return -1;
    }
  }

  vag->slots_resolved = true;
  return 0;
}

int poly_instance_value_and_grad(PolyInstance *inst, const char *entrypoint,
                                 PolyIOBinding *io, int n_io,
                                 float *loss_out) {
  if (!inst || !entrypoint) return -1;

  int ep_idx = find_entrypoint(inst, entrypoint);
  if (ep_idx < 0) {
    fprintf(stderr, "poly_instance_value_and_grad: no '%s' entrypoint\n", entrypoint);
    return -1;
  }

  /* Build combined fwd+bwd graph lazily */
  if (ensure_vag_graph(inst, ep_idx) != 0) return -1;
  VagState *vag = inst->vag;

  /* Build extra bindings for vag output buffers (loss + grads) */
  int n_extra = 1 + inst->n_params;
  PolyUOp **extra_bufs = malloc((size_t)n_extra * sizeof(PolyUOp *));
  PolyBufferHandle *extra_handles = malloc((size_t)n_extra * sizeof(PolyBufferHandle));
  if (!extra_bufs || !extra_handles) {
    free(extra_bufs); free(extra_handles);
    return -1;
  }

  /* Loss output buffer -- always host-resident for readback */
  extra_bufs[0] = vag->loss_out_buf;
  extra_handles[0] = (PolyBufferHandle){
    .ptr = &vag->loss_data, .nbytes = sizeof(float),
    .domain = POLY_DEVICE_CPU, .owned = false,
  };

  /* Gradient output buffers -- host-resident for readback */
  for (int i = 0; i < inst->n_params; i++) {
    NamedBuf *pb = &inst->bufs[inst->param_indices[i]];
    extra_bufs[1 + i] = vag->grad_out_bufs[i];
    extra_handles[1 + i] = (PolyBufferHandle){
      .ptr = vag->grad_datas[i],
      .nbytes = (size_t)pb->numel * sizeof(float),
      .domain = POLY_DEVICE_CPU, .owned = false,
    };
  }

  /* Build combined bindings: instance buffers + IO + vag outputs */
  int n_bindings = 0;
  PolyBufferBinding *bindings = build_bindings_extended(
    inst, ep_idx, io, n_io, extra_bufs, extra_handles, n_extra, &n_bindings);
  free(extra_bufs);
  free(extra_handles);
  if (!bindings) return -1;

  /* Route through core poly_realize -- caching is automatic */
  int ret = poly_realize(inst->ctx, vag->combined_sink, bindings, n_bindings);
  free(bindings);
  if (ret != 0) return ret;

  if (loss_out) *loss_out = vag->loss_data;
  return 0;
}

/* ── Optimizer Graph Builder ─────────────────────────────────────────── */

/* Build optimizer UOp graph (fwd+bwd+optimizer as a single combined SINK).
 * Gradients are consumed directly by ASSIGN ops -- not materialized to
 * separate output buffers (D1: no grad stores in optimizer SINK). */
static int ensure_train_graph(PolyInstance *inst, int loss_ep_idx) {
  if (inst->train) return 0;  /* already built */

  /* Build fwd+bwd first (gives us loss_value and grad UOps) */
  if (ensure_vag_graph(inst, loss_ep_idx) != 0) return -1;
  VagState *vag = inst->vag;

  PolyCtx *ctx = inst->ctx;
  OptimState *o = &inst->optim;
  int np = inst->n_params;

  TrainState *ts = calloc(1, sizeof(TrainState));
  if (!ts) return -1;

  /* Loss output buffer (1 scalar, same as vag) */
  PolyDType out_dt = poly_dtype_scalar(vag->loss_value->dtype);
  if (!poly_dtype_is_float(out_dt)) out_dt = POLY_FLOAT32;
  ts->loss_out_buf = poly_buffer(ctx, out_dt, 1);

  /* Count SINK sources: loss_store + param assigns + moment assigns */
  int has_moments = (o->kind == POLY_OPTIM_ADAM || o->kind == POLY_OPTIM_ADAMW);
  int n_sink_srcs = 1 + np;  /* loss_store + param assigns */
  if (has_moments) n_sink_srcs += 2 * np;  /* + m assigns + v assigns */

  PolyUOp **sink_srcs = calloc((size_t)n_sink_srcs, sizeof(PolyUOp *));
  if (!sink_srcs) { train_free(ts, np); return -1; }

  /* Loss store */
  PolyUOp *loss_flat = vag->loss_value;
  if (uop_numel(ctx, vag->loss_value) != 1) {
    int64_t one_shape[1] = {1};
    loss_flat = poly_reshape(ctx, vag->loss_value, one_shape, 1);
  }
  sink_srcs[0] = poly_store_val(ctx, ts->loss_out_buf, loss_flat);

  /* Allocate moment buffers for Adam/AdamW */
  if (has_moments) {
    ts->m_bufs = calloc((size_t)np, sizeof(PolyUOp *));
    ts->v_bufs = calloc((size_t)np, sizeof(PolyUOp *));
    ts->m_datas = calloc((size_t)np, sizeof(float *));
    ts->v_datas = calloc((size_t)np, sizeof(float *));
    ts->m_handles = calloc((size_t)np, sizeof(PolyBufferHandle));
    ts->v_handles = calloc((size_t)np, sizeof(PolyBufferHandle));
    ts->n_moment_bufs = np;

    /* Bias correction scalar buffers */
    ts->bc1_buf = poly_buffer(ctx, POLY_FLOAT32, 1);
    ts->bc2_buf = poly_buffer(ctx, POLY_FLOAT32, 1);
    ts->bc1_data = 1.0f;  /* will be updated before each step */
    ts->bc2_data = 1.0f;
    ts->bc1_handle = (PolyBufferHandle){
      .ptr = &ts->bc1_data, .nbytes = sizeof(float),
      .domain = POLY_DEVICE_CPU, .owned = false,
    };
    ts->bc2_handle = (PolyBufferHandle){
      .ptr = &ts->bc2_data, .nbytes = sizeof(float),
      .domain = POLY_DEVICE_CPU, .owned = false,
    };

    for (int i = 0; i < np; i++) {
      NamedBuf *pb = &inst->bufs[inst->param_indices[i]];
      int64_t numel = pb->numel;

      ts->m_bufs[i] = poly_buffer(ctx, POLY_FLOAT32, numel);
      ts->v_bufs[i] = poly_buffer(ctx, POLY_FLOAT32, numel);

      ts->m_datas[i] = calloc((size_t)numel, sizeof(float));
      ts->v_datas[i] = calloc((size_t)numel, sizeof(float));

      size_t nbytes = (size_t)numel * sizeof(float);
      ts->m_handles[i] = (PolyBufferHandle){
        .ptr = ts->m_datas[i], .nbytes = nbytes,
        .domain = POLY_DEVICE_CPU, .owned = false,
      };
      ts->v_handles[i] = (PolyBufferHandle){
        .ptr = ts->v_datas[i], .nbytes = nbytes,
        .domain = POLY_DEVICE_CPU, .owned = false,
      };
    }
  }

  /* Build optimizer update graph for each parameter */
  PolyUOp *lr_const = poly_const_float(ctx, (double)o->lr);
  int si = 1;  /* sink_srcs index (0 = loss_store) */

  for (int i = 0; i < np; i++) {
    PolyUOp *param_buf = inst->bufs[inst->param_indices[i]].buffer;
    PolyUOp *grad = vag->grad_uops[i];

    switch (o->kind) {
    case POLY_OPTIM_SGD: {
      /* p_new = p - lr * grad */
      PolyUOp *update = poly_alu2(ctx, POLY_OP_MUL, lr_const, grad);
      PolyUOp *p_new = poly_alu2(ctx, POLY_OP_SUB, param_buf, update);
      sink_srcs[si++] = poly_assign(ctx, param_buf, p_new);
      break;
    }
    case POLY_OPTIM_ADAM:
    case POLY_OPTIM_ADAMW: {
      PolyUOp *m_buf = ts->m_bufs[i];
      PolyUOp *v_buf = ts->v_bufs[i];
      NamedBuf *pb = &inst->bufs[inst->param_indices[i]];
      int64_t numel = pb->numel;

      /* Expand scalar bc buffers to match param shape */
      int64_t param_shape[1] = { numel };
      PolyUOp *bc1_expanded = poly_expand(ctx, ts->bc1_buf, param_shape, 1);
      PolyUOp *bc2_expanded = poly_expand(ctx, ts->bc2_buf, param_shape, 1);

      PolyUOp *beta1 = poly_const_float(ctx, (double)o->beta1);
      PolyUOp *beta2 = poly_const_float(ctx, (double)o->beta2);
      PolyUOp *one_minus_b1 = poly_const_float(ctx, 1.0 - (double)o->beta1);
      PolyUOp *one_minus_b2 = poly_const_float(ctx, 1.0 - (double)o->beta2);
      PolyUOp *eps = poly_const_float(ctx, (double)o->eps);

      /* AdamW: decoupled weight decay on param first */
      PolyUOp *p_cur = param_buf;
      if (o->kind == POLY_OPTIM_ADAMW && o->weight_decay > 0.0f) {
        PolyUOp *wd_factor = poly_const_float(ctx,
          1.0 - (double)o->lr * (double)o->weight_decay);
        p_cur = poly_alu2(ctx, POLY_OP_MUL, param_buf, wd_factor);
      }

      /* m_new = beta1 * m + (1 - beta1) * grad */
      PolyUOp *m_new = poly_alu2(ctx, POLY_OP_ADD,
        poly_alu2(ctx, POLY_OP_MUL, beta1, m_buf),
        poly_alu2(ctx, POLY_OP_MUL, one_minus_b1, grad));

      /* v_new = beta2 * v + (1 - beta2) * grad * grad */
      PolyUOp *g_sq = poly_alu2(ctx, POLY_OP_MUL, grad, grad);
      PolyUOp *v_new = poly_alu2(ctx, POLY_OP_ADD,
        poly_alu2(ctx, POLY_OP_MUL, beta2, v_buf),
        poly_alu2(ctx, POLY_OP_MUL, one_minus_b2, g_sq));

      /* Bias-corrected: m_hat = m_new * bc1, v_hat = v_new * bc2 */
      PolyUOp *m_hat = poly_alu2(ctx, POLY_OP_MUL, m_new, bc1_expanded);
      PolyUOp *v_hat = poly_alu2(ctx, POLY_OP_MUL, v_new, bc2_expanded);

      /* p_new = p_cur - lr * m_hat / (sqrt(v_hat) + eps) */
      PolyUOp *v_sqrt = poly_alu1(ctx, POLY_OP_SQRT, v_hat);
      PolyUOp *denom = poly_alu2(ctx, POLY_OP_ADD, v_sqrt, eps);
      PolyUOp *step_val = poly_alu2(ctx, POLY_OP_MUL, lr_const,
        poly_alu2(ctx, POLY_OP_MUL, m_hat,
          poly_alu1(ctx, POLY_OP_RECIPROCAL, denom)));
      PolyUOp *p_new = poly_alu2(ctx, POLY_OP_SUB, p_cur, step_val);

      /* ASSIGN all three: param, m, v */
      sink_srcs[si++] = poly_assign(ctx, param_buf, p_new);
      sink_srcs[1 + np + 2*i] = poly_assign(ctx, m_buf, m_new);
      sink_srcs[1 + np + 2*i + 1] = poly_assign(ctx, v_buf, v_new);
      break;
    }
    default:
      fprintf(stderr, "ensure_train_graph: unsupported optimizer %d\n", o->kind);
      free(sink_srcs); train_free(ts, np);
      return -1;
    }
  }

  /* For Adam/AdamW, si covered param assigns (1..np), moment assigns
   * were written directly to their positions. Verify: */
  if (has_moments) {
    /* param assigns: indices 1..np (written by si++)
     * moment assigns: indices (1+np)..(1+np+2*np-1) (written directly) */
  }

  ts->combined_sink = poly_sink_n(ctx, sink_srcs, n_sink_srcs);
  free(sink_srcs);

  inst->train = ts;
  return 0;
}

/* ── Train Step ──────────────────────────────────────────────────────── */

int poly_instance_train_step(PolyInstance *inst,
                             PolyIOBinding *io, int n_io,
                             float *loss_out) {
  if (!inst) return -1;
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
  int np = inst->n_params;

  /* Update bias correction scalars (Adam/AdamW) */
  o->step++;
  if (o->kind == POLY_OPTIM_ADAM || o->kind == POLY_OPTIM_ADAMW) {
    float bc1 = 1.0f / (1.0f - powf(o->beta1, (float)o->step));
    float bc2 = 1.0f / (1.0f - powf(o->beta2, (float)o->step));

    /* Device-aware update (D2) */
    if (ts->bc1_handle.domain != POLY_DEVICE_CPU) {
      const PolyBackendDesc *bd = poly_backend_get(ts->bc1_handle.domain);
      const PolyAllocator *a = bd ? bd->get_allocator() : NULL;
      if (a && a->copy_in) {
        a->copy_in(ts->bc1_handle.ptr, &bc1, sizeof(float), a->dev_ctx);
        a->copy_in(ts->bc2_handle.ptr, &bc2, sizeof(float), a->dev_ctx);
      }
    } else {
      ts->bc1_data = bc1;
      ts->bc2_data = bc2;
    }
  }

  /* Build cached slot table on first call */
  SlotCache *c = inst->train_cache;
  if (!c) {
    int n_extra = 1;  /* loss output */
    if (o->kind == POLY_OPTIM_ADAM || o->kind == POLY_OPTIM_ADAMW)
      n_extra += 2 * np + 2;  /* m + v + bc1 + bc2 */

    PolyUOp **extra_bufs = malloc((size_t)n_extra * sizeof(PolyUOp *));
    void **extra_ptrs = malloc((size_t)n_extra * sizeof(void *));
    if (!extra_bufs || !extra_ptrs) {
      free(extra_bufs); free(extra_ptrs);
      o->step--;
      return -1;
    }

    extra_bufs[0] = ts->loss_out_buf;
    extra_ptrs[0] = &ts->loss_data;
    if (o->kind == POLY_OPTIM_ADAM || o->kind == POLY_OPTIM_ADAMW) {
      for (int i = 0; i < np; i++) {
        extra_bufs[1 + i] = ts->m_bufs[i];
        extra_ptrs[1 + i] = ts->m_handles[i].ptr;
        extra_bufs[1 + np + i] = ts->v_bufs[i];
        extra_ptrs[1 + np + i] = ts->v_handles[i].ptr;
      }
      extra_bufs[1 + 2*np] = ts->bc1_buf;
      extra_ptrs[1 + 2*np] = &ts->bc1_data;
      extra_bufs[1 + 2*np + 1] = ts->bc2_buf;
      extra_ptrs[1 + 2*np + 1] = &ts->bc2_data;
    }

    PolyDeviceId device = inst->buf_handles[0].domain;
    inst->train_cache = slot_cache_build(inst, ts->combined_sink, device,
                                          extra_bufs, extra_ptrs, n_extra);
    free(extra_bufs);
    free(extra_ptrs);
    c = inst->train_cache;
    if (!c) { o->step--; return -1; }
  }

  /* Patch IO slots */
  for (int i = 0; i < n_io; i++) {
    if (!io[i].data || !io[i].name) continue;
    int bi = find_buf_by_name(inst, io[i].name);
    if (bi < 0) continue;
    if (inst->buf_handles[bi].domain != POLY_DEVICE_CPU) {
      const PolyBackendDesc *bd = poly_backend_get(inst->buf_handles[bi].domain);
      const PolyAllocator *a = bd ? bd->get_allocator() : NULL;
      if (a && a->copy_in)
        a->copy_in(inst->buf_handles[bi].ptr, io[i].data,
                   inst->buf_handles[bi].nbytes, a->dev_ctx);
    } else {
      for (int j = 0; j < c->n_io; j++) {
        if (c->io_buf_indices[j] == bi) {
          c->slot_data[c->io_slot_indices[j]] = io[i].data;
          break;
        }
      }
    }
  }

  int ret = poly_compiled_plan_run(c->plan, c->slot_data, c->n_slots, NULL, 0);
  if (ret != 0) { o->step--; return ret; }

  /* Read back loss */
  if (loss_out) *loss_out = ts->loss_data;

  /* Update instance "loss" named buffer for consumers */
  int loss_named_idx = find_buf_by_name(inst, "loss");
  if (loss_named_idx >= 0 && inst->bufs[loss_named_idx].data)
    inst->bufs[loss_named_idx].data[0] = ts->loss_data;

  return 0;
}
