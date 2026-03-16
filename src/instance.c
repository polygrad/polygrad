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
  float *m;         /* concatenated first moments (all params) */
  float *v;         /* concatenated second moments (all params) */
  int64_t total_numel; /* total param elements */
} OptimState;

/* ── Execution cache entry types ─────────────────────────────────────── */

typedef struct {
  int ep_idx;
  PolyCompileMode mode;
  PolySchedule *prep;
} PrepCacheEntry;

typedef struct {
  int ep_idx;
  PolyCompileMode mode;
  PolyDeviceId device;
  PolyCompiledPlan *exec;
} ExecCacheEntry;

/* ── Value-and-grad metadata (built lazily on first train call) ──────── */

typedef struct {
  PolyUOp *combined_sink;           /* combined fwd+bwd SINK */
  PolyUOp *loss_out_buf;            /* BUFFER UOp for loss output */
  PolyUOp **grad_out_bufs;          /* [n_params] gradient BUFFER UOps */
  float **grad_datas;               /* [n_params] gradient host data */
  float loss_data;                  /* scalar loss value */

  /* Cached slot indices (set after first prepare_step) */
  int loss_slot;                    /* buf_slot index for loss output */
  int *grad_slots;                  /* [n_params] buf_slot indices for grads */
  bool slots_resolved;              /* true after first resolution */
} VagState;

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

  /* ── Legacy fields (used by value_and_grad, to be migrated) ── */
  PolyDeviceId device;
  const PolyAllocator *allocator;
  PrepCacheEntry *prep_cache;
  int n_prep_cached;
  int prep_cache_cap;
  ExecCacheEntry *exec_cache;
  int n_exec_cached;
  int exec_cache_cap;

  /* ── Value-and-grad state (lazy, per-entrypoint -- currently only "loss") ── */
  VagState *vag;                    /* NULL until first value_and_grad call */

  /* ── Optimizer state ── */
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

/* ── Cache helpers ───────────────────────────────────────────────────── */

static PolySchedule *prep_cache_lookup(const PolyInstance *inst,
                                            int ep_idx, PolyCompileMode mode) {
  for (int i = 0; i < inst->n_prep_cached; i++)
    if (inst->prep_cache[i].ep_idx == ep_idx &&
        inst->prep_cache[i].mode == mode)
      return inst->prep_cache[i].prep;
  return NULL;
}

static void prep_cache_insert(PolyInstance *inst, int ep_idx,
                               PolyCompileMode mode, PolySchedule *prep) {
  if (inst->n_prep_cached >= inst->prep_cache_cap) {
    int new_cap = inst->prep_cache_cap ? inst->prep_cache_cap * 2 : 4;
    inst->prep_cache = realloc(inst->prep_cache, (size_t)new_cap * sizeof(PrepCacheEntry));
    inst->prep_cache_cap = new_cap;
  }
  inst->prep_cache[inst->n_prep_cached++] = (PrepCacheEntry){
    .ep_idx = ep_idx, .mode = mode, .prep = prep
  };
}

static PolyCompiledPlan *exec_cache_lookup(const PolyInstance *inst,
                                              int ep_idx, PolyCompileMode mode,
                                              PolyDeviceId device) {
  for (int i = 0; i < inst->n_exec_cached; i++)
    if (inst->exec_cache[i].ep_idx == ep_idx &&
        inst->exec_cache[i].mode == mode &&
        inst->exec_cache[i].device == device)
      return inst->exec_cache[i].exec;
  return NULL;
}

static void exec_cache_insert(PolyInstance *inst, int ep_idx,
                               PolyCompileMode mode, PolyDeviceId device,
                               PolyCompiledPlan *exec) {
  if (inst->n_exec_cached >= inst->exec_cache_cap) {
    int new_cap = inst->exec_cache_cap ? inst->exec_cache_cap * 2 : 4;
    inst->exec_cache = realloc(inst->exec_cache, (size_t)new_cap * sizeof(ExecCacheEntry));
    inst->exec_cache_cap = new_cap;
  }
  inst->exec_cache[inst->n_exec_cached++] = (ExecCacheEntry){
    .ep_idx = ep_idx, .mode = mode, .device = device, .exec = exec
  };
}

/* ── Slot-data builder ───────────────────────────────────────────────── */
/*
 * Build the slot_data[] array for poly_compiled_plan_run().
 * Maps each prepared step buf_slot to a host data pointer:
 *   - External slots: find the instance buffer whose buf_uop matches
 *   - IO bindings override by name
 *   - Intermediate slots: NULL (the executable step owns those)
 *   - Extra buffers (loss/grad): matched by buf_uop pointer
 */

static void **build_slot_data(PolyInstance *inst,
                               const PolySchedule *prep,
                               PolyIOBinding *io, int n_io,
                               /* extra buf_uop -> data mappings */
                               PolyUOp **extra_uops, float **extra_datas, int n_extra) {
  void **slot_data = calloc((size_t)prep->n_buf_slots, sizeof(void *));
  if (!slot_data) return NULL;

  for (int s = 0; s < prep->n_buf_slots; s++) {
    if (prep->buf_slots[s].is_intermediate) continue;

    PolyUOp *slot_uop = prep->buf_slots[s].buf_uop;

    /* Check extra mappings first (loss/grad buffers) */
    bool found = false;
    for (int e = 0; e < n_extra; e++) {
      if (extra_uops[e] == slot_uop) {
        slot_data[s] = extra_datas[e];
        found = true;
        break;
      }
    }
    if (found) continue;

    /* Match to instance buffer by buf_uop pointer */
    for (int b = 0; b < inst->n_bufs; b++) {
      if (inst->bufs[b].buffer == slot_uop) {
        slot_data[s] = inst->bufs[b].data;
        break;
      }
    }
  }

  /* Override with IO bindings by name */
  for (int i = 0; i < n_io; i++) {
    if (!io[i].data) continue;
    int bi = find_buf_by_name(inst, io[i].name);
    if (bi < 0) continue;
    PolyUOp *buf_uop = inst->bufs[bi].buffer;
    for (int s = 0; s < prep->n_buf_slots; s++) {
      if (prep->buf_slots[s].buf_uop == buf_uop) {
        slot_data[s] = io[i].data;
        break;
      }
    }
  }

  return slot_data;
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

  /* Default device and allocator */
#ifdef __EMSCRIPTEN__
  inst->device = POLY_DEVICE_WASM_JIT;
#else
  inst->device = POLY_DEVICE_CPU;
#endif
  inst->allocator = &POLY_CPU_ALLOCATOR;

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
  free(vag->grad_slots);
  free(vag);
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
   * then prepared cache, then context (arena-frees all UOps). */
  for (int i = 0; i < inst->n_exec_cached; i++)
    poly_compiled_plan_free(inst->exec_cache[i].exec);
  free(inst->exec_cache);

  for (int i = 0; i < inst->n_prep_cached; i++)
    poly_schedule_free(inst->prep_cache[i].prep);
  free(inst->prep_cache);

  /* Free value-and-grad state */
  vag_free(inst->vag, inst->n_params);

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
  int bi = inst->param_indices[i];
  NamedBuf *b = &inst->bufs[bi];
  if (numel_out) *numel_out = b->numel;
  /* Device-memory backends: host data may be stale */
  if (inst->buf_handles && inst->buf_handles[bi].domain == POLY_DEVICE_CUDA)
    return NULL;
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

  /* Legacy fields (used by value_and_grad path) */
  inst->device = resolved;
  inst->allocator = alloc;

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
   * IO data is host memory -- for device-memory domains, the core handles
   * upload inside poly_realize (future: explicit upload). For now, IO
   * overrides point directly at host memory (works for CPU/INTERP/WASM). */
  for (int i = 0; i < n_io; i++) {
    if (!io[i].data) continue;
    int bi = find_buf_by_name(inst, io[i].name);
    if (bi < 0) continue;
    bindings[bi].handle.ptr = io[i].data;
  }

  *n_out = n;
  return bindings;
}

int poly_instance_call(PolyInstance *inst, const char *entrypoint,
                       PolyIOBinding *io, int n_io) {
  if (!inst || !entrypoint) return -1;

  int ep_idx = find_entrypoint(inst, entrypoint);
  if (ep_idx < 0) {
    fprintf(stderr, "poly_instance_call: no '%s' entrypoint\n", entrypoint);
    return -1;
  }

  /* Build bindings from instance buffers + IO overrides */
  int n_bindings = 0;
  PolyBufferBinding *bindings = build_bindings_for_realize(
    inst, ep_idx, io, n_io, &n_bindings);
  if (!bindings) return -1;

  /* Call the core -- device inferred, caching automatic */
  int ret = poly_realize(inst->ctx, inst->entrypoints[ep_idx].sink,
                          bindings, n_bindings);
  free(bindings);
  return ret;
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

  /* Use a synthetic cache key: ep_idx with VALUE_AND_GRAD mode.
   * The combined_sink is different from the original entrypoint sink,
   * so we use a dedicated cache entry. */

  /* Lookup or build prepared step for the combined graph */
  PolySchedule *prep = prep_cache_lookup(inst, ep_idx, POLY_MODE_VALUE_AND_GRAD);
  if (!prep) {
    prep = poly_schedule_for(inst->ctx, vag->combined_sink, POLY_MODE_CALL);
    if (!prep) {
      fprintf(stderr, "poly_instance_value_and_grad: prepare failed\n");
      return -1;
    }
    prep_cache_insert(inst, ep_idx, POLY_MODE_VALUE_AND_GRAD, prep);

    /* Resolve slot indices on first prepare */
    if (resolve_vag_slots(inst, prep) != 0) return -1;
  }

  /* Lookup or build executable step */
  PolyCompiledPlan *exec = exec_cache_lookup(inst, ep_idx,
                                                POLY_MODE_VALUE_AND_GRAD,
                                                inst->device);
  if (!exec) {
    exec = poly_compile_schedule(inst->ctx, prep, inst->device);
    if (!exec) {
      fprintf(stderr, "poly_instance_value_and_grad: lower failed\n");
      return -1;
    }
    exec_cache_insert(inst, ep_idx, POLY_MODE_VALUE_AND_GRAD,
                       inst->device, exec);
  }

  /* Build extra buf_uop -> data mappings for loss and grad outputs */
  int n_extra = 1 + inst->n_params;
  PolyUOp **extra_uops = malloc((size_t)n_extra * sizeof(PolyUOp *));
  float **extra_datas = malloc((size_t)n_extra * sizeof(float *));
  extra_uops[0] = vag->loss_out_buf;
  extra_datas[0] = &vag->loss_data;
  for (int i = 0; i < inst->n_params; i++) {
    extra_uops[1 + i] = vag->grad_out_bufs[i];
    extra_datas[1 + i] = vag->grad_datas[i];
  }

  void **slot_data = build_slot_data(inst, prep, io, n_io,
                                      extra_uops, extra_datas, n_extra);
  free(extra_uops);
  free(extra_datas);
  if (!slot_data) return -1;

  int ret = poly_compiled_plan_run(exec, slot_data, prep->n_buf_slots,
                                      NULL, 0);
  free(slot_data);
  if (ret != 0) return ret;

  if (loss_out) *loss_out = vag->loss_data;
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

  /* Run value_and_grad for the "loss" entrypoint */
  int ret = poly_instance_value_and_grad(inst, "loss", io, n_io, loss_out);
  if (ret != 0) return ret;

  /* Update instance "loss" buffer so consumers see the current value */
  int loss_named_idx = find_buf_by_name(inst, "loss");
  if (loss_named_idx >= 0 && inst->bufs[loss_named_idx].data)
    inst->bufs[loss_named_idx].data[0] = inst->vag->loss_data;

  /* Apply optimizer updates (host-side) */
  inst->optim.step++;
  int64_t moment_offset = 0;
  for (int i = 0; i < inst->n_params; i++) {
    NamedBuf *pb = &inst->bufs[inst->param_indices[i]];
    apply_optimizer_update(inst, i, inst->vag->grad_datas[i], pb->numel, moment_offset);
    moment_offset += pb->numel;
  }

  return 0;
}
