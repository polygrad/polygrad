/*
 * exec_plan.c -- Execution plan: prepare, lower, run, free
 *
 * Contains all exec_plan API implementations:
 *   - poly_schedule_for: backend-neutral scheduling
 *   - poly_compile_schedule: dispatches to backend-specific lowering via vtable
 *   - poly_compiled_plan_run: dispatches to backend-specific execution
 *   - poly_compiled_plan_free: cleanup via backend vtable
 *   - Backend descriptors for CPU, INTERP, WASM_JIT, CUDA
 *   - POLY_CPU_ALLOCATOR, POLY_CUDA_ALLOCATOR
 *
 * Compiled into both native and Emscripten builds. Backend availability
 * depends on build flags (POLY_HAS_CUDA, __EMSCRIPTEN__).
 */

#define _POSIX_C_SOURCE 200809L
#include "exec_plan.h"
#include "frontend_internal.h"
#include "codegen.h"
#include "interp.h"
#include "rangeify.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Prepared step construction ───────────────────────────────────────── */

PolySchedule *poly_schedule_for(PolyCtx *ctx, PolyUOp *sink,
                                    PolyCompileMode mode) {
  if (!sink || sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: prepare_step: expected SINK\n");
    return NULL;
  }

  /* --- Collect pre-strip buffer ordering -------------------------------- */
  PolyUOp **buf_order_orig = calloc(POLY_MAX_REALIZE_BUFS, sizeof(PolyUOp *));
  PolyUOp **dfs_visited = calloc(POLY_MAX_STRUCT_NODES, sizeof(PolyUOp *));
  int n_bufs_orig = 0, n_dfs = 0;
  poly_collect_buf_order(sink, buf_order_orig, &n_bufs_orig, dfs_visited, &n_dfs);

  /* Identify output buffers */
  PolyUOp *output_bufs[POLY_MAX_REALIZE_BUFS];
  int n_output_bufs = poly_collect_output_buffers_in_sink(sink, output_bufs, POLY_MAX_REALIZE_BUFS);
  (void)n_output_bufs;

  /* --- Extract BIND defaults and strip ---------------------------------- */
  PolyVarBinding bind_vals[16];
  int n_bind_vals = 0;
  sink = poly_strip_bind_values(ctx, sink, bind_vals, &n_bind_vals, 16, NULL, 0);

  /* Post-strip buffer ordering (scheduler sees these pointers) */
  PolyUOp **buf_order_post = calloc(POLY_MAX_REALIZE_BUFS, sizeof(PolyUOp *));
  int n_bufs_post = 0;
  n_dfs = 0;
  poly_collect_buf_order(sink, buf_order_post, &n_bufs_post, dfs_visited, &n_dfs);
  free(dfs_visited);

  uint32_t ghash = poly_structural_hash(sink) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);

  /* --- Schedule --------------------------------------------------------- */
  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  if (sr.n_kernels < 1) {
    free(buf_order_orig); free(buf_order_post);
    poly_schedule_result_free(&sr);
    return NULL;
  }

  /* --- Allocate prepared step ------------------------------------------- */
  PolySchedule *ps = calloc(1, sizeof(PolySchedule));
  if (!ps) {
    free(buf_order_orig); free(buf_order_post);
    poly_schedule_result_free(&sr);
    return NULL;
  }
  ps->mode = mode;
  ps->graph_hash = ghash;
  ps->loss_buf_slot = -1;

  /* --- Default variable bindings ---------------------------------------- */
  ps->n_default_vars = n_bind_vals;
  if (n_bind_vals > 0) {
    ps->default_vars = malloc((size_t)n_bind_vals * sizeof(PolyVarBinding));
    memcpy(ps->default_vars, bind_vals, (size_t)n_bind_vals * sizeof(PolyVarBinding));
  }

  /* --- Build buffer slot table ------------------------------------------ */
  int n_external = n_bufs_orig;
  int n_intermediate = sr.n_intermediates;
  ps->n_buf_slots = n_external + n_intermediate;

  if (ps->n_buf_slots > 0) {
    ps->buf_slots = calloc((size_t)ps->n_buf_slots, sizeof(PolyScheduleBufSlot));

    /* External buffer slots */
    for (int i = 0; i < n_external; i++) {
      PolyScheduleBufSlot *slot = &ps->buf_slots[i];
      PolyUOp *buf = buf_order_orig[i];
      slot->buf_uop = buf;
      slot->is_intermediate = false;
      slot->external_buf_idx = i;
      slot->dtype = buf ? poly_dtype_scalar(buf->dtype) : POLY_FLOAT32;
      slot->numel = (buf && buf->arg.kind == POLY_ARG_INT) ? buf->arg.i : 0;
      if (slot->numel > 0)
        slot->nbytes = slot->numel * poly_dtype_itemsize(slot->dtype);
    }

    /* Intermediate buffer slots */
    for (int b = 0; b < n_intermediate; b++) {
      PolyScheduleBufSlot *slot = &ps->buf_slots[n_external + b];
      slot->is_intermediate = true;
      slot->external_buf_idx = -1;
      if (sr.intermediate_buf_uops && sr.intermediate_buf_uops[b]) {
        slot->buf_uop = sr.intermediate_buf_uops[b];
        slot->dtype = poly_dtype_scalar(sr.intermediate_buf_uops[b]->dtype);
      } else {
        slot->dtype = POLY_FLOAT32;
      }
      slot->numel = sr.intermediate_sizes ? sr.intermediate_sizes[b] : 0;
      int itemsize = (sr.intermediate_itemsizes && sr.intermediate_itemsizes[b] > 0)
                       ? sr.intermediate_itemsizes[b] : (int)sizeof(float);
      slot->nbytes = slot->numel * itemsize;
    }
  }

  /* --- Build intermediate UOp -> slot index lookup ---------------------- */
  PolyMap *inter_set = NULL;
  if (n_intermediate > 0 && sr.intermediate_buf_uops) {
    inter_set = poly_map_new((size_t)(n_intermediate < 4 ? 4 : n_intermediate));
    for (int b = 0; b < n_intermediate; b++) {
      PolyUOp *ib = sr.intermediate_buf_uops[b];
      /* Store 1-based index to distinguish from NULL */
      poly_map_set(inter_set, poly_ptr_hash(ib), ib,
                   (PolyUOp *)(intptr_t)(n_external + b + 1), poly_ptr_eq);
    }
  }

  /* --- Build exec items ------------------------------------------------- */
  ps->n_items = sr.n_kernels;
  ps->items = calloc((size_t)sr.n_kernels, sizeof(PolyExecItem));

  for (int k = 0; k < sr.n_kernels; k++) {
    PolyExecItem *item = &ps->items[k];
    item->kind = POLY_EXEC_COMPUTE;
    item->root = sr.kernels[k];

    /* Map each PARAM to a buffer slot index */
    int np = sr.kernel_n_params[k];
    item->n_buf_slots = np;
    item->buf_slot_indices = malloc((size_t)np * sizeof(int));

    for (int i = 0; i < np; i++) {
      PolyUOp *buf = sr.param_to_buf[k][i];
      item->buf_slot_indices[i] = -1;

      if (buf->op == POLY_OP_BUFFER) {
        /* Try external buffer lookup (post-strip ordering) */
        int pos = poly_find_buf_position(buf, buf_order_post, n_bufs_post);
        if (pos >= 0) {
          item->buf_slot_indices[i] = pos;
        } else if (inter_set) {
          PolyUOp *v = poly_map_get(inter_set, poly_ptr_hash(buf), buf, poly_ptr_eq);
          if (v) item->buf_slot_indices[i] = (int)((intptr_t)v - 1);
        }
      }

      if (item->buf_slot_indices[i] < 0) {
        fprintf(stderr, "polygrad: prepare_step: unresolved param %d in kernel %d\n",
                i, k);
        if (inter_set) poly_map_destroy(inter_set);
        goto cleanup;
      }
    }
  }

  if (inter_set) poly_map_destroy(inter_set);

  /* --- Execution order -------------------------------------------------- */
  ps->exec_order = malloc((size_t)sr.n_kernels * sizeof(int));
  if (sr.exec_order)
    memcpy(ps->exec_order, sr.exec_order, (size_t)sr.n_kernels * sizeof(int));
  else
    for (int k = 0; k < sr.n_kernels; k++) ps->exec_order[k] = k;

  free(buf_order_orig);
  free(buf_order_post);
  poly_schedule_result_free(&sr);
  return ps;

cleanup:
  free(buf_order_orig);
  free(buf_order_post);
  poly_schedule_free(ps);
  poly_schedule_result_free(&sr);
  return NULL;
}

void poly_schedule_free(PolySchedule *step) {
  if (!step) return;
  for (int i = 0; i < step->n_items; i++)
    free(step->items[i].buf_slot_indices);
  free(step->items);
  free(step->buf_slots);
  free(step->exec_order);
  free(step->default_vars);
  free(step->grad_buf_slots);
  free(step);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Allocators                                                          */
/* ══════════════════════════════════════════════════════════════════════ */

/* ── CPU allocator ────────────────────────────────────────────────────── */

static void *cpu_alloc(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  return calloc(1, nbytes);
}

static void cpu_free_alloc(void *handle, void *dev_ctx) {
  (void)dev_ctx;
  free(handle);
}

static int cpu_copy_in(void *dst, const void *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst, src, n);
  return 0;
}

static int cpu_copy_out(void *dst, const void *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst, src, n);
  return 0;
}

static int cpu_copy_between(void *dst, const void *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst, src, n);
  return 0;
}

const PolyAllocator POLY_CPU_ALLOCATOR = {
  .alloc = cpu_alloc,
  .free = cpu_free_alloc,
  .copy_in = cpu_copy_in,
  .copy_out = cpu_copy_out,
  .copy_between = cpu_copy_between,
  .dev_ctx = NULL,
};

/* ── CUDA allocator ───────────────────────────────────────────────────── */

#ifdef POLY_HAS_CUDA

static void *cuda_alloc_fn(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  unsigned long long dptr = poly_cuda_alloc(nbytes);
  return dptr ? (void *)(uintptr_t)dptr : NULL;
}

static void cuda_free_fn(void *handle, void *dev_ctx) {
  (void)dev_ctx;
  if (handle) poly_cuda_free((unsigned long long)(uintptr_t)handle);
}

static int cuda_copy_in_fn(void *dst, const void *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_cuda_copy_htod((unsigned long long)(uintptr_t)dst, src, n);
}

static int cuda_copy_out_fn(void *dst, const void *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_cuda_copy_dtoh(dst, (unsigned long long)(uintptr_t)src, n);
}

static int cuda_copy_between_fn(void *dst, const void *src, size_t n, void *dev_ctx) {
  (void)dev_ctx; (void)dst; (void)src; (void)n;
  fprintf(stderr, "polygrad: cuda_copy_between: not implemented\n");
  return -1;
}

const PolyAllocator POLY_CUDA_ALLOCATOR = {
  .alloc = cuda_alloc_fn,
  .free = cuda_free_fn,
  .copy_in = cuda_copy_in_fn,
  .copy_out = cuda_copy_out_fn,
  .copy_between = cuda_copy_between_fn,
  .dev_ctx = NULL,
};

#endif /* POLY_HAS_CUDA */

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend-specific runner handle types                                 */
/* ══════════════════════════════════════════════════════════════════════ */

/* Interpreter: linearized UOp array */
typedef struct {
  PolyUOp **lin;
  int n_lin;
} InterpHandle;

/* CUDA: compiled program handle */
#ifdef POLY_HAS_CUDA
typedef struct {
  PolyCudaProgram *prog;
} CudaRunnerHandle;
#endif

/* WASM JIT: JS-side kernel cache index */
typedef struct {
  int kernel_id;
} WasmJitHandle;

/* ── WASM JIT EM_JS bridge (Emscripten only) ─────────────────────────── */

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

EM_JS(int, js_compile_wasm_kernel, (const uint8_t *bytes, int len), {
  var mod = new WebAssembly.Module(HEAPU8.subarray(bytes, bytes + len));
  var imports = {
    env: { memory: wasmMemory },
    math: {
      exp2f: function(x) { return Math.pow(2, x); },
      log2f: function(x) { return Math.log2(x); },
      sinf:  function(x) { return Math.sin(x); },
      powf:  function(x, y) { return Math.pow(x, y); }
    }
  };
  var inst = new WebAssembly.Instance(mod, imports);
  if (!Module._polyKernelCache) Module._polyKernelCache = [];
  Module._polyKernelCache.push(inst);
  return Module._polyKernelCache.length - 1;
});

EM_JS(int, js_exec_wasm_kernel, (int kernel_id, const int *args, int n_args), {
  var inst = Module._polyKernelCache[kernel_id];
  if (!inst) return -1;
  var params = [];
  for (var i = 0; i < n_args; i++) {
    params.push(HEAP32[(args >> 2) + i]);
  }
  inst.exports.kernel.apply(null, params);
  return 0;
});

EM_JS(void, js_free_wasm_kernel, (int kernel_id), {
  if (Module._polyKernelCache && kernel_id >= 0 &&
      kernel_id < Module._polyKernelCache.length) {
    Module._polyKernelCache[kernel_id] = null;
  }
});
#endif /* __EMSCRIPTEN__ */

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend implementations (lower_item / execute / free_runner)         */
/* ══════════════════════════════════════════════════════════════════════ */

/* ── CPU backend ──────────────────────────────────────────────────────── */

#ifndef __EMSCRIPTEN__

static int cpu_lower_item(PolyCtx *ctx, PolyUOp *scheduled_root,
                          const char *fn_name, PolyRunner *out) {
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, scheduled_root, &n_lin);
  if (!lin) return -1;

  char *src = poly_render_c(lin, n_lin, fn_name);
  free(lin);
  if (!src) return -1;

  if (getenv("POLY_DUMP_KERNELS"))
    fprintf(stderr, "=== LOWER KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

  PolyProgram *prog = poly_compile_c(src, fn_name);
  if (!prog) {
    fprintf(stderr, "=== FAILED LOWER KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);
    free(src);
    return -1;
  }
  free(src);

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = prog;
  out->handle_size = 0;
  return 0;
}

static int cpu_execute(PolyRunner *runner, void **args, int n_args) {
  (void)n_args;
  poly_program_call((PolyProgram *)runner->handle, args, n_args);
  return 0;
}

static void cpu_free_runner(PolyRunner *runner) {
  if (runner->handle) poly_program_destroy((PolyProgram *)runner->handle);
}

static const PolyAllocator *cpu_get_allocator(void) {
  return &POLY_CPU_ALLOCATOR;
}

#endif /* !__EMSCRIPTEN__ */

/* ── Interpreter backend ──────────────────────────────────────────────── */

static int interp_lower_item(PolyCtx *ctx, PolyUOp *scheduled_root,
                             const char *fn_name, PolyRunner *out) {
  (void)fn_name;
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, scheduled_root, &n_lin);
  if (!lin) return -1;

  InterpHandle *ih = malloc(sizeof(InterpHandle));
  if (!ih) { free(lin); return -1; }
  ih->lin = lin;
  ih->n_lin = n_lin;

  out->kind = POLY_RUNNER_INTERP;
  out->handle = ih;
  out->handle_size = 0;
  return 0;
}

static int interp_execute(PolyRunner *runner, void **args, int n_args) {
  InterpHandle *ih = (InterpHandle *)runner->handle;
  return poly_interp_eval(ih->lin, ih->n_lin, args, n_args);
}

static void interp_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    InterpHandle *ih = (InterpHandle *)runner->handle;
    free(ih->lin);
    free(ih);
  }
}

static const PolyAllocator *interp_get_allocator(void) {
  return &POLY_CPU_ALLOCATOR;
}

/* ── WASM JIT backend ─────────────────────────────────────────────────── */

#ifdef __EMSCRIPTEN__

static int wasm_lower_item(PolyCtx *ctx, PolyUOp *scheduled_root,
                           const char *fn_name, PolyRunner *out) {
  (void)fn_name;
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, scheduled_root, &n_lin);
  if (!lin) return -1;

  int wasm_len = 0;
  uint8_t *wasm_bytes = poly_render_wasm(lin, n_lin, &wasm_len, false);
  free(lin);
  if (!wasm_bytes || wasm_len <= 0) return -1;

  int kernel_id = js_compile_wasm_kernel(wasm_bytes, wasm_len);
  free(wasm_bytes);
  if (kernel_id < 0) return -1;

  WasmJitHandle *wh = malloc(sizeof(WasmJitHandle));
  if (!wh) return -1;
  wh->kernel_id = kernel_id;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = wh;
  out->handle_size = 0;
  return 0;
}

static int wasm_execute(PolyRunner *runner, void **args, int n_args) {
  WasmJitHandle *wh = (WasmJitHandle *)runner->handle;
  int *iargs = malloc((size_t)n_args * sizeof(int));
  if (!iargs) return -1;
  for (int i = 0; i < n_args; i++)
    iargs[i] = (int)(intptr_t)args[i];
  int ret = js_exec_wasm_kernel(wh->kernel_id, iargs, n_args);
  free(iargs);
  return ret;
}

static void wasm_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    WasmJitHandle *wh = (WasmJitHandle *)runner->handle;
    js_free_wasm_kernel(wh->kernel_id);
    free(wh);
  }
}

static const PolyAllocator *wasm_get_allocator(void) {
  return &POLY_CPU_ALLOCATOR;
}

#endif /* __EMSCRIPTEN__ */

/* ── CUDA backend ─────────────────────────────────────────────────────── */

#ifdef POLY_HAS_CUDA

static int cuda_lower_item(PolyCtx *ctx, PolyUOp *scheduled_root,
                           const char *fn_name, PolyRunner *out) {
  int n_lin;
  PolyUOp **lin = poly_linearize_cuda(ctx, scheduled_root, &n_lin);
  if (!lin) return -1;

  /* Extract grid/block from SPECIAL ops */
  int grid_size = 0, local_size = 0;
  for (int j = 0; j < n_lin; j++) {
    if (lin[j]->op == POLY_OP_SPECIAL && lin[j]->n_src > 0 &&
        lin[j]->src[0]->op == POLY_OP_CONST) {
      const char *sn = lin[j]->arg.str;
      if (sn && sn[0] == 'l')
        local_size = (int)lin[j]->src[0]->arg.i;
      else
        grid_size = (int)lin[j]->src[0]->arg.i;
    }
  }

  int block_size = local_size > 0 ? local_size : 256;

  char *src = poly_render_cuda(lin, n_lin, fn_name, block_size);
  free(lin);
  if (!src) return -1;

  if (getenv("POLY_DUMP_KERNELS"))
    fprintf(stderr, "=== CUDA KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

  PolyCudaProgram *prog = poly_compile_cuda(src, fn_name);
  if (!prog) {
    fprintf(stderr, "=== FAILED CUDA KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);
    free(src);
    return -1;
  }
  free(src);

  /* Compute grid dimensions */
  int gx;
  if (local_size > 0 && grid_size > 0) gx = grid_size;
  else if (local_size > 0)             gx = 1;
  else if (grid_size > 0)              gx = (grid_size + block_size - 1) / block_size;
  else                                 gx = 1;

  CudaRunnerHandle *ch = malloc(sizeof(CudaRunnerHandle));
  if (!ch) { poly_cuda_program_destroy(prog); return -1; }
  ch->prog = prog;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = ch;
  out->handle_size = 0;
  out->grid[0] = gx;    out->grid[1] = 1; out->grid[2] = 1;
  out->block[0] = block_size; out->block[1] = 1; out->block[2] = 1;
  return 0;
}

static int cuda_execute(PolyRunner *runner, void **args, int n_args) {
  CudaRunnerHandle *ch = (CudaRunnerHandle *)runner->handle;

  /* cuLaunchKernel needs void** where each element points TO a CUdeviceptr.
   * args[] already contains device pointers (cast to void*), but the CUDA
   * driver API requires one level of indirection: args[i] = &dptr[i]. */
  unsigned long long *dptrs = malloc((size_t)n_args * sizeof(unsigned long long));
  void **cuda_args = malloc((size_t)n_args * sizeof(void *));
  if (!dptrs || !cuda_args) {
    free(dptrs); free(cuda_args);
    return -1;
  }
  for (int i = 0; i < n_args; i++) {
    dptrs[i] = (unsigned long long)(uintptr_t)args[i];
    cuda_args[i] = &dptrs[i];
  }

  int ret = poly_cuda_launch(ch->prog, cuda_args, n_args,
                              runner->grid[0], runner->grid[1], runner->grid[2],
                              runner->block[0], runner->block[1], runner->block[2]);
  if (ret == 0) ret = poly_cuda_sync();

  free(dptrs);
  free(cuda_args);
  return ret;
}

static void cuda_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    CudaRunnerHandle *ch = (CudaRunnerHandle *)runner->handle;
    poly_cuda_program_destroy(ch->prog);
    free(ch);
  }
}

static const PolyAllocator *cuda_get_allocator(void) {
  return &POLY_CUDA_ALLOCATOR;
}

#endif /* POLY_HAS_CUDA */

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend registry                                                     */
/* ══════════════════════════════════════════════════════════════════════ */

static const PolyBackendDesc BACKENDS[] = {
  [POLY_DEVICE_AUTO]  = { NULL, POLY_DEVICE_AUTO, false, NULL, NULL, NULL, NULL },
#ifndef __EMSCRIPTEN__
  [POLY_DEVICE_CPU]   = { "cpu",    POLY_DEVICE_CPU,   false,
                          cpu_lower_item, cpu_execute, cpu_free_runner,
                          cpu_get_allocator },
#else
  [POLY_DEVICE_CPU]   = { NULL, POLY_DEVICE_CPU, false, NULL, NULL, NULL, NULL },
#endif
  [POLY_DEVICE_INTERP]= { "interp", POLY_DEVICE_INTERP, false,
                          interp_lower_item, interp_execute, interp_free_runner,
                          interp_get_allocator },
#ifdef POLY_HAS_CUDA
  [POLY_DEVICE_CUDA]  = { "cuda",   POLY_DEVICE_CUDA,  false,
                          cuda_lower_item, cuda_execute, cuda_free_runner,
                          cuda_get_allocator },
#else
  [POLY_DEVICE_CUDA]  = { NULL, POLY_DEVICE_CUDA, false, NULL, NULL, NULL, NULL },
#endif
#ifdef __EMSCRIPTEN__
  [POLY_DEVICE_WASM_JIT] = { "wasm", POLY_DEVICE_WASM_JIT, true,
                             wasm_lower_item, wasm_execute, wasm_free_runner,
                             wasm_get_allocator },
#else
  [POLY_DEVICE_WASM_JIT] = { NULL, POLY_DEVICE_WASM_JIT, false, NULL, NULL, NULL, NULL },
#endif
  [POLY_DEVICE_WEBGPU] = { NULL, POLY_DEVICE_WEBGPU, false, NULL, NULL, NULL, NULL },
};

#define N_BACKENDS (sizeof(BACKENDS) / sizeof(BACKENDS[0]))

const PolyBackendDesc *poly_backend_get(PolyDeviceId device) {
  if (device < 0 || (size_t)device >= N_BACKENDS) return NULL;
  if (!BACKENDS[device].name) return NULL;
  return &BACKENDS[device];
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Executable step: lower, run, free                                    */
/* ══════════════════════════════════════════════════════════════════════ */

PolyCompiledPlan *poly_compile_schedule(PolyCtx *ctx, PolySchedule *schedule,
                                        PolyDeviceId device) {
  if (!ctx || !schedule) return NULL;

  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend) {
    fprintf(stderr, "polygrad: compile_schedule: unsupported device %d\n", device);
    return NULL;
  }

  PolyCompiledPlan *plan = calloc(1, sizeof(PolyCompiledPlan));
  if (!plan) return NULL;
  plan->schedule = schedule;
  plan->device = device;
  plan->allocator = backend->get_allocator();
  plan->n_runners = schedule->n_items;
  plan->runners = calloc((size_t)schedule->n_items, sizeof(PolyRunner));
  if (!plan->runners) { free(plan); return NULL; }

  /* Lower each COMPUTE item via backend vtable */
  static int lower_counter = 0;
  for (int k = 0; k < schedule->n_items; k++) {
    PolyExecItem *item = &schedule->items[k];
    PolyRunner *runner = &plan->runners[k];

    if (item->kind != POLY_EXEC_COMPUTE) {
      fprintf(stderr, "polygrad: compile_schedule: non-COMPUTE item %d not supported\n", k);
      goto cleanup;
    }

    if (!poly_validate_kernel_graph(ctx, item->root)) {
      fprintf(stderr, "polygrad: compile_schedule: kernel %d validation failed\n", k);
      goto cleanup;
    }

    char fn_name[64];
    snprintf(fn_name, sizeof(fn_name), "lower%d_k%d", lower_counter, k);

    int ret = backend->lower_item(ctx, item->root, fn_name, runner);
    if (ret != 0) {
      fprintf(stderr, "polygrad: compile_schedule: backend '%s' failed for kernel %d\n",
              backend->name, k);
      goto cleanup;
    }

    /* Copy buf_slot_indices as param_to_slot */
    runner->n_params = item->n_buf_slots;
    runner->param_to_slot = malloc((size_t)item->n_buf_slots * sizeof(int));
    memcpy(runner->param_to_slot, item->buf_slot_indices,
           (size_t)item->n_buf_slots * sizeof(int));

    runner->n_vars = 0;
    runner->var_indices = NULL;
  }
  lower_counter++;

  return plan;

cleanup:
  poly_compiled_plan_free(plan);
  return NULL;
}

int poly_compiled_plan_run(PolyCompiledPlan *plan,
                           void **slot_data, int n_slots,
                           PolyVarBinding *var_bindings, int n_var_bindings) {
  if (!plan || !plan->schedule) return -1;
  PolySchedule *sched = plan->schedule;

  const PolyBackendDesc *backend = poly_backend_get(plan->device);
  if (!backend) return -1;

  /* ── Allocate per-invocation intermediates ────────────────────────── */
  int n_inter = 0;
  for (int i = 0; i < sched->n_buf_slots; i++)
    if (sched->buf_slots[i].is_intermediate) n_inter++;

  PolyBufferHandle *inter_handles = NULL;
  if (n_inter > 0) {
    inter_handles = calloc((size_t)n_inter, sizeof(PolyBufferHandle));
    if (!inter_handles) return -1;
    int idx = 0;
    for (int i = 0; i < sched->n_buf_slots; i++) {
      if (!sched->buf_slots[i].is_intermediate) continue;
      size_t nbytes = (size_t)sched->buf_slots[i].nbytes;
      if (nbytes == 0) nbytes = sizeof(float);
      void *ptr = plan->allocator->alloc(nbytes, plan->allocator->dev_ctx);
      if (!ptr) goto free_intermediates;
      inter_handles[idx] = (PolyBufferHandle){
        .ptr = ptr, .nbytes = nbytes,
        .domain = plan->device, .owned = true,
      };
      idx++;
    }

    /* Zero intermediates (needed for REDUCE accumulators) */
    for (int i = 0; i < n_inter; i++) {
      PolyBufferHandle *h = &inter_handles[i];
      if (h->domain == POLY_DEVICE_CPU || h->domain == POLY_DEVICE_INTERP
#ifdef __EMSCRIPTEN__
          || h->domain == POLY_DEVICE_WASM_JIT
#endif
      ) {
        memset(h->ptr, 0, h->nbytes);
      }
#ifdef POLY_HAS_CUDA
      else if (h->domain == POLY_DEVICE_CUDA) {
        poly_cuda_memset((unsigned long long)(uintptr_t)h->ptr, 0, h->nbytes);
      }
#endif
    }
  }

  /* ── Build slot_to_data ───────────────────────────────────────────── */
  void *slot_to_data[POLY_MAX_REALIZE_BUFS];
  memset(slot_to_data, 0, sizeof(slot_to_data));

  /* Fill external slots from caller data */
  for (int i = 0; i < sched->n_buf_slots && i < POLY_MAX_REALIZE_BUFS; i++) {
    if (!sched->buf_slots[i].is_intermediate && i < n_slots && slot_data[i])
      slot_to_data[i] = slot_data[i];
  }

  /* Fill intermediate slots from per-run handles */
  {
    int idx = 0;
    for (int i = 0; i < sched->n_buf_slots && i < POLY_MAX_REALIZE_BUFS; i++) {
      if (sched->buf_slots[i].is_intermediate && idx < n_inter) {
        slot_to_data[i] = inter_handles[idx].ptr;
        idx++;
      }
    }
  }

  /* ── Merge default vars with runtime overrides ────────────────────── */
  PolyVarBinding all_vars[32];
  int n_all = 0;
  for (int i = 0; i < sched->n_default_vars && n_all < 32; i++)
    all_vars[n_all++] = sched->default_vars[i];
  for (int i = 0; i < n_var_bindings && n_all < 32; i++) {
    bool found = false;
    for (int j = 0; j < n_all; j++) {
      if (all_vars[j].var == var_bindings[i].var) {
        all_vars[j].value = var_bindings[i].value;
        found = true;
        break;
      }
    }
    if (!found) all_vars[n_all++] = var_bindings[i];
  }

  /* ── Execute runners in exec_order via backend vtable ─────────────── */
  int ret = 0;
  for (int s = 0; s < sched->n_items && ret == 0; s++) {
    int k = sched->exec_order[s];
    PolyRunner *runner = &plan->runners[k];

    if (!runner->handle) {
      fprintf(stderr, "polygrad: plan_run: runner %d has no handle\n", k);
      ret = -1; break;
    }

    (void)all_vars;
    (void)n_all;

    int n_args = runner->n_params;
    void **args = calloc((size_t)n_args, sizeof(void *));

    for (int i = 0; i < runner->n_params; i++) {
      int slot = runner->param_to_slot[i];
      if (slot >= 0 && slot < POLY_MAX_REALIZE_BUFS)
        args[i] = slot_to_data[slot];
      if (!args[i]) {
        fprintf(stderr, "polygrad: plan_run: missing data for param %d "
                "(slot %d) in kernel %d\n", i, slot, k);
        ret = -1; free(args); args = NULL; break;
      }
    }

    if (ret == 0 && args) {
      ret = backend->execute(runner, args, n_args);
      free(args);
    }
  }

  /* ── Free per-invocation intermediates ────────────────────────────── */
free_intermediates:
  if (inter_handles) {
    for (int i = 0; i < n_inter; i++) {
      if (inter_handles[i].ptr)
        plan->allocator->free(inter_handles[i].ptr, plan->allocator->dev_ctx);
    }
    free(inter_handles);
  }

  return ret;
}

void poly_compiled_plan_free(PolyCompiledPlan *plan) {
  if (!plan) return;

  const PolyBackendDesc *backend = poly_backend_get(plan->device);

  for (int i = 0; i < plan->n_runners; i++) {
    PolyRunner *r = &plan->runners[i];
    if (r->handle && backend)
      backend->free_runner(r);
    free(r->param_to_slot);
    free(r->var_indices);
  }
  free(plan->runners);

  /* Note: intermediates are per-invocation (allocated/freed in plan_run).
   * The plan itself is immutable and owns only runners + mappings. */
  free(plan);
}
