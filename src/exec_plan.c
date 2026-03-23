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
  int max_out = n_bufs_orig > 0 ? n_bufs_orig : POLY_MAX_REALIZE_BUFS;
  PolyUOp **output_bufs = calloc((size_t)max_out, sizeof(PolyUOp *));
  int n_output_bufs = poly_collect_output_buffers_in_sink(sink, output_bufs, max_out);
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
    free(buf_order_orig); free(buf_order_post); free(output_bufs);
    poly_schedule_result_free(&sr);
    return NULL;
  }

  /* --- Allocate prepared step ------------------------------------------- */
  PolySchedule *ps = calloc(1, sizeof(PolySchedule));
  if (!ps) {
    free(buf_order_orig); free(buf_order_post); free(output_bufs);
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

    /* Store DEFINE_VAR UOp pointers for this kernel */
    int nv = (sr.kernel_n_vars ? sr.kernel_n_vars[k] : 0);
    item->n_var_uops = nv;
    if (nv > 0 && sr.var_to_buf && sr.var_to_buf[k]) {
      item->var_uops = malloc((size_t)nv * sizeof(PolyUOp *));
      memcpy(item->var_uops, sr.var_to_buf[k], (size_t)nv * sizeof(PolyUOp *));
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
  free(output_bufs);
  poly_schedule_result_free(&sr);
  return ps;

cleanup:
  free(buf_order_orig);
  free(buf_order_post);
  free(output_bufs);
  poly_schedule_free(ps);
  poly_schedule_result_free(&sr);
  return NULL;
}

void poly_schedule_free(PolySchedule *step) {
  if (!step) return;
  for (int i = 0; i < step->n_items; i++) {
    free(step->items[i].buf_slot_indices);
    free(step->items[i].var_uops);
  }
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

static int cpu_execute_fn(void *self, void **args, int n_args);
static void cpu_free_fn(void *self);

static int cpu_lower_item(PolyCtx *ctx, PolyUOp *scheduled_root,
                          const char *fn_name, PolyRunner *out) {
  int n_lin;
  PolyUOp **lin = poly_linearize_env(ctx, scheduled_root, &n_lin);
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
  out->execute = cpu_execute_fn;
  out->free_handle = cpu_free_fn;
  return 0;
}

static int cpu_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  poly_program_call((PolyProgram *)runner->handle, args, n_args);
  return 0;
}
static int cpu_execute(PolyRunner *runner, void **args, int n_args) {
  return cpu_execute_fn(runner, args, n_args);
}

static void cpu_free_fn(void *self) {
  PolyRunner *runner = (PolyRunner *)self;
  if (runner->handle) poly_program_destroy((PolyProgram *)runner->handle);
}
static void cpu_free_runner(PolyRunner *runner) {
  cpu_free_fn(runner);
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
  PolyUOp **lin = poly_linearize_env(ctx, scheduled_root, &n_lin);
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
  PolyUOp **lin = poly_linearize_env(ctx, scheduled_root, &n_lin);
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

/* ── x86-64 JIT backend ─────────────────────────────────────────────── */

#ifdef POLY_HAS_X64

static int x64_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  poly_x64_program_call((PolyX64Program *)runner->handle, args, n_args);
  return 0;
}

static void x64_free_fn(void *self) {
  PolyRunner *runner = (PolyRunner *)self;
  if (runner->handle) poly_x64_program_destroy((PolyX64Program *)runner->handle);
}

/* Check if the kernel uses only dtypes the x64 renderer supports (f32, int32, bool).
 * Returns false if f64, f16, bf16 or other unsupported types are found.
 * Simple iterative DFS over the DAG. */
/* Check if kernel uses only features the x64 renderer handles correctly.
 * Rejects: f64/f16/bf16 dtypes, multi-range reduce patterns (DEFINE_REG with
 * nested RANGEs and AFTER chains — the renderer compiles but produces wrong code).
 * Iterative DFS with simple open-addressing pointer set. */
static bool x64_can_handle(PolyUOp *root) {
  int cap = 256, top = 0;
  PolyUOp **stack = malloc((size_t)cap * sizeof(PolyUOp *));
  if (!stack) return false;
  int set_cap = 512;
  PolyUOp **set = calloc((size_t)set_cap, sizeof(PolyUOp *));
  if (!set) { free(stack); return false; }
  bool ok = true;
  int n_ranges = 0, n_stores = 0;

  stack[top++] = root;
  while (top > 0) {
    PolyUOp *u = stack[--top];
    uint32_t h = (uint32_t)((uintptr_t)u >> 3) % (uint32_t)set_cap;
    bool found = false;
    for (int probe = 0; probe < set_cap; probe++) {
      uint32_t idx = (h + (uint32_t)probe) % (uint32_t)set_cap;
      if (!set[idx]) { set[idx] = u; break; }
      if (set[idx] == u) { found = true; break; }
    }
    if (found) continue;

    /* Reject unsupported dtypes: non-float32 floats (f64, f16, bf16) */
    PolyDType dt = u->dtype;
    if (!dt.is_ptr && !poly_dtype_eq(dt, POLY_VOID) &&
        poly_dtype_is_float(dt) && poly_dtype_scalar(dt).bitsize != 32) {
      ok = false; break;
    }
    /* Reject 64-bit integers (uint64 from THREEFRY, etc.) */
    if (!dt.is_ptr && !poly_dtype_eq(dt, POLY_VOID) &&
        !poly_dtype_is_float(dt) && poly_dtype_scalar(dt).bitsize > 32) {
      ok = false; break;
    }
    /* Reject unsupported ops */
    if (u->op == POLY_OP_THREEFRY) { ok = false; break; }
    if (u->op == POLY_OP_RANGE) n_ranges++;
    if (u->op == POLY_OP_STORE) n_stores++;

    for (int i = 0; i < u->n_src; i++) {
      if (top >= cap) { cap *= 2; stack = realloc(stack, (size_t)cap * sizeof(PolyUOp *)); }
      stack[top++] = u->src[i];
    }
  }
  /* Previously rejected multi-store + multi-range patterns due to SHL R8
   * clobber in nested loops (fixed in commit 439f957). The renderer now
   * handles these correctly. Keeping the check commented for reference:
   * if (ok && n_stores > 1 && n_ranges > 1) ok = false; */

  free(stack);
  free(set);
  return ok;
}

static int x64_lower_item(PolyCtx *ctx, PolyUOp *scheduled_root,
                           const char *fn_name, PolyRunner *out) {
  /* Pre-check: fall back to CPU for unsupported patterns/dtypes */
  if (!x64_can_handle(scheduled_root)) goto fallback;

  int n_lin;
  /* Use CPU-style linearization (poly_linearize_env) to avoid x64-specific
   * caps producing IR with indexing patterns the renderer can't handle yet
   * (broadcast/expand intermediates, fused SIB for reduce consumers). */
  PolyUOp **lin = poly_linearize_env(ctx, scheduled_root, &n_lin);
  if (!lin) goto fallback;

  int code_size;
  uint8_t *code = poly_render_x64(lin, n_lin, &code_size);
  free(lin);
  if (!code) goto fallback;

  PolyX64Program *prog = poly_compile_x64(code, code_size);
  free(code);
  if (!prog) goto fallback;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = prog;
  out->handle_size = 0;
  out->execute = x64_execute_fn;
  out->free_handle = x64_free_fn;
  return 0;

fallback:
  /* x64 renderer doesn't support all ops yet (DEFINE_LOCAL, BARRIER, etc.).
   * Fall back to CPU compiled backend for unsupported kernels.
   * Both use host memory, so buffer layout is compatible. */
#ifndef __EMSCRIPTEN__
  return cpu_lower_item(ctx, scheduled_root, fn_name, out);
#else
  return -1;
#endif
}

static int x64_execute(PolyRunner *runner, void **args, int n_args) {
  return runner->execute(runner, args, n_args);
}

static void x64_free_runner(PolyRunner *runner) {
  if (runner->free_handle) runner->free_handle(runner);
}

#endif /* POLY_HAS_X64 */

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
#ifdef POLY_HAS_X64
  [POLY_DEVICE_X64_JIT] = { "x64_jit", POLY_DEVICE_X64_JIT, false,
                             x64_lower_item, x64_execute, x64_free_runner,
                             cpu_get_allocator },
#else
  [POLY_DEVICE_X64_JIT] = { NULL, POLY_DEVICE_X64_JIT, false, NULL, NULL, NULL, NULL },
#endif
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

  /* x64 JIT uses per-runner dispatch: x64_execute delegates to runner->execute,
   * which is either x64_execute_fn (native) or cpu_execute_fn (fallback). */

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

  /* ── Allocate persistent intermediates ─────────────────────────────── */
  plan->n_intermediates = 0;
  for (int i = 0; i < schedule->n_buf_slots; i++)
    if (schedule->buf_slots[i].is_intermediate) plan->n_intermediates++;

  if (plan->n_intermediates > 0) {
    plan->intermediates = calloc((size_t)plan->n_intermediates, sizeof(PolyBufferHandle));
    if (!plan->intermediates) goto cleanup;
    int idx = 0;
    for (int i = 0; i < schedule->n_buf_slots; i++) {
      if (!schedule->buf_slots[i].is_intermediate) continue;
      size_t nbytes = (size_t)schedule->buf_slots[i].nbytes;
      if (nbytes == 0) nbytes = sizeof(float);
      void *ptr = plan->allocator->alloc(nbytes, plan->allocator->dev_ctx);
      if (!ptr) goto cleanup;
      plan->intermediates[idx] = (PolyBufferHandle){
        .ptr = ptr, .nbytes = nbytes,
        .domain = plan->device, .owned = true,
      };
      idx++;
    }
  }

  /* ── Allocate persistent per-kernel args arrays ────────────────────── */
  plan->kernel_args = calloc((size_t)plan->n_runners, sizeof(void **));
  if (!plan->kernel_args) goto cleanup;
  for (int k = 0; k < plan->n_runners; k++) {
    PolyExecItem *item = &schedule->items[k];
    int n_args = plan->runners[k].n_params + item->n_var_uops;
    plan->kernel_args[k] = calloc((size_t)(n_args > 0 ? n_args : 1), sizeof(void *));
    if (!plan->kernel_args[k]) goto cleanup;
  }

  /* ── Allocate persistent slot_to_data ──────────────────────────────── */
  plan->n_slot_to_data = schedule->n_buf_slots;
  plan->slot_to_data = calloc((size_t)(plan->n_slot_to_data > 0 ? plan->n_slot_to_data : 1),
                              sizeof(void *));
  if (!plan->slot_to_data) goto cleanup;

  /* Pre-fill intermediate slot pointers (these don't change between runs) */
  {
    int idx = 0;
    for (int i = 0; i < plan->n_slot_to_data; i++) {
      if (schedule->buf_slots[i].is_intermediate && idx < plan->n_intermediates) {
        plan->slot_to_data[i] = plan->intermediates[idx].ptr;
        idx++;
      }
    }
  }

  /* ── Allocate merged vars array ────────────────────────────────────── */
  {
    int total_vars = schedule->n_default_vars + 16; /* room for runtime overrides */
    plan->merged_vars = calloc((size_t)total_vars, sizeof(PolyVarBinding));
    plan->merged_vars_cap = total_vars;
  }

  /* ── Allocate var int storage ──────────────────────────────────────── */
  {
    int total_var_ints = 0;
    for (int k = 0; k < schedule->n_items; k++)
      total_var_ints += schedule->items[k].n_var_uops;
    if (total_var_ints == 0) total_var_ints = 16;
    plan->var_int_storage = calloc((size_t)total_var_ints, sizeof(int));
    plan->var_int_cap = total_var_ints;
  }

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

  int ret = 0;

  /* ── Zero persistent intermediates (reduce accumulators need this) ── */
  for (int i = 0; i < plan->n_intermediates; i++) {
    PolyBufferHandle *h = &plan->intermediates[i];
    if (h->domain == POLY_DEVICE_CPU || h->domain == POLY_DEVICE_INTERP
#ifdef POLY_HAS_X64
        || h->domain == POLY_DEVICE_X64_JIT
#endif
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

  /* ── Fill external slots in persistent slot_to_data ────────────────── */
  for (int i = 0; i < plan->n_slot_to_data; i++) {
    if (!sched->buf_slots[i].is_intermediate) {
      plan->slot_to_data[i] = (i < n_slots && slot_data[i]) ? slot_data[i] : NULL;
    }
    /* intermediate slots are pre-filled at compile time and don't change */
  }

  /* ── Merge default vars with runtime overrides ────────────────────── */
  int n_all = 0;

  /* Grow merged_vars if needed */
  int needed = sched->n_default_vars + n_var_bindings;
  if (needed > plan->merged_vars_cap) {
    free(plan->merged_vars);
    plan->merged_vars = calloc((size_t)needed, sizeof(PolyVarBinding));
    plan->merged_vars_cap = needed;
  }

  for (int i = 0; i < sched->n_default_vars; i++)
    plan->merged_vars[n_all++] = sched->default_vars[i];
  for (int i = 0; i < n_var_bindings; i++) {
    bool found = false;
    for (int j = 0; j < n_all; j++) {
      if (plan->merged_vars[j].var == var_bindings[i].var) {
        plan->merged_vars[j].value = var_bindings[i].value;
        found = true;
        break;
      }
    }
    if (!found) plan->merged_vars[n_all++] = var_bindings[i];
  }

  /* ── Execute runners in exec_order via backend vtable ─────────────── */
  int var_int_idx = 0;
  for (int s = 0; s < sched->n_items && ret == 0; s++) {
    int k = sched->exec_order[s];
    PolyRunner *runner = &plan->runners[k];

    if (!runner->handle) {
      fprintf(stderr, "polygrad: plan_run: runner %d has no handle\n", k);
      ret = -1; break;
    }

    PolyExecItem *item = &sched->items[k];
    int n_vars = item->n_var_uops;
    int n_args = runner->n_params + n_vars;
    void **args = plan->kernel_args[k];

    for (int i = 0; i < runner->n_params; i++) {
      int slot = runner->param_to_slot[i];
      if (slot >= 0 && slot < plan->n_slot_to_data)
        args[i] = plan->slot_to_data[slot];
      if (!args[i]) {
        fprintf(stderr, "polygrad: plan_run: missing data for param %d "
                "(slot %d) in kernel %d\n", i, slot, k);
        ret = -1; break;
      }
    }

    /* Fill var params (DEFINE_VAR values as int* pointers) */
    if (ret == 0 && n_vars > 0 && item->var_uops) {
      for (int v = 0; v < n_vars; v++) {
        PolyUOp *var = item->var_uops[v];
        bool found = false;
        for (int vb = 0; vb < n_all; vb++) {
          if (plan->merged_vars[vb].var == var) {
            if (var_int_idx >= plan->var_int_cap) {
              /* grow var int storage */
              int new_cap = plan->var_int_cap * 2;
              plan->var_int_storage = realloc(plan->var_int_storage,
                                              (size_t)new_cap * sizeof(int));
              plan->var_int_cap = new_cap;
            }
            plan->var_int_storage[var_int_idx] = (int)plan->merged_vars[vb].value;
            args[runner->n_params + v] = &plan->var_int_storage[var_int_idx];
            var_int_idx++;
            found = true;
            break;
          }
        }
        if (!found) {
          fprintf(stderr, "polygrad: plan_run: no binding for DEFINE_VAR "
                  "in kernel %d\n", k);
          ret = -1; break;
        }
      }
    }

    if (ret == 0) {
      ret = backend->execute(runner, args, n_args);
    }
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

  /* Free persistent intermediates */
  if (plan->intermediates) {
    for (int i = 0; i < plan->n_intermediates; i++) {
      if (plan->intermediates[i].ptr)
        plan->allocator->free(plan->intermediates[i].ptr, plan->allocator->dev_ctx);
    }
    free(plan->intermediates);
  }

  /* Free persistent per-kernel args arrays */
  if (plan->kernel_args) {
    for (int i = 0; i < plan->n_runners; i++)
      free(plan->kernel_args[i]);
    free(plan->kernel_args);
  }

  free(plan->slot_to_data);
  free(plan->merged_vars);
  free(plan->var_int_storage);
  free(plan);
}
