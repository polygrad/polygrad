/*
 * exec_plan.c -- Execution plan: prepare, lower, run, free
 *
 * Contains all exec_plan API implementations:
 *   - poly_prepare_step: backend-neutral scheduling
 *   - poly_lower_step: backend-specific lowering (CPU, INTERP, WASM_JIT)
 *   - poly_executable_step_run: dispatches to backend-specific execution
 *   - poly_executable_step_free: cleanup
 *   - POLY_CPU_ALLOCATOR: trivial malloc/free allocator
 *
 * Compiled into both native and Emscripten builds. Backend-specific code
 * is guarded by #ifdef __EMSCRIPTEN__ / #ifndef __EMSCRIPTEN__.
 *
 * Shared helpers (strip_bind_values, validate_kernel_graph, etc.) are
 * declared in frontend_internal.h and implemented in frontend.c.
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

PolyPreparedStep *poly_prepare_step(PolyCtx *ctx, PolyUOp *sink,
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
  PolyPreparedStep *ps = calloc(1, sizeof(PolyPreparedStep));
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
    ps->buf_slots = calloc((size_t)ps->n_buf_slots, sizeof(PolyPreparedBufSlot));

    /* External buffer slots */
    for (int i = 0; i < n_external; i++) {
      PolyPreparedBufSlot *slot = &ps->buf_slots[i];
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
      PolyPreparedBufSlot *slot = &ps->buf_slots[n_external + b];
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
  ps->items = calloc((size_t)sr.n_kernels, sizeof(PolyExecItemSpec));

  for (int k = 0; k < sr.n_kernels; k++) {
    PolyExecItemSpec *item = &ps->items[k];
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
  poly_prepared_step_free(ps);
  poly_schedule_result_free(&sr);
  return NULL;
}

void poly_prepared_step_free(PolyPreparedStep *step) {
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

/* ── CPU allocator ────────────────────────────────────────────────────── */

static void *cpu_alloc(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  return calloc(1, nbytes);
}

static void cpu_free(void *handle, void *dev_ctx) {
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
  .free = cpu_free,
  .copy_in = cpu_copy_in,
  .copy_out = cpu_copy_out,
  .copy_between = cpu_copy_between,
  .dev_ctx = NULL,
};

/* ── Executable step construction ─────────────────────────────────────── */

/* Interpreter runner handle: linearized UOp array + count */
typedef struct {
  PolyUOp **lin;
  int n_lin;
} InterpHandle;

/* ── WASM JIT EM_JS bridge (Emscripten only) ─────────────────────────── */
/*
 * Two-phase design: compile during poly_lower_step (once per kernel),
 * execute during poly_executable_step_run (many times).
 *
 * js_compile_wasm_kernel: compiles WASM bytes into a WebAssembly.Instance,
 *   stores it in a JS-side cache, returns an integer kernel_id.
 *   Uses synchronous WebAssembly.Module() -- works for kernels under 4KB
 *   on browser main thread, no limit in Node.js or Web Workers.
 *
 * js_exec_wasm_kernel: executes a previously compiled kernel by kernel_id.
 *   args is a heap pointer to an int32 array of buffer offsets.
 */
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

/* WASM JIT runner handle: stores JS-side kernel_id for cached instance */
typedef struct {
  int kernel_id;
} WasmJitHandle;

PolyExecutableStep *poly_lower_step(PolyCtx *ctx, PolyPreparedStep *prepared,
                                    PolyDeviceId device) {
  if (!ctx || !prepared) return NULL;

  /* Validate supported devices for this build */
#ifdef __EMSCRIPTEN__
  if (device != POLY_DEVICE_INTERP && device != POLY_DEVICE_WASM_JIT) {
    fprintf(stderr, "polygrad: lower_step: unsupported device %d in WASM build\n", device);
    return NULL;
  }
#else
  if (device != POLY_DEVICE_CPU && device != POLY_DEVICE_INTERP) {
    fprintf(stderr, "polygrad: lower_step: unsupported device %d\n", device);
    return NULL;
  }
#endif

  PolyExecutableStep *es = calloc(1, sizeof(PolyExecutableStep));
  if (!es) return NULL;
  es->prepared = prepared;
  es->device = device;
  es->allocator = &POLY_CPU_ALLOCATOR;
  es->n_runners = prepared->n_items;
  es->runners = calloc((size_t)prepared->n_items, sizeof(PolyRunner));
  if (!es->runners) { free(es); return NULL; }

  /* Allocate intermediate buffers (malloc works in all builds) */
  int n_inter = 0;
  for (int i = 0; i < prepared->n_buf_slots; i++)
    if (prepared->buf_slots[i].is_intermediate) n_inter++;

  es->n_intermediates = n_inter;
  if (n_inter > 0) {
    es->intermediate_handles = calloc((size_t)n_inter, sizeof(PolyBufferHandle));
    int idx = 0;
    for (int i = 0; i < prepared->n_buf_slots; i++) {
      if (!prepared->buf_slots[i].is_intermediate) continue;
      size_t nbytes = (size_t)prepared->buf_slots[i].nbytes;
      if (nbytes == 0) nbytes = sizeof(float);
      void *ptr = calloc(1, nbytes);
      if (!ptr) goto cleanup;
      es->intermediate_handles[idx] = (PolyBufferHandle){
        .ptr = ptr, .nbytes = nbytes,
        .domain = device, .owned = true,
      };
      idx++;
    }
  }

  /* Lower each COMPUTE item */
  static int lower_counter = 0;
  for (int k = 0; k < prepared->n_items; k++) {
    PolyExecItemSpec *item = &prepared->items[k];
    PolyRunner *runner = &es->runners[k];

    if (item->kind != POLY_EXEC_COMPUTE) {
      fprintf(stderr, "polygrad: lower_step: non-COMPUTE item %d not supported\n", k);
      goto cleanup;
    }

    if (!poly_validate_kernel_graph(ctx, item->root)) {
      fprintf(stderr, "polygrad: lower_step: kernel %d validation failed\n", k);
      goto cleanup;
    }

    /* Linearize (shared by all backends) */
    int n_lin;
    PolyUOp **lin = poly_linearize(ctx, item->root, &n_lin);
    if (!lin) goto cleanup;

    if (device == POLY_DEVICE_INTERP) {
      /* INTERP: store linearized UOps directly, no render/compile */
      InterpHandle *ih = malloc(sizeof(InterpHandle));
      if (!ih) { free(lin); goto cleanup; }
      ih->lin = lin;
      ih->n_lin = n_lin;
      runner->kind = POLY_RUNNER_INTERP;
      runner->handle = ih;
      runner->handle_size = 0;
#ifdef __EMSCRIPTEN__
    } else if (device == POLY_DEVICE_WASM_JIT) {
      /* WASM JIT: render to WASM bytes, compile via EM_JS, cache kernel_id */
      int wasm_len = 0;
      uint8_t *wasm_bytes = poly_render_wasm(lin, n_lin, &wasm_len, false);
      free(lin);
      if (!wasm_bytes || wasm_len <= 0) {
        fprintf(stderr, "polygrad: lower_step: WASM render failed for kernel %d\n", k);
        goto cleanup;
      }

      int kernel_id = js_compile_wasm_kernel(wasm_bytes, wasm_len);
      free(wasm_bytes);
      if (kernel_id < 0) {
        fprintf(stderr, "polygrad: lower_step: WASM compile failed for kernel %d\n", k);
        goto cleanup;
      }

      WasmJitHandle *wh = malloc(sizeof(WasmJitHandle));
      if (!wh) goto cleanup;
      wh->kernel_id = kernel_id;
      runner->kind = POLY_RUNNER_COMPILED;
      runner->handle = wh;
      runner->handle_size = 0;
#endif /* __EMSCRIPTEN__ */
#ifndef __EMSCRIPTEN__
    } else {
      /* CPU: render C -> compile -> dlopen (native builds only) */
      char fn_name[64];
      snprintf(fn_name, sizeof(fn_name), "lower%d_k%d", lower_counter, k);
      char *src = poly_render_c(lin, n_lin, fn_name);
      free(lin);
      if (!src) goto cleanup;

      if (getenv("POLY_DUMP_KERNELS"))
        fprintf(stderr, "=== LOWER KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

      PolyProgram *prog = poly_compile_c(src, fn_name);
      if (!prog) {
        fprintf(stderr, "=== FAILED LOWER KERNEL %d ===\n%s\n=== END ===\n", k, src);
        free(src);
        goto cleanup;
      }
      free(src);

      runner->kind = POLY_RUNNER_COMPILED;
      runner->handle = prog;
      runner->handle_size = 0;
#endif /* !__EMSCRIPTEN__ */
    }

    /* Copy buf_slot_indices as param_to_slot */
    runner->n_params = item->n_buf_slots;
    runner->param_to_slot = malloc((size_t)item->n_buf_slots * sizeof(int));
    memcpy(runner->param_to_slot, item->buf_slot_indices,
           (size_t)item->n_buf_slots * sizeof(int));

    /* No var mapping yet -- vars handled at run time via prepared step */
    runner->n_vars = 0;
    runner->var_indices = NULL;
  }
  lower_counter++;

  return es;

cleanup:
  poly_executable_step_free(es);
  return NULL;
}

int poly_executable_step_run(PolyExecutableStep *step,
                             void **slot_data, int n_slots,
                             PolyVarBinding *var_bindings, int n_var_bindings) {
  if (!step || !step->prepared) return -1;
  PolyPreparedStep *ps = step->prepared;

  /* Zero intermediate buffers (needed for REDUCE accumulators) */
  for (int i = 0; i < step->n_intermediates; i++)
    memset(step->intermediate_handles[i].ptr, 0,
           step->intermediate_handles[i].nbytes);

  /* Build slot_to_data: maps buf_slot index -> data pointer.
   * External slots come from slot_data, intermediates from our handles. */
  void *slot_to_data[POLY_MAX_REALIZE_BUFS];
  memset(slot_to_data, 0, sizeof(slot_to_data));

  /* Fill external slots from caller data */
  for (int i = 0; i < ps->n_buf_slots && i < POLY_MAX_REALIZE_BUFS; i++) {
    if (!ps->buf_slots[i].is_intermediate && i < n_slots && slot_data[i])
      slot_to_data[i] = slot_data[i];
  }

  /* Fill intermediate slots from owned handles */
  int inter_idx = 0;
  for (int i = 0; i < ps->n_buf_slots && i < POLY_MAX_REALIZE_BUFS; i++) {
    if (ps->buf_slots[i].is_intermediate && inter_idx < step->n_intermediates) {
      slot_to_data[i] = step->intermediate_handles[inter_idx].ptr;
      inter_idx++;
    }
  }

  /* Merge default vars with runtime overrides */
  PolyVarBinding all_vars[32];
  int n_all = 0;
  for (int i = 0; i < ps->n_default_vars && n_all < 32; i++)
    all_vars[n_all++] = ps->default_vars[i];
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

  /* Execute runners in exec_order */
  int ret = 0;
  for (int s = 0; s < ps->n_items && ret == 0; s++) {
    int k = ps->exec_order[s];
    PolyRunner *runner = &step->runners[k];

    if (!runner->handle) {
      fprintf(stderr, "polygrad: exec_step_run: runner %d has no handle\n", k);
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
        fprintf(stderr, "polygrad: exec_step_run: missing data for param %d "
                "(slot %d) in kernel %d\n", i, slot, k);
        ret = -1; free(args); args = NULL; break;
      }
    }

    if (ret == 0 && args) {
      if (runner->kind == POLY_RUNNER_COMPILED) {
#ifdef __EMSCRIPTEN__
        /* WASM JIT: build int32 arg array and dispatch via EM_JS */
        if (step->device == POLY_DEVICE_WASM_JIT) {
          WasmJitHandle *wh = (WasmJitHandle *)runner->handle;
          int *iargs = malloc((size_t)n_args * sizeof(int));
          if (!iargs) { ret = -1; } else {
            for (int p = 0; p < n_args; p++)
              iargs[p] = (int)(intptr_t)args[p];
            ret = js_exec_wasm_kernel(wh->kernel_id, iargs, n_args);
            free(iargs);
          }
        } else {
          fprintf(stderr, "polygrad: exec_step_run: COMPILED runner not "
                  "supported on device %d in WASM build\n", step->device);
          ret = -1;
        }
#else
        poly_program_call((PolyProgram *)runner->handle, args, n_args);
#endif
      } else if (runner->kind == POLY_RUNNER_INTERP) {
        InterpHandle *ih = (InterpHandle *)runner->handle;
        ret = poly_interp_eval(ih->lin, ih->n_lin, args, n_args);
      } else {
        fprintf(stderr, "polygrad: exec_step_run: unsupported runner kind %d\n",
                runner->kind);
        ret = -1;
      }
      free(args);
    }
  }

  return ret;
}

void poly_executable_step_free(PolyExecutableStep *step) {
  if (!step) return;
  for (int i = 0; i < step->n_runners; i++) {
    PolyRunner *r = &step->runners[i];
    if (r->kind == POLY_RUNNER_COMPILED && r->handle) {
#ifdef __EMSCRIPTEN__
      /* WASM JIT: free JS-side cached instance, then the handle */
      if (step->device == POLY_DEVICE_WASM_JIT) {
        WasmJitHandle *wh = (WasmJitHandle *)r->handle;
        js_free_wasm_kernel(wh->kernel_id);
        free(wh);
      }
#else
      poly_program_destroy((PolyProgram *)r->handle);
#endif
    }
    if (r->kind == POLY_RUNNER_INTERP && r->handle) {
      InterpHandle *ih = (InterpHandle *)r->handle;
      free(ih->lin);
      free(ih);
    }
    free(r->param_to_slot);
    free(r->var_indices);
  }
  free(step->runners);
  if (step->intermediate_handles) {
    for (int i = 0; i < step->n_intermediates; i++)
      if (step->intermediate_handles[i].owned)
        free(step->intermediate_handles[i].ptr);
    free(step->intermediate_handles);
  }
  free(step);
}
