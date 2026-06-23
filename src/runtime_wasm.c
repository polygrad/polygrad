#include "runtime_wasm.h"

#ifdef __EMSCRIPTEN__

#include "codegen.h"

#include <emscripten.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int kernel_id;
} PolyWasmJitRunnerHandle;

static int poly_wasm_execute_fn(void *self, void **args, int n_args);
static void poly_wasm_free_fn(void *self);

static PolyUOp *wasm_program_kernel_body(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 1) return NULL;
  return program->src[0];
}

EM_JS(int, js_host_copy_out_to_wasm, (uintptr_t src_key, uint8_t *dst, int nbytes), {
  const map = Module.__polygradHostBuffers;
  const src = map && map.get(String(src_key));
  if (!src) return -1;
  const bytes = new Uint8Array(src.buffer, src.byteOffset, Math.min(nbytes, src.byteLength));
  HEAPU8.set(bytes, dst);
  return 0;
})

EM_JS(int, js_host_copy_in_from_wasm, (uintptr_t dst_key, const uint8_t *src, int nbytes), {
  const map = Module.__polygradHostBuffers;
  const dst = map && map.get(String(dst_key));
  if (!dst) return -1;
  const out = new Uint8Array(dst.buffer, dst.byteOffset, Math.min(nbytes, dst.byteLength));
  out.set(HEAPU8.subarray(src, src + out.byteLength));
  return 0;
})

EM_JS(int, js_compile_wasm_kernel, (const uint8_t *bytes, int len), {
  var mod = new WebAssembly.Module(HEAPU8.subarray(bytes, bytes + len));
  var imports = {env : {memory : wasmMemory}, math : {exp2f : function(x){return Math.pow(2, x); },
      log2f: function(x) {
  return Math.log2(x); },
      sinf:  function(x) {
  return Math.sin(x); },
      powf:  function(x, y) {
  return Math.pow(x, y); }
}
}
;
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
  if (Module._polyKernelCache && kernel_id >= 0 && kernel_id < Module._polyKernelCache.length) {
    Module._polyKernelCache[kernel_id] = null;
  }
});

int poly_browser_host_copy_out(uintptr_t src_buffer_key, void *dst_ptr, size_t nbytes) {
  return js_host_copy_out_to_wasm(src_buffer_key, (uint8_t *)dst_ptr, (int)nbytes);
}

int poly_browser_host_copy_in(uintptr_t dst_buffer_key, const void *src_ptr, size_t nbytes) {
  return js_host_copy_in_from_wasm(dst_buffer_key, (const uint8_t *)src_ptr, (int)nbytes);
}

static void *wasm_alloc(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  return calloc(1, nbytes);
}

static void wasm_free_alloc(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  free(buffer ? buffer->ptr : NULL);
}

static int wasm_copy_in(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  if (!dst || !dst->ptr || !src) return -1;
  if (src->device == POLY_DEVICE_HOST) {
    if (src->ptr) {
      memcpy(dst->ptr, src->ptr, n);
      return 0;
    }
    return poly_browser_host_copy_out((uintptr_t)(src->src ? src->src : src), dst->ptr, n);
  }
  if (!src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int wasm_copy_out(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  if (!dst || !src || !src->ptr) return -1;
  if (dst->device == POLY_DEVICE_HOST) {
    if (dst->ptr) {
      memcpy(dst->ptr, src->ptr, n);
      return 0;
    }
    return poly_browser_host_copy_in((uintptr_t)(dst->src ? dst->src : dst), src->ptr, n);
  }
  if (!dst->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int wasm_copy_between(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static const PolyAllocator POLY_WASM_ALLOCATOR = {
    .alloc = wasm_alloc,
    .free = wasm_free_alloc,
    .copy_in = wasm_copy_in,
    .copy_out = wasm_copy_out,
    .copy_between = wasm_copy_between,
    .host_addressable = true,
    .dev_ctx = NULL,
};

int poly_wasm_lower_item(
    PolyCtx *ctx,
    PolyUOp *program,
    const char *fn_name,
    PolyRunner *out
) {
  (void)fn_name;
  PolyUOp *scheduled_root = wasm_program_kernel_body(program);
  if (!scheduled_root) return -1;
  int n_lin;
  bool lin_owned = false;
  PolyUOp *linear = poly_program_linear(program);
  PolyUOp **lin = NULL;
  if (linear) {
    n_lin = linear->n_src;
    lin = linear->src;
  } else {
    lin = poly_linearize_rewritten(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return -1;

  int wasm_len = 0;
  uint8_t *wasm_bytes = poly_render_wasm(lin, n_lin, &wasm_len, false);
  if (lin_owned) free(lin);
  if (!wasm_bytes || wasm_len <= 0) return -1;

  int kernel_id = js_compile_wasm_kernel(wasm_bytes, wasm_len);
  free(wasm_bytes);
  if (kernel_id < 0) return -1;

  PolyWasmJitRunnerHandle *wh = malloc(sizeof(PolyWasmJitRunnerHandle));
  if (!wh) {
    js_free_wasm_kernel(kernel_id);
    return -1;
  }
  wh->kernel_id = kernel_id;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = wh;
  out->handle_size = 0;
  out->execute = poly_wasm_execute_fn;
  out->free_handle = poly_wasm_free_fn;
  return 0;
}

int poly_wasm_execute(PolyRunner *runner, void **args, int n_args) {
  PolyWasmJitRunnerHandle *wh = (PolyWasmJitRunnerHandle *)runner->handle;
  if (!wh) return -1;
  int *iargs = malloc((size_t)n_args * sizeof(int));
  if (!iargs) return -1;
  for (int i = 0; i < n_args; i++) iargs[i] = (int)(intptr_t)args[i];
  int ret = js_exec_wasm_kernel(wh->kernel_id, iargs, n_args);
  free(iargs);
  return ret;
}

void poly_wasm_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    PolyWasmJitRunnerHandle *wh = (PolyWasmJitRunnerHandle *)runner->handle;
    js_free_wasm_kernel(wh->kernel_id);
    free(wh);
  }
}

const PolyAllocator *poly_wasm_get_allocator(void) {
  return &POLY_WASM_ALLOCATOR;
}

static int poly_wasm_execute_fn(void *self, void **args, int n_args) {
  return poly_wasm_execute((PolyRunner *)self, args, n_args);
}

static void poly_wasm_free_fn(void *self) {
  poly_wasm_free_runner((PolyRunner *)self);
}

#endif /* __EMSCRIPTEN__ */
