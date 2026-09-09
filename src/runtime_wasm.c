#include "runtime_wasm.h"

#ifdef __EMSCRIPTEN__

#include "codegen/codegen.h"
#include "utils.h"

#include <emscripten.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int kernel_id;
  int *iargs;
  int iargs_cap;
} PolyWasmJitRunnerHandle;

static int poly_wasm_execute_fn(void *self, void **args, int n_args);
static void poly_wasm_free_fn(void *self);

static PolyUOp *wasm_program_kernel_body(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 1) return NULL;
  return program->src[0];
}

/* Embedded JavaScript is not C; clang-format corrupts its operators, regexes,
 * and template literals. Keep this region out of the C formatting gate. */
// clang-format off
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
  var mod;
  try {
    mod = new WebAssembly.Module(HEAPU8.subarray(bytes, bytes + len));
  } catch (e) {
    var dbg = 0;
    if (typeof globalThis !== 'undefined' && globalThis.__polygradDebugLevel)
      dbg = Number(globalThis.__polygradDebugLevel) | 0;
    if (typeof process !== 'undefined' && process.env) {
      var envDbg = Number(process.env.POLY_DEBUG || process.env.DEBUG || 0);
      if (Number.isFinite(envDbg) && envDbg > dbg) dbg = envDbg | 0;
    }
    if (dbg >= 4)
      console.error(
          '[polygrad:wasm] WebAssembly.Module failed len=' + len + ': ' +
          (e && e.message ? e.message : String(e))
      );
    if (dbg >= 4) {
      var msg = e && e.message ? e.message : String(e);
      var m = /@\\+(\\d+)/.exec(msg);
      if (m) {
        var off = Number(m[1]) | 0;
        var start = Math.max(0, off - 32), end = Math.min(len, off + 32);
        var windowBytes = HEAPU8.subarray(bytes + start, bytes + end);
        var hex = [];
        for (var i = 0; i < windowBytes.length; i++)
          hex.push(windowBytes[i].toString(16).padStart(2, '0'));
        console.error('[polygrad:wasm] bytes ' + start + '..' + end + ': ' + hex.join(' '));
      }
    }
    return -1;
  }
  var imports = {env : {memory : wasmMemory}, math : {exp2f : function(x){return Math.pow(2, x); },
      log2f: function(x) {
  return Math.log2(x); },
      sinf:  function(x) {
  return Math.sin(x); },
      exp2: function(x) {
  return Math.pow(2, x); },
      log2: function(x) {
  return Math.log2(x); },
      sin: function(x) {
  return Math.sin(x); },
      powf:  function(x, y) {
  return Math.pow(x, y); },
      pow:  function(x, y) {
  return Math.pow(x, y); }
}
}
;
var inst = new WebAssembly.Instance(mod, imports);
if (!Module._polyKernelCache) Module._polyKernelCache = [];
var kernel = inst.exports.kernel;
var n_params = kernel.length;
var launcher;
switch (n_params) {
case 0:
  launcher = function(args) {
    kernel();
  };
  break;
case 1:
  launcher = function(args) {
    var h = HEAP32, p = args >> 2;
    kernel(h[p]);
  };
  break;
case 2:
  launcher = function(args) {
    var h = HEAP32, p = args >> 2;
    kernel(h[p], h[p + 1]);
  };
  break;
case 3:
  launcher = function(args) {
    var h = HEAP32, p = args >> 2;
    kernel(h[p], h[p + 1], h[p + 2]);
  };
  break;
case 4:
  launcher = function(args) {
    var h = HEAP32, p = args >> 2;
    kernel(h[p], h[p + 1], h[p + 2], h[p + 3]);
  };
  break;
case 5:
  launcher = function(args) {
    var h = HEAP32, p = args >> 2;
    kernel(h[p], h[p + 1], h[p + 2], h[p + 3], h[p + 4]);
  };
  break;
case 6:
  launcher = function(args) {
    var h = HEAP32, p = args >> 2;
    kernel(h[p], h[p + 1], h[p + 2], h[p + 3], h[p + 4], h[p + 5]);
  };
  break;
case 7:
  launcher = function(args) {
    var h = HEAP32, p = args >> 2;
    kernel(h[p], h[p + 1], h[p + 2], h[p + 3], h[p + 4], h[p + 5], h[p + 6]);
  };
  break;
case 8:
  launcher = function(args) {
    var h = HEAP32, p = args >> 2;
    kernel(h[p], h[p + 1], h[p + 2], h[p + 3], h[p + 4], h[p + 5], h[p + 6], h[p + 7]);
  };
  break;
default:
  launcher = function(args) {
    var h = HEAP32, p = args >> 2, params = [];
    for (var i = 0; i < n_params; i++)
      params.push(h[p + i]);
    kernel.apply(null, params);
  };
  break;
}
Module._polyKernelCache.push({inst : inst, launch : launcher});
return Module._polyKernelCache.length - 1;
});

EM_JS(int, js_exec_wasm_kernel, (int kernel_id, const int *args, int n_args), {
  n_args = n_args | 0;
  var entry = Module._polyKernelCache && Module._polyKernelCache[kernel_id];
  if (!entry || !entry.launch) return -1;
  entry.launch(args);
  return 0;
});

EM_JS(void, js_free_wasm_kernel, (int kernel_id), {
  if (Module._polyKernelCache && kernel_id >= 0 && kernel_id < Module._polyKernelCache.length) {
    Module._polyKernelCache[kernel_id] = null;
  }
});
// clang-format on

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

static int wasm_copy_between(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t n,
    void *dev_ctx
) {
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

int poly_wasm_lower_item(PolyCtx *ctx, PolyUOp *program, const char *fn_name, PolyRunner *out) {
  (void)fn_name;
  PolyUOp *scheduled_root = wasm_program_kernel_body(program);
  if (!scheduled_root) return -1;
  int n_lin;
  bool lin_owned = false;
  PolyUOp *linear = poly_program_linear(program);
  PolyUOp **lin = NULL;

  int wasm_len = 0;
  int kernel_id = -1;
  uint8_t *wasm_bytes = poly_render_wasm_matmul(scheduled_root, &wasm_len, true);
  if (wasm_bytes && wasm_len > 0) {
    kernel_id = js_compile_wasm_kernel(wasm_bytes, wasm_len);
    if (kernel_id >= 0 && out->capture_binary)
      out->compiled_binary =
          poly_uop0(ctx, POLY_OP_BINARY, POLY_UINT8, poly_arg_bytes(wasm_bytes, wasm_len));
    free(wasm_bytes);
    wasm_bytes = NULL;
    wasm_len = 0;
    if (kernel_id < 0) wasm_bytes = poly_render_wasm_matmul(scheduled_root, &wasm_len, false);
  }
  if (kernel_id < 0 && !wasm_bytes) {
    wasm_bytes = poly_render_wasm_reduce(scheduled_root, &wasm_len);
  }
  if (kernel_id < 0 && !wasm_bytes) {
    if (linear) {
      n_lin = linear->n_src;
      lin = linear->src;
    } else {
      lin = poly_do_linearize(ctx, scheduled_root, &n_lin);
      lin_owned = true;
    }
    if (!lin) return -1;
    wasm_bytes = poly_render_wasm(ctx, lin, n_lin, &wasm_len, true);
    if (lin_owned) free(lin);
  }
  if (kernel_id < 0 && (!wasm_bytes || wasm_len <= 0)) return -1;

  if (kernel_id < 0) {
    kernel_id = js_compile_wasm_kernel(wasm_bytes, wasm_len);
    if (kernel_id >= 0 && out->capture_binary)
      out->compiled_binary =
          poly_uop0(ctx, POLY_OP_BINARY, POLY_UINT8, poly_arg_bytes(wasm_bytes, wasm_len));
  }
  free(wasm_bytes);
  if (kernel_id < 0) return -1;
  if (out->capture_binary && !out->compiled_binary) {
    js_free_wasm_kernel(kernel_id);
    return -1;
  }

  PolyWasmJitRunnerHandle *wh = malloc(sizeof(PolyWasmJitRunnerHandle));
  if (!wh) {
    js_free_wasm_kernel(kernel_id);
    return -1;
  }
  wh->kernel_id = kernel_id;
  wh->iargs = NULL;
  wh->iargs_cap = 0;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = wh;
  out->handle_size = (int)sizeof(*wh);
  out->execute = poly_wasm_execute_fn;
  out->free_handle = poly_wasm_free_fn;
  return 0;
}

int poly_wasm_execute(PolyRunner *runner, void **args, int n_args) {
  PolyWasmJitRunnerHandle *wh = (PolyWasmJitRunnerHandle *)runner->handle;
  if (!wh || n_args < 0) return -1;
  bool timing = poly_debug_at_least(7);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (n_args > wh->iargs_cap) {
    int new_cap = wh->iargs_cap > 0 ? wh->iargs_cap : 8;
    while (new_cap < n_args)
      new_cap *= 2;
    int *new_iargs = realloc(wh->iargs, (size_t)new_cap * sizeof(*new_iargs));
    if (!new_iargs) return -1;
    wh->iargs = new_iargs;
    wh->iargs_cap = new_cap;
  }
  /* The shared runner ABI passes buffers as addresses, then scalar values
   * by address. Wasm parameters are i32 values in both cases, like the
   * buffer/vals split in Tinygrad Program.__call__, not all pointer casts. */
  for (int i = 0; i < n_args; i++)
    wh->iargs[i] = i < runner->n_params ? (int)(intptr_t)args[i] : *(const int *)args[i];
  double t_args = timing ? poly_now_ms() : 0.0;
  int ret = js_exec_wasm_kernel(wh->kernel_id, wh->iargs, n_args);
  double t_exec = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:wasm_execute] kernel=%d args=%d pack=%.3fms exec=%.3fms total=%.3fms ret=%d\n",
        wh->kernel_id, n_args, t_args - t0, t_exec - t_args, t_exec - t0, ret
    );
    fflush(stderr);
  }
  return ret;
}

void poly_wasm_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    PolyWasmJitRunnerHandle *wh = (PolyWasmJitRunnerHandle *)runner->handle;
    js_free_wasm_kernel(wh->kernel_id);
    free(wh->iargs);
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
