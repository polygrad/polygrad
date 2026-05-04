#include "runtime_webgpu.h"

#ifdef __EMSCRIPTEN__

#include "codegen.h"
#include "utils.h"

#include <emscripten.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  char *wgsl;
  char *entry;
  uintptr_t pipeline_id;
  int n_bindings;
} PolyWebGpuRunnerHandle;

static void webgpu_extract_dims(PolyUOp **lin, int n_lin, int grid[3], int local[3]) {
  grid[0] = 1;
  grid[1] = 1;
  grid[2] = 1;
  local[0] = 1;
  local[1] = 1;
  local[2] = 1;

  for (int i = 0; i < n_lin; i++) {
    if (lin[i]->op != POLY_OP_SPECIAL || !lin[i]->arg.str || lin[i]->n_src < 1) continue;
    if (lin[i]->src[0]->op != POLY_OP_CONST) continue;
    const char *name = lin[i]->arg.str;
    int slen = (int)strlen(name);
    int dim_idx = (slen > 0) ? name[slen - 1] - '0' : 0;
    if (dim_idx < 0 || dim_idx > 2) dim_idx = 0;
    int bound = (int)lin[i]->src[0]->arg.i;
    if (name[0] == 'l')
      local[dim_idx] = bound;
    else
      grid[dim_idx] = bound;
  }
}

static bool webgpu_dtype_is_unsupported(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  /* tinygrad's WEBGPU dtype support excludes float64. WGSL has no normal f64
   * arithmetic path, so treating double as f32 here silently corrupts values. */
  return s.priority == POLY_FLOAT64.priority && s.bitsize == POLY_FLOAT64.bitsize;
}

static bool webgpu_graph_has_unsupported_dtype(PolyCtx *ctx, PolyUOp *root) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    if (webgpu_dtype_is_unsupported(topo[i]->dtype)) return true;
  }
  return false;
}

EM_JS(uintptr_t, js_webgpu_create_buffer, (size_t nbytes), {
  const st = Module.__polygradWebGpuState;
  if (!st || !st.device) return 0;
  const id = st.nextBufferId++;
  const size = Math.max(4, Math.ceil(nbytes / 4) * 4);
  const buf = st.device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
  });
  st.buffers.set(id, buf);
  st.bufferSizes.set(id, size);
  return id;
})

EM_JS(void, js_webgpu_destroy_buffer, (uintptr_t handle), {
  const st = Module.__polygradWebGpuState;
  if (!st) return;
  const buf = st.buffers.get(handle);
  if (buf) buf.destroy();
  st.buffers.delete(handle);
  st.bufferSizes.delete(handle);
})

EM_JS(int, js_webgpu_write_buffer_from_wasm, (uintptr_t handle, const uint8_t *src, int nbytes), {
  const st = Module.__polygradWebGpuState;
  const buf = st && st.buffers.get(handle);
  if (!buf) return -1;
  const logical = Math.max(0, nbytes | 0);
  if (logical === 0) return 0;
  const writeBytes = Math.max(4, Math.ceil(logical / 4) * 4);
  /* WebGPU requires queue.writeBuffer data length to be a multiple of 4.
   * The GPU allocation is rounded up, but typed-array inputs can be uint8. */
  const data = new Uint8Array(writeBytes);
  data.set(HEAPU8.subarray(src, src + logical));
  st.device.queue.writeBuffer(buf, 0, data);
  return 0;
})

EM_JS(int, js_webgpu_write_buffer_from_hostkey, (uintptr_t handle, uintptr_t src_key, int nbytes), {
  const st = Module.__polygradWebGpuState;
  const buf = st && st.buffers.get(handle);
  const map = Module.__polygradHostBuffers;
  const src = map && map.get(String(src_key));
  if (!buf || !src) return -1;
  const logical = Math.max(0, nbytes | 0);
  if (logical === 0) return 0;
  const writeBytes = Math.max(4, Math.ceil(logical / 4) * 4);
  /* Browser HOST buffers are JS-owned TypedArrays. Pad the upload bytes, not
   * the logical tensor size, so readback still returns exactly nbytes. */
  const copyBytes = Math.min(logical, src.byteLength);
  const data = new Uint8Array(writeBytes);
  data.set(new Uint8Array(src.buffer, src.byteOffset, copyBytes));
  st.device.queue.writeBuffer(buf, 0, data);
  return 0;
})

EM_ASYNC_JS(int, js_webgpu_read_buffer_to_wasm, (uint8_t *dst, uintptr_t handle, int nbytes), {
  const st = await Module.__polygradEnsureWebGPU();
  const src = st.buffers.get(handle);
  if (!src) return -1;
  const logical = Math.max(0, nbytes | 0);
  if (logical === 0) return 0;
  const copyBytes = Math.max(4, Math.ceil(logical / 4) * 4);
  const staging = st.device.createBuffer({
    size: copyBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });
  const enc = st.device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, copyBytes);
  st.device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const mapped = staging.getMappedRange();
  HEAPU8.set(new Uint8Array(mapped, 0, logical), dst);
  staging.unmap();
  staging.destroy();
  return 0;
})

EM_ASYNC_JS(int, js_webgpu_read_buffer_to_hostkey, (uintptr_t dst_key, uintptr_t handle, int nbytes), {
  const st = await Module.__polygradEnsureWebGPU();
  const src = st.buffers.get(handle);
  const map = Module.__polygradHostBuffers;
  const dst = map && map.get(String(dst_key));
  if (!src || !dst) return -1;
  const logical = Math.max(0, nbytes | 0);
  if (logical === 0) return 0;
  const copyBytes = Math.max(4, Math.ceil(logical / 4) * 4);
  const staging = st.device.createBuffer({
    size: copyBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });
  const enc = st.device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, copyBytes);
  st.device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const mapped = staging.getMappedRange();
  const out = new Uint8Array(dst.buffer, dst.byteOffset, Math.min(logical, dst.byteLength));
  out.set(new Uint8Array(mapped, 0, out.byteLength));
  staging.unmap();
  staging.destroy();
  return 0;
})

EM_JS(int, js_webgpu_copy_buffer_to_buffer, (uintptr_t dst_handle, uintptr_t src_handle, int nbytes), {
  const st = Module.__polygradWebGpuState;
  const dst = st && st.buffers.get(dst_handle);
  const src = st && st.buffers.get(src_handle);
  if (!dst || !src) return -1;
  const logical = Math.max(0, nbytes | 0);
  if (logical === 0) return 0;
  const copyBytes = Math.max(4, Math.ceil(logical / 4) * 4);
  const dstSize = st.bufferSizes.get(dst_handle) || copyBytes;
  const srcSize = st.bufferSizes.get(src_handle) || copyBytes;
  const enc = st.device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, dst, 0, Math.min(copyBytes, dstSize, srcSize));
  st.device.queue.submit([enc.finish()]);
  return 0;
})

EM_JS(int, js_webgpu_memset_zero_impl, (uintptr_t handle, int nbytes), {
  const st = Module.__polygradWebGpuState;
  const buf = st && st.buffers.get(handle);
  if (!buf) return -1;
  const logical = Math.max(0, nbytes | 0);
  if (logical === 0) return 0;
  const writeBytes = st.bufferSizes.get(handle) || Math.max(4, Math.ceil(logical / 4) * 4);
  /* Zero the rounded allocation. Later copy/read calls may use rounded WebGPU
   * transfer sizes even when the logical tensor byte count is smaller. */
  st.device.queue.writeBuffer(buf, 0, new Uint8Array(writeBytes));
  return 0;
})

EM_ASYNC_JS(
    uintptr_t,
    js_webgpu_get_or_create_pipeline,
    (const char *wgsl_ptr, const char *entry_ptr, int n_args, int n_params),
    {
  const st = await Module.__polygradEnsureWebGPU();
  const wgsl = UTF8ToString(wgsl_ptr);
  const entry = UTF8ToString(entry_ptr);
  const key = wgsl + '::' + entry + '::' + n_args + '::' + n_params;
  const cached = st.pipelineKeyToId.get(key);
  if (cached) return cached;

  const shaderModule = st.device.createShaderModule({ code: wgsl });
  const info = await shaderModule.getCompilationInfo();
  let hasError = false;
  for (const msg of info.messages) {
    if (msg.type === 'error') hasError = true;
    if (msg.type === 'error' || msg.type === 'warning') {
      console.log(
          '[polygrad:webgpu:shader] ' + msg.type +
          ' line=' + msg.lineNum + ' pos=' + msg.linePos +
          ' len=' + msg.length + ' : ' + msg.message);
    }
  }
  if (hasError) {
    console.log('[polygrad:webgpu:shader] WGSL source for failed pipeline ' + entry + ':\n' + wgsl);
  }
  const entries = [{
    binding: 0,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: 'uniform' }
  }];
  for (let i = 0; i < n_args; i++) {
    entries.push({
      binding: i + 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: i < n_params ? 'storage' : 'uniform' }
    });
  }
  const bindGroupLayout = st.device.createBindGroupLayout({ entries });
  const pipelineLayout = st.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const pipeline = await st.device.createComputePipelineAsync({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: entry }
  });

  const id = st.nextPipelineId++;
  st.pipelines.set(id, { pipeline, bindGroupLayout, entry, wgsl });
  st.pipelineKeyToId.set(key, id);
  return id;
})

EM_ASYNC_JS(
    int,
    js_webgpu_dispatch,
    (uintptr_t pipeline_id, const uintptr_t *args, int n_args, int n_params, int gx, int gy, int gz, int debug_level),
    {
  const st = await Module.__polygradEnsureWebGPU();
  const rec = st.pipelines.get(pipeline_id);
  if (!rec) return -1;

  const bgEntries = [{ binding: 0, resource: { buffer: st.infinityBuf } }];
  const tempUniforms = [];
  const tempCopies = [];
  const outHandle = n_params > 0 ? HEAPU32[args >> 2] : 0;

  const paramHandles = [];
  for (let i = 0; i < n_params; i++) {
    const handle = HEAPU32[(args >> 2) + i];
    paramHandles.push(handle);
    let buf = st.buffers.get(handle);
    if (!buf) return -1;
    if (i > 0 && handle === outHandle) {
      const copyBuf = st.device.createBuffer({
        size: st.bufferSizes.get(handle),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      });
      const copyEnc = st.device.createCommandEncoder();
      copyEnc.copyBufferToBuffer(buf, 0, copyBuf, 0, st.bufferSizes.get(handle));
      st.device.queue.submit([copyEnc.finish()]);
      tempCopies.push(copyBuf);
      buf = copyBuf;
    }
    bgEntries.push({ binding: i + 1, resource: { buffer: buf } });
  }

  for (let i = n_params; i < n_args; i++) {
    const valuePtr = HEAPU32[(args >> 2) + i];
    const value = HEAP32[valuePtr >> 2];
    const ubuf = st.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    st.device.queue.writeBuffer(ubuf, 0, new Int32Array([value]));
    tempUniforms.push(ubuf);
    bgEntries.push({ binding: i + 1, resource: { buffer: ubuf } });
  }

  if (debug_level >= 7) {
    const desc = paramHandles.map((handle, i) => {
      const size = st.bufferSizes.get(handle) || 0;
      return `p${i}=h${handle}[${size}]`;
    }).join(' ');
    console.log(
      `[polygrad:webgpu:dispatch] entry=${rec.entry} pipeline=${pipeline_id} ` +
      `grid=${gx},${gy},${gz} n_params=${n_params} n_args=${n_args} ${desc}`
    );
  }

  if (debug_level >= 8) {
    const dumpBuffer = async (handle, label) => {
      const src = st.buffers.get(handle);
      const nbytes = st.bufferSizes.get(handle) || 0;
      const dumpBytes = Math.min(nbytes, 64);
      if (!src || dumpBytes <= 0) return;
      const staging = st.device.createBuffer({
        size: Math.max(4, (dumpBytes + 3) & ~3),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      });
      const enc = st.device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, staging, 0, dumpBytes);
      st.device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const mapped = staging.getMappedRange();
      const u8 = Array.from(new Uint8Array(mapped, 0, dumpBytes));
      const wordCount = Math.floor(dumpBytes / 4);
      const f32 = Array.from(new Float32Array(mapped, 0, wordCount));
      const i32 = Array.from(new Int32Array(mapped, 0, wordCount));
      console.log(
        `[polygrad:webgpu:buffer] ${label} handle=${handle} nbytes=${nbytes} ` +
        `i32=${JSON.stringify(i32)} f32=${JSON.stringify(f32)} u8=${JSON.stringify(u8)}`
      );
      staging.unmap();
      staging.destroy();
    };
    for (let i = 0; i < paramHandles.length; i++) {
      const handle = paramHandles[i];
      const size = st.bufferSizes.get(handle) || 0;
      if (size <= 64) await dumpBuffer(handle, `param${i}`);
    }
  }

  const bindGroup = st.device.createBindGroup({
    layout: rec.bindGroupLayout,
    entries: bgEntries
  });

  const encoder = st.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(rec.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(gx, gy, gz);
  pass.end();
  st.device.queue.submit([encoder.finish()]);

  if (tempUniforms.length || tempCopies.length) {
    await st.device.queue.onSubmittedWorkDone();
  }
  for (const ubuf of tempUniforms) ubuf.destroy();
  for (const buf of tempCopies) buf.destroy();
  return 0;
})

static void *webgpu_alloc(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  uintptr_t handle = js_webgpu_create_buffer(nbytes);
  return handle ? (void *)handle : NULL;
}

static void webgpu_free(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  if (!buffer || !buffer->ptr) return;
  js_webgpu_destroy_buffer((uintptr_t)buffer->ptr);
}

static int webgpu_copy_in(const PolyBuffer *dst, const PolyBuffer *src, size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  if (!dst || !dst->ptr || !src) return -1;
  if (src->device == POLY_DEVICE_HOST) {
    return js_webgpu_write_buffer_from_hostkey(
        (uintptr_t)dst->ptr, (uintptr_t)(src->src ? src->src : src), (int)nbytes
    );
  }
  return js_webgpu_write_buffer_from_wasm((uintptr_t)dst->ptr, (const uint8_t *)src->ptr, (int)nbytes);
}

static int webgpu_copy_out(const PolyBuffer *dst, const PolyBuffer *src, size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  if (!dst || !src || !src->ptr) return -1;
  if (dst->device == POLY_DEVICE_HOST) {
    return js_webgpu_read_buffer_to_hostkey(
        (uintptr_t)(dst->src ? dst->src : dst), (uintptr_t)src->ptr, (int)nbytes
    );
  }
  return js_webgpu_read_buffer_to_wasm((uint8_t *)dst->ptr, (uintptr_t)src->ptr, (int)nbytes);
}

static int webgpu_copy_between(const PolyBuffer *dst, const PolyBuffer *src, size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  if (!dst || !src || !dst->ptr || !src->ptr) return -1;
  return js_webgpu_copy_buffer_to_buffer((uintptr_t)dst->ptr, (uintptr_t)src->ptr, (int)nbytes);
}

static const PolyAllocator POLY_WEBGPU_ALLOCATOR = {
  .alloc = webgpu_alloc,
  .free = webgpu_free,
  .copy_in = webgpu_copy_in,
  .copy_out = webgpu_copy_out,
  .copy_between = webgpu_copy_between,
  .dev_ctx = NULL,
  .host_addressable = false,
};

int poly_webgpu_memset_zero(uintptr_t handle, size_t nbytes) {
  return js_webgpu_memset_zero_impl(handle, (int)nbytes);
}

int poly_webgpu_lower_item(PolyCtx *ctx, PolyUOp *scheduled_root, const char *fn_name, PolyRunner *out) {
  if (webgpu_graph_has_unsupported_dtype(ctx, scheduled_root)) {
    fprintf(stderr, "polygrad: webgpu: float64 kernels are not supported\n");
    return -1;
  }

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, scheduled_root, &n_lin);
  if (!lin) return -1;

  int grid[3], local[3];
  webgpu_extract_dims(lin, n_lin, grid, local);

  char *wgsl = poly_render_wgsl(lin, n_lin, fn_name);
  free(lin);
  if (!wgsl) return -1;
  if (poly_dump_kernels_enabled())
    fprintf(stderr, "=== WEBGPU KERNEL %s ===\n%s\n=== END ===\n", fn_name, wgsl);

  PolyWebGpuRunnerHandle *wh = calloc(1, sizeof(PolyWebGpuRunnerHandle));
  if (!wh) {
    free(wgsl);
    return -1;
  }
  wh->wgsl = wgsl;
  wh->entry = malloc(strlen(fn_name) + 1);
  if (wh->entry) strcpy(wh->entry, fn_name);
  if (!wh->entry) {
    free(wh->wgsl);
    free(wh);
    return -1;
  }

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = wh;
  out->handle_size = (int)strlen(wgsl);
  out->grid[0] = grid[0];
  out->grid[1] = grid[1];
  out->grid[2] = grid[2];
  out->block[0] = local[0];
  out->block[1] = local[1];
  out->block[2] = local[2];
  return 0;
}

int poly_webgpu_execute(PolyRunner *runner, void **args, int n_args) {
  if (!runner || !runner->handle) return -1;
  PolyWebGpuRunnerHandle *wh = (PolyWebGpuRunnerHandle *)runner->handle;
  if (!wh->pipeline_id || wh->n_bindings != n_args) {
    wh->pipeline_id = js_webgpu_get_or_create_pipeline(wh->wgsl, wh->entry, n_args, runner->n_params);
    wh->n_bindings = n_args;
    if (!wh->pipeline_id) return -1;
  }
  return js_webgpu_dispatch(
      wh->pipeline_id,
      (const uintptr_t *)args,
      n_args,
      runner->n_params,
      runner->grid[0],
      runner->grid[1],
      runner->grid[2],
      poly_debug_level());
}

void poly_webgpu_free_runner(PolyRunner *runner) {
  if (!runner || !runner->handle) return;
  PolyWebGpuRunnerHandle *wh = (PolyWebGpuRunnerHandle *)runner->handle;
  free(wh->wgsl);
  free(wh->entry);
  free(wh);
  runner->handle = NULL;
}

const PolyAllocator *poly_webgpu_get_allocator(void) {
  return &POLY_WEBGPU_ALLOCATOR;
}

#endif
