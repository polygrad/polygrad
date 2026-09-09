#include "runtime_webgpu.h"

#ifndef __EMSCRIPTEN__

bool poly_webgpu_supports_float16(void) {
  return false;
}

#else

#include "codegen/codegen.h"
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

/* Embedded JavaScript is not C; keep its operators out of clang-format. */
// clang-format off
EM_JS(int, js_webgpu_supports_float16, (), {
  const st = Module.__polygradWebGpuState;
  return st && st.hasShaderF16 ? 1 : 0;
})
// clang-format on

bool poly_webgpu_supports_float16(void) {
  return js_webgpu_supports_float16() != 0;
}

static PolyUOp *webgpu_program_kernel_body(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 1) return NULL;
  return program->src[0];
}

static const char *webgpu_program_source_text(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 3) return NULL;
  PolyUOp *source = program->src[2];
  if (!source || source->op != POLY_OP_SOURCE || source->arg.kind != POLY_ARG_STRING) return NULL;
  return source->arg.str;
}

static char *webgpu_strdup(const char *s) {
  if (!s) return NULL;
  size_t n = strlen(s) + 1;
  char *out = malloc(n);
  if (out) memcpy(out, s, n);
  return out;
}

static int webgpu_launch_dim_upper_bound(PolyCtx *ctx, PolyUOp *expr) {
  if (!expr) return 1;
  int64_t lo = 0, hi = 1;
  poly_uop_minmax(ctx, expr, &lo, &hi);
  if (hi <= 0) return 1;
  if (hi > INT32_MAX) return INT32_MAX;
  return (int)hi;
}

static void webgpu_extract_dims(PolyCtx *ctx, PolyUOp **lin, int n_lin, int grid[3], int local[3]) {
  grid[0] = 1;
  grid[1] = 1;
  grid[2] = 1;
  local[0] = 1;
  local[1] = 1;
  local[2] = 1;

  for (int i = 0; i < n_lin; i++) {
    if (lin[i]->op != POLY_OP_SPECIAL || !lin[i]->arg.str || lin[i]->n_src < 1) continue;
    const char *name = lin[i]->arg.str;
    int slen = (int)strlen(name);
    int dim_idx = (slen > 0) ? name[slen - 1] - '0' : 0;
    if (dim_idx < 0 || dim_idx > 2) dim_idx = 0;
    int bound = webgpu_launch_dim_upper_bound(ctx, lin[i]->src[0]);
    if (name[0] == 'l')
      local[dim_idx] = bound;
    else
      grid[dim_idx] = bound;
  }
}

static bool webgpu_dtype_is_unsupported(PolyDType dt) {
  PolyDType s = dt;
  /* tinygrad's WEBGPU dtype support excludes float64. WGSL has no normal f64
   * arithmetic path, so treating double as f32 here silently corrupts values. */
  return s.priority == POLY_FLOAT64.priority && s.bitsize == POLY_FLOAT64.bitsize;
}

static bool webgpu_graph_has_unsupported_dtype(PolyCtx *ctx, PolyUOp *root) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n_topo);
  bool unsupported = false;
  for (int i = 0; i < n_topo; i++) {
    if (webgpu_dtype_is_unsupported(topo[i]->dtype)) {
      unsupported = true;
      break;
    }
  }
  poly_toposort_free(topo);
  return unsupported;
}

// clang-format off
EM_JS(uintptr_t, js_webgpu_create_buffer, (size_t nbytes), {
  const st = Module.__polygradWebGpuState;
  if (!st || !st.device) return 0;
  const id = st.nextBufferId++;
  const size = Math.max(4, Math.ceil(nbytes / 4) * 4);
  const buf = st.device.createBuffer({
    size,
    usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST |
        GPUBufferUsage.UNIFORM
  });
  st.buffers.set(id, buf);
  st.bufferSizes.set(id, size);
  if (!st.bufferOffsets) st.bufferOffsets = new Map();
  if (!st.bufferViews) st.bufferViews = new Set();
  st.bufferOffsets.set(id, 0);
  return id;
})

EM_JS(void, js_webgpu_destroy_buffer, (uintptr_t handle), {
  const st = Module.__polygradWebGpuState;
  if (!st) return;
  const buf = st.buffers.get(handle);
  const isView = st.bufferViews && st.bufferViews.has(handle);
  if (buf && !isView) buf.destroy();
  st.buffers.delete(handle);
  st.bufferSizes.delete(handle);
  if (st.bufferOffsets) st.bufferOffsets.delete(handle);
  if (st.bufferViews) st.bufferViews.delete(handle);
})

EM_JS(
    uintptr_t,
    js_webgpu_create_buffer_view,
    (uintptr_t base_handle, int byte_offset, int nbytes),
    {
      const st = Module.__polygradWebGpuState;
      if (!st || !st.device) return 0;
      const base = st.buffers.get(base_handle);
      if (!base) return 0;
      const baseOffset = st.bufferOffsets ? (st.bufferOffsets.get(base_handle) || 0) : 0;
      const baseSize = st.bufferSizes.get(base_handle) || 0;
      const off = Math.max(0, byte_offset | 0);
      const logical = Math.max(0, nbytes | 0);
      const size = Math.max(4, Math.ceil(logical / 4) * 4);
      if (off % 256 !== 0) return 0;
      if (logical <= 0 || off > baseSize || size > baseSize - off) return 0;
      if (!st.bufferOffsets) st.bufferOffsets = new Map();
      if (!st.bufferViews) st.bufferViews = new Set();
      const id = st.nextBufferId++;
      st.buffers.set(id, base);
      st.bufferOffsets.set(id, baseOffset + off);
      st.bufferSizes.set(id, size);
      st.bufferViews.add(id);
      return id;
    }
)

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
  const dstOffset = st.bufferOffsets ? (st.bufferOffsets.get(handle) || 0) : 0;
  st.device.queue.writeBuffer(buf, dstOffset, data);
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
  const dstOffset = st.bufferOffsets ? (st.bufferOffsets.get(handle) || 0) : 0;
  st.device.queue.writeBuffer(buf, dstOffset, data);
  return 0;
})

EM_ASYNC_JS(int, js_webgpu_read_buffer_to_wasm, (uint8_t * dst, uintptr_t handle, int nbytes), {
  const st = await Module.__polygradEnsureWebGPU();
  const src = st.buffers.get(handle);
  if (!src) return -1;
  const logical = Math.max(0, nbytes | 0);
  if (logical === 0) return 0;
  const copyBytes = Math.max(4, Math.ceil(logical / 4) * 4);
  const srcOffset = st.bufferOffsets ? (st.bufferOffsets.get(handle) || 0) : 0;
  const staging = st.device.createBuffer(
      {size : copyBytes, usage : GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST}
  );
  const enc = st.device.createCommandEncoder();
  enc.copyBufferToBuffer(src, srcOffset, staging, 0, copyBytes);
  st.device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const mapped = staging.getMappedRange();
  HEAPU8.set(new Uint8Array(mapped, 0, logical), dst);
  staging.unmap();
  staging.destroy();
  return 0;
})

EM_ASYNC_JS(
    int,
    js_webgpu_read_buffer_to_hostkey,
    (uintptr_t dst_key, uintptr_t handle, int nbytes),
    {
      const st = await Module.__polygradEnsureWebGPU();
      const src = st.buffers.get(handle);
      const map = Module.__polygradHostBuffers;
      const dst = map && map.get(String(dst_key));
      if (!src || !dst) return -1;
      const logical = Math.max(0, nbytes | 0);
      if (logical === 0) return 0;
      const copyBytes = Math.max(4, Math.ceil(logical / 4) * 4);
      const srcOffset = st.bufferOffsets ? (st.bufferOffsets.get(handle) || 0) : 0;
      const staging = st.device.createBuffer(
          {size : copyBytes, usage : GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST}
      );
      const enc = st.device.createCommandEncoder();
      enc.copyBufferToBuffer(src, srcOffset, staging, 0, copyBytes);
      st.device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const mapped = staging.getMappedRange();
      const out = new Uint8Array(dst.buffer, dst.byteOffset, Math.min(logical, dst.byteLength));
      out.set(new Uint8Array(mapped, 0, out.byteLength));
      staging.unmap();
      staging.destroy();
      return 0;
    }
)

EM_JS(
    int,
    js_webgpu_copy_buffer_to_buffer,
    (uintptr_t dst_handle, uintptr_t src_handle, int nbytes),
    {
      const st = Module.__polygradWebGpuState;
      const dst = st && st.buffers.get(dst_handle);
      const src = st && st.buffers.get(src_handle);
      if (!dst || !src) return -1;
      const logical = Math.max(0, nbytes | 0);
      if (logical === 0) return 0;
      const copyBytes = Math.max(4, Math.ceil(logical / 4) * 4);
      const dstSize = st.bufferSizes.get(dst_handle) || copyBytes;
      const srcSize = st.bufferSizes.get(src_handle) || copyBytes;
      const dstOffset = st.bufferOffsets ? (st.bufferOffsets.get(dst_handle) || 0) : 0;
      const srcOffset = st.bufferOffsets ? (st.bufferOffsets.get(src_handle) || 0) : 0;
      const enc = st.device.createCommandEncoder();
      enc.copyBufferToBuffer(src, srcOffset, dst, dstOffset, Math.min(copyBytes, dstSize, srcSize));
      st.device.queue.submit([enc.finish()]);
      return 0;
    }
)

EM_JS(int, js_webgpu_memset_zero_impl, (uintptr_t handle, int nbytes), {
  const st = Module.__polygradWebGpuState;
  const buf = st && st.buffers.get(handle);
  if (!buf) return -1;
  const logical = Math.max(0, nbytes | 0);
  if (logical === 0) return 0;
  const writeBytes = Math.max(4, Math.ceil(logical / 4) * 4);
  const dstOffset = st.bufferOffsets ? (st.bufferOffsets.get(handle) || 0) : 0;
  /* Zero the rounded allocation. Later copy/read calls may use rounded WebGPU
   * transfer sizes even when the logical tensor byte count is smaller. */
  st.device.queue.writeBuffer(buf, dstOffset, new Uint8Array(writeBytes));
  return 0;
})

EM_JS(
    uintptr_t,
    js_webgpu_get_or_create_pipeline,
    (const char *wgsl_ptr, const char *entry_ptr, int n_args, int n_params, int debug_level, int cache),
    {
      const t0 = performance.now();
      const st = Module.__polygradWebGpuState;
      if (!st || !st.device) {
        if (debug_level >= 1)
          console.log('[polygrad:webgpu:pipeline] WebGPU device is not initialized');
        return 0;
      }
      const wgsl = UTF8ToString(wgsl_ptr);
      const entry = UTF8ToString(entry_ptr);
      const key = wgsl + '::' + entry + '::' + n_args + '::' + n_params;
      const cached = cache ? st.pipelineKeyToId.get(key) : 0;
      if (cached) {
        if (debug_level >= 7) {
          console.log(`[polygrad:webgpu:pipeline] hit entry=${entry} id=${cached} wgsl=${wgsl.length}`);
        }
        return cached;
      }
      if (debug_level >= 7) {
        console.log(
      `[polygrad:webgpu:pipeline] miss entry=${entry} wgsl=${wgsl.length} ` +
      `n_args=${n_args} n_params=${n_params}`);
      }

      let shaderModule;
      let bindGroupLayout;
      let pipeline;
      try {
        shaderModule = st.device.createShaderModule({code : wgsl});
        const entries =
            [ {binding : 0, visibility : GPUShaderStage.COMPUTE, buffer : {type : 'uniform'}} ];
        for (let i = 0; i < n_args; i++) {
          entries.push({
            binding : i + 1,
            visibility : GPUShaderStage.COMPUTE,
            buffer : {type : i < n_params ? 'storage' : 'uniform'}
          });
        }
        bindGroupLayout = st.device.createBindGroupLayout({entries});
        const pipelineLayout =
            st.device.createPipelineLayout({bindGroupLayouts : [bindGroupLayout]});
        pipeline = st.device.createComputePipeline(
            {layout : pipelineLayout, compute : {module : shaderModule, entryPoint : entry}}
        );
      } catch (err) {
        console.log(
            '[polygrad:webgpu:pipeline] failed entry=' + entry + ' : ' +
            (err && err.message ? err.message : String(err))
        );
        if (debug_level >= 4)
          console.log(
              '[polygrad:webgpu:shader] WGSL source for failed pipeline ' + entry + ':\n' + wgsl
          );
        return 0;
      }

      const id = st.nextPipelineId++;
      st.pipelines.set(id, {pipeline, bindGroupLayout, entry, wgsl});
      if (cache) st.pipelineKeyToId.set(key, id);
      if (debug_level >= 7) {
        console.log(
      `[polygrad:webgpu:pipeline] ready entry=${entry} id=${id} ` +
      `ms=${(performance.now() - t0).toFixed(3)}`);
      }
      return id;
    }
)

EM_JS(void, js_webgpu_free_search_pipeline, (uintptr_t id), {
  const st = Module.__polygradWebGpuState;
  if (st) st.pipelines.delete(id);
});

EM_ASYNC_JS(
    int,
    js_webgpu_dispatch,
    (uintptr_t pipeline_id,
     const uintptr_t *args,
     int n_args,
     int n_params,
     int gx,
     int gy,
     int gz,
     int debug_level,
     double *elapsed_us),
    {
      const st = await Module.__polygradEnsureWebGPU();
      const rec = st.pipelines.get(pipeline_id);
      if (!rec) return -1;
      // Pinned WebGPU wait uses device timestamps, not queue wall time.
      if (elapsed_us && !st.device.features.has('timestamp-query')) return -1;

      const bgEntries = [ {binding : 0, resource : {buffer : st.infinityBuf}} ];
      const tempUniforms = [];
      const tempCopies = [];
      let querySet = null, queryBuffer = null, queryReadback = null;
      try {
      const outHandle = n_params > 0 ? HEAPU32[args >> 2] : 0;

      const paramHandles = [];
      for (let i = 0; i < n_params; i++) {
        const handle = HEAPU32[(args >> 2) + i];
        paramHandles.push(handle);
        let buf = st.buffers.get(handle);
        if (!buf) return -1;
        let offset = st.bufferOffsets ? (st.bufferOffsets.get(handle) || 0) : 0;
        let size = st.bufferSizes.get(handle) || 0;
        if (i > 0 && handle === outHandle) {
          const copyBuf = st.device.createBuffer({
            size,
            usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
          });
          const copyEnc = st.device.createCommandEncoder();
          copyEnc.copyBufferToBuffer(buf, offset, copyBuf, 0, size);
          st.device.queue.submit([copyEnc.finish()]);
          tempCopies.push(copyBuf);
          buf = copyBuf;
          offset = 0;
        }
        bgEntries.push({binding : i + 1, resource : {buffer : buf, offset, size}});
      }

      for (let i = n_params; i < n_args; i++) {
        const valuePtr = HEAPU32[(args >> 2) + i];
        const value = HEAP32[valuePtr >> 2];
        const ubuf = st.device.createBuffer(
            {size : 4, usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST}
        );
        st.device.queue.writeBuffer(ubuf, 0, new Int32Array([value]));
        tempUniforms.push(ubuf);
        bgEntries.push({binding : i + 1, resource : {buffer : ubuf}});
      }

      if (debug_level >= 7) {
        const desc =
            paramHandles
                .map(
                    (handle, i) =>
                                  {
                                    const size = st.bufferSizes.get(handle) || 0;
                                    const offset =
                                        st.bufferOffsets ? (st.bufferOffsets.get(handle) || 0) : 0;
                                    return `p${i} = h${handle} + ${offset}[${size}]`;
                                  }
                )
                .join(' ');
        console.log(
      `[polygrad:webgpu:dispatch] entry=${rec.entry} pipeline=${pipeline_id} ` +
      `grid=${gx},${gy},${gz} n_params=${n_params} n_args=${n_args} ${desc}`);
      }

      if (debug_level >= 8) {
        const dumpBuffer = async (handle, label) => {
          const src = st.buffers.get(handle);
          const nbytes = st.bufferSizes.get(handle) || 0;
          const offset = st.bufferOffsets ? (st.bufferOffsets.get(handle) || 0) : 0;
          const dumpBytes = Math.min(nbytes, 64);
          if (!src || dumpBytes <= 0) return;
          const staging = st.device.createBuffer({
            size : Math.max(4, (dumpBytes + 3) & ~3),
            usage : GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
          });
          const enc = st.device.createCommandEncoder();
          enc.copyBufferToBuffer(src, offset, staging, 0, dumpBytes);
          st.device.queue.submit([enc.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          const mapped = staging.getMappedRange();
          const u8 = Array.from(new Uint8Array(mapped, 0, dumpBytes));
          const wordCount = Math.floor(dumpBytes / 4);
          const f32 = Array.from(new Float32Array(mapped, 0, wordCount));
          const i32 = Array.from(new Int32Array(mapped, 0, wordCount));
          console.log(
        `[polygrad:webgpu:buffer] ${label} handle=${handle} nbytes=${nbytes} ` +
        `i32=${JSON.stringify(i32)} f32=${JSON.stringify(f32)} u8=${JSON.stringify(u8)}`);
          staging.unmap();
          staging.destroy();
        };
        for (let i = 0; i < paramHandles.length; i++) {
          const handle = paramHandles[i];
          const size = st.bufferSizes.get(handle) || 0;
          if (size <= 64)
            await dumpBuffer(
                handle, `param${i}`
            );
        }
      }

      const bindGroup =
          st.device.createBindGroup({layout : rec.bindGroupLayout, entries : bgEntries});

      const encoder = st.device.createCommandEncoder();
      const passDescriptor = {};
      if (elapsed_us) {
        querySet = st.device.createQuerySet({type: 'timestamp', count: 2});
        queryBuffer = st.device.createBuffer({size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC});
        queryReadback = st.device.createBuffer({size: 16,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
        passDescriptor.timestampWrites = {querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1};
      }
      const pass = encoder.beginComputePass(passDescriptor);
      pass.setPipeline(rec.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(gx, gy, gz);
      pass.end();
      if (elapsed_us) {
        encoder.resolveQuerySet(querySet, 0, 2, queryBuffer, 0);
        encoder.copyBufferToBuffer(queryBuffer, 0, queryReadback, 0, 16);
      }
      st.device.queue.submit([encoder.finish()]);

      if (elapsed_us) {
        await queryReadback.mapAsync(GPUMapMode.READ);
        const times = new BigUint64Array(queryReadback.getMappedRange());
        HEAPF64[elapsed_us >> 3] = Number(times[1] - times[0]) / 1000;
        queryReadback.unmap();
      } else if (tempUniforms.length || tempCopies.length) {
        await st.device.queue.onSubmittedWorkDone();
      }
      return 0;
      } catch (error) {
        console.error('polygrad: WebGPU dispatch failed:', error);
        return -1;
      } finally {
      if (queryReadback) queryReadback.destroy();
      if (queryBuffer) queryBuffer.destroy();
      if (querySet) querySet.destroy();
      for (const ubuf of tempUniforms)
        ubuf.destroy();
      for (const buf of tempCopies)
        buf.destroy();
      }
    }
)
// clang-format on

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

static int webgpu_copy_in(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t nbytes,
    void *dev_ctx
) {
  (void)dev_ctx;
  if (!dst || !dst->ptr || !src) return -1;
  if (src->device == POLY_DEVICE_HOST) {
    if (src->ptr)
      return js_webgpu_write_buffer_from_wasm(
          (uintptr_t)dst->ptr, (const uint8_t *)src->ptr, (int)nbytes
      );
    return js_webgpu_write_buffer_from_hostkey(
        (uintptr_t)dst->ptr, (uintptr_t)(src->src ? src->src : src), (int)nbytes
    );
  }
  return js_webgpu_write_buffer_from_wasm(
      (uintptr_t)dst->ptr, (const uint8_t *)src->ptr, (int)nbytes
  );
}

static int webgpu_copy_out(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t nbytes,
    void *dev_ctx
) {
  (void)dev_ctx;
  if (!dst || !src || !src->ptr) return -1;
  if (dst->device == POLY_DEVICE_HOST) {
    if (dst->ptr)
      return js_webgpu_read_buffer_to_wasm((uint8_t *)dst->ptr, (uintptr_t)src->ptr, (int)nbytes);
    return js_webgpu_read_buffer_to_hostkey(
        (uintptr_t)(dst->src ? dst->src : dst), (uintptr_t)src->ptr, (int)nbytes
    );
  }
  return js_webgpu_read_buffer_to_wasm((uint8_t *)dst->ptr, (uintptr_t)src->ptr, (int)nbytes);
}

static int webgpu_copy_between(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t nbytes,
    void *dev_ctx
) {
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

uintptr_t poly_webgpu_create_buffer_view(uintptr_t base_handle, size_t byte_offset, size_t nbytes) {
  if (byte_offset > (size_t)INT32_MAX || nbytes > (size_t)INT32_MAX) return 0;
  return js_webgpu_create_buffer_view(base_handle, (int)byte_offset, (int)nbytes);
}

char *poly_webgpu_render_source(PolyCtx *ctx, PolyUOp *program, const char *fn_name) {
  PolyUOp *scheduled_root = webgpu_program_kernel_body(program);
  if (!scheduled_root) return NULL;
  if (webgpu_graph_has_unsupported_dtype(ctx, scheduled_root)) {
    fprintf(stderr, "polygrad: webgpu: float64 kernels are not supported\n");
    return NULL;
  }

  int n_lin = 0;
  bool lin_owned = false;
  PolyUOp *linear = poly_program_linear(program);
  PolyUOp **lin = NULL;
  if (linear) {
    n_lin = linear->n_src;
    lin = linear->src;
  } else {
    lin = poly_do_linearize(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return NULL;

  char *wgsl = poly_render_wgsl(ctx, lin, n_lin, fn_name);
  if (lin_owned) free(lin);
  return wgsl;
}

int poly_webgpu_lower_item(PolyCtx *ctx, PolyUOp *program, const char *fn_name, PolyRunner *out) {
  PolyUOp *scheduled_root = webgpu_program_kernel_body(program);
  if (!scheduled_root) return -1;
  bool timing = poly_debug_at_least(7);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:webgpu:lower] begin fn=%s root=%p\n", fn_name, (void *)scheduled_root
    );
    fflush(stderr);
  }
  if (webgpu_graph_has_unsupported_dtype(ctx, scheduled_root)) {
    fprintf(stderr, "polygrad: webgpu: float64 kernels are not supported\n");
    return -1;
  }
  double t_check = timing ? poly_now_ms() : 0.0;

  int n_lin = 0;
  bool lin_owned = false;
  PolyUOp *linear = poly_program_linear(program);
  PolyUOp **lin = NULL;
  if (linear) {
    n_lin = linear->n_src;
    lin = linear->src;
  } else {
    lin = poly_do_linearize(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return -1;
  double t_lin = timing ? poly_now_ms() : 0.0;

  int grid[3], local[3];
  webgpu_extract_dims(ctx, lin, n_lin, grid, local);

  const char *source = webgpu_program_source_text(program);
  char *wgsl = source ? webgpu_strdup(source) : poly_render_wgsl(ctx, lin, n_lin, fn_name);
  if (lin_owned) free(lin);
  if (!wgsl) return -1;
  double t_render = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:webgpu:lower] rendered fn=%s lin=%d wgsl=%zu grid=%d,%d,%d local=%d,%d,%d "
        "check=%.3fms lin=%.3fms render=%.3fms\n",
        fn_name, n_lin, strlen(wgsl), grid[0], grid[1], grid[2], local[0], local[1], local[2],
        t_check - t0, t_lin - t_check, t_render - t_lin
    );
    fflush(stderr);
  }
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
  out->handle_size = (int)(sizeof(*wh) + strlen(wgsl) + strlen(fn_name) + 1);
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
  bool timing = poly_debug_at_least(7);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (!wh->pipeline_id || wh->n_bindings != n_args) {
    if (timing) {
      fprintf(
          stderr,
          "[polygrad:webgpu:execute] pipeline begin entry=%s n_args=%d n_params=%d wgsl=%zu\n",
          wh->entry, n_args, runner->n_params, strlen(wh->wgsl)
      );
      fflush(stderr);
    }
    wh->pipeline_id = js_webgpu_get_or_create_pipeline(
        wh->wgsl, wh->entry, n_args, runner->n_params, poly_debug_level(), !runner->capture_binary
    );
    wh->n_bindings = n_args;
    if (!wh->pipeline_id) return -1;
    if (timing) {
      double t_pipeline = poly_now_ms();
      fprintf(
          stderr, "[polygrad:webgpu:execute] pipeline done entry=%s id=%lu ms=%.3f\n", wh->entry,
          (unsigned long)wh->pipeline_id, t_pipeline - t0
      );
      fflush(stderr);
    }
  }
  if (timing) {
    fprintf(
        stderr, "[polygrad:webgpu:execute] dispatch begin entry=%s pipeline=%lu\n", wh->entry,
        (unsigned long)wh->pipeline_id
    );
    fflush(stderr);
  }
  int ret = js_webgpu_dispatch(
      wh->pipeline_id, (const uintptr_t *)args, n_args, runner->n_params, runner->grid[0],
      runner->grid[1], runner->grid[2], poly_debug_level(),
      runner->wait ? &runner->elapsed_us : NULL
  );
  if (timing) {
    double t_done = poly_now_ms();
    fprintf(
        stderr, "[polygrad:webgpu:execute] dispatch done entry=%s ret=%d total=%.3fms\n", wh->entry,
        ret, t_done - t0
    );
    fflush(stderr);
  }
  return ret;
}

void poly_webgpu_free_runner(PolyRunner *runner) {
  if (!runner || !runner->handle) return;
  PolyWebGpuRunnerHandle *wh = (PolyWebGpuRunnerHandle *)runner->handle;
  if (runner->capture_binary && wh->pipeline_id) js_webgpu_free_search_pipeline(wh->pipeline_id);
  free(wh->wgsl);
  free(wh->entry);
  free(wh);
  runner->handle = NULL;
}

const PolyAllocator *poly_webgpu_get_allocator(void) {
  return &POLY_WEBGPU_ALLOCATOR;
}

#endif
