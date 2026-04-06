/**
 * exec_webgpu.js -- WebGPU kernel executor for polygrad.
 *
 * Compiles WGSL shaders and dispatches compute workgroups via the
 * WebGPU API. Data lives in GPU buffers, uploaded via writeBuffer,
 * read back via mapAsync.
 *
 * Built on core_wasm.js (Emscripten target binding).
 * Parity target: tinygrad runtime/ops_webgpu.py
 */

'use strict'

const { createWasmCore } = require('./core_wasm')

// LRU cache helpers
function lruGet(map, key) {
  const val = map.get(key)
  if (val === undefined) return undefined
  map.delete(key)
  map.set(key, val)
  return val
}

function lruSet(map, key, val, maxSize) {
  if (map.has(key)) map.delete(key)
  map.set(key, val)
  if (map.size > maxSize) {
    const oldest = map.keys().next().value
    map.delete(oldest)
  }
}

function hashString(str) {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0).toString(16) + ':' + str.length
}

const MAX_PIPELINE_CACHE = 256

/**
 * Create a WebGPU execution backend.
 */
async function createWebGpuBackend() {
  const core = await createWasmCore('webgpu')
  const { Module, ctx, heap32, heapU8, serializeRealize } = core

  // --- Initialize WebGPU ---
  if (typeof navigator === 'undefined' || !navigator.gpu)
    throw new Error('polygrad: WebGPU not available')

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter)
    throw new Error('polygrad: no WebGPU adapter found')

  const features = []
  if (adapter.features.has('shader-f16')) features.push('shader-f16')

  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
      maxBufferSize: adapter.limits.maxBufferSize
    }
  })

  // INFINITY uniform buffer: binding 0 for every kernel (tinygrad ops_webgpu.py:69,129)
  const infinityBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  })
  device.queue.writeBuffer(infinityBuf, 0, new Float32Array([Infinity]))

  // Pipeline cache: WGSL hash -> { pipeline, bindGroupLayout }
  const _pipelineCache = new Map()

  async function getOrCreatePipeline(wgslSrc, entryPoint, nBindings) {
    const key = hashString(wgslSrc)
    const cached = lruGet(_pipelineCache, key)
    if (cached) return cached

    const shaderModule = device.createShaderModule({ code: wgslSrc })

    // Bind group layout: binding 0 = uniform (INFINITY), binding 1..N = storage
    const entries = [{
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform' }
    }]
    for (let i = 1; i <= nBindings; i++) {
      entries.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      })
    }
    const bindGroupLayout = device.createBindGroupLayout({ entries })
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] })

    const pipeline = await device.createComputePipelineAsync({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint }
    })

    const result = { pipeline, bindGroupLayout }
    lruSet(_pipelineCache, key, result, MAX_PIPELINE_CACHE)
    return result
  }

  // --- WebGPU kernel execution ---
  async function renderAndExecWebGpu(ctx, sink, numel, leafMap, isF64) {
    if (isF64) throw new Error('polygrad: WebGPU does not support float64')

    return serializeRealize(ctx, async () => {
      const plan = Module._poly_render_step_webgpu_plan(ctx, sink)
      if (!plan) throw new Error('poly_render_step_webgpu_plan failed')

      const _scratchLenPtr = core._scratchLenPtr

      try {
        const nKernels = Module._poly_webgpu_stepplan_n_kernels(plan)
        const nBufs = Module._poly_webgpu_stepplan_n_buffers(plan)
        const nBindable = Module._poly_webgpu_stepplan_n_bindable_buffers(plan)

        // Allocate GPU buffers (4-byte aligned, tinygrad ops_webgpu.py:181)
        const gpuBufs = new Array(nBufs)
        const bufNbytes = new Array(nBufs)

        for (let i = 0; i < nBufs; i++) {
          bufNbytes[i] = Number(Module._poly_webgpu_stepplan_buf_nbytes(plan, i))
          const aligned = Math.max(4, (bufNbytes[i] + 3) & ~3)
          gpuBufs[i] = device.createBuffer({
            size: aligned,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
          })
        }

        // Copy leaf data and const registry data to GPU
        for (let bi = 0; bi < nBindable; bi++) {
          const bufIdx = Module._poly_webgpu_stepplan_bindable_buf_index(plan, bi)
          const bufUop = Module._poly_kernel_buf(ctx, bi)
          const data = leafMap.get(bufUop)
          if (data) {
            device.queue.writeBuffer(gpuBufs[bufIdx], 0,
              new Uint8Array(data.buffer, data.byteOffset, data.byteLength))
          } else {
            // Const registry fallback (arange, causal mask, etc.)
            const constPtr = Module._poly_const_buffer_data(ctx, bufUop)
            if (constPtr) {
              const nb = bufNbytes[bufIdx]
              device.queue.writeBuffer(gpuBufs[bufIdx], 0,
                new Uint8Array(heapU8().buffer, constPtr, nb).slice())
            }
          }
        }

        // Read exec order
        const execOrderPtr = Module._poly_webgpu_stepplan_exec_order(plan, _scratchLenPtr)
        const execOrder = []
        for (let i = 0; i < nKernels; i++) {
          execOrder.push(heap32()[(execOrderPtr >> 2) + i])
        }

        // Execute kernels
        for (const ki of execOrder) {
          const wgslPtr = Module._poly_webgpu_stepplan_kernel_wgsl(plan, ki, _scratchLenPtr)
          // Debug: log WGSL source for shader validation failures
          if (typeof console !== 'undefined' && console.debug) {
            const _dbgLen = heap32()[_scratchLenPtr >> 2]
            if (wgslPtr && _dbgLen > 0) {
              const _dbgSrc = new TextDecoder().decode(new Uint8Array(heapU8().buffer, wgslPtr, _dbgLen))
              console.debug('[webgpu] kernel k' + ki + ' WGSL (' + _dbgLen + ' bytes):\n' + _dbgSrc)
            }
          }
          const wgslLen = heap32()[_scratchLenPtr >> 2]
          if (!wgslPtr || wgslLen <= 0) throw new Error(`No WGSL source for kernel ${ki}`)

          const wgslSrc = new TextDecoder().decode(
            new Uint8Array(heapU8().buffer, wgslPtr, wgslLen))

          const gx = Module._poly_webgpu_stepplan_kernel_grid(plan, ki, 0)
          const gy = Module._poly_webgpu_stepplan_kernel_grid(plan, ki, 1)
          const gz = Module._poly_webgpu_stepplan_kernel_grid(plan, ki, 2)

          const nParams = Module._poly_webgpu_stepplan_kernel_n_params(plan, ki)
          const fnName = 'k' + ki

          // Compile/cache pipeline
          const { pipeline, bindGroupLayout } = await getOrCreatePipeline(wgslSrc, fnName, nParams)

          // Build bind group: binding 0 = INFINITY, binding 1+ = storage buffers
          const bgEntries = [{ binding: 0, resource: { buffer: infinityBuf } }]
          for (let p = 0; p < nParams; p++) {
            const bufIdx = Module._poly_webgpu_stepplan_kernel_param_buf_index(plan, ki, p)
            if (bufIdx < 0 || bufIdx >= nBufs)
              throw new Error(`Invalid buffer index ${bufIdx} for kernel ${ki} param ${p}`)

            // Same-buffer input/output: WebGPU doesn't allow same buffer as both
            // read and write in one dispatch (tinygrad ops_webgpu.py:98-103)
            let buf = gpuBufs[bufIdx]
            const outBufIdx = Module._poly_webgpu_stepplan_kernel_param_buf_index(plan, ki, 0)
            if (p > 0 && bufIdx === outBufIdx) {
              const tmpBuf = device.createBuffer({
                size: gpuBufs[bufIdx].size,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
              })
              const enc = device.createCommandEncoder()
              enc.copyBufferToBuffer(gpuBufs[bufIdx], 0, tmpBuf, 0, gpuBufs[bufIdx].size)
              device.queue.submit([enc.finish()])
              buf = tmpBuf
            }
            bgEntries.push({ binding: p + 1, resource: { buffer: buf } })
          }

          const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: bgEntries })

          const encoder = device.createCommandEncoder()
          const pass = encoder.beginComputePass()
          pass.setPipeline(pipeline)
          pass.setBindGroup(0, bindGroup)
          pass.dispatchWorkgroups(gx, gy, gz)
          pass.end()
          device.queue.submit([encoder.finish()])
        }

        // Readback output buffer (index 0 = output, same convention as WASM step plan)
        const outBytes = bufNbytes[0]
        const staging = device.createBuffer({
          size: outBytes,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        })
        const enc = device.createCommandEncoder()
        enc.copyBufferToBuffer(gpuBufs[0], 0, staging, 0, outBytes)
        device.queue.submit([enc.finish()])

        await staging.mapAsync(GPUMapMode.READ)
        const mapped = staging.getMappedRange()
        const result = new Float32Array(numel)
        result.set(new Float32Array(mapped, 0, numel))
        staging.unmap()
        staging.destroy()

        // Cleanup GPU buffers
        for (const buf of gpuBufs) if (buf) buf.destroy()

        return result
      } finally {
        Module._poly_webgpu_stepplan_destroy(plan)
      }
    })
  }

  return {
    ffi: core.ffi,
    ctx: core.ctx,
    ops: core.ops,
    instance: core.instance,
    int64: core.int64,
    readShape: core.readShape,
    realize: renderAndExecWebGpu,
    caps: { simd: true, f64: false, target: 'wasm', device: 'webgpu' },
    destroy: () => {
      infinityBuf.destroy()
      core.destroy()
    }
  }
}

module.exports = { createWebGpuBackend }
