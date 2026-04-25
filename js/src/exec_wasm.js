/**
 * exec_wasm.js -- WASM kernel executor for polygrad.
 *
 * Compiles WASM kernel bytes and runs them via the WebAssembly API.
 * Data lives in a separate WebAssembly.Memory, copied in/out.
 *
 * Built on core_wasm.js (Emscripten target binding).
 */

'use strict'

const { createWasmCore } = require('./core_wasm')

// LRU helpers
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

function hashBytes(bytes) {
  let h = 0x811c9dc5
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i]
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

function moduleCacheKey(hash, len) {
  return `${hash}:${len}`
}

/**
 * Create a WASM execution backend.
 * @param {string} device - 'auto', 'cpu', 'wasm', or 'interp'
 */
async function createWasmBackend(device) {
  const core = await createWasmCore(device || 'auto')
  const { Module, ctx, heap32, heapU8, serializeRealize } = core
  if (core.deviceName === 'webgpu' && Module.__polygradEnsureWebGPU) {
    await Module.__polygradEnsureWebGPU()
  }

  // --- C math imports for WASM kernels ---
  const mathImports = {
    exp2f: Module._exp2f,
    log2f: Module._log2f,
    sinf: Module._sinf,
    powf: Module._powf
  }

  // --- Caching infrastructure ---
  const MAX_MODULE_CACHE = 512
  const _moduleCache = new Map()

  const MAX_INSTANCE_CACHE = 512
  const _instanceCache = new WeakMap()

  function getOrCreateInstance(mod, cacheKey, memory, imports) {
    let perMemory = _instanceCache.get(memory)
    if (!perMemory) {
      perMemory = new Map()
      _instanceCache.set(memory, perMemory)
    }
    let inst = lruGet(perMemory, cacheKey)
    if (!inst) {
      inst = new WebAssembly.Instance(mod, imports)
      lruSet(perMemory, cacheKey, inst, MAX_INSTANCE_CACHE)
    }
    return inst
  }

  // --- Memory pool ---
  const _ctxMemory = new Map()

  function getOrGrowMemory(ctxPtr, neededPages) {
    let entry = _ctxMemory.get(ctxPtr)
    if (!entry) {
      const memory = new WebAssembly.Memory({ initial: neededPages })
      entry = { memory, pages: neededPages }
      _ctxMemory.set(ctxPtr, entry)
      return entry.memory
    }
    if (neededPages > entry.pages) {
      entry.memory.grow(neededPages - entry.pages)
      entry.pages = neededPages
    }
    return entry.memory
  }

  // --- Core WASM kernel execution ---
  async function renderAndExec(ctx, sink, numel, leafMap, isF64) {
    return serializeRealize(ctx, async () => {
      const plan = Module._poly_render_step_wasm_plan(ctx, sink)
      if (!plan) throw new Error('poly_render_step_wasm_plan failed')

      const ArrayType = isF64 ? Float64Array : Float32Array
      const _scratchLenPtr = core._scratchLenPtr

      try {
        const nKernels = Module._poly_wasm_stepplan_n_kernels(plan)
        const nBufs = Module._poly_wasm_stepplan_n_buffers(plan)
        const nBindable = Module._poly_wasm_stepplan_n_bindable_buffers(plan)

        const bufNbytes = new Array(nBufs)
        const bufData = new Array(nBufs)

        for (let bi = 0; bi < nBindable; bi++) {
          const bufIdx = Module._poly_wasm_stepplan_bindable_buf_index(plan, bi)
          const bufUop = Module._poly_kernel_buf(ctx, bi)
          const data = leafMap.get(bufUop)
          if (!data) {
            // Phase E: const-registry has been removed from the C core, so
            // every bindable buffer must come from the caller's leafMap.
            // The previous fallback (Module._poly_const_buffer_data) read
            // from g_const_bindings, which no longer exists.
            throw new Error(`No data binding for bindable buffer ${bi}`)
          }
          bufData[bufIdx] = data
          bufNbytes[bufIdx] = data.byteLength
        }

        for (let i = nBindable; i < nBufs; i++) {
          bufNbytes[i] = Number(Module._poly_wasm_stepplan_buf_nbytes(plan, i))
          bufData[i] = null
        }

        const offsets = new Array(nBufs)
        let totalBytes = 0
        for (let i = 0; i < nBufs; i++) {
          totalBytes = (totalBytes + 7) & ~7
          offsets[i] = totalBytes
          totalBytes += bufNbytes[i]
        }

        const neededPages = Math.max(1, Math.ceil(totalBytes / 65536))
        const memory = getOrGrowMemory(ctx, neededPages)
        const memBytes = new Uint8Array(memory.buffer)

        for (let bi = 0; bi < nBindable; bi++) {
          const bufIdx = Module._poly_wasm_stepplan_bindable_buf_index(plan, bi)
          const data = bufData[bufIdx]
          if (data) {
            if (data instanceof Uint8Array) {
              memBytes.set(data, offsets[bufIdx])
            } else {
              memBytes.set(
                new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
                offsets[bufIdx]
              )
            }
          }
        }

        const execOrderPtr = Module._poly_wasm_stepplan_exec_order(plan, _scratchLenPtr)
        const execOrder = []
        for (let i = 0; i < nKernels; i++) {
          execOrder.push(heap32()[(execOrderPtr >> 2) + i])
        }

        const imports = { env: { memory }, math: mathImports }

        for (const ki of execOrder) {
          const bytesPtr = Module._poly_wasm_stepplan_kernel_bytes(plan, ki, _scratchLenPtr)
          const bytesLen = heap32()[_scratchLenPtr >> 2]
          if (!bytesPtr || bytesLen <= 0) throw new Error(`No WASM bytes for kernel ${ki}`)

          const wasmView = new Uint8Array(heapU8().buffer, bytesPtr, bytesLen)
          const hash = hashBytes(wasmView)
          const cacheKey = moduleCacheKey(hash, bytesLen)

          let mod = lruGet(_moduleCache, cacheKey)
          if (!mod) {
            mod = await WebAssembly.compile(wasmView.slice())
            lruSet(_moduleCache, cacheKey, mod, MAX_MODULE_CACHE)
          }

          const nParams = Module._poly_wasm_stepplan_kernel_n_params(plan, ki)
          const paramOffsets = []
          for (let p = 0; p < nParams; p++) {
            const bufIdx = Module._poly_wasm_stepplan_kernel_param_buf_index(plan, ki, p)
            if (bufIdx < 0 || bufIdx >= nBufs) {
              throw new Error(`Invalid buffer index ${bufIdx} for kernel ${ki} param ${p}`)
            }
            paramOffsets.push(offsets[bufIdx])
          }

          const instance = getOrCreateInstance(mod, cacheKey, memory, imports)
          instance.exports.kernel(...paramOffsets)
        }

        const outView = new ArrayType(memory.buffer, offsets[0], numel)
        const result = new ArrayType(numel)
        result.set(outView)
        return result
      } finally {
        Module._poly_wasm_stepplan_destroy(plan)
      }
    })
  }

  // --- Device routing ---
  const realize = (core.deviceId === core.deviceIds.wasm) ? renderAndExec : core.realizeViaBackend

  return {
    ffi: core.ffi,
    dtypeIds: core.dtypeIds,
    ctx: core.ctx,
    ops: core.ops,
    instance: core.instance,
    int64: core.int64,
    readShape: core.readShape,
    realize,
    registerHostBuffer: core.registerHostBuffer,
    unregisterHostBuffer: core.unregisterHostBuffer,
    caps: { simd: true, f64: core.deviceName !== 'webgpu', target: 'wasm', device: core.deviceName },
    destroy: () => {
      _ctxMemory.delete(ctx)
      core.destroy()
    }
  }
}

module.exports = { createWasmBackend }
