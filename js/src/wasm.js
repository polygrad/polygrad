/**
 * wasm.js -- WASM backend adapter for polygrad.
 *
 * Wraps the Emscripten-compiled polygrad core into a backend object.
 * Handles: Emscripten heap management, int64 marshalling, WASM kernel
 * compilation/caching/execution, memory pool, instance cache.
 */

'use strict'

// Lazy-loaded Emscripten module factory.
// Published: ../wasm/polygrad.js (SINGLE_FILE build).
// Dev (Node only): ../../build/polygrad.js (two-file Emscripten build).
//
// The dev fallback uses a computed require() so bundlers (esbuild, webpack, vite)
// cannot statically resolve it. This is intentional: the dev path only runs in
// Node.js during development -- published packages always have wasm/polygrad.js.
let _moduleFactory = null
function getModuleFactory() {
  if (_moduleFactory) return _moduleFactory
  try {
    _moduleFactory = require('../wasm/polygrad.js')
  } catch (e) {
    // Dev fallback -- Node.js only. The computed require() prevents bundlers
    // from following this branch (they can't resolve a runtime string).
    if (typeof process !== 'undefined' && process.versions && process.versions.node) {
      // eslint-disable-next-line no-eval
      const nodeRequire = typeof __non_webpack_require__ !== 'undefined'
        ? __non_webpack_require__ : eval('require')
      const path = nodeRequire('path')
      _moduleFactory = nodeRequire(
        path.resolve(__dirname, '..', '..', 'build', 'polygrad.js')
      )
    } else {
      throw new Error(
        'polygrad: WASM module not found. The package may be installed incorrectly.'
      )
    }
  }
  return _moduleFactory
}

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
 * Create a WASM backend instance.
 * Returns a Promise<backend>.
 */
async function createWasmBackend() {
  const Module = await getModuleFactory()()

  // --- Heap accessors ---
  function heap32() {
    if (Module.HEAP32) return Module.HEAP32
    const mem = Module.wasmMemory || (Module.asm && Module.asm.memory)
    if (!mem) throw new Error('No wasmMemory')
    if (!Module.__heap32 || Module.__heap32.buffer !== mem.buffer) {
      Module.__heap32 = new Int32Array(mem.buffer)
    }
    return Module.__heap32
  }

  function heapU8() {
    if (Module.HEAPU8) return Module.HEAPU8
    const mem = Module.wasmMemory || (Module.asm && Module.asm.memory)
    if (!mem) throw new Error('No wasmMemory')
    if (!Module.__heapU8 || Module.__heapU8.buffer !== mem.buffer) {
      Module.__heapU8 = new Uint8Array(mem.buffer)
    }
    return Module.__heapU8
  }

  function heapF64() {
    if (Module.HEAPF64) return Module.HEAPF64
    const mem = Module.wasmMemory || (Module.asm && Module.asm.memory)
    if (!mem) throw new Error('No wasmMemory')
    if (!Module.__heapF64 || Module.__heapF64.buffer !== mem.buffer) {
      Module.__heapF64 = new Float64Array(mem.buffer)
    }
    return Module.__heapF64
  }

  function heapF32() {
    if (Module.HEAPF32) return Module.HEAPF32
    const mem = Module.wasmMemory || (Module.asm && Module.asm.memory)
    if (!mem) throw new Error('No wasmMemory')
    if (!Module.__heapF32 || Module.__heapF32.buffer !== mem.buffer) {
      Module.__heapF32 = new Float32Array(mem.buffer)
    }
    return Module.__heapF32
  }

  // --- Scratch pointers ---
  const _scratchLenPtr = Module._malloc(4)
  const _scratchNumelPtr = Module._malloc(8)
  const _scratchAxisPtr = Module._malloc(64)  // 8 dims * 8 bytes
  const _scratchOutShapePtr = Module._malloc(64)
  const _scratchOutNdimPtr = Module._malloc(4)

  // --- Int64 marshalling helpers ---
  function writeInt64Array(arr) {
    const ptr = Module._malloc(arr.length * 8)
    for (let i = 0; i < arr.length; i++) {
      const base = (ptr >> 2) + i * 2
      const val = arr[i]
      heap32()[base] = val & 0xFFFFFFFF
      heap32()[base + 1] = val < 0 ? -1 : 0
    }
    return ptr
  }

  function writeInt64Scratch(arr) {
    for (let i = 0; i < arr.length; i++) {
      const base = (_scratchAxisPtr >> 2) + i * 2
      const val = arr[i]
      heap32()[base] = val & 0xFFFFFFFF
      heap32()[base + 1] = val < 0 ? -1 : 0
    }
    return _scratchAxisPtr
  }

  function readOutShape() {
    const ndim = heap32()[_scratchOutNdimPtr >> 2]
    return readShapeFromPtr(_scratchOutShapePtr, ndim)
  }

  function readInt64At(ptr) {
    const base = ptr >> 2
    const lo = heap32()[base] >>> 0
    const hi = heap32()[base + 1]
    return hi >= 0
      ? hi * 0x100000000 + lo
      : -(~hi * 0x100000000 + (~lo >>> 0) + 1)
  }

  function readShapeFromPtr(ptr, ndim) {
    const shape = []
    for (let i = 0; i < ndim; i++) shape.push(readInt64At(ptr + i * 8))
    return shape
  }

  function callWithInt64(fn, ctx, uop, arr, ...extra) {
    if (arr.length <= 8) {
      return fn(ctx, uop, writeInt64Scratch(arr), ...extra)
    }
    const ptr = writeInt64Array(arr)
    const result = fn(ctx, uop, ptr, ...extra)
    Module._free(ptr)
    return result
  }

  function allocString(str) {
    const bytes = new TextEncoder().encode(str + '\0')
    const ptr = Module._malloc(bytes.length)
    heapU8().set(bytes, ptr)
    return ptr
  }

  function allocBytes(bytes) {
    const ptr = Module._malloc(bytes.length || 1)
    if (bytes.length > 0) heapU8().set(bytes, ptr)
    return ptr
  }

  function readCString(ptr) {
    if (!ptr) return null
    const bytes = heapU8()
    let end = ptr
    while (bytes[end] !== 0) end++
    return new TextDecoder().decode(bytes.subarray(ptr, end))
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

  // --- Realize serialization ---
  const _realizeChains = new Map()

  function serializeRealize(ctxPtr, fn) {
    const prev = _realizeChains.get(ctxPtr) || Promise.resolve()
    const next = prev.then(fn, fn)
    _realizeChains.set(ctxPtr, next)
    return next
  }

  // --- Core kernel execution ---
  async function renderAndExec(ctx, sink, numel, leafMap, isF64) {
    return serializeRealize(ctx, async () => {
      const plan = Module._poly_render_step_wasm_plan(ctx, sink)
      if (!plan) throw new Error('poly_render_step_wasm_plan failed')

      const ArrayType = isF64 ? Float64Array : Float32Array

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
          if (data) {
            bufData[bufIdx] = data
            bufNbytes[bufIdx] = data.byteLength
          } else {
            const constPtr = Module._poly_const_buffer_data(ctx, bufUop)
            if (!constPtr) throw new Error(`No data binding for bindable buffer ${bi}`)
            const nbytes = Number(Module._poly_wasm_stepplan_buf_nbytes(plan, bufIdx))
            bufData[bufIdx] = new Uint8Array(heapU8().buffer, constPtr, nbytes)
            bufNbytes[bufIdx] = nbytes
          }
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

  // --- Build normalized FFI table ---
  const cwrapName = Module.cwrap('poly_op_name', 'string', ['number'])
  const cwrapReshape = Module.cwrap('poly_reshape', 'number', ['number', 'number', 'number', 'number'])
  const cwrapExpand = Module.cwrap('poly_expand', 'number', ['number', 'number', 'number', 'number'])
  const cwrapPermute = Module.cwrap('poly_permute', 'number', ['number', 'number', 'number', 'number'])
  const cwrapShrink = Module.cwrap('poly_shrink', 'number', ['number', 'number', 'number', 'number'])
  const cwrapFlip = Module.cwrap('poly_flip', 'number', ['number', 'number', 'number', 'number'])
  const cwrapPad = Module.cwrap('poly_pad', 'number', ['number', 'number', 'number', 'number'])

  // Build OPS enum
  const ops = {}
  const opCount = Module._poly_op_count()
  for (let i = 0; i < opCount; i++) {
    const name = cwrapName(i)
    if (name) ops[name] = i
  }

  // Normalized FFI: int64 arrays accept plain JS number[], shape-returning fns return { uop, shape }
  const ffi = {
    // Simple ops (no int64 arrays)
    poly_ctx_new: Module._poly_ctx_new,
    poly_ctx_destroy: Module._poly_ctx_destroy,
    poly_const_float: Module._poly_const_float,
    poly_const_double: Module._poly_const_double,
    poly_const_int: (ctx, val) => Module._poly_const_int(ctx, BigInt(val)),
    poly_alu1: Module._poly_alu1,
    poly_alu2: Module._poly_alu2,
    poly_alu3: Module._poly_alu3,
    poly_store_val: Module._poly_store_val,
    poly_sink1: Module._poly_sink1,
    poly_buffer_f32: (ctx, size) => Module._poly_buffer_f32(ctx, BigInt(size)),
    poly_buffer_f64: (ctx, size) => Module._poly_buffer_f64(ctx, BigInt(size)),
    poly_grad: Module._poly_grad,
    poly_detach: Module._poly_detach,
    poly_cast_by_id: Module._poly_cast_by_id,

    // Shape-taking ops (accept JS number[])
    poly_reshape: (ctx, uop, shape, len) => callWithInt64(cwrapReshape, ctx, uop, shape, len),
    poly_expand: (ctx, uop, shape, len) => callWithInt64(cwrapExpand, ctx, uop, shape, len),
    poly_permute: (ctx, uop, order, len) => callWithInt64(cwrapPermute, ctx, uop, order, len),
    poly_flip: (ctx, uop, axes, len) => callWithInt64(cwrapFlip, ctx, uop, axes, len),

    poly_shrink: (ctx, uop, flat, npairs) => {
      const ptr = writeInt64Array(flat)
      const result = cwrapShrink(ctx, uop, ptr, npairs)
      Module._free(ptr)
      return result
    },

    poly_pad: (ctx, uop, flat, npairs) => {
      const ptr = writeInt64Array(flat)
      const result = cwrapPad(ctx, uop, ptr, npairs)
      Module._free(ptr)
      return result
    },

    poly_reduce_axis: (ctx, op, uop, axes, naxes) => {
      if (axes.length <= 8) {
        return Module._poly_reduce_axis(ctx, op, uop, writeInt64Scratch(axes), naxes)
      }
      const ptr = writeInt64Array(axes)
      const result = Module._poly_reduce_axis(ctx, op, uop, ptr, naxes)
      Module._free(ptr)
      return result
    },

    // Shape-returning ops (return { uop, shape })
    poly_max_reduce: (ctx, uop, shape, nshape, axis, keepdim) => {
      const shPtr = writeInt64Array(shape)
      heap32()[_scratchOutNdimPtr >> 2] = 0
      const result = Module._poly_max_reduce(ctx, uop, shPtr, nshape, axis, keepdim,
        _scratchOutShapePtr, _scratchOutNdimPtr)
      Module._free(shPtr)
      return { uop: result, shape: readOutShape() }
    },

    poly_mean_reduce: (ctx, uop, shape, nshape, axis, keepdim) => {
      const shPtr = writeInt64Array(shape)
      heap32()[_scratchOutNdimPtr >> 2] = 0
      const result = Module._poly_mean_reduce(ctx, uop, shPtr, nshape, axis, keepdim,
        _scratchOutShapePtr, _scratchOutNdimPtr)
      Module._free(shPtr)
      return { uop: result, shape: readOutShape() }
    },

    poly_dot: (ctx, aUop, aShape, aNshape, bUop, bShape, bNshape) => {
      const aPtr = writeInt64Array(aShape)
      const bPtr = writeInt64Array(bShape)
      heap32()[_scratchOutNdimPtr >> 2] = 0
      const result = Module._poly_dot(ctx, aUop, aPtr, aNshape, bUop, bPtr, bNshape,
        _scratchOutShapePtr, _scratchOutNdimPtr)
      Module._free(aPtr)
      Module._free(bPtr)
      return { uop: result, shape: readOutShape() }
    },

    poly_cross_entropy: (ctx, logitsUop, logitsShape, logitsNshape, targetUop, targetShape, targetNshape, axis) => {
      const logitsPtr = writeInt64Array(logitsShape)
      const targetPtr = writeInt64Array(targetShape)
      heap32()[_scratchOutNdimPtr >> 2] = 0
      const result = Module._poly_cross_entropy(
        ctx, logitsUop, logitsPtr, logitsNshape,
        targetUop, targetPtr, targetNshape,
        axis, _scratchOutShapePtr, _scratchOutNdimPtr
      )
      Module._free(logitsPtr)
      Module._free(targetPtr)
      return { uop: result, shape: readOutShape() }
    },

    poly_einsum: (ctx, formula, operands) => {
      const n = operands.length

      const tensorPtrs = Module._malloc(n * 4)
      for (let i = 0; i < n; i++) {
        heap32()[(tensorPtrs >> 2) + i] = operands[i]._uop
      }

      const shapePtrs = Module._malloc(n * 4)
      const shapeArrays = []
      for (let i = 0; i < n; i++) {
        const shPtr = writeInt64Array(operands[i]._shape)
        shapeArrays.push(shPtr)
        heap32()[(shapePtrs >> 2) + i] = shPtr
      }

      const ndimPtr = Module._malloc(n * 4)
      for (let i = 0; i < n; i++) {
        heap32()[(ndimPtr >> 2) + i] = operands[i]._shape.length
      }

      const formulaPtr = allocString(formula)
      heap32()[_scratchOutNdimPtr >> 2] = 0

      const result = Module._poly_einsum(ctx, formulaPtr,
        tensorPtrs, shapePtrs, ndimPtr, n,
        _scratchOutShapePtr, _scratchOutNdimPtr)

      Module._free(formulaPtr)
      Module._free(ndimPtr)
      for (const ptr of shapeArrays) Module._free(ptr)
      Module._free(shapePtrs)
      Module._free(tensorPtrs)

      return { uop: result, shape: readOutShape() }
    },

    poly_rearrange: (ctx, formula, uop, shape, kwargs) => {
      const names = Object.keys(kwargs)
      const values = names.map(k => kwargs[k])
      const n = names.length

      const formulaPtr = allocString(formula)
      const shPtr = writeInt64Array(shape)

      let namesPtr = 0
      let valuesPtr = 0
      if (n > 0) {
        namesPtr = allocString(names.join(' '))
        valuesPtr = writeInt64Array(values)
      }

      heap32()[_scratchOutNdimPtr >> 2] = 0
      const result = Module._poly_rearrange(ctx, formulaPtr,
        uop, shPtr, shape.length,
        namesPtr, valuesPtr, n,
        _scratchOutShapePtr, _scratchOutNdimPtr)

      Module._free(formulaPtr)
      Module._free(shPtr)
      if (namesPtr) Module._free(namesPtr)
      if (valuesPtr) Module._free(valuesPtr)

      return { uop: result, shape: readOutShape() }
    },

    // Composed elementwise ops
    poly_exp: Module._poly_exp,
    poly_log: Module._poly_log,
    poly_log1p: Module._poly_log1p,
    poly_expm1: Module._poly_expm1,
    poly_sin: Module._poly_sin,
    poly_cos: Module._poly_cos,
    poly_tan: Module._poly_tan,
    poly_erf: Module._poly_erf,
    poly_erfc: Module._poly_erfc,
    poly_erfinv: Module._poly_erfinv,
    poly_ndtri: Module._poly_ndtri,
    poly_digamma: Module._poly_digamma,
    poly_lgamma: Module._poly_lgamma,
    poly_sigmoid: Module._poly_sigmoid,
    poly_tanh_act: Module._poly_tanh_act,
    poly_abs: Module._poly_abs,
    poly_sign: Module._poly_sign,
    poly_square: Module._poly_square,
    poly_rsqrt: Module._poly_rsqrt,
    poly_ceil: Module._poly_ceil,
    poly_floor: Module._poly_floor,
    poly_round_f: Module._poly_round_f,
    poly_isinf: Module._poly_isinf,
    poly_isnan: Module._poly_isnan,

    // Activations
    poly_relu: Module._poly_relu,
    poly_relu6: Module._poly_relu6,
    poly_leaky_relu: Module._poly_leaky_relu,
    poly_gelu: Module._poly_gelu,
    poly_quick_gelu: Module._poly_quick_gelu,
    poly_silu: Module._poly_silu,
    poly_elu: Module._poly_elu,
    poly_softplus: Module._poly_softplus,
    poly_mish: Module._poly_mish,
    poly_hardtanh: Module._poly_hardtanh,
    poly_hardswish: Module._poly_hardswish,
    poly_hardsigmoid: Module._poly_hardsigmoid,

    // Comparisons
    poly_eq: Module._poly_eq,
    poly_ne: Module._poly_ne,
    poly_gt: Module._poly_gt,
    poly_ge: Module._poly_ge,
    poly_le: Module._poly_le,
    poly_where_op: Module._poly_where_op,
    poly_maximum: Module._poly_maximum,
    poly_minimum: Module._poly_minimum,
    poly_clamp: Module._poly_clamp,

    // Creation (shape-taking, seed is uint64)
    poly_rand: (ctx, shape, ndim, seed) => {
      const shPtr = writeInt64Array(shape)
      const result = Module._poly_rand(ctx, shPtr, ndim, BigInt(seed))
      Module._free(shPtr)
      return result
    },
    poly_randn: (ctx, shape, ndim, seed) => {
      const shPtr = writeInt64Array(shape)
      const result = Module._poly_randn(ctx, shPtr, ndim, BigInt(seed))
      Module._free(shPtr)
      return result
    },
    poly_arange: Module._poly_arange,
    poly_eye: Module._poly_eye,
    poly_linspace: Module._poly_linspace,
    poly_full: Module._poly_full,
    poly_tril: (ctx, uop, shape, ndim, diagonal) => {
      const ptr = writeInt64Array(shape)
      const result = Module._poly_tril(ctx, uop, ptr, ndim, diagonal)
      Module._free(ptr)
      return result
    },
    poly_triu: (ctx, uop, shape, ndim, diagonal) => {
      const ptr = writeInt64Array(shape)
      const result = Module._poly_triu(ctx, uop, ptr, ndim, diagonal)
      Module._free(ptr)
      return result
    },
    poly_cholesky: Module._poly_cholesky,
    poly_triangular_solve: Module._poly_triangular_solve,

    // Reduction (non-shape-returning)
    poly_sum_reduce: Module._poly_sum_reduce,
    poly_logsumexp: Module._poly_logsumexp,

    // ABI
    poly_abi_version: Module._poly_abi_version,
    poly_op_count: Module._poly_op_count
  }

  // ABI version check
  const EXPECTED_ABI = 1
  const abi = ffi.poly_abi_version()
  if (abi !== EXPECTED_ABI) {
    throw new Error(
      `polygrad WASM ABI mismatch: expected version ${EXPECTED_ABI}, got ${abi}. ` +
      'Rebuild polygrad.wasm or update the polygrad package.'
    )
  }

  // Create context
  const ctx = ffi.poly_ctx_new()

  const instance = {
    fromIR(irBytes, weightsBytes) {
      const irPtr = allocBytes(irBytes)
      let weightsPtr = 0
      let weightsLen = 0
      if (weightsBytes && weightsBytes.length > 0) {
        weightsPtr = allocBytes(weightsBytes)
        weightsLen = weightsBytes.length
      }
      const inst = Module._poly_instance_from_ir(irPtr, irBytes.length, weightsPtr, weightsLen)
      Module._free(irPtr)
      if (weightsPtr) Module._free(weightsPtr)
      return inst || null
    },

    mlp(specJson) {
      const bytes = new TextEncoder().encode(specJson)
      const specPtr = allocBytes(bytes)
      const inst = Module._poly_mlp_instance(specPtr, bytes.length)
      Module._free(specPtr)
      return inst || null
    },

    tabm(specJson) {
      const bytes = new TextEncoder().encode(specJson)
      const specPtr = allocBytes(bytes)
      const inst = Module._poly_tabm_instance(specPtr, bytes.length)
      Module._free(specPtr)
      return inst || null
    },

    nam(specJson) {
      const bytes = new TextEncoder().encode(specJson)
      const specPtr = allocBytes(bytes)
      const inst = Module._poly_nam_instance(specPtr, bytes.length)
      Module._free(specPtr)
      return inst || null
    },

    free(instPtr) {
      Module._poly_instance_free(instPtr)
    },

    paramCount(instPtr) {
      return Module._poly_instance_param_count(instPtr)
    },

    paramName(instPtr, i) {
      return readCString(Module._poly_instance_param_name(instPtr, i))
    },

    paramShape(instPtr, i) {
      const ndim = Module._poly_instance_param_shape(instPtr, i, _scratchOutShapePtr, 8)
      return readShapeFromPtr(_scratchOutShapePtr, ndim)
    },

    paramData(instPtr, i) {
      const dataPtr = Module._poly_instance_param_data(instPtr, i, _scratchNumelPtr)
      if (!dataPtr) return null
      const numel = readInt64At(_scratchNumelPtr)
      return new Float32Array(heapF32().buffer.slice(dataPtr, dataPtr + numel * 4))
    },

    bufCount(instPtr) {
      return Module._poly_instance_buf_count(instPtr)
    },

    bufName(instPtr, i) {
      return readCString(Module._poly_instance_buf_name(instPtr, i))
    },

    bufRole(instPtr, i) {
      return Module._poly_instance_buf_role(instPtr, i)
    },

    bufShape(instPtr, i) {
      const ndim = Module._poly_instance_buf_shape(instPtr, i, _scratchOutShapePtr, 8)
      return readShapeFromPtr(_scratchOutShapePtr, ndim)
    },

    bufData(instPtr, i) {
      const dataPtr = Module._poly_instance_buf_data(instPtr, i, _scratchNumelPtr)
      if (!dataPtr) return null
      const numel = readInt64At(_scratchNumelPtr)
      return new Float32Array(heapF32().buffer.slice(dataPtr, dataPtr + numel * 4))
    },

    exportWeights(instPtr) {
      const bytesPtr = Module._poly_instance_export_weights(instPtr, _scratchLenPtr)
      if (!bytesPtr) return null
      const len = heap32()[_scratchLenPtr >> 2]
      const bytes = new Uint8Array(heapU8().buffer.slice(bytesPtr, bytesPtr + len))
      Module._free(bytesPtr)
      return bytes
    },

    importWeights(instPtr, bytes) {
      const bytesPtr = allocBytes(bytes)
      const rc = Module._poly_instance_import_weights(instPtr, bytesPtr, bytes.length)
      Module._free(bytesPtr)
      return rc
    },

    exportIR(instPtr) {
      const bytesPtr = Module._poly_instance_export_ir(instPtr, _scratchLenPtr)
      if (!bytesPtr) return null
      const len = heap32()[_scratchLenPtr >> 2]
      const bytes = new Uint8Array(heapU8().buffer.slice(bytesPtr, bytesPtr + len))
      Module._free(bytesPtr)
      return bytes
    },

    setOptimizer(instPtr, kind, lr, beta1, beta2, eps, weightDecay) {
      return Module._poly_instance_set_optimizer(
        instPtr, kind, lr, beta1, beta2, eps, weightDecay)
    },

    forward(instPtr, names, arrays) {
      const n = names.length
      const bindingPtr = Module._malloc(Math.max(1, n) * 8)
      const namePtrs = new Array(n)
      const dataPtrs = new Array(n)

      for (let i = 0; i < n; i++) {
        namePtrs[i] = allocString(names[i])
        dataPtrs[i] = allocBytes(new Uint8Array(arrays[i].buffer, arrays[i].byteOffset, arrays[i].byteLength))
        const base = (bindingPtr >> 2) + i * 2
        heap32()[base] = namePtrs[i]
        heap32()[base + 1] = dataPtrs[i]
      }

      const rc = Module._poly_instance_forward(instPtr, bindingPtr, n)

      for (const ptr of dataPtrs) Module._free(ptr)
      for (const ptr of namePtrs) Module._free(ptr)
      Module._free(bindingPtr)
      return rc
    },

    trainStep(instPtr, names, arrays) {
      const n = names.length
      const bindingPtr = Module._malloc(Math.max(1, n) * 8)
      const namePtrs = new Array(n)
      const dataPtrs = new Array(n)

      for (let i = 0; i < n; i++) {
        namePtrs[i] = allocString(names[i])
        dataPtrs[i] = allocBytes(new Uint8Array(arrays[i].buffer, arrays[i].byteOffset, arrays[i].byteLength))
        const base = (bindingPtr >> 2) + i * 2
        heap32()[base] = namePtrs[i]
        heap32()[base + 1] = dataPtrs[i]
      }

      const lossPtr = Module._malloc(4)
      const rc = Module._poly_instance_train_step(instPtr, bindingPtr, n, lossPtr)
      const loss = rc === 0 ? heapF32()[lossPtr >> 2] : null

      Module._free(lossPtr)
      for (const ptr of dataPtrs) Module._free(ptr)
      for (const ptr of namePtrs) Module._free(ptr)
      Module._free(bindingPtr)
      return loss
    }
  }

  return {
    ffi,
    ctx,
    ops,
    int64: BigInt,
    readShape: readOutShape,
    realize: renderAndExec,
    caps: { simd: true, f64: true, target: 'wasm', device: 'cpu' },
    destroy: () => {
      ffi.poly_ctx_destroy(ctx)
      _ctxMemory.delete(ctx)
      _realizeChains.delete(ctx)
      Module._free(_scratchLenPtr)
      Module._free(_scratchNumelPtr)
      Module._free(_scratchAxisPtr)
      Module._free(_scratchOutShapePtr)
      Module._free(_scratchOutNdimPtr)
    }
  }
}

module.exports = { createWasmBackend }
