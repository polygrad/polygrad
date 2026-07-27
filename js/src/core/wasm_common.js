/**
 * core/wasm_common.js -- Emscripten core binding for polygrad.
 *
 * Wraps an already-loaded C core compiled to WASM, builds the FFI table,
 * creates a context, and exposes heap/marshalling helpers. Execution stays in
 * the C core; JS only adapts memory and frontend calls.
 *
 * This is a core binding, not a device executor. It does not decide how
 * kernels are compiled or dispatched.
 */

'use strict'

const { PolyAsyncRequired } = require('../errors')

/**
 * Wrap an already-loaded Emscripten module into a polygrad core binding.
 *
 * @param {object} Module - Ready Emscripten Module.
 * @param {string} device - Device name ('auto', 'cpu', 'wasm', 'interp', 'webgpu')
 * @returns {object} Internal core object.
 */
function createWasmCoreFromModule(Module, device) {
  const deviceName = device || 'auto'
  Module.__polygradHostBuffers = Module.__polygradHostBuffers || new Map()

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
  let _scratchPtrArrayPtr = 0
  let _scratchPtrArrayCap = 0

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

  function readUopShape(ctx, uop) {
    if (!uop) return []
    const ndim = Module._poly_uop_ndim(ctx, uop)
    if (ndim <= 0) return []
    const dimsPtr = Module._poly_uop_max_shape_dims(ctx, uop)
    if (!dimsPtr) return []
    const result = []
    const h32 = heap32()
    for (let i = 0; i < ndim; i++) {
      const lo = h32[(dimsPtr >> 2) + i * 2]
      const hi = h32[(dimsPtr >> 2) + i * 2 + 1]
      result.push(lo + hi * 0x100000000)
    }
    return result
  }

  function readInt64At(ptr) {
    const base = ptr >> 2
    const lo = heap32()[base] >>> 0
    const hi = heap32()[base + 1]
    return hi >= 0
      ? hi * 0x100000000 + lo
      : -(~hi * 0x100000000 + (~lo >>> 0) + 1)
  }

  function callUopPair(fn, args, name) {
    const outPtr = Module._malloc(8)
    try {
      const rc = fn(...args, outPtr, outPtr + 4)
      if (rc !== 0) throw new Error(`${name} failed (rc=${rc})`)
      const h32 = heap32()
      return [h32[outPtr >> 2], h32[(outPtr >> 2) + 1]]
    } finally {
      Module._free(outPtr)
    }
  }

  function readShapeFromPtr(ptr, ndim) {
    const shape = []
    for (let i = 0; i < ndim; i++) shape.push(readInt64At(ptr + i * 8))
    return shape
  }

  function shapeNumel(shape) {
    if (!shape || shape.length === 0) return 1
    let numel = 1
    for (const d of shape) numel *= d
    return numel
  }

  function writePtrArray(arr) {
    if (!arr || arr.length === 0) return 0
    const ptr = Module._malloc(arr.length * 4)
    const h32 = heap32()
    for (let i = 0; i < arr.length; i++) h32[(ptr >> 2) + i] = arr[i] || 0
    return ptr
  }

  function writeCString(s) {
    const bytes = new TextEncoder().encode(String(s))
    const ptr = Module._malloc(bytes.length + 1)
    heapU8().set(bytes, ptr)
    heapU8()[ptr + bytes.length] = 0
    return ptr
  }

  function writePtrArrayScratch(arr) {
    if (!arr || arr.length === 0) return 0
    if (arr.length > _scratchPtrArrayCap) {
      if (_scratchPtrArrayPtr) Module._free(_scratchPtrArrayPtr)
      _scratchPtrArrayCap = Math.max(arr.length, _scratchPtrArrayCap ? _scratchPtrArrayCap * 2 : 8)
      _scratchPtrArrayPtr = Module._malloc(_scratchPtrArrayCap * 4)
    }
    const h32 = heap32()
    for (let i = 0; i < arr.length; i++) h32[(_scratchPtrArrayPtr >> 2) + i] = arr[i] || 0
    return _scratchPtrArrayPtr
  }

  function writeI32Array(arr) {
    if (!arr || arr.length === 0) return 0
    const ptr = Module._malloc(arr.length * 4)
    const h32 = heap32()
    for (let i = 0; i < arr.length; i++) h32[(ptr >> 2) + i] = arr[i] | 0
    return ptr
  }

  function readPtrArray(ptr, n) {
    const out = new Array(n)
    const h32 = heap32()
    for (let i = 0; i < n; i++) out[i] = h32[(ptr >> 2) + i]
    return out
  }

  const ctxStatsFields = [
    'arenaBytes',
    'arenaHighWater',
    'scratchBytes',
    'scratchHighWater',
    'cseEntries',
    'scheduleCacheEntries',
    'toProgramCacheEntries',
    'runtimeCacheEntries',
    'programCacheEntries',
    'shapeCacheEntries',
    'bufferEntries',
    'bufferOwnedBytes',
    'bufferOwnedCurrentBytes',
    'bufferOwnedSourceBytes',
    'tensorEntries',
    'tensorRecords',
    'registryEntries',
    'entrypointEntries',
    'compiledArtifactBytes',
    'runtimeArtifactEntries',
    'launchCount',
    'scheduleCacheHits',
    'scheduleCacheMisses',
    'runtimeCacheHits',
    'runtimeCacheMisses',
    'bufferReadCount',
    'bufferReadBytes',
    'bufferWriteCount',
    'bufferWriteBytes',
    'bufferCopyCount',
    'bufferCopyBytes'
  ]

  function readCtxStats(ctx) {
    const ptr = Module._malloc(168)
    try {
      const rc = Module._poly_ctx_stats(ctx, ptr)
      if (rc !== 0) throw new Error('poly_ctx_stats failed (rc=' + rc + ')')
      const h32 = heap32()
      const out = {}
      for (let i = 0; i < ctxStatsFields.length; i++) {
        out[ctxStatsFields[i]] = h32[(ptr >> 2) + i] >>> 0
      }
      const view = new DataView(heapU8().buffer)
      const readU64 = (offset) => {
        const lo = view.getUint32(ptr + offset, true)
        const hi = view.getUint32(ptr + offset + 4, true)
        return hi * 0x100000000 + lo
      }
      out.globalOps = readU64(128)
      out.globalMem = readU64(136)
      out.timeSumS = view.getFloat64(ptr + 144, true)
      out.kernelCount = readU64(152)
      out.memUsed = readU64(160)
      return out
    } finally {
      Module._free(ptr)
    }
  }

  function canRunOp(device, op, dtypeId, shape) {
    const dims = Array.from(shape || [])
    const opPtr = allocString(op)
    const shapePtr = dims.length > 0 ? writeInt64Array(dims) : 0
    try {
      return Module._poly_can_run_op(ctx, device, opPtr, dtypeId, shapePtr, dims.length)
    } finally {
      if (shapePtr) Module._free(shapePtr)
      Module._free(opPtr)
    }
  }

  function writeOptimConfig(cfg) {
    const ptr = Module._malloc(28)
    const u8 = heapU8()
    const h32 = heap32()
    const f32 = heapF32()
    u8.fill(0, ptr, ptr + 28)
    h32[ptr >> 2] = cfg.kind || 0
    f32[(ptr >> 2) + 1] = cfg.beta1 == null ? 0.9 : cfg.beta1
    f32[(ptr >> 2) + 2] = cfg.beta2 == null ? 0.999 : cfg.beta2
    f32[(ptr >> 2) + 3] = cfg.eps == null ? 1e-8 : cfg.eps
    f32[(ptr >> 2) + 4] = cfg.weightDecay == null ? 0 : cfg.weightDecay
    f32[(ptr >> 2) + 5] = cfg.momentum == null ? 0 : cfg.momentum
    u8[ptr + 24] = cfg.nesterov ? 1 : 0
    u8[ptr + 25] = cfg.classic ? 1 : 0
    return ptr
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

  // Lookup device ID by name via C API. 
  // This allows the C core to control the device enum and supported devices, and avoids hardcoding IDs in JS.
  function coreDeviceId(name) {
    const ptr = allocString(name)
    const id = Module._poly_device_by_name(ptr)
    Module._free(ptr)
    return id
  }

  function coreDeviceName(device) {
    const ptr = Module._poly_device_name(device)
    return ptr ? Module.UTF8ToString(ptr) : 'auto'
  }

  function coreDTypeId(name) {
    const ptr = allocString(name)
    const id = Module._poly_dtype_id_by_name(ptr)
    Module._free(ptr)
    return id
  }

  const AUTO_DEVICE_ID = coreDeviceId('auto')

  // Public cpu and runtime-default auto resolve to the wasm execution backend
  // on wasm targets. AUTO_DEVICE_ID remains the core's device-less graph value.
  const DEVICE_IDS = {
    auto: coreDeviceId('wasm'),
    cpu: coreDeviceId('wasm'),
    interp: coreDeviceId('interp'),
    wasm: coreDeviceId('wasm'),
    webgpu: coreDeviceId('webgpu')
  }
  const HOST_DEVICE_ID = coreDeviceId('host')
  const DTYPE_IDS = {
    bool: coreDTypeId('bool'),
    int8: coreDTypeId('int8'),
    uint8: coreDTypeId('uint8'),
    int16: coreDTypeId('int16'),
    uint16: coreDTypeId('uint16'),
    int32: coreDTypeId('int32'),
    uint32: coreDTypeId('uint32'),
    int64: coreDTypeId('int64'),
    uint64: coreDTypeId('uint64'),
    float16: coreDTypeId('float16'),
    bfloat16: coreDTypeId('bfloat16'),
    float32: coreDTypeId('float32'),
    float64: coreDTypeId('float64')
  }
  if (!(deviceName in DEVICE_IDS))
    throw new Error('polygrad: unsupported device \'' + deviceName + '\'')
  const resolvedDeviceName = deviceName === 'auto' ? 'wasm' : deviceName
  const deviceId = DEVICE_IDS[deviceName]
  let webgpuSupportsF16 = false

  async function ensureWebGPU() {
    if (deviceName !== 'webgpu') {
      throw new Error('polygrad: WebGPU state requested for non-webgpu runtime')
    }
    if (Module.__polygradWebGpuState && Module.__polygradWebGpuState.device) {
      return Module.__polygradWebGpuState
    }
    if (Module.__polygradWebGpuInit) return Module.__polygradWebGpuInit

    Module.__polygradWebGpuInit = (async () => {
      if (typeof navigator === 'undefined' || !navigator.gpu) {
        throw new Error('polygrad: WebGPU not available')
      }

      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
      if (!adapter) throw new Error('polygrad: no WebGPU adapter found')

      const features = []
      const hasShaderF16 = adapter.features.has('shader-f16')
      if (hasShaderF16) features.push('shader-f16')

      const device = await adapter.requestDevice({
        requiredFeatures: features,
        requiredLimits: {
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
          maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
          maxBufferSize: adapter.limits.maxBufferSize
        }
      })

      const infinityBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      })
      device.queue.writeBuffer(infinityBuf, 0, new Float32Array([Infinity]))

      Module.__polygradWebGpuState = {
        adapter,
        device,
        adapterFeatures: [...adapter.features],
        hasShaderF16,
        infinityBuf,
        buffers: new Map(),
        bufferSizes: new Map(),
        bufferOffsets: new Map(),
        bufferViews: new Set(),
        nextBufferId: 1,
        pipelines: new Map(),
        pipelineKeyToId: new Map(),
        nextPipelineId: 1
      }
      webgpuSupportsF16 = hasShaderF16
      return Module.__polygradWebGpuState
    })()

    try {
      return await Module.__polygradWebGpuInit
    } finally {
      Module.__polygradWebGpuInit = null
    }
  }

  Module.__polygradEnsureWebGPU = ensureWebGPU

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

  // --- Build FFI table ---
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

  function requireSyncBackend(method, asyncMethod) {
    if (deviceName === 'webgpu') throw new PolyAsyncRequired(method, asyncMethod)
  }

  function realizeUopsSync(ctx, uops) {
    requireSyncBackend('poly_realize_uops', 'poly_realize_uops_async')
    const n = uops.length
    if (n === 0) return []
    const inPtr = Module._malloc(n * 4)
    const outPtr = Module._malloc(n * 4)
    const h32 = heap32()
    for (let i = 0; i < n; i++) h32[(inPtr >> 2) + i] = uops[i]
    for (let i = 0; i < n; i++) h32[(outPtr >> 2) + i] = 0
    const rc = Module._poly_realize_uops(ctx, inPtr, n, outPtr)
    const out = new Array(n)
    if (rc === 0) {
      const h32b = heap32()
      for (let i = 0; i < n; i++) out[i] = h32b[(outPtr >> 2) + i]
    }
    Module._free(inPtr)
    Module._free(outPtr)
    return rc === 0 ? out : null
  }

  async function realizeUopsAsync(ctx, uops) {
    if (deviceName !== 'webgpu' || !Module.ccall) return realizeUopsSync(ctx, uops)
    await ensureWebGPU()
    const n = uops.length
    if (n === 0) return []
    const inPtr = Module._malloc(n * 4)
    const outPtr = Module._malloc(n * 4)
    const h32 = heap32()
    for (let i = 0; i < n; i++) h32[(inPtr >> 2) + i] = uops[i]
    for (let i = 0; i < n; i++) h32[(outPtr >> 2) + i] = 0
    const rc = await Module.ccall(
      'poly_realize_uops',
      'number',
      ['number', 'number', 'number', 'number'],
      [ctx, inPtr, n, outPtr],
      { async: true }
    )
    const out = new Array(n)
    if (rc === 0) {
      const h32b = heap32()
      for (let i = 0; i < n; i++) out[i] = h32b[(outPtr >> 2) + i]
    }
    Module._free(inPtr)
    Module._free(outPtr)
    return rc === 0 ? out : null
  }

  function realizeTensorsSync(ctx, tensors) {
    requireSyncBackend('poly_realize_tensors', 'poly_realize_tensors_async')
    const n = tensors.length
    if (n === 0) return []
    const inPtr = Module._malloc(n * 4)
    const outPtr = Module._malloc(n * 4)
    const h32 = heap32()
    for (let i = 0; i < n; i++) h32[(inPtr >> 2) + i] = tensors[i]
    for (let i = 0; i < n; i++) h32[(outPtr >> 2) + i] = 0
    const rc = Module._poly_realize_tensors(ctx, inPtr, n, outPtr)
    const out = new Array(n)
    if (rc === 0) {
      const h32b = heap32()
      for (let i = 0; i < n; i++) out[i] = h32b[(outPtr >> 2) + i]
    }
    Module._free(inPtr)
    Module._free(outPtr)
    return rc === 0 ? out : null
  }

  async function realizeTensorsAsync(ctx, tensors) {
    if (deviceName !== 'webgpu' || !Module.ccall) return realizeTensorsSync(ctx, tensors)
    await ensureWebGPU()
    const n = tensors.length
    if (n === 0) return []
    const inPtr = Module._malloc(n * 4)
    const outPtr = Module._malloc(n * 4)
    const h32 = heap32()
    for (let i = 0; i < n; i++) h32[(inPtr >> 2) + i] = tensors[i]
    for (let i = 0; i < n; i++) h32[(outPtr >> 2) + i] = 0
    const rc = await Module.ccall(
      'poly_realize_tensors',
      'number',
      ['number', 'number', 'number', 'number'],
      [ctx, inPtr, n, outPtr],
      { async: true }
    )
    const out = new Array(n)
    if (rc === 0) {
      const h32b = heap32()
      for (let i = 0; i < n; i++) out[i] = h32b[(outPtr >> 2) + i]
    }
    Module._free(inPtr)
    Module._free(outPtr)
    return rc === 0 ? out : null
  }

  function jitRunSync(jit, tensors) {
    requireSyncBackend('poly_jit_run', 'poly_jit_run_async')
    const ptr = writePtrArrayScratch(tensors)
    return Module._poly_jit_run(jit, ptr, tensors.length)
  }

  async function jitRunAsync(jit, tensors) {
    if (deviceName !== 'webgpu' || !Module.ccall) return jitRunSync(jit, tensors)
    await ensureWebGPU()
    const ptr = writePtrArrayScratch(tensors)
    return await Module.ccall(
      'poly_jit_run',
      'number',
      ['number', 'number', 'number'],
      [jit, ptr, tensors.length],
      { async: true }
    )
  }

  function bufferReadSync(ctx, buf, nbytes) {
    requireSyncBackend('poly_buffer_read', 'poly_buffer_read_async')
    if (nbytes <= 0) return new Uint8Array(0)
    const dst = Module._malloc(nbytes)
    try {
      const rc = Module._poly_buffer_read(ctx, buf, dst, nbytes)
      if (rc !== 0) throw new Error('poly_buffer_read failed (rc=' + rc + ')')
      return heapU8().slice(dst, dst + nbytes)
    } finally {
      Module._free(dst)
    }
  }

  async function bufferReadAsync(ctx, buf, nbytes) {
    if (deviceName !== 'webgpu' || !Module.ccall) return bufferReadSync(ctx, buf, nbytes)
    if (nbytes <= 0) return new Uint8Array(0)
    await ensureWebGPU()
    const dst = Module._malloc(nbytes)
    try {
      const rc = await Module.ccall(
        'poly_buffer_read',
        'number',
        ['number', 'number', 'number', 'number'],
        [ctx, buf, dst, nbytes],
        { async: true }
      )
      if (rc !== 0) throw new Error('poly_buffer_read failed (rc=' + rc + ')')
      return heapU8().slice(dst, dst + nbytes)
    } finally {
      Module._free(dst)
    }
  }

  const ffi = {
    // Simple ops (no int64 arrays)
    poly_ctx_new: Module._poly_ctx_new,
    poly_ctx_destroy: Module._poly_ctx_destroy,
    poly_ctx_named_count: Module._poly_ctx_named_count,
    poly_const_float: Module._poly_const_float,
    poly_const_double: Module._poly_const_double,
    poly_const_int: (ctx, val) => Module._poly_const_int(ctx, BigInt(val)),
    poly_const_float_by_id: Module._poly_const_float_by_id,
    poly_const_int_by_id: (ctx, val, dtypeId) =>
      Module._poly_const_int_by_id(ctx, BigInt(val), dtypeId),
    // Tensor.contiguous() is used as the explicit materialization/readback
    // boundary for views, matching the native N-API adapter surface.
    poly_contiguous: Module._poly_contiguous,
    poly_alu1: Module._poly_alu1,
    poly_alu2: Module._poly_alu2,
    poly_alu3: Module._poly_alu3,
    poly_store_val: Module._poly_store_val,
    poly_sink1: Module._poly_sink1,
    poly_sink_n: (ctx, stores) => {
      const ptr = writePtrArray(stores || [])
      try {
        return Module._poly_sink_n(ctx, ptr, (stores || []).length)
      } finally {
        if (ptr) Module._free(ptr)
      }
    },
    poly_uop_placeholder_like: Module._poly_uop_placeholder_like,
    poly_uop_range: (ctx, bound, axisId, axisType) =>
      Module._poly_uop_range(ctx, BigInt(bound), BigInt(axisId), axisType),
    poly_uop_index: (ctx, base, indices, keepPtr) => {
      const idx = indices || []
      const ptr = writePtrArray(idx)
      try {
        return Module._poly_uop_index(ctx, base, ptr, idx.length, keepPtr ? 1 : 0)
      } finally {
        if (ptr) Module._free(ptr)
      }
    },
    poly_uop_load: Module._poly_uop_load,
    poly_uop_store: Module._poly_uop_store,
    poly_uop_set: (ctx, addr, value, ranges) => {
      const rs = ranges || []
      const ptr = rs.length ? writePtrArray(rs) : 0
      try {
        return Module._poly_uop_set(ctx, addr, value, ptr, rs.length)
      } finally {
        if (ptr) Module._free(ptr)
      }
    },
    poly_uop_group: (ctx, srcs) => {
      const xs = srcs || []
      const ptr = writePtrArray(xs)
      try {
        return Module._poly_uop_group(ctx, ptr, xs.length)
      } finally {
        if (ptr) Module._free(ptr)
      }
    },
    poly_uop_end: (ctx, body, ranges) => {
      const rs = ranges || []
      const ptr = writePtrArray(rs)
      try {
        return Module._poly_uop_end(ctx, body, ptr, rs.length)
      } finally {
        if (ptr) Module._free(ptr)
      }
    },
    poly_uop_sink: (ctx, srcs) => {
      const xs = srcs || []
      const ptr = writePtrArray(xs)
      try {
        return Module._poly_uop_sink(ctx, ptr, xs.length)
      } finally {
        if (ptr) Module._free(ptr)
      }
    },
    poly_uop_sink_ex: (ctx, srcs, name, optimize) => {
      const xs = srcs || []
      const ptr = writePtrArray(xs)
      const namePtr = name ? writeCString(String(name)) : 0
      try {
        return Module._poly_uop_sink_ex(ctx, ptr, xs.length, namePtr, optimize ? 1 : 0)
      } finally {
        if (ptr) Module._free(ptr)
        if (namePtr) Module._free(namePtr)
      }
    },
    poly_uop_call: (ctx, body, args) => {
      const xs = args || []
      const ptr = writePtrArray(xs)
      try {
        return Module._poly_uop_call(ctx, body, ptr, xs.length)
      } finally {
        if (ptr) Module._free(ptr)
      }
    },
    poly_uop_after: Module._poly_uop_after,
    poly_uop_reduce: (ctx, op, expr, ranges) => {
      const rs = ranges || []
      const ptr = writePtrArray(rs)
      try {
        return Module._poly_uop_reduce(ctx, op, expr, ptr, rs.length)
      } finally {
        if (ptr) Module._free(ptr)
      }
    },
    poly_uop_flatten: Module._poly_uop_flatten,
    poly_uop_numel: (ctx, uop) => Number(Module._poly_uop_numel(ctx, uop)),
    poly_register_buffer_by_id: (ctx, role, dtypeId, shape, name) => {
      const shapePtr = writeInt64Array(shape || [])
      const namePtr = allocString(name)
      try {
        return Module._poly_register_buffer_by_id(
          ctx, role, dtypeId, shapePtr, (shape || []).length, namePtr
        )
      } finally {
        Module._free(namePtr)
        if (shapePtr) Module._free(shapePtr)
      }
    },
    poly_register_existing_buffer: (ctx, role, buffer, shape, name, trainable) => {
      const shapePtr = writeInt64Array(shape || [])
      const namePtr = allocString(name)
      try {
        return Module._poly_register_existing_buffer(
          ctx, role, buffer, shapePtr, (shape || []).length, namePtr, Boolean(trainable)
        )
      } finally {
        Module._free(namePtr)
        if (shapePtr) Module._free(shapePtr)
      }
    },
    poly_buffer_by_id: (ctx, dtypeId, size) => Module._poly_buffer_by_id(ctx, dtypeId, BigInt(size)),
    poly_buffer_on_device_by_id: (ctx, dtypeId, size, device) =>
      Module._poly_buffer_on_device_by_id(ctx, dtypeId, BigInt(size), device),
    poly_buffer_f32: (ctx, size) => Module._poly_buffer_f32(ctx, BigInt(size)),
    poly_buffer_f64: (ctx, size) => Module._poly_buffer_f64(ctx, BigInt(size)),

    // Matches poly_buffer_from_host (device.h). For the unified WebGPU path,
    // browser HOST stays JS-owned and the C core stores only a HOST residency
    // key; no eager copy into wasm memory happens at import. Non-WebGPU wasm
    // runtimes still stage bytes into wasm memory at construction.
    poly_buffer_from_host: (ctx, typedArray, nbytes, dtypeId, dims, ndim) => {
      let ptr = 0
      if (deviceName !== 'webgpu') {
        ptr = Module._malloc(nbytes)
        heapU8().set(new Uint8Array(typedArray.buffer, typedArray.byteOffset, nbytes), ptr)
      }
      const dimsPtr = (dims && ndim > 0) ? writeInt64Array(Array.from(dims)) : 0
      const uop = Module._poly_buffer_from_host(ctx, ptr, nbytes, dtypeId, dimsPtr, ndim)
      if (dimsPtr) Module._free(dimsPtr)
      return uop
    },

    poly_uop_has_buffer_identity: (uop) => !!Module._poly_uop_has_buffer_identity(uop),
    poly_uop_get_buffer_identity: (uop) => Module._poly_uop_get_buffer_identity(uop),
    poly_uop_op: (uop) => Module._poly_uop_op(uop),
    poly_uop_device: (uop) => Module._poly_uop_device(uop),
    poly_uop_dtype_id: (ctx, uop) => Module._poly_uop_dtype_id(ctx, uop),
    poly_uop_key: (uop) => BigInt(uop || 0),
    poly_uop_n_src: (uop) => Module._poly_uop_n_src(uop || 0),
    poly_uop_src: (uop, idx) => Module._poly_uop_src(uop || 0, idx | 0),
    poly_uop_reachable: (ctx, root, target) =>
      !!Module._poly_uop_reachable(ctx, root || 0, target || 0),
    poly_uop_substitute: (ctx, root, from, to) => {
      const n = Math.min(from.length, to.length)
      if (n <= 0) return root
      const fromPtr = Module._malloc(n * 4)
      const toPtr = Module._malloc(n * 4)
      const h32 = heap32()
      for (let i = 0; i < n; i++) {
        h32[(fromPtr >> 2) + i] = from[i] || 0
        h32[(toPtr >> 2) + i] = to[i] || 0
      }
      const out = Module._poly_uop_substitute(ctx, root, fromPtr, toPtr, n)
      Module._free(fromPtr)
      Module._free(toPtr)
      return out
    },
    poly_buffer_get_ptr: (ctx, buf) => Module._poly_buffer_get_ptr(ctx, buf),
    poly_buffer_is_allocated: (ctx, buf) => !!Module._poly_buffer_is_allocated(ctx, buf),
    poly_buffer_get_key: (ctx, buf) => Module._poly_buffer_get_key(ctx, buf),
    poly_device_by_name: (name) => {
      const key = String(name).toLowerCase()
      if (key === 'auto') return AUTO_DEVICE_ID
      return DEVICE_IDS[key] !== undefined ? DEVICE_IDS[key] : coreDeviceId(key)
    },
    poly_device_name: (device) => coreDeviceName(device),
    poly_device_is_host_addressable: (device) =>
      Boolean(Module._poly_device_is_host_addressable(device)),
    poly_tensor_create: (ctx, uop, role, device) =>
      Module._poly_tensor_create(ctx, uop, role, device),
    poly_tensor_empty_by_id: (ctx, dtypeId, shape, ndim, device) => {
      const dimsPtr = writeInt64Array(shape || [])
      try {
        return Module._poly_tensor_empty_by_id(ctx, dtypeId, dimsPtr, ndim, device)
      } finally {
        if (dimsPtr) Module._free(dimsPtr)
      }
    },
    poly_tensor_create_with_roots: (ctx, logical, physical, role, device) =>
      Module._poly_tensor_create_with_roots(ctx, logical, physical, role, device),
    poly_tensor_update: (ctx, tensor, logical, physical, role, device) =>
      Module._poly_tensor_update(ctx, tensor, logical || 0, physical || 0, role, device),
    poly_tensor_to_device: (ctx, tensor, device) =>
      Module._poly_tensor_to_device(ctx, tensor, device),
    poly_tensor_assign: (ctx, target, value) =>
      Module._poly_tensor_assign(ctx, target, value),
    poly_tensor_alu1: (ctx, op, src) =>
      Module._poly_tensor_alu1(ctx, op, src),
    poly_tensor_alu2: (ctx, op, a, b) =>
      Module._poly_tensor_alu2(ctx, op, a, b),
    poly_tensor_alu3: (ctx, op, a, b, c) =>
      Module._poly_tensor_alu3(ctx, op, a, b, c),
    poly_tensor_clone_into: (ctx, target, source) =>
      Module._poly_tensor_clone_into(ctx, target, source),
    poly_tensor_uop: (tensor) => Module._poly_tensor_uop(tensor),
    poly_tensor_uop_logical: (tensor) => Module._poly_tensor_uop_logical(tensor),
    poly_tensor_uop_physical: (tensor) => Module._poly_tensor_uop_physical(tensor),
    poly_tensor_device: (tensor) => Module._poly_tensor_device(tensor),
    poly_tensor_requires_grad: (tensor) => Boolean(Module._poly_tensor_requires_grad(tensor)),
    poly_tensor_set_requires_grad: (tensor, requiresGrad) =>
      Module._poly_tensor_set_requires_grad(tensor, Boolean(requiresGrad)),
    poly_set_frontend_buffer_release: (fn) => {
      Module.__polygradFrontendBufferRelease = fn
      if (!Module.__polygradFrontendBufferReleasePtr) {
        Module.__polygradFrontendBufferReleasePtr = Module.addFunction((bufferKey) => {
          if (Module.__polygradFrontendBufferRelease) {
            Module.__polygradFrontendBufferRelease(bufferKey)
          }
        }, 'vi')
      }
      Module._poly_set_frontend_buffer_release(Module.__polygradFrontendBufferReleasePtr)
    },
    poly_ctx_set_frontend_buffer_release: (ctx, fn) => {
      Module.__polygradFrontendBufferRelease = fn
      if (!Module.__polygradFrontendBufferReleasePtr) {
        Module.__polygradFrontendBufferReleasePtr = Module.addFunction((bufferKey) => {
          if (Module.__polygradFrontendBufferRelease) {
            Module.__polygradFrontendBufferRelease(bufferKey)
          }
        }, 'vi')
      }
      Module._poly_ctx_set_frontend_buffer_release(ctx, Module.__polygradFrontendBufferReleasePtr)
    },

    // Batched raw-UOp/tensor realize helpers.
    poly_realize_uops: realizeUopsSync,
    poly_realize_uops_async: realizeUopsAsync,
    poly_realize_tensors: realizeTensorsSync,
    poly_realize_tensors_async: realizeTensorsAsync,

    poly_jit_new: (ctx) => Module._poly_jit_new(ctx),
    poly_jit_free: (jit) => Module._poly_jit_free(jit),
    poly_jit_set_prune: (jit, prune) => Module._poly_jit_set_prune(jit, Boolean(prune)),
    poly_jit_begin_capture: (jit, tensors) => {
      const ptr = writePtrArray(tensors)
      try {
        return Module._poly_jit_begin_capture(jit, ptr, tensors.length)
      } finally {
        if (ptr) Module._free(ptr)
      }
    },
    poly_jit_end_capture: (jit) => Module._poly_jit_end_capture(jit),
    poly_jit_cancel_capture: (jit) => Module._poly_jit_cancel_capture(jit),
    poly_jit_is_captured: (jit) => Boolean(Module._poly_jit_is_captured(jit)),
    poly_jit_schedule_count: (jit) => Module._poly_jit_schedule_count(jit),
    poly_jit_run: jitRunSync,
    poly_jit_run_async: jitRunAsync,

    poly_optim_build_step: (ctx, cfg, lr, params, grads, mTensors, vTensors, bc1, bc2) => {
      const n = params.length
      const cfgPtr = writeOptimConfig(cfg || {})
      const paramsPtr = writePtrArray(params)
      const gradsPtr = writePtrArray(grads)
      const mPtr = writePtrArray(mTensors)
      const vPtr = writePtrArray(vTensors)
      let outPtr = 0
      try {
        /* src/optim.c owns the optimizer math. JS only marshals PolyTensor*
         * arrays and receives the tensors whose AFTER/STORE effects must be
         * realized together. */
        const needed = Module._poly_optim_build_step(
          ctx, cfgPtr, lr, paramsPtr, gradsPtr, n, mPtr, vPtr, bc1 || 0, bc2 || 0, 0, 0
        )
        if (needed < 0) return null
        outPtr = Module._malloc(Math.max(1, needed) * 4)
        const rc = Module._poly_optim_build_step(
          ctx, cfgPtr, lr, paramsPtr, gradsPtr, n, mPtr, vPtr,
          bc1 || 0, bc2 || 0, outPtr, needed
        )
        if (rc < 0) return null
        return readPtrArray(outPtr, rc)
      } finally {
        if (outPtr) Module._free(outPtr)
        if (vPtr) Module._free(vPtr)
        if (mPtr) Module._free(mPtr)
        if (gradsPtr) Module._free(gradsPtr)
        if (paramsPtr) Module._free(paramsPtr)
        Module._free(cfgPtr)
      }
    },

    // Backend-aware readback helpers.
    poly_buffer_read: bufferReadSync,
    poly_buffer_read_async: bufferReadAsync,
    poly_buffer_ensure_device_allocated: (ctx, buf, device) => {
      if (deviceName === 'webgpu' && device === DEVICE_IDS.webgpu &&
          !(Module.__polygradWebGpuState && Module.__polygradWebGpuState.device)) return
      const rc = Module._poly_buffer_ensure_device_allocated(ctx, buf, device)
      if (rc !== 0) {
        throw new Error('poly_buffer_ensure_device_allocated failed (rc=' + rc + ')')
      }
    },
    poly_buffer_write: (ctx, buf, src) => {
      let bytes
      if (src instanceof Uint8Array) {
        bytes = src
      } else if (ArrayBuffer.isView(src)) {
        bytes = new Uint8Array(src.buffer, src.byteOffset, src.byteLength)
      } else if (src instanceof ArrayBuffer) {
        bytes = new Uint8Array(src)
      } else {
        throw new TypeError('poly_buffer_write expects a TypedArray or ArrayBuffer')
      }
      const nbytes = bytes.byteLength
      const useFrontendHostKey = deviceName === 'webgpu' &&
        !(Module.__polygradWebGpuState && Module.__polygradWebGpuState.device)
      const frontendBytes = useFrontendHostKey ? bytes.slice() : null
      const ptr = nbytes ? Module._malloc(nbytes) : 0
      try {
        if (nbytes) heapU8().set(bytes, ptr)
        const rc = Module._poly_buffer_write(ctx, buf, ptr, nbytes)
        if (rc !== 0) throw new Error('poly_buffer_write failed (rc=' + rc + ')')
        if (frontendBytes) {
          Module._poly_buffer_set(ctx, buf, 0, nbytes, HOST_DEVICE_ID)
          const bufferKey = Module._poly_buffer_get_key(ctx, buf)
          if (!bufferKey) throw new Error('poly_buffer_set did not create a frontend host key')
          Module.__polygradHostBuffers.set(String(bufferKey), frontendBytes)
        }
      } finally {
        if (ptr) Module._free(ptr)
      }
    },
    poly_ctx_stats: readCtxStats,
    poly_ctx_reset_counters: (ctx) => Module._poly_ctx_reset_counters(ctx),
    poly_grad: Module._poly_grad,
    poly_grad_many: (ctx, loss, initialGrad, targets) => {
      const n = targets.length
      const wrtsPtr = Module._malloc(n * 4)
      const outPtr = Module._malloc(n * 4)
      const presentPtr = Module._malloc(n)
      const h32 = heap32()
      const u8 = heapU8()
      for (let i = 0; i < n; i++) {
        h32[(wrtsPtr >> 2) + i] = targets[i] || 0
        h32[(outPtr >> 2) + i] = 0
        u8[presentPtr + i] = 0
      }
      /* Match tinygrad's single gradient pass over all live targets. The
       * pointer arrays are wasm32 handles, so each slot is 4 bytes. */
      const rc = Module._poly_grad_many_ex(
        ctx, loss || 0, initialGrad || 0, wrtsPtr, n, outPtr, presentPtr
      )
      if (rc !== 0) {
        Module._free(wrtsPtr)
        Module._free(outPtr)
        Module._free(presentPtr)
        return null
      }
      /* Any C call may grow Emscripten memory. Reacquire the heap views before
       * reading result pointers, just as the realization bridges above do. */
      const h32b = heap32()
      const u8b = heapU8()
      const grads = new Array(n)
      const present = new Array(n)
      for (let i = 0; i < n; i++) {
        grads[i] = h32b[(outPtr >> 2) + i]
        present[i] = u8b[presentPtr + i] !== 0
      }
      Module._free(wrtsPtr)
      Module._free(outPtr)
      Module._free(presentPtr)
      return { grads, present }
    },
    poly_detach: Module._poly_detach,
    poly_cast_by_id: Module._poly_cast_by_id,

    // Shape-on-UOp accessors
    poly_uop_ndim: (ctx, uop) => Module._poly_uop_ndim(ctx, uop),
    poly_uop_max_shape_dims: (ctx, uop) => {
      const ndim = Module._poly_uop_ndim(ctx, uop)
      if (ndim <= 0) return []
      const dimsPtr = Module._poly_uop_max_shape_dims(ctx, uop)
      if (!dimsPtr) return []
      const result = []
      const h32 = heap32()
      for (let i = 0; i < ndim; i++) {
        const lo = h32[(dimsPtr >> 2) + i * 2]
        const hi = h32[(dimsPtr >> 2) + i * 2 + 1]
        result.push(lo + hi * 0x100000000)
      }
      return result
    },

    // Shape-taking ops
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

    poly_shrink_uop: (ctx, uop, starts, sizes, ndim) => {
      const n = Number(ndim)
      const startsPtr = Module._malloc(n * 4)
      const sizesPtr = Module._malloc(n * 4)
      const h32 = heap32()
      for (let i = 0; i < n; i++) {
        h32[(startsPtr >> 2) + i] = starts[i] || 0
        h32[(sizesPtr >> 2) + i] = sizes[i] || 0
      }
      const result = Module._poly_shrink_uop(ctx, uop, startsPtr, sizesPtr, n)
      Module._free(startsPtr)
      Module._free(sizesPtr)
      return result
    },

    poly_pad: (ctx, uop, flat, npairs) => {
      const ptr = writeInt64Array(flat)
      const result = cwrapPad(ctx, uop, ptr, npairs)
      Module._free(ptr)
      return result
    },

    poly_pad_value: (ctx, uop, flat, npairs, value) => {
      const ptr = writeInt64Array(flat)
      const result = Module._poly_pad_value(ctx, uop, ptr, npairs, value)
      Module._free(ptr)
      return result
    },

    poly_pool: (ctx, uop, k, nk, stride, dilation) => {
      const kPtr = writeInt64Array(k)
      const stridePtr = writeInt64Array(stride)
      const dilationPtr = writeInt64Array(dilation)
      const result = Module._poly_pool(ctx, uop, kPtr, nk, stridePtr, dilationPtr)
      Module._free(kPtr)
      Module._free(stridePtr)
      Module._free(dilationPtr)
      return result
    },

    poly_max_pool2d: (ctx, x, k, nk, stride, dilation, padding, npadding) => {
      const kPtr = writeInt64Array(k)
      const stridePtr = writeInt64Array(stride)
      const dilationPtr = writeInt64Array(dilation)
      const paddingPtr = writeInt64Array(padding)
      const result = Module._poly_max_pool2d(
        ctx, x, kPtr, nk, stridePtr, dilationPtr, paddingPtr, npadding
      )
      Module._free(kPtr)
      Module._free(stridePtr)
      Module._free(dilationPtr)
      Module._free(paddingPtr)
      return result
    },

    poly_conv2d: (ctx, x, weight, bias, groups, stride, dilation, padding, npadding) => {
      const stridePtr = writeInt64Array(stride)
      const dilationPtr = writeInt64Array(dilation)
      const paddingPtr = writeInt64Array(padding)
      const result = Module._poly_conv2d(
        ctx, x, weight, bias || 0, groups, stridePtr, dilationPtr, paddingPtr, npadding
      )
      Module._free(stridePtr)
      Module._free(dilationPtr)
      Module._free(paddingPtr)
      return result
    },

    poly_batchnorm: (ctx, x, weight, bias, mean, invstd, axes, naxes) => {
      const axesPtr = writeInt64Array(axes)
      const result = Module._poly_batchnorm(
        ctx, x, weight || 0, bias || 0, mean, invstd, axesPtr, naxes
      )
      Module._free(axesPtr)
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

    // Shape-on-UOp: C computes shapes internally, JS passes bare UOp pointers.
    poly_max_reduce: Module._poly_max_reduce,
    poly_mean_reduce: Module._poly_mean_reduce,
    poly_var_reduce: Module._poly_var_reduce,
    poly_one_hot: (ctx, x, numClasses) =>
      Module._poly_one_hot(ctx, x, BigInt(numClasses)),
    poly_index_select: Module._poly_index_select,
    poly_softmax: Module._poly_softmax,
    poly_log_softmax: Module._poly_log_softmax,
    poly_dot: Module._poly_dot,
    poly_qr: (ctx, uop) => callUopPair(Module._poly_qr, [ctx, uop], 'poly_qr'),
    poly_qr_ex: (ctx, uop, mode) => callUopPair(Module._poly_qr_ex, [ctx, uop, mode], 'poly_qr_ex'),
    poly_cross_entropy: Module._poly_cross_entropy,
    poly_gather_dim: Module._poly_gather_dim,
    poly_scatter: (ctx, self, dim, index, src, reduce) => {
      const reducePtr = allocString(reduce || '')
      const result = Module._poly_scatter(ctx, self, dim, index, src, reducePtr)
      Module._free(reducePtr)
      return result
    },
    poly_scatter_reduce: (ctx, self, dim, index, src, reduce, includeSelf) => {
      const reducePtr = allocString(reduce)
      const result = Module._poly_scatter_reduce(ctx, self, dim, index, src, reducePtr, includeSelf ? 1 : 0)
      Module._free(reducePtr)
      return result
    },
    poly_argmax: Module._poly_argmax,
    poly_argsort: Module._poly_argsort,
    poly_sort: (ctx, uop, dim, descending) =>
      callUopPair(Module._poly_sort, [ctx, uop, dim, descending ? 1 : 0], 'poly_sort'),
    poly_topk: (ctx, uop, k, dim, largest, sorted) =>
      callUopPair(
        Module._poly_topk,
        [ctx, uop, BigInt(k), dim, largest ? 1 : 0, sorted ? 1 : 0],
        'poly_topk'
      ),

    poly_einsum: (ctx, formula, operands) => {
      const n = operands.length
      const tensorPtrs = Module._malloc(n * 4)
      for (let i = 0; i < n; i++) {
        heap32()[(tensorPtrs >> 2) + i] = operands[i]._uop
      }
      const formulaPtr = allocString(formula)
      const result = Module._poly_einsum(ctx, formulaPtr, tensorPtrs, n)
      Module._free(formulaPtr)
      Module._free(tensorPtrs)
      return { uop: result, shape: readUopShape(ctx, result) }
    },

    poly_rearrange: (ctx, formula, uop, shape, kwargs) => {
      const names = Object.keys(kwargs)
      const values = names.map(k => kwargs[k])
      const n = names.length
      const formulaPtr = allocString(formula)
      let namesPtr = 0
      let valuesPtr = 0
      if (n > 0) {
        namesPtr = allocString(names.join(' '))
        valuesPtr = writeInt64Array(values)
      }
      const result = Module._poly_rearrange(ctx, formulaPtr, uop, namesPtr, valuesPtr, n)
      Module._free(formulaPtr)
      if (namesPtr) Module._free(namesPtr)
      if (valuesPtr) Module._free(valuesPtr)
      return { uop: result, shape: readUopShape(ctx, result) }
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

    // Creation
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
    // Shape-on-UOp: C reads shape from UOp internally.
    poly_tril: Module._poly_tril,
    poly_triu: Module._poly_triu,
    poly_cholesky: Module._poly_cholesky,
    poly_cholesky_solve: Module._poly_cholesky_solve,
    poly_triangular_solve: Module._poly_triangular_solve,
    poly_solve: Module._poly_solve,
    poly_lstsq: Module._poly_lstsq,

    // Reduction (non-shape-returning)
    poly_sum_reduce: Module._poly_sum_reduce,
    poly_logsumexp: Module._poly_logsumexp,

    // ABI
    poly_abi_version: Module._poly_abi_version,
    poly_op_count: Module._poly_op_count
  }

  // ABI version check
  const EXPECTED_ABI = 23
  const abi = ffi.poly_abi_version()
  if (abi !== EXPECTED_ABI) {
    throw new Error(
      `polygrad WASM ABI mismatch: expected version ${EXPECTED_ABI}, got ${abi}. ` +
      'Rebuild polygrad.wasm or update the polygrad package.'
    )
  }

  // Create context
  const ctx = ffi.poly_ctx_new()
  if (Module._poly_ctx_set_preferred_device) {
    Module._poly_ctx_set_preferred_device(ctx, deviceId)
  }

  // --- Instance API ---
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
      if (inst && Module._poly_instance_set_device(inst, deviceId) !== 0) {
        Module._poly_instance_free(inst)
        throw new Error('polygrad: set_device failed for device ' + deviceName)
      }
      return inst || null
    },

    mlp(specJson) {
      const bytes = new TextEncoder().encode(specJson)
      const specPtr = allocBytes(bytes)
      const inst = Module._poly_mlp_from_json(specPtr, bytes.length)
      Module._free(specPtr)
      if (inst && Module._poly_instance_set_device(inst, deviceId) !== 0) {
        Module._poly_instance_free(inst)
        throw new Error('polygrad: set_device failed for device ' + deviceName)
      }
      return inst || null
    },

    tabm(specJson) {
      const bytes = new TextEncoder().encode(specJson)
      const specPtr = allocBytes(bytes)
      const inst = Module._poly_tabm_instance(specPtr, bytes.length)
      Module._free(specPtr)
      if (inst && Module._poly_instance_set_device(inst, deviceId) !== 0) {
        Module._poly_instance_free(inst)
        throw new Error('polygrad: set_device failed for device ' + deviceName)
      }
      return inst || null
    },

    nam(specJson) {
      const bytes = new TextEncoder().encode(specJson)
      const specPtr = allocBytes(bytes)
      const inst = Module._poly_nam_instance(specPtr, bytes.length)
      Module._free(specPtr)
      if (inst && Module._poly_instance_set_device(inst, deviceId) !== 0) {
        Module._poly_instance_free(inst)
        throw new Error('polygrad: set_device failed for device ' + deviceName)
      }
      return inst || null
    },

    free(instPtr) { Module._poly_instance_free(instPtr) },
    paramCount(instPtr) { return Module._poly_instance_param_count(instPtr) },
    paramName(instPtr, i) { return readCString(Module._poly_instance_param_name(instPtr, i)) },
    paramShape(instPtr, i) {
      const ndim = Module._poly_instance_param_shape(instPtr, i, _scratchOutShapePtr, 8)
      return readShapeFromPtr(_scratchOutShapePtr, ndim)
    },
    paramData(instPtr, i) {
      const read = (dataPtr) => {
        if (!dataPtr) return null
        const numel = readInt64At(_scratchNumelPtr)
        return new Float32Array(heapF32().buffer.slice(dataPtr, dataPtr + numel * 4))
      }
      if (deviceName === 'webgpu' && Module.ccall) {
        const nbytes = shapeNumel(this.paramShape(instPtr, i)) * 4
        if (nbytes <= 0) return Promise.resolve(new Float32Array(0))
        const dst = Module._malloc(nbytes)
        return Module.ccall(
          'poly_instance_readback_param',
          'number',
          ['number', 'number', 'number', 'number'],
          [instPtr, i, dst, nbytes],
          { async: true }
        ).then(rc => {
          if (rc !== 0) return null
          return new Float32Array(heapF32().buffer.slice(dst, dst + nbytes))
        }).finally(() => Module._free(dst))
      }
      const dataPtr = Module._poly_instance_param_data(instPtr, i, _scratchNumelPtr)
      return read(dataPtr)
    },
    paramTrainable(instPtr, i) {
      return Boolean(Module._poly_instance_param_trainable(instPtr, i))
    },
    setParamTrainable(instPtr, i, trainable) {
      return Module._poly_instance_set_param_trainable(instPtr, i, Boolean(trainable))
    },
    bufCount(instPtr) { return Module._poly_instance_buf_count(instPtr) },
    bufName(instPtr, i) { return readCString(Module._poly_instance_buf_name(instPtr, i)) },
    bufRole(instPtr, i) { return Module._poly_instance_buf_role(instPtr, i) },
    bufTrainable(instPtr, i) {
      return Boolean(Module._poly_instance_buf_trainable(instPtr, i))
    },
    setBufTrainable(instPtr, i, trainable) {
      return Module._poly_instance_set_buf_trainable(instPtr, i, Boolean(trainable))
    },
    bufShape(instPtr, i) {
      const ndim = Module._poly_instance_buf_shape(instPtr, i, _scratchOutShapePtr, 8)
      return readShapeFromPtr(_scratchOutShapePtr, ndim)
    },
    bufData(instPtr, i) {
      const read = (dataPtr) => {
        if (!dataPtr) return null
        const numel = readInt64At(_scratchNumelPtr)
        return new Float32Array(heapF32().buffer.slice(dataPtr, dataPtr + numel * 4))
      }
      if (deviceName === 'webgpu' && Module.ccall) {
        const nbytes = shapeNumel(this.bufShape(instPtr, i)) * 4
        if (nbytes <= 0) return Promise.resolve(new Float32Array(0))
        const dst = Module._malloc(nbytes)
        return Module.ccall(
          'poly_instance_readback_buf',
          'number',
          ['number', 'number', 'number', 'number'],
          [instPtr, i, dst, nbytes],
          { async: true }
        ).then(rc => {
          if (rc !== 0) return null
          return new Float32Array(heapF32().buffer.slice(dst, dst + nbytes))
        }).finally(() => Module._free(dst))
      }
      const dataPtr = Module._poly_instance_buf_data(instPtr, i, _scratchNumelPtr)
      return read(dataPtr)
    },
    exportWeights(instPtr, flags) {
      const exportFlags = flags == null ? 3 : flags
      if (deviceName === 'webgpu' && Module.ccall) {
        return Module.ccall(
          'poly_instance_export_weights_ex',
          'number',
          ['number', 'number', 'number'],
          [instPtr, _scratchLenPtr, exportFlags],
          { async: true }
        ).then(bytesPtr => {
          if (!bytesPtr) return null
          const len = heap32()[_scratchLenPtr >> 2]
          const bytes = new Uint8Array(heapU8().buffer.slice(bytesPtr, bytesPtr + len))
          Module._free(bytesPtr)
          return bytes
        })
      }
      const bytesPtr = Module._poly_instance_export_weights_ex(instPtr, _scratchLenPtr, exportFlags)
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
    saveBundle(instPtr, flags) {
      const exportFlags = flags == null ? 3 : flags
      if (deviceName === 'webgpu' && Module.ccall) {
        return Module.ccall(
          'poly_instance_save_bundle_ex',
          'number',
          ['number', 'number', 'number'],
          [instPtr, _scratchLenPtr, exportFlags],
          { async: true }
        ).then(bytesPtr => {
          if (!bytesPtr) return null
          const len = heap32()[_scratchLenPtr >> 2]
          const bytes = new Uint8Array(heapU8().buffer.slice(bytesPtr, bytesPtr + len))
          Module._free(bytesPtr)
          return bytes
        })
      }
      const bytesPtr = Module._poly_instance_save_bundle_ex(instPtr, _scratchLenPtr, exportFlags)
      if (!bytesPtr) return null
      const len = heap32()[_scratchLenPtr >> 2]
      const bytes = new Uint8Array(heapU8().buffer.slice(bytesPtr, bytesPtr + len))
      Module._free(bytesPtr)
      return bytes
    },
    fromBundle(bytes) {
      const ptr = allocBytes(bytes)
      const inst = Module._poly_instance_from_bundle(ptr, bytes.length)
      Module._free(ptr)
      if (inst && Module._poly_instance_set_device(inst, deviceId) !== 0) {
        Module._poly_instance_free(inst)
        throw new Error('polygrad: set_device failed for device ' + deviceName)
      }
      return inst || null
    },

    fromSinks(ctxPtr, names, sinks) {
      const n = Math.min(names.length, sinks.length)
      const namesPtr = Module._malloc(Math.max(1, n) * 4)
      const sinksPtr = Module._malloc(Math.max(1, n) * 4)
      const namePtrs = []
      try {
        for (let i = 0; i < n; i++) {
          const p = allocString(names[i])
          namePtrs.push(p)
          heap32()[(namesPtr >> 2) + i] = p
          heap32()[(sinksPtr >> 2) + i] = sinks[i] || 0
        }
        const inst = Module._poly_instance_from_sinks(ctxPtr, namesPtr, sinksPtr, n)
        if (inst && Module._poly_instance_set_device(inst, deviceId) !== 0) {
          Module._poly_instance_free(inst)
          throw new Error('polygrad: set_device failed for device ' + deviceName)
        }
        return inst || null
      } finally {
        for (const p of namePtrs) Module._free(p)
        Module._free(namesPtr)
        Module._free(sinksPtr)
      }
    },

    fromBindings(ctxPtr, bindings, entries) {
      const bindingNames = bindings.map(b => b.name)
      const bindingRoles = bindings.map(b => b.role | 0)
      const bindingTensors = bindings.map(b => b.tensor || 0)
      const bindingFlags = bindings.map(b => b.flags || 0)
      const entryNames = entries.map(e => e.name)
      const entryInputs = entries.flatMap(e => e.inputs || [])
      const entryInputCounts = entries.map(e => (e.inputs || []).length)
      const entryOutputs = entries.flatMap(e => e.outputs || [])
      const entryOutputCounts = entries.map(e => (e.outputs || []).length)
      const entryObjectives = entries.map(e => e.objective || null)
      const entryFlags = entries.map(e => e.flags || 0)

      const stringPtrs = []
      const allocStringArray = (items, nullable = false) => {
        const ptr = Module._malloc(Math.max(1, items.length) * 4)
        for (let i = 0; i < items.length; i++) {
          const item = items[i]
          const sp = nullable && item == null ? 0 : allocString(item)
          if (sp) stringPtrs.push(sp)
          heap32()[(ptr >> 2) + i] = sp
        }
        return ptr
      }

      const bindingNamesPtr = allocStringArray(bindingNames)
      const bindingRolesPtr = writeI32Array(bindingRoles)
      const bindingTensorsPtr = writePtrArray(bindingTensors)
      const bindingFlagsPtr = writeI32Array(bindingFlags)
      const entryNamesPtr = allocStringArray(entryNames)
      const entryInputsPtr = allocStringArray(entryInputs)
      const entryInputCountsPtr = writeI32Array(entryInputCounts)
      const entryOutputsPtr = allocStringArray(entryOutputs)
      const entryOutputCountsPtr = writeI32Array(entryOutputCounts)
      const entryObjectivesPtr = allocStringArray(entryObjectives, true)
      const entryFlagsPtr = writeI32Array(entryFlags)

      try {
        const inst = Module._poly_instance_from_binding_arrays(
          ctxPtr,
          bindingNamesPtr, bindingRolesPtr, bindingTensorsPtr, bindingFlagsPtr, bindings.length,
          entryNamesPtr, entryInputsPtr, entryInputCountsPtr,
          entryOutputsPtr, entryOutputCountsPtr, entryObjectivesPtr, entryFlagsPtr,
          entries.length, 0, 0
        )
        if (inst && Module._poly_instance_set_device(inst, deviceId) !== 0) {
          Module._poly_instance_free(inst)
          throw new Error('polygrad: set_device failed for device ' + deviceName)
        }
        return inst || null
      } finally {
        for (const p of stringPtrs) Module._free(p)
        Module._free(bindingNamesPtr)
        if (bindingRolesPtr) Module._free(bindingRolesPtr)
        if (bindingTensorsPtr) Module._free(bindingTensorsPtr)
        if (bindingFlagsPtr) Module._free(bindingFlagsPtr)
        Module._free(entryNamesPtr)
        if (entryInputsPtr) Module._free(entryInputsPtr)
        if (entryInputCountsPtr) Module._free(entryInputCountsPtr)
        if (entryOutputsPtr) Module._free(entryOutputsPtr)
        if (entryOutputCountsPtr) Module._free(entryOutputCountsPtr)
        if (entryObjectivesPtr) Module._free(entryObjectivesPtr)
        if (entryFlagsPtr) Module._free(entryFlagsPtr)
      }
    },

    loadHF(configBytes, weightFilesBytes, maxBatch, maxSeqLen) {
      const cfgPtr = allocBytes(configBytes)
      const n = weightFilesBytes.length
      const ptrArr = Module._malloc(n * 4)
      const lenArr = Module._malloc(n * 8)
      const filePtrs = []
      for (let i = 0; i < n; i++) {
        const fp = allocBytes(weightFilesBytes[i])
        filePtrs.push(fp)
        heap32()[(ptrArr >> 2) + i] = fp
        heap32()[(lenArr >> 2) + i * 2] = weightFilesBytes[i].length
        heap32()[(lenArr >> 2) + i * 2 + 1] = 0
      }
      const inst = Module._poly_hf_load(
        cfgPtr, configBytes.length, ptrArr, lenArr, n,
        maxBatch || 1, maxSeqLen || 0)
      for (const fp of filePtrs) Module._free(fp)
      Module._free(ptrArr); Module._free(lenArr); Module._free(cfgPtr)
      if (inst && Module._poly_instance_set_device(inst, deviceId) !== 0) {
        Module._poly_instance_free(inst)
        throw new Error('polygrad: set_device failed')
      }
      return inst || null
    },

    loadGGUF(ggufBytes, maxBatch, maxSeqLen) {
      const ptr = allocBytes(ggufBytes)
      const inst = Module._poly_gguf_load(
        ptr, BigInt(ggufBytes.length), maxBatch || 1, maxSeqLen || 0)
      Module._free(ptr)
      if (inst && Module._poly_instance_set_device(inst, deviceId) !== 0) {
        Module._poly_instance_free(inst)
        throw new Error('polygrad: set_device failed')
      }
      return inst || null
    },

    importLastError() {
      const code = Module._poly_import_last_error_code()
      if (code === 0) return null
      const msgPtr = Module._poly_import_last_error_message()
      return { code, message: msgPtr ? readCString(msgPtr) : 'unknown' }
    },

    tokenizerFromGGUF(ggufBytes) {
      const ptr = allocBytes(ggufBytes)
      const outPtr = Module._malloc(4)
      Module._poly_gguf_decode(ptr, BigInt(ggufBytes.length), outPtr)
      const decoded = heap32()[outPtr >> 2]
      Module._free(outPtr)
      if (!decoded) { Module._free(ptr); return null }
      const tok = Module._poly_tokenizer_from_gguf(decoded)
      Module._poly_gguf_decoded_free(decoded)
      Module._free(ptr)
      return tok || null
    },

    tokenizerFromJSON(jsonBytes) {
      const ptr = allocBytes(jsonBytes)
      const tok = Module._poly_tokenizer_from_json(ptr, jsonBytes.length)
      Module._free(ptr)
      return tok || null
    },

    tokenize(tokPtr, text) {
      const textPtr = allocString(text)
      const idsPtr = Module._malloc(4096 * 4)
      const n = Module._poly_tokenize(tokPtr, textPtr, idsPtr, 4096)
      const ids = new Int32Array(n)
      for (let i = 0; i < n; i++) ids[i] = heap32()[(idsPtr >> 2) + i]
      Module._free(idsPtr); Module._free(textPtr)
      return ids
    },

    detokenize(tokPtr, ids) {
      const idsPtr = Module._malloc(ids.length * 4)
      for (let i = 0; i < ids.length; i++) heap32()[(idsPtr >> 2) + i] = ids[i]
      const bufPtr = Module._malloc(8192)
      Module._poly_detokenize(tokPtr, idsPtr, ids.length, bufPtr, 8192)
      const text = readCString(bufPtr)
      Module._free(bufPtr); Module._free(idsPtr)
      return text
    },

    tokenizerFree(tokPtr) { Module._poly_tokenizer_free(tokPtr) },
    tokenizerVocabSize(tokPtr) { return Module._poly_tokenizer_vocab_size(tokPtr) },
    tokenizerBosId(tokPtr) { return Module._poly_tokenizer_bos_id(tokPtr) },
    tokenizerEosId(tokPtr) { return Module._poly_tokenizer_eos_id(tokPtr) },

    setOptimizer(instPtr, kind, lr, beta1, beta2, eps, weightDecay, momentum, nesterov, classic) {
      return Module._poly_instance_set_optimizer_ex(
        instPtr, kind, lr, beta1, beta2, eps, weightDecay, momentum || 0, !!nesterov, !!classic)
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
      const cleanup = () => {
        for (const ptr of dataPtrs) Module._free(ptr)
        for (const ptr of namePtrs) Module._free(ptr)
        Module._free(bindingPtr)
      }
      if (deviceName === 'webgpu' && Module.ccall) {
        return Module.ccall(
          'poly_instance_forward',
          'number',
          ['number', 'number', 'number'],
          [instPtr, bindingPtr, n],
          { async: true }
        ).finally(cleanup)
      }
      const rc = Module._poly_instance_forward(instPtr, bindingPtr, n)
      cleanup()
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
      const readLoss = (rc) => rc === 0 ? heapF32()[lossPtr >> 2] : null
      const cleanup = () => {
        Module._free(lossPtr)
        for (const ptr of dataPtrs) Module._free(ptr)
        for (const ptr of namePtrs) Module._free(ptr)
        Module._free(bindingPtr)
      }
      if (deviceName === 'webgpu' && Module.ccall) {
        return Module.ccall(
          'poly_instance_train_step',
          'number',
          ['number', 'number', 'number', 'number'],
          [instPtr, bindingPtr, n, lossPtr],
          { async: true }
        ).then(readLoss).finally(cleanup)
      }
      const rc = Module._poly_instance_train_step(instPtr, bindingPtr, n, lossPtr)
      const loss = readLoss(rc)
      cleanup()
      return loss
    }
  }

  return {
    Module,
    deviceName,
    deviceId,
    deviceIds: DEVICE_IDS,
    dtypeIds: DTYPE_IDS,
    ffi,
    ctx,
    ops,
    instance,
    int64: BigInt,
    readShape: readOutShape,
    canRunOp,
    get caps() {
      return {
        simd: true,
        f16: deviceName !== 'webgpu' || webgpuSupportsF16,
        f64: deviceName !== 'webgpu',
        core: 'wasm',
        device: resolvedDeviceName
      }
    },
    // Heap/marshalling helpers for frontend adapters and host buffer views.
    heap32,
    heapU8,
    heapF32,
    heapF64,
    allocBytes,
    allocString,
    readCString,
    _scratchLenPtr,
    registerHostBuffer(bufferKey, value) {
      Module.__polygradHostBuffers.set(String(bufferKey), value)
    },
    unregisterHostBuffer(bufferKey) {
      Module.__polygradHostBuffers.delete(String(bufferKey))
    },
    destroy() {
      ffi.poly_ctx_destroy(ctx)
      if (_scratchPtrArrayPtr) {
        Module._free(_scratchPtrArrayPtr)
        _scratchPtrArrayPtr = 0
        _scratchPtrArrayCap = 0
      }
      Module._free(_scratchLenPtr)
      Module._free(_scratchNumelPtr)
      Module._free(_scratchAxisPtr)
      Module._free(_scratchOutShapePtr)
      Module._free(_scratchOutNdimPtr)
    }
  }
}

module.exports = { createWasmCoreFromModule }
