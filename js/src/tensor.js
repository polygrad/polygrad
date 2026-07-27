/**
 * tensor.js -- Runtime-bound Tensor class factory for polygrad.
 *
 * All graph construction ops are synchronous (use _runtime._core).
 * Realize/data-extraction are synchronous for native/CPU/WASM and have
 * explicit Async variants for WebGPU or async WASM loading paths.
 *
 * The core ffi table normalizes int64 marshalling:
 *   - Functions taking shape/axis arrays accept plain JS number[]
 *   - JS currently reads concrete max_shape dims from UOps
 */

'use strict'

const { UOp } = require('./uop/ops')
const { PolyAsyncRequired } = require('./errors')

/**
 * Strong frontend owner registry. Maps C-side PolyBuffer* address value ->
 * TypedArray that backs that imported HOST residency. This is intentionally
 * keyed by the retired residency object, not by BUFFER UOp identity.
 * The C core calls frontend_buffer_release(buffer) when the HOST PolyBuffer
 * is retired, and only then do we drop the JS owner entry.
 */
const hostBuffers = new Map()

// JS-side materialization/storage type by dtype name. The numeric dtype ids
// come from the C core; this map is only the frontend's ArrayBuffer view choice.
const TA_BY_DTYPE = {
  bool: Uint8Array,
  int8: Int8Array,
  uint8: Uint8Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  float16: Uint16Array,
  bfloat16: Uint16Array,
  float32: Float32Array,
  float64: Float64Array
}

// --- Utility helpers ---

function flattenArray(arr, dtype) {
  const ArrayType = TA_BY_DTYPE[dtype] || Float32Array
  if (typeof arr === 'number') {
    const v = dtype === 'bool' ? (arr ? 1 : 0) : arr
    return { data: new ArrayType([v]), shape: [1] }
  }
  if (arr instanceof Float32Array || arr instanceof Float64Array) {
    return { data: new ArrayType(arr), shape: [arr.length] }
  }
  if (!Array.isArray(arr)) {
    throw new Error('Expected number, array, Float32Array, or Float64Array')
  }

  const shape = []
  let cur = arr
  while (Array.isArray(cur)) {
    shape.push(cur.length)
    cur = cur[0]
  }

  const flat = []
  const recurse = (a) => {
    if (Array.isArray(a)) {
      for (const el of a) recurse(el)
    } else {
      flat.push(dtype === 'bool' ? (a ? 1 : 0) : a)
    }
  }
  recurse(arr)

  return { data: new ArrayType(flat), shape }
}

function arraysEqual(a, b) {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}

function normalizeExpandShape(currentShape, requestedShape) {
  const ndim = Math.max(currentShape.length, requestedShape.length)
  const cur = new Array(ndim - currentShape.length).fill(1).concat(currentShape)
  const req = new Array(ndim - requestedShape.length).fill(1).concat(requestedShape)
  return req.map((r, i) => (r === -1 || r === null || r === undefined) ? cur[i] : r)
}

function normalizeBufferKey(key) {
  return typeof key === 'bigint' ? key.toString() : String(key)
}

function hasRuntimePtr(ptr) {
  return ptr !== null && ptr !== undefined && ptr !== 0 && ptr !== 0n
}

function decodeFloat16Bits(h) {
  const s = (h & 0x8000) ? -1 : 1
  const e = (h >> 10) & 0x1F
  const f = h & 0x03FF
  if (e === 0) return f ? s * Math.pow(2, -14) * (f / 1024) : s * 0
  if (e === 0x1F) return f ? NaN : s * Infinity
  return s * Math.pow(2, e - 15) * (1 + f / 1024)
}

function decodeFloat16Array(bits) {
  const out = new Float32Array(bits.length)
  for (let i = 0; i < bits.length; i++) out[i] = decodeFloat16Bits(bits[i])
  return out
}

function decodeBfloat16Array(bits) {
  const out = new Float32Array(bits.length)
  const tmp = new Uint32Array(1)
  const f32 = new Float32Array(tmp.buffer)
  for (let i = 0; i < bits.length; i++) {
    tmp[0] = bits[i] << 16
    out[i] = f32[0]
  }
  return out
}

function _buildNested(data, shape, dim, offset) {
  if (dim === shape.length - 1) {
    const arr = []
    for (let i = 0; i < shape[dim]; i++) arr.push(data[offset + i])
    return { value: arr, consumed: shape[dim] }
  }
  const arr = []
  let pos = offset
  for (let i = 0; i < shape[dim]; i++) {
    const r = _buildNested(data, shape, dim + 1, pos)
    arr.push(r.value)
    pos += r.consumed
  }
  return { value: arr, consumed: pos - offset }
}

// --- Class factory ---

function _broadcastShapes(a, b) {
  if (!a.length) return [...b]
  if (!b.length) return [...a]
  const ndim = Math.max(a.length, b.length)
  const pa = new Array(ndim - a.length).fill(1).concat(a)
  const pb = new Array(ndim - b.length).fill(1).concat(b)
  const result = []
  for (let i = 0; i < ndim; i++) {
    if (pa[i] === pb[i]) result.push(pa[i])
    else if (pa[i] === 1) result.push(pb[i])
    else if (pb[i] === 1) result.push(pa[i])
    else throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`)
  }
  return result
}

function makeTuple(value, count) {
  if (typeof value === 'number') return new Array(count).fill(Number(value))
  return Array.from(value).map(x => Number(x))
}

function resolvePoolPads(padding, dims) {
  if (typeof padding === 'number') return new Array(2 * dims).fill(Number(padding))
  const p = Array.from(padding).map(x => Number(x))
  if (p.length === 2 * dims) return p
  if (p.length === dims) {
    const out = []
    for (let i = p.length - 1; i >= 0; i--) out.push(p[i], p[i])
    return out
  }
  throw new Error(`padding must be a number or an array of length ${dims} or ${2 * dims}`)
}

function normalizePadArg(padding, ndim) {
  const p = Array.from(padding)
  if (!p.some(x => Array.isArray(x) || x === null || x === undefined)) {
    if (p.length % 2 !== 0) throw new Error('Flat padding must have even number of pads')
    const grouped = []
    for (let i = p.length - 2; i >= 0; i -= 2) grouped.push([Number(p[i]), Number(p[i + 1])])
    while (grouped.length < ndim) grouped.unshift([0, 0])
    if (grouped.length !== ndim) throw new Error(`padding length is improper, ndim=${ndim}`)
    return grouped
  }
  if (p.length !== ndim) throw new Error(`padding length is improper, ndim=${ndim}`)
  return p.map(x => x == null ? [0, 0] : [Number(x[0]), Number(x[1])])
}

function product(xs) {
  let out = 1
  for (const x of xs) out *= Number(x)
  return out
}

function isIntegerDtype(dtype) {
  return ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'].includes(dtype)
}

function createBoundTensorClass(runtime) {
  const _runtime = runtime
  const DTYPE_ID = _runtime._core.dtypeIds
  if (!DTYPE_ID) throw new Error('polygrad: core missing dtypeIds')
  const DTYPE_NAME_BY_ID = new Map(
    Object.entries(DTYPE_ID).map(([name, id]) => [Number(id), name])
  )
  let _seed = 0
  const ffi = _runtime._core.ffi
  const ops = _runtime._core.ops || {}
  const POLY_TENSOR_VALUE = 0
  const POLY_TENSOR_PLACE = 1
  const normalizeDevice = (device) => String(device || _runtime.device || 'cpu').toLowerCase()
  const deviceId = (device) => ffi.poly_device_by_name(normalizeDevice(device))
  const tensorCreate = (ctx, uop, role, device) =>
    ffi.poly_tensor_create(ctx, uop, role, deviceId(device))
  const tensorCreateWithRoots = (ctx, logical, physical, role, device) =>
    ffi.poly_tensor_create_with_roots(
      ctx, rawUop(logical), physical ? rawUop(physical) : null, role, deviceId(device)
    )
  const tensorUop = (tensor) => ffi.poly_tensor_uop(tensor)
  const tensorUopPhysical = (tensor) => ffi.poly_tensor_uop_physical(tensor)
  const tensorUopLogical = (tensor) => ffi.poly_tensor_uop_logical(tensor)
  const tensorDevice = (tensor) => ffi.poly_tensor_device(tensor)
  const uopKey = (uop) => {
    if (!uop) return '0'
    if (ffi.poly_uop_key) return String(ffi.poly_uop_key(uop))
    return String(uop)
  }
  const rawUop = (uop) => uop instanceof UOp ? uop.raw : uop
  const dtypeNameForUop = (ctx, uop, fallback) => {
    const raw = rawUop(uop)
    if (!raw) return fallback || 'float32'
    if (!ffi.poly_uop_dtype_id) {
      if (fallback) return fallback
      throw new Error('poly_uop_dtype_id is required for JS Tensor dtype inference')
    }
    const id = Number(ffi.poly_uop_dtype_id(ctx, raw))
    const name = DTYPE_NAME_BY_ID.get(id)
    if (name) return name
    if (fallback) return fallback
    throw new Error(`unknown Polygrad dtype id ${id}`)
  }
  const gradManyRaw = (ctx, root, initialGrad, wrts) => {
    const clean = wrts.map(rawUop).filter(Boolean)
    if (!clean.length) return { grads: [], present: [] }
    const out = ffi.poly_grad_many(ctx, rawUop(root) || 0, rawUop(initialGrad) || 0, clean)
    if (!out || !Array.isArray(out.grads) || !Array.isArray(out.present) ||
        out.grads.length !== clean.length || out.present.length !== clean.length) {
      throw new Error('poly_grad_many failed')
    }
    return out
  }
  const customKernelGradRecords = []
  const customGradRecordsFor = (ctx, root) => {
    const rootRaw = rawUop(root)
    if (!rootRaw || !ffi.poly_uop_reachable) return []
    const out = []
    for (const rec of customKernelGradRecords) {
      if (rec.ctx !== ctx) continue
      const active = []
      const activeSeen = new Set()
      for (let i = 0; i < rec.afters.length; i++) {
        const logicalAfter = rawUop(rec.afters[i])
        let activeAfter = null
        if (ffi.poly_uop_reachable(ctx, rootRaw, logicalAfter)) {
          activeAfter = logicalAfter
        } else {
          const physicalAfter = rec.physicalAfters && rawUop(rec.physicalAfters[i])
          if (physicalAfter && ffi.poly_uop_reachable(ctx, rootRaw, physicalAfter)) {
            activeAfter = physicalAfter
          }
        }
        const activeKey = uopKey(activeAfter)
        // Hash-consed duplicate sources share one AFTER. tinygrad uses the
        // first matching CALL slot and passes its accumulated gradient once.
        if (activeAfter && !activeSeen.has(activeKey)) {
          const afterSrc = new UOp(ctx, ffi, activeAfter).src
          if (afterSrc.length !== 2 || afterSrc[1].op !== ops.CALL) {
            throw new Error('customKernel active output is not AFTER(data, CALL)')
          }
          const callSrc = afterSrc[1].src
          if (callSrc.length !== rec.args.length + 1) {
            throw new Error('customKernel active CALL argument count changed')
          }
          activeSeen.add(activeKey)
          // Pinned tinygrad mixin/gradient.py:90-91 and :25-31 passes the
          // exact reachable CALL and its matching argument slot to
          // call_gradient. Select both from this active AFTER representation.
          active.push({ after: activeAfter, arg: callSrc[i + 1].raw, call: afterSrc[1].raw })
        }
      }
      if (active.length) out.push({ rec, active })
    }
    return out
  }
  const gradResultRaw = (grad) => {
    if (!grad) return null
    if (grad instanceof Tensor) return grad._graphUopRaw()
    if (grad instanceof UOp) return grad.raw
    return rawUop(grad)
  }
  const accumulateGradRaw = (ctx, current, next) => {
    const cur = rawUop(current)
    const nxt = rawUop(next)
    if (!cur) return nxt
    if (!nxt) return cur
    return ffi.poly_alu2(ctx, ops.ADD, cur, nxt)
  }
  const usesAsyncHostBridge = () => {
    const caps = _runtime && _runtime._core && _runtime._core.caps
    return caps && caps.device === 'webgpu'
  }
  const requireSyncHostBridge = (method, asyncMethod) => {
    if (usesAsyncHostBridge()) throw new PolyAsyncRequired(method, asyncMethod)
  }
  const liveTensors = []
  const useWeakRef = typeof WeakRef !== 'undefined'
  const registerTensor = (tensor) => {
    liveTensors.push(useWeakRef ? new WeakRef(tensor) : tensor)
  }
  const liveTensorSnapshot = () => {
    const kept = []
    const out = []
    for (const ref of liveTensors) {
      const t = useWeakRef ? ref.deref() : ref
      if (!t) continue
      kept.push(ref)
      out.push(t)
    }
    liveTensors.length = 0
    liveTensors.push(...kept)
    return out
  }
  const canRegisterFrontendRelease =
    ffi.poly_ctx_set_frontend_buffer_release || ffi.poly_set_frontend_buffer_release
  if (canRegisterFrontendRelease && !_runtime._core.__frontendBufferReleaseRegistered) {
    const releaseCore = _runtime._core
    const releaseFrontendBuffer = (bufferKey) => {
      const key = normalizeBufferKey(bufferKey)
      hostBuffers.delete(key)
      if (releaseCore.unregisterHostBuffer) {
        releaseCore.unregisterHostBuffer(key)
      }
    }
    if (ffi.poly_ctx_set_frontend_buffer_release) {
      ffi.poly_ctx_set_frontend_buffer_release(releaseCore.ctx, releaseFrontendBuffer)
    } else {
      ffi.poly_set_frontend_buffer_release(releaseFrontendBuffer)
    }
    releaseCore.__frontendBufferReleaseRegistered = true
  }

  class Tensor {
    /**
     * @param {number|number[]|Float32Array|Float64Array|null} data
     * @param {object} opts
     */
    constructor(data, opts) {
      if (!opts) opts = {}
      if (data instanceof UOp && !opts._uop) {
        opts = Object.assign({}, opts, { _uop: data, _ctx: opts._ctx || data.ctx })
        data = null
      }
      const core = _runtime._core
      this._rt = _runtime
      this._ctx = opts._ctx || core.ctx
      this._requiresGrad = Boolean(opts.requiresGrad)
      this._grad = null
      this._isParam = Boolean(opts.isParam || opts.is_param || opts._isParam)
      this._device = normalizeDevice(opts._device || opts.device || _runtime.device || 'cpu')
      this._tensor = opts._tensor || null
      let currentUop = null
      let importedFromHost = false
      let importedTensorFromHost = false
      const optUop = opts._uop || (this._tensor ? tensorUop(this._tensor) : null)

      if (optUop) {
        // Internal construction from ops -- shape is on the UOp.
        // Accept either a UOp wrapper instance or a raw handle.
        const raw = rawUop(optUop)
        currentUop = optUop instanceof UOp ? optUop : new UOp(this._ctx, core.ffi, raw)
        this._data = opts._data || null
        this._dtype = dtypeNameForUop(this._ctx, raw, opts._dtype)
      } else {
        const scalarData = typeof data === 'number' || typeof data === 'boolean'
        // User construction from data. Resolve dtype and flatten to a single
        // TypedArray. Then call UOp.fromHost (one FFI call) which creates
        // the BUFFER UOp, registers a PolyBuffer wrapping the TypedArray's
        // bytes in ctx->buffers, and wraps in RESHAPE when ndim > 1.
        // JS also keeps a strong owner entry keyed by the C-side PolyBuffer*
        // address value, not by the BUFFER UOp.
        let dt, flat, shape
        if (scalarData) {
          dt = opts.dtype || (
            typeof data === 'boolean' ? 'bool' : Number.isInteger(data) ? 'int32' : 'float32'
          )
          const dtypeId = DTYPE_ID[dt]
          if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dt}`)
          const targetDeviceId = deviceId(this._device)
          if (dt === 'bool' || isIntegerDtype(dt)) {
            const value = dt === 'bool' ? (data ? 1 : 0) : Math.trunc(Number(data))
            this._tensor = ffi.poly_tensor_const_int_by_id(
              this._ctx, value, dtypeId, targetDeviceId
            )
          } else {
            this._tensor = ffi.poly_tensor_const_float_by_id(
              this._ctx, Number(data), dtypeId, targetDeviceId
            )
          }
          if (!this._tensor) throw new Error('C-owned scalar Tensor construction failed')
          const physical = tensorUopPhysical(this._tensor)
          if (!physical) throw new Error('scalar Tensor has no physical root')
          currentUop = new UOp(this._ctx, core.ffi, physical)
          this._dtype = dt
          this._data = null
        } else if (data instanceof Float64Array) {
          importedFromHost = true
          dt = 'float64'; flat = new Float64Array(data); shape = [data.length]
        } else if (data instanceof Float32Array && (!opts.dtype || opts.dtype === 'float32')) {
          importedFromHost = true
          dt = 'float32'; flat = new Float32Array(data); shape = [data.length]
        } else if (ArrayBuffer.isView(data) && !(data instanceof DataView)) {
          importedFromHost = true
          dt = (opts && opts.dtype) || 'float32'
          const ArrayType = TA_BY_DTYPE[dt] || Float32Array
          flat = new ArrayType(data)
          shape = [data.length]
        } else {
          importedFromHost = true
          dt = (opts && opts.dtype) || 'float32'
          const r = flattenArray(data, dt)
          flat = r.data; shape = r.shape
        }
        if (!scalarData) {
          this._dtype = dt
          this._data = flat
          const dtypeId = DTYPE_ID[dt]
          if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dt}`)
          // Preserve one-dimensional zero shapes. Without the explicit [0],
          // poly_buffer_from_host cannot distinguish an empty vector from an
          // unspecified/scalar host buffer and applies its scalar numel fallback.
          const dims = shape.length ? shape : null
          if (dt !== 'bfloat16' && ffi.poly_tensor_from_host_by_id) {
            this._tensor = ffi.poly_tensor_from_host_by_id(
              this._ctx, flat, flat.byteLength, dtypeId, dims, dims ? dims.length : 0
            )
            if (!this._tensor) throw new Error('poly_tensor_from_host_by_id failed')
            const physical = tensorUopPhysical(this._tensor)
            if (!physical) throw new Error('host Tensor source has no physical root')
            currentUop = new UOp(this._ctx, core.ffi, physical)
            importedTensorFromHost = true
          } else {
            currentUop = UOp.fromHost(this._ctx, core.ffi, flat, dtypeId, dims)
          }
          const buffer = currentUop.buffer ? currentUop.buffer.raw : null
          const needsFrontendHostOwner =
            Boolean(buffer && core.ffi.poly_buffer_get_key) &&
            (!core.caps || core.caps.core !== 'wasm' || core.caps.device === 'webgpu')
          if (needsFrontendHostOwner && buffer && core.ffi.poly_buffer_get_key) {
            const bufferKey = core.ffi.poly_buffer_get_key(this._ctx, buffer)
            if (bufferKey) {
              const key = normalizeBufferKey(bufferKey)
              hostBuffers.set(key, flat)
              if (core.registerHostBuffer) core.registerHostBuffer(key, flat)
            }
          }
        }
      }
      if (importedTensorFromHost) {
        const targetDeviceId = deviceId(this._device)
        const sourceDeviceId = tensorDevice(this._tensor)
        if (targetDeviceId !== sourceDeviceId) {
          this._tensor = ffi.poly_tensor_to_device(this._ctx, this._tensor, targetDeviceId)
          if (!this._tensor) throw new Error(`poly_tensor_to_device failed for ${this._device}`)
        }
      } else if (!this._tensor && currentUop) {
        const sourceDevice = 'cpu'
        const targetDeviceId = deviceId(this._device)
        const sourceDeviceId = deviceId(sourceDevice)
        const targetUsesHostStorage = Boolean(
          ffi.poly_device_is_host_addressable(targetDeviceId)
        )
        if (importedFromHost && targetDeviceId !== sourceDeviceId && !targetUsesHostStorage) {
          // Match pinned tinygrad's direct PYTHON -> target-device creation
          // COPY. Explicit CPU construction followed by .to(device) still
          // retains its separate CPU COPY boundary.
          const source = ffi.poly_tensor_create(
            this._ctx, currentUop.raw, POLY_TENSOR_VALUE, deviceId('host')
          )
          this._tensor = ffi.poly_tensor_to_device(this._ctx, source, targetDeviceId)
          if (!this._tensor) throw new Error(`poly_tensor_to_device failed for ${this._device}`)
        } else {
          this._tensor = this._coreCreate(currentUop.raw, POLY_TENSOR_VALUE, this._device)
        }
      }
      this._syncCoreRequiresGrad()
      registerTensor(this)
    }

    _coreCreate(uop, role, device) {
      return tensorCreate(this._ctx, uop, role, device || this._device)
    }

    _coreCreateWithRoots(logical, physical, role, device) {
      return tensorCreateWithRoots(this._ctx, logical, physical, role, device || this._device)
    }

    _syncCoreRequiresGrad() {
      if (this._tensor && ffi.poly_tensor_set_requires_grad) {
        ffi.poly_tensor_set_requires_grad(this._tensor, Boolean(this._requiresGrad))
      }
    }

    _currentUopRaw() { return this._tensor ? tensorUop(this._tensor) : null }
    _logicalUopRaw() { return this._tensor ? tensorUopLogical(this._tensor) : null }
    _physicalUopRaw() { return this._tensor ? tensorUopPhysical(this._tensor) : null }
    _graphUopRaw() {
      const logical = this._logicalUopRaw()
      const physical = this._physicalUopRaw()
      if (logical && physical && ffi.poly_uop_op && ops && ffi.poly_uop_op(logical) === ops.AFTER) {
        return physical
      }
      return logical || this._currentUopRaw()
    }
    _graphBufferRaw() {
      const raw = this._graphUopRaw()
      if (!raw) return null
      if (ffi.poly_uop_get_buffer_identity) return ffi.poly_uop_get_buffer_identity(raw)
      const u = new UOp(this._ctx, this._rt._core.ffi, raw)
      return u.buffer ? u.buffer.raw : null
    }
    _coreToDevice(device) {
      if (!this._tensor) throw new Error('Tensor has no core PolyTensor')
      return ffi.poly_tensor_to_device(this._ctx, this._tensor, deviceId(device))
    }
    static _coreRealizeBatch(ctx, targets) {
      const realized = ffi.poly_realize_tensors(ctx, targets.map(t => t._tensor))
      if (!realized || realized.length !== targets.length) {
        throw new Error('poly_realize_tensors failed')
      }
      return realized
    }

    static async _coreRealizeBatchAsync(ctx, targets) {
      const fn = ffi.poly_realize_tensors_async || ffi.poly_realize_tensors
      const realized = await fn(ctx, targets.map(t => t._tensor))
      if (!realized || realized.length !== targets.length) {
        throw new Error('poly_realize_tensors failed')
      }
      return realized
    }

    get _uop() { return this._currentUopRaw() }
    get _graphUop() { return this._graphUopRaw() }
    get _buffer() {
      const u = this.uop
      return u && u.buffer ? u.buffer.raw : null
    }
    get uop() {
      const raw = this._currentUopRaw()
      return raw ? new UOp(this._ctx, this._rt._core.ffi, raw) : null
    }
    get uopLogical() {
      const raw = this._logicalUopRaw()
      return raw ? new UOp(this._ctx, this._rt._core.ffi, raw) : null
    }
    get uopPhysical() {
      const raw = this._physicalUopRaw()
      return raw ? new UOp(this._ctx, this._rt._core.ffi, raw) : null
    }

    get shape() {
      const { ffi } = this._rt._core
      if (!this._uop) return []
      return ffi.poly_uop_max_shape_dims(this._ctx, this._uop)
    }
    get dtype() { return this._dtype }
    get device() {
      if (!this._tensor) throw new Error('Tensor has no core PolyTensor')
      /* Public device is a placement label. The WASM core maps public CPU onto
       * the WASM execution backend internally, but user-facing Tensor.device
       * should stay CPU for parity with Python/tinygrad-style APIs. */
      return this._device.toUpperCase()
    }
    get ndim() { return this._rt._core.ffi.poly_uop_ndim(this._ctx, this._uop) || 0 }
    get requiresGrad() { return this._requiresGrad }
    set requiresGrad(v) {
      this._requiresGrad = Boolean(v)
      this._syncCoreRequiresGrad()
    }
    get isParam() { return this._isParam }
    set isParam(v) { this._isParam = Boolean(v) }
    get is_param() { return this._isParam }
    set is_param(v) { this._isParam = Boolean(v) }
    is_param_(isParam = true) {
      this._isParam = Boolean(isParam)
      return this
    }
    get grad() { return this._grad }
    get T() { return this.transpose() }

    numel() {
      return this.shape.reduce((a, b) => a * b, 1)
    }

    size(dim) {
      if (dim === undefined || dim === null) return [...this.shape]
      if (dim < 0) dim += this.shape.length
      return this.shape[dim]
    }

    _liveGradTargets() {
      if (!ffi.poly_uop_reachable) {
        throw new Error('poly_uop_reachable is required for tinygrad-style backward')
      }
      const root = this._currentUopRaw()
      const targets = []
      /* tinygrad discovers backward targets from all_tensors, not from a saved
       * input list. Polygrad accepts reachability through either half of its
       * logical/current alias pair, but prefers the executable current root;
       * the logical root can retain stale pre-realization RNG provenance. */
      for (const t of liveTensorSnapshot()) {
        if (!t || t._ctx !== this._ctx || !t._tensor || !t._requiresGrad) continue
        const target = t._graphUopRaw()
        const current = t._currentUopRaw()
        let gradRoot = null
        if (current && ffi.poly_uop_reachable(this._ctx, root, current)) {
          gradRoot = current
        } else if (target && uopKey(current) !== uopKey(target) &&
                   ffi.poly_uop_reachable(this._ctx, root, target)) {
          gradRoot = target
        }
        if (gradRoot) {
          targets.push({ tensor: t, root: gradRoot })
        }
      }
      return targets
    }

    _realizeWith(realizeBatch, ...lst) {
      // Triggers the computation needed to create these Tensor(s). The core
      // decides whether a buffer identity is already current on the requested
      // device or still needs an allocation/copy.
      const { ffi, ctx } = this._rt._core
      const tensors = [this, ...lst]
      if (!ffi.poly_realize_tensors) throw new Error('poly_realize_tensors is required')
      const targets = []
      const seen = new Set()
      for (const t of tensors) {
        const key = `${uopKey(t._currentUopRaw())}:${tensorDevice(t._tensor)}`
        if (seen.has(key)) continue
        seen.add(key)
        targets.push(t)
      }
      if (!targets.length) return this
      realizeBatch.call(Tensor, ctx, targets)
      return this
    }

    realize(...lst) {
      requireSyncHostBridge('realize()', 'realizeAsync()')
      return this._realizeWith(Tensor._coreRealizeBatch, ...lst)
    }

    async realizeAsync(...lst) {
      const { ffi, ctx } = this._rt._core
      const tensors = [this, ...lst]
      if (!ffi.poly_realize_tensors_async && !ffi.poly_realize_tensors) {
        throw new Error('poly_realize_tensors is required')
      }
      const targets = []
      const seen = new Set()
      for (const t of tensors) {
        const key = `${uopKey(t._currentUopRaw())}:${tensorDevice(t._tensor)}`
        if (seen.has(key)) continue
        seen.add(key)
        targets.push(t)
      }
      if (!targets.length) return this
      await Tensor._coreRealizeBatchAsync(ctx, targets)
      return this
    }

    _readBufferBytesWith(readBuffer) {
      const { ffi, ctx } = this._rt._core
      const numel = this.numel()
      const AT = TA_BY_DTYPE[this._dtype] || Float32Array
      const itemsize = AT.BYTES_PER_ELEMENT
      const bufRaw = this.uop && this.uop.buffer ? this.uop.buffer.raw : null
      if (!bufRaw) throw new Error('toArray: tensor has no buffer identity')
      let raw
      const bufferKey = ffi.poly_buffer_get_key ? ffi.poly_buffer_get_key(ctx, bufRaw) : 0
      const normKey = bufferKey ? normalizeBufferKey(bufferKey) : null
      const ptr = ffi.poly_buffer_get_ptr ? ffi.poly_buffer_get_ptr(ctx, bufRaw) : null
      if (!hasRuntimePtr(ptr) && normKey && hostBuffers.has(normKey)) {
        /* JS-owned HOST bytes are only authoritative when the core has no
         * current runtime pointer, which happens for zero-copy browser imports.
         * Once WASM/native kernels have a pointer, read through C so in-place
         * STORE/assign updates are visible instead of returning stale TypedArray
         * contents captured at tensor construction. */
        raw = hostBuffers.get(normKey)
      } else {
        const nbytes = numel * itemsize
        raw = readBuffer(ctx, bufRaw, nbytes)
      }
      if (this._dtype === 'float16') {
        const bits = raw instanceof Uint16Array ? raw :
          new Uint16Array(raw.buffer, raw.byteOffset, numel)
        return decodeFloat16Array(bits)
      }
      if (this._dtype === 'bfloat16') {
        const bits = raw instanceof Uint16Array ? raw :
          new Uint16Array(raw.buffer, raw.byteOffset, numel)
        return decodeBfloat16Array(bits)
      }
      if (raw instanceof AT) return raw
      return new AT(raw.buffer, raw.byteOffset, numel)
    }

    _readBufferBytes() {
      requireSyncHostBridge('toArray()', 'toArrayAsync()')
      return this._readBufferBytesWith(ffi.poly_buffer_read)
    }

    async _readBufferBytesAsync() {
      const { ffi, ctx } = this._rt._core
      const numel = this.numel()
      const AT = TA_BY_DTYPE[this._dtype] || Float32Array
      const itemsize = AT.BYTES_PER_ELEMENT
      const bufRaw = this.uop && this.uop.buffer ? this.uop.buffer.raw : null
      if (!bufRaw) throw new Error('toArray: tensor has no buffer identity')
      let raw
      const bufferKey = ffi.poly_buffer_get_key ? ffi.poly_buffer_get_key(ctx, bufRaw) : 0
      const normKey = bufferKey ? normalizeBufferKey(bufferKey) : null
      const ptr = ffi.poly_buffer_get_ptr ? ffi.poly_buffer_get_ptr(ctx, bufRaw) : null
      if (!hasRuntimePtr(ptr) && normKey && hostBuffers.has(normKey)) {
        raw = hostBuffers.get(normKey)
      } else {
        const nbytes = numel * itemsize
        const readBuffer = ffi.poly_buffer_read_async || ffi.poly_buffer_read
        raw = await readBuffer(ctx, bufRaw, nbytes)
      }
      if (this._dtype === 'float16') {
        const bits = raw instanceof Uint16Array ? raw :
          new Uint16Array(raw.buffer, raw.byteOffset, numel)
        return decodeFloat16Array(bits)
      }
      if (this._dtype === 'bfloat16') {
        const bits = raw instanceof Uint16Array ? raw :
          new Uint16Array(raw.buffer, raw.byteOffset, numel)
        return decodeBfloat16Array(bits)
      }
      if (raw instanceof AT) return raw
      return new AT(raw.buffer, raw.byteOffset, numel)
    }

    toArray() {
      requireSyncHostBridge('toArray()', 'toArrayAsync()')
      const numel = this.numel()
      if (numel === 0) {
        const AT = TA_BY_DTYPE[this._dtype] || Float32Array
        return new AT(0)
      }
      let t = this
      if (this._dtype === 'float16' || this._dtype === 'bfloat16') t = t.cast('float32')
      if (!t.uop.hasBufferIdentity()) t = t.contiguous()
      t.realize()
      return t._readBufferBytes()
    }

    async toArrayAsync() {
      const numel = this.numel()
      if (numel === 0) {
        const AT = TA_BY_DTYPE[this._dtype] || Float32Array
        return new AT(0)
      }
      let t = this
      if (this._dtype === 'float16' || this._dtype === 'bfloat16') t = t.cast('float32')
      if (!t.uop.hasBufferIdentity()) t = t.contiguous()
      await t.realizeAsync()
      return await t._readBufferBytesAsync()
    }

    toTypedArray() {
      requireSyncHostBridge('toTypedArray()', 'toTypedArrayAsync()')
      return this.toArray()
    }

    async toTypedArrayAsync() {
      return await this.toArrayAsync()
    }

    static toTypedArrays(...tensors) {
      if (tensors.length === 1 && Array.isArray(tensors[0])) tensors = tensors[0]
      if (!tensors.length) return []
      for (const t of tensors) {
        if (!(t instanceof Tensor)) throw new TypeError('Tensor.toTypedArrays expects Tensor arguments')
      }
      requireSyncHostBridge('Tensor.toTypedArrays()', 'Tensor.toTypedArraysAsync()')
      const prepared = tensors.map(t => {
        if (t.numel() === 0) return t
        let out = t
        if (out._dtype === 'float16' || out._dtype === 'bfloat16') out = out.cast('float32')
        if (!out.uop.hasBufferIdentity()) out = out.contiguous()
        return out
      })
      const targets = prepared.filter(t => t.numel() !== 0)
      if (targets.length) targets[0].realize(...targets.slice(1))
      const out = []
      for (const t of prepared) {
        if (t.numel() === 0) {
          const AT = TA_BY_DTYPE[t._dtype] || Float32Array
          out.push(new AT(0))
        } else {
          out.push(t._readBufferBytes())
        }
      }
      return out
    }

    static async toTypedArraysAsync(...tensors) {
      if (tensors.length === 1 && Array.isArray(tensors[0])) tensors = tensors[0]
      if (!tensors.length) return []
      for (const t of tensors) {
        if (!(t instanceof Tensor)) throw new TypeError('Tensor.toTypedArraysAsync expects Tensor arguments')
      }
      const prepared = tensors.map(t => {
        if (t.numel() === 0) return t
        let out = t
        if (out._dtype === 'float16' || out._dtype === 'bfloat16') out = out.cast('float32')
        if (!out.uop.hasBufferIdentity()) out = out.contiguous()
        return out
      })
      const targets = prepared.filter(t => t.numel() !== 0)
      if (targets.length) await targets[0].realizeAsync(...targets.slice(1))
      const out = []
      for (const t of prepared) {
        if (t.numel() === 0) {
          const AT = TA_BY_DTYPE[t._dtype] || Float32Array
          out.push(new AT(0))
        } else {
          out.push(await t._readBufferBytesAsync())
        }
      }
      return out
    }

    item() {
      requireSyncHostBridge('item()', 'itemAsync()')
      const arr = this.toArray()
      if (arr.length !== 1) {
        throw new Error(`item() requires scalar tensor, got shape [${this.shape}]`)
      }
      return arr[0]
    }

    async itemAsync() {
      const arr = await this.toArrayAsync()
      if (arr.length !== 1) {
        throw new Error(`item() requires scalar tensor, got shape [${this.shape}]`)
      }
      return arr[0]
    }

    tolist() {
      requireSyncHostBridge('tolist()', 'tolistAsync()')
      return _buildNested(this.toArray(), this.shape, 0, 0).value
    }

    async tolistAsync() {
      return _buildNested(await this.toArrayAsync(), this.shape, 0, 0).value
    }

    detach() {
      const { ffi } = this._rt._core
      const uop = ffi.poly_detach(this._ctx, this._graphUopRaw())
      if (!uop) throw new Error('poly_detach failed')
      const physical = this._physicalizeResult(uop, [this])
      return new Tensor(null, {
        _ctx: this._ctx,
        _tensor: this._coreCreateWithRoots(uop, physical, POLY_TENSOR_VALUE, this._device),
        _dtype: this._dtype,
        _device: this._device,
        requiresGrad: false
      })
    }

    async detachAsync() {
      return this.detach()
    }

    clone(device) {
      const dev = normalizeDevice(device == null ? this._device : device)
      const t = Tensor.empty(this.shape, {
        _ctx: this._ctx,
        dtype: this._dtype,
        device: dev,
        requiresGrad: this._requiresGrad
      })
      const cloned = ffi.poly_tensor_clone_into(this._ctx, t._tensor, this._tensor)
      if (!cloned) throw new Error('poly_tensor_clone_into failed')
      t._tensor = cloned
      t._isParam = this._isParam
      if (this._grad) t._grad = this._grad.clone(dev)
      return t
    }

    async cloneAsync(device) {
      return this.clone(device)
    }

    assign(x) {
      if (!(x instanceof Tensor)) x = new Tensor(x, { dtype: this._dtype, device: this._device })
      if (!arraysEqual(this.shape, x.shape)) {
        const outShape = _broadcastShapes(x.shape, this.shape)
        if (!arraysEqual(outShape, this.shape)) {
          throw new Error(`assign shape mismatch [${this.shape}] != [${x.shape}]`)
        }
        // Assignment broadcasting is represented as normal lazy EXPAND/RESHAPE
        // on the RHS before the core builds tinygrad-style AFTER(target, STORE).
        x = new Tensor(null, {
          _ctx: this._ctx,
          _uop: x._broadcastUop(this.shape),
          _dtype: x._dtype,
          _device: x._device,
          requiresGrad: x._requiresGrad
        })
      }
      if (this._device !== x._device) {
        throw new Error(`assign device mismatch ${this.device} != ${x.device}`)
      }
      if (this._dtype !== x._dtype) {
        throw new Error(`assign dtype mismatch ${this._dtype} != ${x._dtype}`)
      }
      const assigned = ffi.poly_tensor_assign(this._ctx, this._tensor, x._tensor)
      if (!assigned) throw new Error('poly_tensor_assign failed')
      this._tensor = assigned
      this._syncCoreRequiresGrad()
      this._data = null
      return this
    }

    copyFrom(data) {
      if (!ffi.poly_buffer_write || !ffi.poly_buffer_ensure_device_allocated) {
        throw new Error(
          'poly_buffer_write and poly_buffer_ensure_device_allocated are required for Tensor.copyFrom'
        )
      }
      const logicalRaw = this._logicalUopRaw() || this._currentUopRaw()
      const logical = logicalRaw ? new UOp(this._ctx, this._rt._core.ffi, logicalRaw) : null
      const buf = logical && logical.buffer ? logical.buffer.raw : null
      if (!buf) throw new Error('copyFrom requires a tensor backed by a BUFFER UOp')
      const AT = TA_BY_DTYPE[this._dtype] || Float32Array
      const expectedBytes = this.numel() * AT.BYTES_PER_ELEMENT
      let view
      if (data instanceof AT) {
        view = data
      } else if (Array.isArray(data) || typeof data === 'number') {
        view = flattenArray(data, this._dtype).data
      } else if (data instanceof ArrayBuffer) {
        view = new AT(data)
      } else if (ArrayBuffer.isView(data) && !(data instanceof DataView)) {
        if (!(data instanceof AT)) {
          throw new TypeError(`copyFrom dtype mismatch: expected ${AT.name}, got ${data.constructor.name}`)
        }
        view = data
      } else {
        throw new TypeError('copyFrom expects a TypedArray, ArrayBuffer, number, or array')
      }
      if (view.byteLength !== expectedBytes) {
        throw new Error(`copyFrom byte size mismatch ${view.byteLength} != ${expectedBytes}`)
      }
      let physicalRaw = this._physicalUopRaw()
      let writeBuf = buf
      if (physicalRaw) {
        const physical = new UOp(this._ctx, this._rt._core.ffi, physicalRaw)
        const physicalBuf = physical && physical.buffer ? physical.buffer.raw : null
        if (physicalBuf && uopKey(physicalBuf) !== uopKey(buf)) {
          writeBuf = physicalBuf
        } else if (!physicalBuf) {
          physicalRaw = null
        }
      }
      const targetDevice = deviceId(this._device)
      ffi.poly_buffer_ensure_device_allocated(this._ctx, writeBuf, targetDevice)
      ffi.poly_buffer_write(this._ctx, writeBuf, view)
      if (ffi.poly_tensor_replace_roots && logicalRaw) {
        const rc = ffi.poly_tensor_replace_roots(
          this._ctx, this._tensor, logicalRaw, physicalRaw, POLY_TENSOR_VALUE, deviceId(this._device)
        )
        if (rc !== 0) throw new Error('poly_tensor_replace_roots failed during copyFrom')
      }
      this._data = null
      return this
    }

    updateFrom(data) {
      return this.copyFrom(data)
    }

    to(device) {
      const dev = normalizeDevice(device)
      if (dev === this._device) return this
      const coreTensor = this._coreToDevice(dev)
      const t = new Tensor(null, {
        _ctx: this._ctx,
        _tensor: coreTensor,
        _data: this._data,
        _dtype: this._dtype,
        _device: dev,
        requiresGrad: this._requiresGrad
      })
      t._grad = this._grad ? this._grad.to(dev) : null
      t._isParam = this._isParam
      return t
    }

    to_(device) {
      const moved = this.to(device)
      if (moved === this) return this
      this._tensor = moved._tensor
      this._data = moved._data
      this._dtype = moved._dtype
      this._device = moved._device
      this._requiresGrad = moved._requiresGrad
      this._grad = moved._grad
      this._isParam = moved._isParam
      this._syncCoreRequiresGrad()
      return this
    }

    shard_(devices, axis) {
      if (typeof devices === 'string') return this.to_(devices)
      const devs = Array.from(devices || [])
      if (devs.length === 1) return this.to_(devs[0])
      throw new Error('Polygrad JS does not yet support multi-device shard_')
    }

    cpu() { return this.to('cpu') }
    cuda() { return this.to('cuda') }

    contiguous() {
      const { ffi } = this._rt._core
      /* tinygrad applies UOp.contiguous to its one current Tensor.uop, where
       * an existing BUFFER identity is returned unchanged. Keep Polygrad's
       * logical provenance, but run the same existing fold independently on
       * the executable current root instead of substituting into a prebuilt
       * CONTIGUOUS(logical) graph. */
      const logical = ffi.poly_contiguous(this._ctx, this._graphUopRaw())
      const current = ffi.poly_contiguous(this._ctx, this._currentUopRaw())
      if (!logical || !current) throw new Error('poly_contiguous failed')
      const physical = uopKey(current) === uopKey(logical) ? null : current
      return new Tensor(null, {
        _ctx: this._ctx,
        _tensor: this._coreCreateWithRoots(
          logical, physical, POLY_TENSOR_VALUE, this._device
        ),
        _dtype: this._dtype,
        _device: this._device,
        requiresGrad: this._requiresGrad
      })
    }

    // --- Internal helpers ---

    _physicalizeResult(logical, inputs) {
      if (!this._rt._core.ffi.poly_uop_substitute) return null
      const from = []
      const to = []
      const seen = new Set()
      for (const input of inputs || []) {
        if (!input || !(input instanceof Tensor)) continue
        const logicalRoot = input._graphUopRaw()
        const currentRoot = input._currentUopRaw()
        if (!logicalRoot || !currentRoot) continue
        const logicalKey = uopKey(logicalRoot)
        if (logicalKey === uopKey(currentRoot) || seen.has(logicalKey)) continue
        seen.add(logicalKey)
        from.push(logicalRoot)
        to.push(currentRoot)
      }
      if (!from.length) return null
      const physical = this._rt._core.ffi.poly_uop_substitute(this._ctx, logical, from, to)
      return physical && uopKey(physical) !== uopKey(logical) ? physical : null
    }

    _makeResult(uop, inputs, forcedDtype) {
      const dt = dtypeNameForUop(this._ctx, uop, forcedDtype)
      const device = this._inferDevice(inputs)
      const physical = this._physicalizeResult(uop, inputs)
      const t = new Tensor(null, {
        _ctx: this._ctx,
        _tensor: this._coreCreateWithRoots(uop, physical, POLY_TENSOR_VALUE, device),
        _dtype: dt,
        _device: device
      })
      for (let i = 0; i < inputs.length; i++) {
        if (inputs[i]._requiresGrad) { t.requiresGrad = true; break }
      }
      return t
    }

    _makeResultFromCore(coreTensor, inputs, forcedDtype) {
      if (!coreTensor) throw new Error('core Tensor operation failed')
      const current = tensorUop(coreTensor)
      if (!current) throw new Error('core Tensor operation returned no current UOp')
      const device = this._inferDevice(inputs)
      const t = new Tensor(null, {
        _ctx: this._ctx,
        _tensor: coreTensor,
        _dtype: dtypeNameForUop(this._ctx, current, forcedDtype || this._dtype),
        _device: device
      })
      for (let i = 0; i < inputs.length; i++) {
        if (inputs[i]._requiresGrad) { t.requiresGrad = true; break }
      }
      return t
    }

    _inferDevice(inputs) {
      const devices = new Set()
      for (const input of inputs) {
        if (input && input._device) devices.add(input._device)
      }
      if (devices.size === 0) return this._device
      if (devices.size > 1) {
        throw new Error(`Mixed devices are not supported: ${Array.from(devices).sort().join(', ')}`)
      }
      return Array.from(devices)[0]
    }

    customKernel(...args) {
      let fxn = null
      let gradFxn = null
      if (args.length && typeof args[args.length - 1] === 'function') {
        fxn = args.pop()
      } else if (args.length && args[args.length - 1] && typeof args[args.length - 1] === 'object'
        && typeof args[args.length - 1].fxn === 'function') {
        const opts = args.pop()
        fxn = opts.fxn
        gradFxn = opts.gradFxn || opts.grad_fxn || null
      }
      if (!fxn) throw new TypeError('customKernel requires a kernel function')
      const srcs = [this, ...args]
      for (const t of srcs) {
        if (!(t instanceof Tensor)) throw new TypeError('customKernel expects Tensor arguments')
        if (t._ctx !== this._ctx) throw new Error('customKernel tensors must share a context')
      }
      const contig = srcs.map(t => {
        const graph = t._graphUopRaw()
        return graph && ffi.poly_uop_op && ffi.poly_uop_op(graph) === ops.AFTER ? t : t.contiguous()
      })
      const placeholders = contig.map((t, i) => UOp.placeholderLike(new UOp(t._ctx, ffi, t._graphUopRaw()), i))
      const body = fxn(...placeholders)
      if (!(body instanceof UOp)) throw new TypeError('customKernel function must return a UOp SINK body')
      const call = body.call(...contig.map(t => new UOp(t._ctx, ffi, t._graphUopRaw())))
      const afters = []
      const physicalAfters = []
      const outs = contig.map(t => {
        const logical = new UOp(t._ctx, ffi, t._graphUopRaw()).after(call)
        const physical = t._physicalizeResult(logical.raw, contig)
        afters.push(logical)
        physicalAfters.push(physical)
        return new Tensor(null, {
          _ctx: t._ctx,
          _tensor: tensorCreateWithRoots(
            t._ctx, logical.raw, physical, POLY_TENSOR_VALUE, t._device
          ),
          _dtype: t._dtype,
          _device: t._device,
          requiresGrad: t._requiresGrad
        })
      })
      customKernelGradRecords.push({
        ctx: this._ctx,
        call,
        args: contig.map(t => new UOp(t._ctx, ffi, t._graphUopRaw())),
        afters,
        physicalAfters,
        gradFxn
      })
      return outs
    }

    custom_kernel(...args) { return this.customKernel(...args) }

    _ensureTensor(other) {
      if (other instanceof Tensor) return other
      if (typeof other === 'number') {
        const { ffi } = this._rt._core
        const c = this._dtype === 'float64'
          ? ffi.poly_const_double(this._ctx, other)
          : ffi.poly_const_float(this._ctx, other)
        return new Tensor(null, { _ctx: this._ctx, _uop: c,
                                  _dtype: this._dtype, _device: this._device })
      }
      throw new TypeError(`Cannot convert ${typeof other} to Tensor`)
    }

    _broadcastShape(otherShape) {
      const a = this.shape
      const b = otherShape
      if (!a.length) return [...b]
      if (!b.length) return [...a]
      const ndim = Math.max(a.length, b.length)
      const pa = new Array(ndim - a.length).fill(1).concat(a)
      const pb = new Array(ndim - b.length).fill(1).concat(b)
      const result = []
      for (let i = 0; i < ndim; i++) {
        if (pa[i] === pb[i]) result.push(pa[i])
        else if (pa[i] === 1) result.push(pb[i])
        else if (pb[i] === 1) result.push(pa[i])
        else throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`)
      }
      return result
    }

    _broadcastUop(targetShape) {
      if (arraysEqual(this.shape, targetShape)) return this._graphUopRaw()
      const { ffi } = this._rt._core
      let uop = this._graphUopRaw()
      let curShape = [...this.shape]
      const targetNd = targetShape.length
      if (curShape.length < targetNd) {
        curShape = new Array(targetNd - curShape.length).fill(1).concat(curShape)
        uop = ffi.poly_reshape(this._ctx, uop, curShape, curShape.length)
      }
      if (!arraysEqual(curShape, targetShape)) {
        uop = ffi.poly_expand(this._ctx, uop, targetShape, targetShape.length)
      }
      return uop
    }

    _broadcastTensor(targetShape) {
      if (arraysEqual(this.shape, targetShape)) return this
      if (this.shape.length > targetShape.length) {
        throw new Error(
          `cannot broadcast tensor to fewer dimensions. shape=[${this.shape}] ` +
          `to newShape=[${targetShape}]`
        )
      }
      const aligned = new Array(targetShape.length - this.shape.length)
        .fill(1).concat(this.shape)
      for (let i = 0; i < aligned.length; i++) {
        if (aligned[i] !== targetShape[i] && aligned[i] !== 1) {
          throw new Error(`cannot broadcast [${this.shape}] to newShape=[${targetShape}]`)
        }
      }
      const reshaped = this.reshape(aligned)
      const expanded = reshaped.expand(targetShape)
      return arraysEqual(expanded.shape, reshaped.shape) ? reshaped : expanded
    }

    _binop(other, opName) {
      const { ffi, ops } = this._rt._core
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const x = this._broadcastTensor(outShape)
      const y = other._broadcastTensor(outShape)
      const core = ffi.poly_tensor_alu2(this._ctx, ops[opName], x._tensor, y._tensor)
      return this._makeResultFromCore(core, [x, y])
    }

    // --- Element-wise arithmetic ---

    add(other) { return this._binop(other, 'ADD') }
    sub(other) { return this._binop(other, 'SUB') }
    mul(other) { return this._binop(other, 'MUL') }
    div(other) { return this._binop(other, 'FDIV') }
    pow(other) { return this._binop(other, 'POW') }
    lt(other) { return this._binop(other, 'CMPLT') }

    neg() {
      const { ffi, ops } = this._rt._core
      const core = ffi.poly_tensor_alu1(this._ctx, ops.NEG, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    // --- Comparisons (C core) ---

    eq(other) {
      const { ffi } = this._rt._core
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const uop = ffi.poly_eq(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      return this._makeResult(uop, [this, other])
    }

    ne(other) {
      const { ffi } = this._rt._core
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const uop = ffi.poly_ne(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      return this._makeResult(uop, [this, other])
    }

    gt(other) {
      const { ffi } = this._rt._core
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const uop = ffi.poly_gt(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      return this._makeResult(uop, [this, other])
    }

    ge(other) {
      const { ffi } = this._rt._core
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const uop = ffi.poly_ge(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      return this._makeResult(uop, [this, other])
    }

    le(other) {
      const { ffi } = this._rt._core
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const uop = ffi.poly_le(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      return this._makeResult(uop, [this, other])
    }

    where(x, y) {
      const { ffi } = this._rt._core
      x = this._ensureTensor(x)
      y = this._ensureTensor(y)
      let outShape = _broadcastShapes(this.shape, x.shape)
      outShape = _broadcastShapes(outShape, y.shape)
      const cUop = this._broadcastUop(outShape)
      const xUop = x._broadcastUop(outShape)
      const yUop = y._broadcastUop(outShape)
      const uop = ffi.poly_where_op(this._ctx, cUop, xUop, yUop)
      return this._makeResult(uop, [this, x, y])
    }

    maximum(other) {
      return this._binop(other, 'MAX')
    }

    minimum(other) {
      const { ffi } = this._rt._core
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const uop = ffi.poly_minimum(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      return this._makeResult(uop, [this, other])
    }

    clamp(lo, hi) {
      if (lo === undefined && hi === undefined) {
        throw new Error("at least one of 'lo' or 'hi' must not be undefined")
      }
      const { ffi } = this._rt._core
      lo = lo !== undefined ? lo : -1e38
      hi = hi !== undefined ? hi : 1e38
      const uop = ffi.poly_clamp(this._ctx, this._graphUopRaw(), lo, hi)
      return this._makeResult(uop, [this])
    }

    // --- Cast ---

    cast(dtype) {
      if (dtype === this._dtype) return this
      const id = DTYPE_ID[dtype]
      if (id === undefined) throw new Error(`unsupported cast target dtype: ${dtype}`)
      const caps = this._rt.caps || {}
      if (dtype === 'float16' && caps.f16 === false) {
        throw new Error(`float16 is not supported on ${this._rt.device} (shader-f16 unavailable)`)
      }
      if (dtype === 'float64' && caps.f64 === false) {
        throw new Error(`float64 is not supported on ${this._rt.device}`)
      }
      const { ffi } = this._rt._core
      const uop = ffi.poly_cast_by_id(this._ctx, this._graphUopRaw(), id)
      if (!uop) throw new Error(`poly_cast_by_id failed for dtype ${dtype}`)
      return this._makeResult(uop, [this], dtype)
    }

    half() { return this.cast('float16') }
    double() { return this.cast('float64') }

    // --- Triu/Tril ---

    triu(diagonal = 0) {
      const { ffi } = this._rt._core
      const uop = ffi.poly_triu(this._ctx, this._graphUopRaw(), diagonal)
      return this._makeResult(uop, [this])
    }

    tril(diagonal = 0) {
      const { ffi } = this._rt._core
      const uop = ffi.poly_tril(this._ctx, this._graphUopRaw(), diagonal)
      return this._makeResult(uop, [this])
    }

    // --- Unary math (C core composed ops) ---

    exp2() {
      const { ffi, ops } = this._rt._core
      const core = ffi.poly_tensor_alu1(this._ctx, ops.EXP2, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    log2() {
      const { ffi, ops } = this._rt._core
      const core = ffi.poly_tensor_alu1(this._ctx, ops.LOG2, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    sqrt() {
      const { ffi, ops } = this._rt._core
      const core = ffi.poly_tensor_alu1(this._ctx, ops.SQRT, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    reciprocal() {
      const { ffi, ops } = this._rt._core
      const core = ffi.poly_tensor_alu1(this._ctx, ops.RECIPROCAL, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    trunc() {
      const { ffi, ops } = this._rt._core
      const core = ffi.poly_tensor_alu1(this._ctx, ops.TRUNC, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    exp() {
      const uop = this._rt._core.ffi.poly_exp(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    log() {
      const uop = this._rt._core.ffi.poly_log(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    log1p() {
      const uop = this._rt._core.ffi.poly_log1p(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    expm1() {
      const uop = this._rt._core.ffi.poly_expm1(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    sin() {
      const uop = this._rt._core.ffi.poly_sin(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    cos() {
      const uop = this._rt._core.ffi.poly_cos(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    tan() {
      const uop = this._rt._core.ffi.poly_tan(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    sigmoid() {
      const uop = this._rt._core.ffi.poly_sigmoid(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    tanh() {
      const uop = this._rt._core.ffi.poly_tanh_act(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    abs() {
      const uop = this._rt._core.ffi.poly_abs(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    sign() {
      const uop = this._rt._core.ffi.poly_sign(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    square() {
      const uop = this._rt._core.ffi.poly_square(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    rsqrt() {
      const uop = this._rt._core.ffi.poly_rsqrt(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    ceil() {
      const uop = this._rt._core.ffi.poly_ceil(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    floor() {
      const uop = this._rt._core.ffi.poly_floor(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    round() {
      const uop = this._rt._core.ffi.poly_round_f(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    isinf() {
      const uop = this._rt._core.ffi.poly_isinf(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    isnan() {
      const uop = this._rt._core.ffi.poly_isnan(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    // --- Activations (C core composed ops) ---

    relu() {
      const uop = this._rt._core.ffi.poly_relu(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    relu6() {
      const uop = this._rt._core.ffi.poly_relu6(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    leakyRelu(negSlope) {
      if (negSlope === undefined) negSlope = 0.01
      const uop = this._rt._core.ffi.poly_leaky_relu(this._ctx, this._graphUopRaw(), negSlope)
      return this._makeResult(uop, [this])
    }

    gelu() {
      const uop = this._rt._core.ffi.poly_gelu(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    quickGelu() {
      const uop = this._rt._core.ffi.poly_quick_gelu(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    silu() {
      const uop = this._rt._core.ffi.poly_silu(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    swish() { return this.silu() }

    elu(alpha) {
      if (alpha === undefined) alpha = 1.0
      const uop = this._rt._core.ffi.poly_elu(this._ctx, this._graphUopRaw(), alpha)
      return this._makeResult(uop, [this])
    }

    softplus(beta) {
      if (beta === undefined) beta = 1.0
      const uop = this._rt._core.ffi.poly_softplus(this._ctx, this._graphUopRaw(), beta)
      return this._makeResult(uop, [this])
    }

    mish() {
      const uop = this._rt._core.ffi.poly_mish(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    hardtanh(minVal, maxVal) {
      if (minVal === undefined) minVal = -1
      if (maxVal === undefined) maxVal = 1
      const uop = this._rt._core.ffi.poly_hardtanh(this._ctx, this._graphUopRaw(), minVal, maxVal)
      return this._makeResult(uop, [this])
    }

    hardswish() {
      const uop = this._rt._core.ffi.poly_hardswish(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    hardsigmoid() {
      const uop = this._rt._core.ffi.poly_hardsigmoid(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    // --- Softmax ---

    softmax(axis) {
      if (axis === undefined) axis = -1
      // Keep softmax lazy so backward sees the same current UOp graph as
      // tinygrad; scheduling decides where kernel boundaries belong.
      const uop = this._rt._core.ffi.poly_softmax(this._ctx, this._graphUopRaw(), axis)
      if (!uop) throw new Error('poly_softmax failed')
      return this._makeResult(uop, [this])
    }

    logSoftmax(axis) {
      if (axis === undefined) axis = -1
      const uop = this._rt._core.ffi.poly_log_softmax(this._ctx, this._graphUopRaw(), axis)
      if (!uop) throw new Error('poly_log_softmax failed')
      return this._makeResult(uop, [this])
    }

    // --- Movement ops ---

    reshape(...shape) {
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      shape = shape.map((s, i) => s === null ? this.shape[i] : s)
      const inferred = shape.filter(s => s === -1).length
      if (inferred > 1) {
        throw new Error(`only one dimension can be inferred using -1, getting (${shape})`)
      }
      if (inferred) {
        const total = this.numel()
        const divisor = shape.reduce((a, b) => a * b, 1)
        if (divisor === 0) throw new RangeError('division by zero')
        shape = shape.map(s => s === -1 ? Math.floor(-total / divisor) : s)
      }
      const targetNumel = shape.reduce((a, b) => a * b, 1)
      if (this.numel() !== targetNumel) {
        throw new Error(`size mismatch, can't reshape ((${this.shape})) -> ((${shape}))`)
      }
      if (shape.length === this.shape.length && shape.every((s, i) => s === this.shape[i])) return this
      const core = this._rt._core.ffi.poly_tensor_reshape(
        this._ctx, this._tensor, shape, shape.length
      )
      return this._makeResultFromCore(core, [this])
    }

    permute(...order) {
      if (order.length === 1 && Array.isArray(order[0])) order = order[0]
      const core = this._rt._core.ffi.poly_tensor_permute(
        this._ctx, this._tensor, order, order.length
      )
      return this._makeResultFromCore(core, [this])
    }

    expand(...shape) {
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      shape = normalizeExpandShape(this.shape, shape)
      if (arraysEqual(this.shape, shape)) return this
      const aligned = new Array(shape.length - this.shape.length).fill(1).concat(this.shape)
      const reshaped = this.reshape(aligned)
      const core = this._rt._core.ffi.poly_tensor_expand(
        this._ctx, reshaped._tensor, shape, shape.length
      )
      return reshaped._makeResultFromCore(core, [reshaped])
    }

    shrink(arg) {
      const flat = []
      for (let i = 0; i < arg.length; i++) {
        flat.push(arg[i][0], arg[i][1])
      }
      const core = this._rt._core.ffi.poly_tensor_shrink(
        this._ctx, this._tensor, flat, arg.length
      )
      return this._makeResultFromCore(core, [this])
    }

    pad(arg, mode = 'constant', value = 0.0) {
      if (mode !== 'constant') throw new Error(`mode=${mode} is not supported`)
      arg = normalizePadArg(arg, this.shape.length)
      const flat = []
      for (let i = 0; i < arg.length; i++) {
        flat.push(arg[i][0], arg[i][1])
      }
      // Pinned _pad_constant shrinks negative pads before emitting a
      // non-negative PAD (mixin/__init__.py:359-368). The shared C boundary
      // owns that policy for zero and nonzero fill values alike.
      const core = this._rt._core.ffi.poly_tensor_pad_value(
        this._ctx, this._tensor, flat, arg.length, Number(value)
      )
      return this._makeResultFromCore(core, [this])
    }

    flip(axis, ...args) {
      let axes = Array.isArray(axis) ? Array.from(axis) : [axis]
      axes = axes.concat(args)
      axes = axes.map(a => {
        a = Number(a)
        return a < 0 ? a + this.shape.length : a
      })
      if (new Set(axes).size !== axes.length) {
        throw new Error(`dim can appear at most once, got ${axes}`)
      }
      const core = this._rt._core.ffi.poly_tensor_flip(
        this._ctx, this._tensor, axes, axes.length
      )
      return this._makeResultFromCore(core, [this])
    }

    transpose(dim0, dim1) {
      if (dim0 === undefined) dim0 = -2
      if (dim1 === undefined) dim1 = -1
      const nd = this.shape.length
      if (nd < 2) return this
      if (dim0 < 0) dim0 += nd
      if (dim1 < 0) dim1 += nd
      const order = [...Array(nd).keys()]
      const tmp = order[dim0]
      order[dim0] = order[dim1]
      order[dim1] = tmp
      return this.permute(...order)
    }

    squeeze(dim) {
      if (dim !== undefined && dim !== null) {
        if (dim < 0) dim += this.shape.length
        if (this.shape[dim] !== 1) return this
        const newShape = this.shape.filter((_, i) => i !== dim)
        if (!newShape.length) return this.reshape(1)
        return this.reshape(newShape)
      }
      const newShape = this.shape.filter(s => s !== 1)
      if (!newShape.length) return this.reshape(1)
      if (arraysEqual(newShape, this.shape)) return this
      return this.reshape(newShape)
    }

    unsqueeze(dim) {
      if (dim < 0) dim += this.shape.length + 1
      const newShape = [...this.shape]
      newShape.splice(dim, 0, 1)
      return this.reshape(newShape)
    }

    flatten(startDim, endDim) {
      if (startDim === undefined) startDim = 0
      if (endDim === undefined) endDim = -1
      if (endDim < 0) endDim += this.shape.length
      const before = this.shape.slice(0, startDim)
      let flatDim = 1
      for (let i = startDim; i <= endDim; i++) flatDim *= this.shape[i]
      const after = this.shape.slice(endDim + 1)
      return this.reshape([...before, flatDim, ...after])
    }

    unflatten(dim, sizes) {
      if (dim < 0) dim += this.shape.length
      const before = this.shape.slice(0, dim)
      const after = this.shape.slice(dim + 1)
      return this.reshape([...before, ...sizes, ...after])
    }

    view(...shape) { return this.reshape(...shape) }

    repeat(...repeats) {
      if (repeats.length === 1 && Array.isArray(repeats[0])) repeats = repeats[0]
      const nd = Math.max(this.shape.length, repeats.length)
      const shape = new Array(nd - this.shape.length).fill(1).concat(this.shape)
      repeats = new Array(nd - repeats.length).fill(1).concat(repeats)
      const newShape = []
      const expShape = []
      for (let i = 0; i < nd; i++) {
        newShape.push(1, shape[i])
        expShape.push(repeats[i], shape[i])
      }
      const finalShape = shape.map((s, i) => s * repeats[i])
      return this.reshape(newShape).expand(expShape).reshape(finalShape)
    }

    // --- Reduction ops ---

    sum(axis, keepdim) {
      if (keepdim === undefined) keepdim = false
      if (axis === undefined || axis === null) {
        axis = this.shape.map((_, i) => i)
      } else if (typeof axis === 'number') {
        axis = [axis]
      }
      const nd = this.shape.length
      axis = axis.map(a => a < 0 ? a + nd : a)

      const { ffi, ops } = this._rt._core
      let uop = ffi.poly_reduce_axis(this._ctx, ops.ADD, this._graphUopRaw(), axis, axis.length)

      // REDUCE_AXIS keeps all dims (reduced→1). If !keepdim, reshape to squeeze.
      if (!keepdim && axis.length > 0) {
        const axisSet = new Set(axis)
        const newShape = this.shape.filter((_, i) => !axisSet.has(i))
        if (newShape.length > 0) {
          uop = ffi.poly_reshape(this._ctx, uop, newShape, newShape.length)
        } else {
          uop = ffi.poly_reshape(this._ctx, uop, [], 0)
        }
      }
      return this._makeResult(uop, [this])
    }

    max(opts) {
      if (!opts) opts = {}
      let axis, keepdim
      if (typeof opts === 'object' && !Array.isArray(opts)) {
        axis = opts.axis
        keepdim = opts.keepdim || false
      } else {
        axis = opts
        keepdim = arguments[1] || false
      }

      const { ffi } = this._rt._core

      if (axis === undefined || axis === null) {
        let result = this
        for (let i = this.shape.length - 1; i >= 0; i--) {
          const uop = ffi.poly_max_reduce(this._ctx, result._graphUopRaw(), i, keepdim ? 1 : 0)
          result = this._makeResult(uop, [result])
        }
        return result
      }

      if (Array.isArray(axis)) {
        const axes = axis.map(a => {
          a = Number(a)
          return a < 0 ? a + this.shape.length : a
        })
        let result = this
        if (keepdim) {
          for (const ax of axes) {
            const uop = ffi.poly_max_reduce(this._ctx, result._graphUopRaw(), ax, 1)
            result = result._makeResult(uop, [result])
          }
          return result
        }
        for (const ax of axes.slice().sort((a, b) => b - a)) {
          const uop = ffi.poly_max_reduce(this._ctx, result._graphUopRaw(), ax, 0)
          result = result._makeResult(uop, [result])
        }
        return result
      }

      if (axis < 0) axis += this.shape.length
      const uop = ffi.poly_max_reduce(this._ctx, this._graphUopRaw(), axis, keepdim ? 1 : 0)
      return this._makeResult(uop, [this])
    }

    argmax(axis, keepdim) {
      if (keepdim === undefined) keepdim = false
      if (axis === undefined || axis === null) return this.flatten().argmax(0, false)
      if (axis < 0) axis += this.shape.length
      const { ffi } = this._rt._core
      const uop = ffi.poly_argmax(this._ctx, this._graphUopRaw(), axis)
      if (!uop) throw new Error('poly_argmax failed')
      let result = this._makeResult(uop, [this])
      if (keepdim) {
        const keepShape = [...this.shape]
        keepShape[axis] = 1
        result = result.reshape(keepShape)
      }
      return result
    }

    sort(dim, descending) {
      if (dim === undefined) dim = -1
      if (descending === undefined) descending = false
      if (dim < 0) dim += this.shape.length
      const { ffi } = this._rt._core
      if (!ffi.poly_sort) throw new Error('poly_sort is required for Tensor.sort')
      const pair = ffi.poly_sort(this._ctx, this._graphUopRaw(), dim, descending ? 1 : 0)
      if (!pair || pair.length !== 2 || !pair[0] || !pair[1]) throw new Error('poly_sort failed')
      const values = this._makeResult(pair[0], [this])
      const indices = new Tensor(null, {
        _ctx: this._ctx,
        _tensor: this._coreCreateWithRoots(
          pair[1], this._physicalizeResult(pair[1], [this]), POLY_TENSOR_VALUE, this._device
        ),
        _device: this._device,
        _dtype: dtypeNameForUop(this._ctx, pair[1], 'int32'),
        requiresGrad: false
      })
      return [values, indices]
    }

    argsort(dim, descending) {
      return this.sort(dim, descending)[1]
    }

    topk(k, dim, largest, sorted_) {
      if (dim === undefined) dim = -1
      if (largest === undefined) largest = true
      if (sorted_ === undefined) sorted_ = true
      if (!sorted_) throw new Error('topk with sorted_=False is not supported')
      if (dim < 0) dim += this.shape.length
      if (k > this.shape[dim]) throw new Error(`selected index k=${k} is out of range`)
      const { ffi } = this._rt._core
      if (!ffi.poly_topk) throw new Error('poly_topk is required for Tensor.topk')
      const pair = ffi.poly_topk(this._ctx, this._graphUopRaw(), k, dim, largest ? 1 : 0, sorted_ ? 1 : 0)
      if (!pair || pair.length !== 2 || !pair[0] || !pair[1]) throw new Error('poly_topk failed')
      const values = this._makeResult(pair[0], [this])
      const indices = new Tensor(null, {
        _ctx: this._ctx,
        _tensor: this._coreCreateWithRoots(
          pair[1], this._physicalizeResult(pair[1], [this]), POLY_TENSOR_VALUE, this._device
        ),
        _device: this._device,
        _dtype: dtypeNameForUop(this._ctx, pair[1], 'int32'),
        requiresGrad: false
      })
      return [values, indices]
    }

    min(opts) {
      if (!opts) opts = {}
      return this.neg().max(opts).neg()
    }

    mean(axis, keepdim) {
      if (keepdim === undefined) keepdim = false
      if (axis === undefined || axis === null) {
        return this.sum(null, keepdim).div(this.numel())
      }
      if (Array.isArray(axis)) {
        const axes = axis.map(a => {
          a = Number(a)
          return a < 0 ? a + this.shape.length : a
        })
        return this.sum(axes, keepdim).div(product(axes.map(a => this.shape[a])))
      }
      if (axis < 0) axis += this.shape.length
      const { ffi } = this._rt._core
      const uop = ffi.poly_mean_reduce(this._ctx, this._graphUopRaw(), axis, keepdim ? 1 : 0)
      return this._makeResult(uop, [this])
    }

    var(axis, keepdim, correction) {
      if (keepdim === undefined) keepdim = false
      if (correction === undefined) correction = 1
      if (axis === undefined || axis === null) {
        const m = this.mean()
        const diff = this.sub(m)
        const sq = diff.mul(diff)
        return sq.sum().div(this.numel() - correction)
      }
      if (axis < 0) axis += this.shape.length
      const { ffi } = this._rt._core
      // Keep var lazy like tinygrad. The previous mean.realize() was a stale
      // scheduler workaround that cut the differentiable graph in JS only.
      const uop = ffi.poly_var_reduce(
        this._ctx, this._graphUopRaw(), axis, keepdim ? 1 : 0, correction)
      if (!uop) throw new Error('poly_var_reduce failed')
      return this._makeResult(uop, [this])
    }

    std(axis, keepdim, correction) {
      if (keepdim === undefined) keepdim = false
      if (correction === undefined) correction = 1
      return this.var(axis, keepdim, correction).sqrt()
    }

    gather(dim, index) {
      if (!(index instanceof Tensor)) index = new Tensor(index, { dtype: 'int32', device: this._device })
      if (index._device !== this._device) {
        throw new Error(`expected index and self on the same device, index.device=${index.device}, self.device=${this.device}`)
      }
      if (index.ndim !== this.ndim) {
        throw new Error(`self.ndim must equal index.ndim, self.ndim=${this.ndim}, index.ndim=${index.ndim}`)
      }
      if (dim < 0) dim += this.ndim
      if (dim < 0 || dim >= this.ndim) throw new Error(`dim=${dim} out of range`)
      for (let d = 0; d < this.ndim; d++) {
        if (d !== dim && this.shape[d] < index.shape[d]) {
          throw new Error('requires self.shape[d] >= index.shape[d] for all d != dim')
        }
      }

      const uop = this._rt._core.ffi.poly_gather_dim(this._ctx, this._graphUopRaw(), dim, index._graphUopRaw())
      if (!uop) throw new Error('poly_gather_dim failed')
      return this._makeResult(uop, [this, index])
    }

    takeAlongAxis(index, axis) {
      return this.gather(axis, index)
    }

    oneHot(numClasses) {
      const uop = this._rt._core.ffi.poly_one_hot(this._ctx, this._graphUopRaw(), Number(numClasses))
      if (!uop) throw new Error('poly_one_hot failed')
      return this._makeResult(uop, [this])
    }

    one_hot(numClasses) { return this.oneHot(numClasses) }

    _preScatterValidate(dim, index, src) {
      if (!(index instanceof Tensor)) index = new Tensor(index, { dtype: 'int32', device: this._device })
      if (!(src instanceof Tensor)) src = Tensor.full(index.shape, src, { dtype: this._dtype, device: this._device })
      if (index._device !== this._device) {
        throw new Error(`expected index and self on the same device, index.device=${index.device}, self.device=${this.device}`)
      }
      if (src._device !== this._device) {
        throw new Error(`expected src and self on the same device, src.device=${src.device}, self.device=${this.device}`)
      }
      if (dim < 0) dim += this.ndim
      if (dim < 0 || dim >= this.ndim) throw new Error(`dim=${dim} out of range`)
      if (index.ndim !== this.ndim || src.ndim !== this.ndim) {
        throw new Error(`index.ndim, self.ndim and src.ndim must all match, index.ndim=${index.ndim}, self.ndim=${this.ndim}, src.ndim=${src.ndim}`)
      }
      for (let d = 0; d < this.ndim; d++) {
        if ((d !== dim && this.shape[d] < index.shape[d]) || src.shape[d] < index.shape[d]) {
          throw new Error('requires self.shape[d] >= index.shape[d] for all d != dim and src.shape[d] >= index.shape[d] for all d')
        }
      }
      if (this._dtype !== src._dtype) {
        throw new Error(`expected self and src to have the same dtype, self.dtype=${this._dtype}, src.dtype=${src._dtype}`)
      }
      return { dim, index, src }
    }

    scatterReduce(dim, index, src, reduce, includeSelf) {
      if (includeSelf === undefined) includeSelf = true
      if (!['sum', 'prod', 'mean', 'amax', 'amin'].includes(reduce)) {
        throw new Error(`reduce=${JSON.stringify(reduce)} must be one of 'sum', 'prod', 'mean', 'amax', 'amin'`)
      }
      if (!(src instanceof Tensor)) src = new Tensor(src, { dtype: this._dtype, device: this._device })
      const p = this._preScatterValidate(dim, index, src)
      const { ffi } = this._rt._core
      const uop = ffi.poly_scatter_reduce(this._ctx, this._graphUopRaw(), p.dim, p.index._graphUopRaw(), p.src._graphUopRaw(), reduce, includeSelf ? 1 : 0)
      if (!uop) throw new Error('poly_scatter_reduce failed')
      return this._makeResult(uop, [this, p.index, p.src])
    }

    scatter_reduce(dim, index, src, reduce, includeSelf) {
      return this.scatterReduce(dim, index, src, reduce, includeSelf)
    }

    scatter(dim, index, src, reduce) {
      if (reduce === undefined) reduce = null
      if (![null, 'add', 'multiply'].includes(reduce)) {
        throw new TypeError(`reduce=${JSON.stringify(reduce)} must be one of None, 'multiply', or 'add'`)
      }
      const srcIsTensor = src instanceof Tensor
      if (!srcIsTensor) {
        const idxShape = index instanceof Tensor ? index.shape : flattenArray(index, 'int32').shape
        src = Tensor.full(idxShape, src, { dtype: this._dtype, device: this._device })
      } else if (reduce !== null) {
        throw new TypeError('non-scalar src is not supported with reduce arg. use scatter_reduce')
      }
      const p = this._preScatterValidate(dim, index, src)
      const { ffi } = this._rt._core
      const uop = ffi.poly_scatter(this._ctx, this._graphUopRaw(), p.dim, p.index._graphUopRaw(), p.src._graphUopRaw(), reduce || '')
      if (!uop) throw new Error('poly_scatter failed')
      return this._makeResult(uop, [this, p.index, p.src])
    }

    // --- Matmul (C core dot) ---

    dot(w) {
      if (!(w instanceof Tensor)) {
        throw new TypeError(`Expected Tensor, got ${typeof w}`)
      }
      const { ffi } = this._rt._core
      const uop = ffi.poly_dot(this._ctx, this._graphUopRaw(), w._graphUopRaw())
      if (!uop) {
        throw new Error(`cannot dot ${JSON.stringify(this.shape)} and ${JSON.stringify(w.shape)}`)
      }
      return this._makeResult(uop, [this, w])
    }

    matmul(other) { return this.dot(other) }

    qr(opts) {
      let mode = 'complete'
      if (typeof opts === 'string') mode = opts
      else if (opts && typeof opts.mode === 'string') mode = opts.mode
      const modeId = mode === 'complete' ? 0 : mode === 'reduced' ? 1 : mode === 'r' ? 2 : -1
      if (modeId < 0) throw new Error("qr mode must be 'complete', 'reduced', or 'r'")
      const { ffi } = this._rt._core
      if (!ffi.poly_qr_ex && !ffi.poly_qr) throw new Error('poly_qr_ex is required for Tensor.qr')
      const qrFn = ffi.poly_qr_ex || ((ctx, uop, m) => {
        if (m !== 0) throw new Error('poly_qr_ex is required for non-complete Tensor.qr modes')
        return ffi.poly_qr(ctx, uop)
      })
      const pair = qrFn(this._ctx, this._graphUopRaw(), modeId)
      if (!pair || pair.length !== 2 || !pair[1] || (modeId !== 2 && !pair[0])) throw new Error('poly_qr_ex failed')
      if (modeId === 2) return this._makeResult(pair[1], [this])
      return [this._makeResult(pair[0], [this]), this._makeResult(pair[1], [this])]
    }

    triangularSolve(b, opts) {
      if (!(b instanceof Tensor)) b = new Tensor(b)
      opts = opts || {}
      const { ffi } = this._rt._core
      if (!ffi.poly_triangular_solve) {
        throw new Error('poly_triangular_solve is required for Tensor.triangularSolve')
      }
      const uop = ffi.poly_triangular_solve(
        this._ctx, this._graphUopRaw(), b._graphUopRaw(),
        opts.upper ? 1 : 0,
        opts.transposeA || opts.transpose_a ? 1 : 0,
        opts.unitDiagonal || opts.unit_diagonal ? 1 : 0
      )
      if (!uop) {
        throw new Error(`cannot triangularSolve A.shape=${JSON.stringify(this.shape)} and b.shape=${JSON.stringify(b.shape)}`)
      }
      return this._makeResult(uop, [this, b])
    }

    triangular_solve(b, upper, transpose_a, unit_diagonal) {
      return this.triangularSolve(b, { upper, transpose_a, unit_diagonal })
    }

    solveTriangular(b, opts) { return this.triangularSolve(b, opts) }

    cholesky(opts) {
      opts = opts || {}
      const { ffi } = this._rt._core
      if (!ffi.poly_cholesky) throw new Error('poly_cholesky is required for Tensor.cholesky')
      const uop = ffi.poly_cholesky(this._ctx, this._graphUopRaw(), opts.upper ? 1 : 0)
      if (!uop) throw new Error(`cannot cholesky shape=${JSON.stringify(this.shape)}`)
      return this._makeResult(uop, [this])
    }

    choleskySolve(b, opts) {
      if (!(b instanceof Tensor)) b = new Tensor(b)
      opts = opts || {}
      const { ffi } = this._rt._core
      if (!ffi.poly_cholesky_solve) {
        throw new Error('poly_cholesky_solve is required for Tensor.choleskySolve')
      }
      const uop = ffi.poly_cholesky_solve(this._ctx, this._graphUopRaw(), b._graphUopRaw(), opts.upper ? 1 : 0)
      if (!uop) {
        throw new Error(`cannot choleskySolve factor.shape=${JSON.stringify(this.shape)} and b.shape=${JSON.stringify(b.shape)}`)
      }
      return this._makeResult(uop, [this, b])
    }

    cholesky_solve(b, upper) { return this.choleskySolve(b, { upper }) }

    solve(b) {
      if (!(b instanceof Tensor)) b = new Tensor(b)
      const { ffi } = this._rt._core
      if (!ffi.poly_solve) throw new Error('poly_solve is required for Tensor.solve')
      const uop = ffi.poly_solve(this._ctx, this._graphUopRaw(), b._graphUopRaw())
      if (!uop) throw new Error(`cannot solve A.shape=${JSON.stringify(this.shape)} and b.shape=${JSON.stringify(b.shape)}`)
      return this._makeResult(uop, [this, b])
    }

    lstsq(b) {
      if (!(b instanceof Tensor)) b = new Tensor(b)
      const { ffi } = this._rt._core
      if (!ffi.poly_lstsq) throw new Error('poly_lstsq is required for Tensor.lstsq')
      const uop = ffi.poly_lstsq(this._ctx, this._graphUopRaw(), b._graphUopRaw())
      if (!uop) throw new Error(`cannot lstsq A.shape=${JSON.stringify(this.shape)} and b.shape=${JSON.stringify(b.shape)}`)
      return this._makeResult(uop, [this, b])
    }

    linear(weight, bias) {
      let result = this.dot(weight.transpose(-1, -2))
      if (bias) result = result.add(bias)
      return result
    }

    sequential(list) {
      let result = this
      for (const fn of list) result = fn(result)
      return result
    }

    maxPool2d(kernelSize = [2, 2], opts = {}) {
      if (typeof opts !== 'object' || Array.isArray(opts)) opts = { stride: opts }
      if (opts.ceilMode || opts.ceil_mode) {
        throw new Error('maxPool2d ceil_mode is not implemented in Polygrad yet')
      }
      if (opts.returnIndices || opts.return_indices) {
        throw new Error('maxPool2d return_indices is not implemented in Polygrad yet')
      }
      const k = makeTuple(kernelSize, 2)
      const stride = opts.stride == null ? k : makeTuple(opts.stride, k.length)
      const dilation = opts.dilation == null ? makeTuple(1, k.length) : makeTuple(opts.dilation, k.length)
      const padding = resolvePoolPads(opts.padding == null ? 0 : opts.padding, k.length)
      const uop = this._rt._core.ffi.poly_max_pool2d(
        this._ctx, this._graphUopRaw(), k, k.length, stride, dilation, padding, padding.length
      )
      if (!uop) throw new Error('poly_max_pool2d failed')
      return this._makeResult(uop, [this])
    }

    max_pool2d(kernelSize, stride, dilation, padding, ceilMode, returnIndices) {
      return this.maxPool2d(kernelSize === undefined ? [2, 2] : kernelSize, {
        stride: stride === undefined ? null : stride,
        dilation: dilation === undefined ? 1 : dilation,
        padding: padding === undefined ? 0 : padding,
        ceilMode: Boolean(ceilMode),
        returnIndices: Boolean(returnIndices)
      })
    }

    conv2d(weight, bias = null, groupsOrOpts = 1, stride = 1, dilation = 1, padding = 0, dtype = null) {
      if (!(weight instanceof Tensor)) weight = this._ensureTensor(weight)
      if (bias !== null && !(bias instanceof Tensor)) bias = this._ensureTensor(bias)
      let opts = groupsOrOpts
      if (typeof opts !== 'object' || Array.isArray(opts)) {
        opts = { groups: Number(groupsOrOpts), stride, dilation, padding, dtype }
      }
      const hw = weight.shape.slice(2)
      const strideTuple = opts.stride == null ? makeTuple(1, hw.length) : makeTuple(opts.stride, hw.length)
      const dilationTuple = opts.dilation == null ? makeTuple(1, hw.length) : makeTuple(opts.dilation, hw.length)
      const paddingTuple = resolvePoolPads(opts.padding == null ? 0 : opts.padding, hw.length)
      const groups = opts.groups == null ? 1 : Number(opts.groups)
      const uop = this._rt._core.ffi.poly_conv2d(
        this._ctx, this._graphUopRaw(), weight._graphUopRaw(),
        bias ? bias._graphUopRaw() : null,
        groups, strideTuple, dilationTuple, paddingTuple, paddingTuple.length
      )
      if (!uop) throw new Error('poly_conv2d failed')
      const inputs = bias ? [this, weight, bias] : [this, weight]
      return this._makeResult(uop, inputs)
    }

    batchnorm(weight, bias, mean, invstd, axis = 1) {
      const axes = (Array.isArray(axis) ? axis : [axis]).map(a => {
        a = Number(a)
        return a < 0 ? a + this.shape.length : a
      })
      const uop = this._rt._core.ffi.poly_batchnorm(
        this._ctx, this._graphUopRaw(),
        weight ? weight._graphUopRaw() : null,
        bias ? bias._graphUopRaw() : null,
        mean._graphUopRaw(), invstd._graphUopRaw(), axes, axes.length
      )
      if (!uop) throw new Error('poly_batchnorm failed')
      const inputs = [this, mean, invstd]
      if (weight) inputs.push(weight)
      if (bias) inputs.push(bias)
      return this._makeResult(uop, inputs)
    }

    // --- Loss functions ---

    crossEntropy(target, axis) {
      if (axis === undefined) axis = this.shape.length === 1 ? 0 : 1
      if (!(target instanceof Tensor)) target = new Tensor(target)
      const { ffi } = this._rt._core
      const uop = ffi.poly_cross_entropy(this._ctx,
        this._graphUopRaw(), target._graphUopRaw(), axis)
      if (!uop) {
        throw new Error(`shape mismatch: self.shape=${JSON.stringify(this.shape)}, target.shape=${JSON.stringify(target.shape)}`)
      }
      return this._makeResult(uop, [this, target])
    }

    binaryCrossEntropy(target) {
      const t1 = target.mul(this.log())
      const t2 = this._ensureTensor(1.0).sub(target).mul(this._ensureTensor(1.0).sub(this).log())
      return t1.add(t2).neg().mean()
    }

    layernorm(axis, eps) {
      if (axis === undefined) axis = -1
      if (eps === undefined) eps = 1e-5
      // Layernorm is normal lazy graph construction. A hidden realize here
      // would make JS diverge from tinygrad and Python autograd boundaries.
      const m = this.mean(axis, true)
      const v = this.var(axis, true, 0)
      return this.sub(m).div(v.add(eps).sqrt())
    }

    // --- Indexing ---

    getitem(...idx) {
      if (idx.length === 1 && Array.isArray(idx[0])) idx = idx[0]

      let result = this
      let dim = 0
      for (const i of idx) {
        if (i === null || i === undefined) {
          result = result.unsqueeze(dim)
          dim += 1
        } else if (typeof i === 'number') {
          let ii = i
          if (ii < 0) ii += result.shape[dim]
          const arg = result.shape.map((s, d) => d === dim ? [ii, ii + 1] : [0, s])
          result = result.shrink(arg)
          result = result.squeeze(dim)
        } else if (Array.isArray(i) && i.length === 2) {
          const [start, stop] = i
          const arg = result.shape.map((s, d) => d === dim ? [start, stop] : [0, s])
          result = result.shrink(arg)
          dim += 1
        } else if (i instanceof Tensor) {
          if (!isIntegerDtype(i.dtype)) throw new Error(`index dtype ${i.dtype} is not supported`)
          if (i._device !== result._device) {
            throw new Error(`expected index and self on the same device, index.device=${i.device}, self.device=${result.device}`)
          }
          const uop = this._rt._core.ffi.poly_index_select(
            result._ctx, result._graphUopRaw(), dim, i._graphUopRaw()
          )
          if (!uop) throw new Error('poly_index_select failed')
          result = result._makeResult(uop, [result, i])
          dim += i.shape.length
        } else if (typeof i === 'object' && i !== null && 'step' in i) {
          // Slice with step: {start, stop, step}
          // Reimplements Python's slice.indices(size)
          const size = result.shape[dim]
          let step = i.step != null ? i.step : 1
          if (step === 0) throw new Error('slice step cannot be zero')
          let start, stop
          if (step > 0) {
            start = i.start != null ? (i.start < 0 ? Math.max(i.start + size, 0) : Math.min(i.start, size)) : 0
            stop = i.stop != null ? (i.stop < 0 ? Math.max(i.stop + size, 0) : Math.min(i.stop, size)) : size
          } else {
            start = i.start != null ? (i.start < 0 ? Math.max(i.start + size, -1) : Math.min(i.start, size - 1)) : size - 1
            stop = i.stop != null ? (i.stop < 0 ? Math.max(i.stop + size, -1) : Math.min(i.stop, size - 1)) : -1
          }
          // Compute boundary and stride (matching tinygrad _getitem)
          let boundary = [start, stop]
          const stride = step
          if (stride * (boundary[1] - boundary[0]) < 0) {
            boundary = [0, 0]
          } else if (stride < 0) {
            boundary = [boundary[1] + 1, boundary[0] + 1]
          }
          // shrink to boundary
          const shrinkArg = result.shape.map((s, d) => d === dim ? boundary : [0, s])
          result = result.shrink(shrinkArg)
          // flip if negative stride
          if (stride < 0) result = result.flip(dim)
          const absStride = Math.abs(stride)
          // apply stride via pad+reshape+shrink+reshape
          if (absStride !== 1) {
            const sh = [...result.shape]
            // pad to multiple of stride
            const rem = sh[dim] % absStride
            if (rem !== 0) {
              const padAmt = absStride - rem
              const padding = sh.map((_, d) => d === dim ? [0, padAmt] : [0, 0])
              result = result.pad(padding)
              sh[dim] += padAmt
            }
            // reshape: split dim into (n_groups, stride)
            const newSh = [...sh.slice(0, dim), sh[dim] / absStride, absStride, ...sh.slice(dim + 1)]
            result = result.reshape(...newSh)
            // shrink to first element of each stride group
            const shrinkArg2 = result.shape.map((s, d) => d === dim + 1 ? [0, 1] : [0, s])
            result = result.shrink(shrinkArg2)
            // reshape back, collapsing the stride dim
            const finalSh = [...result.shape.slice(0, dim), result.shape[dim], ...result.shape.slice(dim + 2)]
            result = result.reshape(...finalSh)
          }
          dim += 1
        } else {
          throw new Error(`Unsupported index type: ${typeof i}`)
        }
      }
      return result
    }

    // --- Einsum (C core) ---

    static einsum(formula, ...operands) {
      if (operands.length === 1 && Array.isArray(operands[0])) operands = operands[0]
      if (!operands.length) throw new Error('einsum requires at least one operand')
      const { ffi } = _runtime._core
      const ctx = operands[0]._ctx
      const graphOperands = operands.map(t => ({ _uop: t._graphUopRaw() }))
      const r = ffi.poly_einsum(ctx, formula, graphOperands)
      if (!r.uop) throw new Error(`poly_einsum failed for formula: ${formula}`)
      const t0 = operands[0]
      const physical = t0._physicalizeResult(r.uop, operands)
      const t = new Tensor(null, {
        _ctx: ctx,
        _tensor: t0._coreCreateWithRoots(r.uop, physical, POLY_TENSOR_VALUE, t0._device),
        _dtype: dtypeNameForUop(ctx, r.uop, t0._dtype),
        _device: t0._device
      })
      t.requiresGrad = operands.some(x => x._requiresGrad)
      t._device = operands[0]._device
      return t
    }

    // --- Rearrange (C core, einops-style) ---

    rearrange(formula, kwargs) {
      if (!kwargs) kwargs = {}
      const { ffi } = this._rt._core
      const r = ffi.poly_rearrange(this._ctx, formula, this._graphUopRaw(), this.shape, kwargs)
      if (!r.uop) throw new Error(`poly_rearrange failed for formula: ${formula}`)
      return this._makeResult(r.uop, [this])
    }

    // --- Autograd ---

    backward() {
      const { ffi } = this._rt._core
      const targetEntries = this._liveGradTargets()
      if (!targetEntries.length) {
        throw new Error('No leaf tensors require grad')
      }

      const root = this._currentUopRaw()
      const gradLeaves = targetEntries.map(entry => entry.tensor)
      const targetUops = targetEntries.map(entry => entry.root)
      const customRecords = customGradRecordsFor(this._ctx, root)
      let gradUops
      if (customRecords.length) {
        // tinygrad splits AFTER(data, CALL) into a direct data edge and a
        // callback edge carrying the exact upstream at that AFTER. The core
        // handles the data edge; retain every active AFTER as a WRT here.
        const tempWrts = []
        const seen = new Set()
        const addWrt = (raw) => {
          raw = rawUop(raw)
          const key = uopKey(raw)
          if (!raw || seen.has(key)) return
          seen.add(key)
          tempWrts.push(raw)
        }
        for (const target of targetUops) addWrt(target)
        const activeByCall = []
        for (const { rec, active } of customRecords) {
          const activeAliases = []
          let activeCall = null
          for (const alias of active) {
            const afterRaw = rawUop(alias.after)
            const argRaw = rawUop(alias.arg)
            const callRaw = rawUop(alias.call)
            if (activeCall && uopKey(activeCall) !== uopKey(callRaw)) {
              throw new Error('customKernel outputs resolve to different CALLs')
            }
            activeCall = callRaw
            activeAliases.push({ afterRaw, argRaw })
            addWrt(afterRaw)
          }
          activeByCall.push({ rec, activeAliases, activeCall })
        }
        const tempResult = gradManyRaw(this._ctx, root, 0, tempWrts)
        const gradByRoot = new Map()
        const presentByRoot = new Map()
        for (let i = 0; i < tempWrts.length; i++) {
          const key = uopKey(tempWrts[i])
          gradByRoot.set(key, tempResult.grads[i])
          presentByRoot.set(key, tempResult.present[i])
        }

        for (const { rec, activeAliases, activeCall } of [...activeByCall].reverse()) {
          const upstreams = activeAliases
            .filter(({ afterRaw }) => presentByRoot.get(uopKey(afterRaw)) === true)
            .map(({ afterRaw }) => gradByRoot.get(uopKey(afterRaw)))
            .map(raw => new UOp(this._ctx, ffi, raw))
          if (!upstreams.length) continue
          const call = new UOp(this._ctx, ffi, activeCall)
          const callArgs = call.src.slice(1)
          if (!rec.gradFxn) {
            const needsCallGrad = callArgs.some(arg => targetUops.some(target =>
              ffi.poly_uop_reachable(this._ctx, rawUop(arg), rawUop(target))
            ))
            if (needsCallGrad) {
              const bodyOp = call.src[0].op
              const bodyName = Object.keys(ops).find(name => ops[name] === bodyOp) || String(bodyOp)
              throw new Error(`expected TUPLE body for gradient, got Ops.${bodyName}`)
            }
            continue
          }
          const returned = upstreams.length > 1
            ? rec.gradFxn(...upstreams, call)
            : rec.gradFxn(upstreams[0], call)
          if (!returned) continue
          const returnedList = Array.isArray(returned) ? returned : [returned]
          if (returnedList.length !== callArgs.length) {
            throw new Error(`customKernel gradFxn returned ${returnedList.length} grads, expected ${callArgs.length}`)
          }
          for (let i = 0; i < callArgs.length; i++) {
            const argGradRaw = gradResultRaw(returnedList[i])
            if (!argGradRaw) continue
            const propagated = gradManyRaw(
              this._ctx, rawUop(callArgs[i]), argGradRaw, tempWrts
            )
            for (let j = 0; j < tempWrts.length; j++) {
              if (!propagated.present[j]) continue
              const key = uopKey(tempWrts[j])
              gradByRoot.set(key, accumulateGradRaw(
                this._ctx, gradByRoot.get(key), propagated.grads[j]
              ))
              presentByRoot.set(key, true)
            }
          }
        }
        gradUops = targetUops.map(raw => gradByRoot.get(uopKey(raw)))
      } else {
        // tinygrad computes all target gradients in one reverse pass. Keeping
        // JS backward lazy also avoids realize-time live-retargeting while the
        // gradient set is still being built.
        gradUops = gradManyRaw(this._ctx, root, 0, targetUops).grads
      }

      for (let i = 0; i < gradLeaves.length; i++) {
        const leaf = gradLeaves[i]
        const gradUop = gradUops[i]
        if (!gradUop) throw new Error('poly_grad_many returned NULL for a leaf tensor')
        let gradTensor = new Tensor(null, {
          _ctx: this._ctx,
          _uop: gradUop,
          _dtype: leaf._dtype,
          _device: leaf._device
        })
        if (leaf.shape.length > 1) gradTensor = gradTensor.reshape(...leaf.shape)
        if (Number(ffi.poly_uop_device(gradUop)) === deviceId('auto')) {
          gradTensor = gradTensor.clone(leaf._device)
        }
        if (leaf._grad) {
          leaf._grad.assign(leaf._grad.add(gradTensor.to(leaf._grad._device)))
        } else {
          leaf._grad = gradTensor
        }
      }
    }

    // --- Static constructors ---

    static _resolveArrayType(opts) {
      return (opts && opts.dtype === 'float64') ? Float64Array : Float32Array
    }

    static zeros(...args) {
      let shape = args, opts
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !(args[args.length - 1] instanceof Array)) {
        opts = args[args.length - 1]; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      shape = shape.map(x => Number(x))
      return Tensor.full(shape, 0, { ...(opts || {}), dtype: (opts && opts.dtype) || 'float32' })
    }

    static ones(...args) {
      let shape = args, opts
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !(args[args.length - 1] instanceof Array)) {
        opts = args[args.length - 1]; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      shape = shape.map(x => Number(x))
      return Tensor.full(shape, 1, { ...(opts || {}), dtype: (opts && opts.dtype) || 'float32' })
    }

    static full(shape, fillValue, opts) {
      if (typeof shape === 'number') shape = [shape]
      shape = Array.from(shape, Number)
      opts = opts ? { ...opts } : {}
      const ctx = opts._ctx || _runtime._core.ctx
      const device = normalizeDevice(opts._device || opts.device || _runtime.device || 'cpu')
      const dtype = opts.dtype || (
        typeof fillValue === 'boolean' ? 'bool' : Number.isInteger(fillValue) ? 'int32' : 'float32'
      )
      const dtypeId = DTYPE_ID[dtype]
      if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dtype}`)
      const targetDevice = deviceId(device)
      const tensor = (dtype === 'bool' || isIntegerDtype(dtype))
        ? ffi.poly_tensor_full_int_by_id(
            ctx, shape, shape.length,
            typeof fillValue === 'boolean' ? (fillValue ? 1 : 0) : Math.trunc(Number(fillValue)),
            dtypeId, targetDevice
          )
        : ffi.poly_tensor_full_float_by_id(
            ctx, shape, shape.length, Number(fillValue), dtypeId, targetDevice
          )
      if (!tensor) throw new Error('C-owned Tensor.full construction failed')
      const value = new Tensor(null, {
        _ctx: ctx, _tensor: tensor, _dtype: dtype, _device: device,
        requiresGrad: Boolean(opts.requiresGrad || opts.requires_grad)
      })
      return opts.buffer === false ? value : value.clone(device)
    }

    static arange(start, stop, step, opts) {
      if (typeof stop === 'object' && stop !== null) { opts = stop; stop = undefined; step = undefined }
      if (typeof step === 'object' && step !== null) { opts = step; step = undefined }
      if (stop === undefined) { stop = start; start = 0 }
      if (step === undefined) step = 1
      if (step === 0) throw new Error('Tensor.arange step must not be zero')
      opts = opts ? { ...opts } : {}
      const ctx = opts._ctx || _runtime._core.ctx
      const device = normalizeDevice(opts._device || opts.device || _runtime.device || 'cpu')
      const dtype = opts.dtype || (
        [start, stop, step].every(Number.isInteger) ? 'int32' : 'float32'
      )
      const dtypeId = DTYPE_ID[dtype]
      if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dtype}`)
      const targetDevice = deviceId(device)
      const tensor = (dtype === 'bool' || isIntegerDtype(dtype))
        ? ffi.poly_tensor_arange_int_by_id(
            ctx, Math.trunc(start), Math.trunc(stop), Math.trunc(step), dtypeId, targetDevice
          )
        : ffi.poly_tensor_arange_float_by_id(
            ctx, Number(start), Number(stop), Number(step), dtypeId, targetDevice
          )
      if (!tensor) throw new Error('C-owned Tensor.arange construction failed')
      return new Tensor(null, {
        _ctx: ctx, _tensor: tensor, _dtype: dtype, _device: device,
        requiresGrad: Boolean(opts.requiresGrad || opts.requires_grad)
      })
    }

    static manual_seed(seed) {
      _seed = seed >>> 0
    }

    static rand(...args) {
      let shape = args, opts
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !(args[args.length - 1] instanceof Array)) {
        opts = args[args.length - 1]; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      const { ffi, ctx } = _runtime._core
      const seed = _seed++
      const uop = ffi.poly_rand(ctx, shape, shape.length, seed)
      if (!uop) throw new Error('poly_rand failed')
      return new Tensor(null, {
        _ctx: ctx, _uop: uop,
        _dtype: (opts && opts.dtype) || 'float32'
      })
    }

    static randn(...args) {
      let shape = args, opts
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !(args[args.length - 1] instanceof Array)) {
        opts = args[args.length - 1]; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      const { ffi, ctx } = _runtime._core
      const seed = _seed++
      const uop = ffi.poly_randn(ctx, shape, shape.length, seed)
      if (!uop) throw new Error('poly_randn failed')
      return new Tensor(null, {
        _ctx: ctx, _uop: uop,
        _dtype: (opts && opts.dtype) || 'float32'
      })
    }

    static randint(...args) {
      let opts = {}
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !Array.isArray(args[args.length - 1])) {
        opts = { ...args.pop() }
      }

      let shape
      let low = opts.low == null ? 0 : Number(opts.low)
      let high = opts.high == null ? 10 : Number(opts.high)
      if (opts.shape !== undefined) {
        shape = typeof opts.shape === 'number' ? [opts.shape] : Array.from(opts.shape)
        if (args.length >= 2 && opts.low == null && opts.high == null) {
          low = Number(args[0])
          high = Number(args[1])
        } else if (args.length > 0) {
          throw new Error('Tensor.randint got both positional shape and shape option')
        }
      } else if (args.length >= 3 && (Array.isArray(args[2]) || typeof args[2] === 'number')) {
        low = Number(args[0])
        high = Number(args[1])
        shape = typeof args[2] === 'number' ? [args[2]] : Array.from(args[2])
      } else {
        shape = args.length === 1 && Array.isArray(args[0]) ? Array.from(args[0]) : args.map(Number)
      }
      if (!shape.length) shape = [1]
      if (!Number.isInteger(low) || !Number.isInteger(high)) {
        throw new Error(`low=${low} and high=${high} must be integers`)
      }
      if (high <= low) throw new Error(`high must be greater than low, got low=${low}, high=${high}`)
      const dtype = opts.dtype || 'int32'
      const randOpts = { ...opts, dtype: 'float32' }
      delete randOpts.low
      delete randOpts.high
      delete randOpts.shape
      return Tensor.rand(...shape, randOpts).mul(high - low).add(low).cast(dtype)
    }

    static randperm(n, opts) {
      opts = opts ? { ...opts } : {}
      const dtype = opts.dtype || 'int32'
      opts.dtype = 'float32'
      return Tensor.rand(Number(n), opts).argsort().cast(dtype)
    }

    static linspace(start, stop, steps, opts) {
      opts = opts ? { ...opts } : {}
      const ctx = opts._ctx || _runtime._core.ctx
      const device = normalizeDevice(opts._device || opts.device || _runtime.device || 'cpu')
      const dtype = opts.dtype || 'float32'
      const dtypeId = DTYPE_ID[dtype]
      if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dtype}`)
      const tensor = ffi.poly_tensor_linspace_by_id(
        ctx, Number(start), Number(stop), Number(steps), dtypeId, deviceId(device)
      )
      if (!tensor) throw new Error('C-owned Tensor.linspace construction failed')
      return new Tensor(null, {
        _ctx: ctx, _tensor: tensor, _dtype: dtype, _device: device,
        requiresGrad: Boolean(opts.requiresGrad || opts.requires_grad)
      })
    }

    static eye(n, m, opts) {
      if (typeof m === 'object' && m !== null) { opts = m; m = undefined }
      opts = opts ? { ...opts } : {}
      const ctx = opts._ctx || _runtime._core.ctx
      const device = normalizeDevice(opts._device || opts.device || _runtime.device || 'cpu')
      const dtype = opts.dtype || 'float32'
      const dtypeId = DTYPE_ID[dtype]
      if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dtype}`)
      const rows = Number(n)
      const cols = m === undefined ? rows : Number(m)
      const tensor = ffi.poly_tensor_eye_by_id(
        ctx, rows, cols, dtypeId, deviceId(device)
      )
      if (!tensor) throw new Error('C-owned Tensor.eye construction failed')
      return new Tensor(null, {
        _ctx: ctx, _tensor: tensor, _dtype: dtype, _device: device,
        requiresGrad: Boolean(opts.requiresGrad || opts.requires_grad)
      })
    }

    static empty(...args) {
      let shape = args, opts
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !(args[args.length - 1] instanceof Array)) {
        opts = args[args.length - 1]; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      shape = shape.map(x => Number(x))
      if (shape.some(x => x < 0)) throw new Error(`negative dimensions are not allowed: ${shape}`)
      if (opts && Object.prototype.hasOwnProperty.call(opts, 'name')) {
        throw new TypeError('Tensor.empty does not accept name; pass names to Instance.fromTensors')
      }
      const ctx = (opts && opts._ctx) || _runtime._core.ctx
      const dtype = (opts && opts.dtype) || 'float32'
      const dtypeId = DTYPE_ID[dtype] || DTYPE_ID.float32
      const tensorDevice = normalizeDevice(
        (opts && (opts._device || opts.device)) || _runtime.device || 'cpu'
      )
      const tensor = ffi.poly_tensor_empty_by_id(
        ctx, dtypeId, shape, shape.length, deviceId(tensorDevice)
      )
      if (!tensor) throw new Error('poly_tensor_empty_by_id failed')
      return new Tensor(null, {
        _ctx: ctx,
        _tensor: tensor,
        _dtype: dtype,
        _device: tensorDevice,
        requiresGrad: opts && opts.requiresGrad
      })
    }

    cat(...tensors) {
      return Tensor.cat(this, ...tensors)
    }

    static cat(...tensors) {
      let dim = 0
      if (tensors.length >= 2 && typeof tensors[tensors.length - 1] === 'object'
        && !(tensors[tensors.length - 1] instanceof Tensor)
        && !Array.isArray(tensors[tensors.length - 1])) {
        dim = tensors.pop().dim || 0
      }
      if (tensors.length === 1 && Array.isArray(tensors[0])) tensors = tensors[0]
      if (!tensors.length) throw new Error('cat requires at least one tensor')

      const ndim = tensors[0].shape.length
      if (dim < 0) dim += ndim
      const outShape = [...tensors[0].shape]
      outShape[dim] = tensors.reduce((acc, t) => acc + t.shape[dim], 0)

      let offset = 0
      let result = null
      for (const t of tensors) {
        const padBefore = new Array(ndim).fill(0)
        const padAfter = new Array(ndim).fill(0)
        padBefore[dim] = offset
        padAfter[dim] = outShape[dim] - offset - t.shape[dim]
        const padArg = Array.from({ length: ndim }, (_, i) => [padBefore[i], padAfter[i]])
        const padded = t.pad(padArg)
        result = result ? result.add(padded) : padded
        offset += t.shape[dim]
      }
      return result
    }

    stack(...tensors) {
      return Tensor.stack(this, ...tensors)
    }

    static stack(...tensors) {
      let dim = 0
      if (tensors.length >= 2 && typeof tensors[tensors.length - 1] === 'object'
        && !(tensors[tensors.length - 1] instanceof Tensor)
        && !Array.isArray(tensors[tensors.length - 1])) {
        dim = tensors.pop().dim || 0
      }
      if (tensors.length === 1 && Array.isArray(tensors[0])) tensors = tensors[0]
      return Tensor.cat(tensors.map(t => t.unsqueeze(dim)), { dim })
    }

    split(sizes, dim) {
      if (dim === undefined) dim = 0
      if (dim < 0) dim += this.shape.length
      if (typeof sizes === 'number') {
        const total = this.shape[dim]
        const chunkSize = sizes
        sizes = []
        for (let i = 0; i < total; i += chunkSize) {
          sizes.push(Math.min(chunkSize, total - i))
        }
      }
      const results = []
      let offset = 0
      for (const sz of sizes) {
        const arg = this.shape.map((s, d) => d === dim ? [offset, offset + sz] : [0, s])
        results.push(this.shrink(arg))
        offset += sz
      }
      return results
    }

    chunk(n, dim) {
      if (dim === undefined) dim = 0
      if (dim < 0) dim += this.shape.length
      const total = this.shape[dim]
      const chunkSize = Math.ceil(total / n)
      return this.split(chunkSize, dim)
    }

    toString() {
      return `Tensor(shape=[${this.shape}], dtype=${this.dtype}, realized=${this.uop.hasBufferIdentity()})`
    }
  }

  return Tensor
}

module.exports = { createBoundTensorClass, flattenArray, arraysEqual, _buildNested }
