/**
 * tensor.js -- Runtime-bound Tensor class factory for polygrad.
 *
 * All graph construction ops are synchronous (use _runtime._core).
 * All realize/data-extraction ops are async (core already resolved by create()).
 *
 * The core ffi table normalizes int64 marshalling:
 *   - Functions taking shape/axis arrays accept plain JS number[]
 *   - JS currently reads concrete max_shape dims from UOps
 */

'use strict'

const { UOp } = require('./uop/ops')

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
  const applyMapToTensors = (ctx, replacements) => {
    if (!replacements.length || !ffi.poly_uop_substitute) return

    const byDevice = new Map()
    for (const r of replacements) {
      if (!r.oldRaw || !r.newRaw) continue
      const key = String(r.deviceId)
      if (!byDevice.has(key)) byDevice.set(key, [])
      byDevice.get(key).push(r)
    }
    if (!byDevice.size) return

    const kept = []
    for (const ref of liveTensors) {
      const t = useWeakRef ? ref.deref() : ref
      if (!t) continue
      kept.push(ref)
      if (t._ctx !== ctx || !t._tensor) continue

      const devId = tensorDevice(t._tensor)
      const entries = byDevice.get(String(devId))
      if (!entries) continue

      const currentRaw = t._currentUopRaw()
      if (!currentRaw) continue

      const newRaw = ffi.poly_uop_substitute(
        t._ctx,
        currentRaw,
        entries.map(r => r.oldRaw),
        entries.map(r => r.newRaw)
      )
      if (!newRaw || uopKey(newRaw) === uopKey(currentRaw)) continue

      if (!ffi.poly_tensor_update) {
        throw new Error('poly_tensor_update is required for live UOp retargeting')
      }
      const rc = ffi.poly_tensor_update(
        t._ctx, t._tensor, 0, newRaw, POLY_TENSOR_VALUE, devId)
      if (rc !== 0) throw new Error('poly_tensor_update failed during live UOp retarget')
      t._data = null
      /* Retargeting changes the current UOp, not the frontend placement label.
       * In the WASM core, public "cpu" executes on POLY_DEVICE_WASM, so deriving
       * _device from the core backend id would turn a CPU tensor into WASM. */
    }

    liveTensors.length = 0
    liveTensors.push(...kept)
  }
  const canRegisterFrontendRelease =
    ffi.poly_ctx_set_frontend_buffer_release || ffi.poly_set_frontend_buffer_release
  if (canRegisterFrontendRelease && !_runtime._core.__frontendBufferReleaseRegistered) {
    const releaseFrontendBuffer = (bufferKey) => {
      const key = normalizeBufferKey(bufferKey)
      hostBuffers.delete(key)
      if (_runtime._core.unregisterHostBuffer) {
        _runtime._core.unregisterHostBuffer(key)
      }
    }
    if (ffi.poly_ctx_set_frontend_buffer_release) {
      ffi.poly_ctx_set_frontend_buffer_release(_runtime._core.ctx, releaseFrontendBuffer)
    } else {
      ffi.poly_set_frontend_buffer_release(releaseFrontendBuffer)
    }
    _runtime._core.__frontendBufferReleaseRegistered = true
  }

  class Tensor {
    /**
     * @param {number|number[]|Float32Array|Float64Array|null} data
     * @param {object} opts
     */
    constructor(data, opts) {
      if (!opts) opts = {}
      const core = _runtime._core
      this._rt = _runtime
      this._ctx = opts._ctx || core.ctx
      this._requiresGrad = Boolean(opts.requiresGrad)
      this._grad = null
      this._device = normalizeDevice(opts._device || opts.device || _runtime.device || 'cpu')
      this._tensor = opts._tensor || null
      let currentUop = null
      const optUop = opts._uop || (this._tensor ? tensorUop(this._tensor) : null)

      if (optUop) {
        // Internal construction from ops -- shape is on the UOp.
        // Accept either a UOp wrapper instance or a raw handle.
        const raw = rawUop(optUop)
        currentUop = optUop instanceof UOp ? optUop : new UOp(this._ctx, core.ffi, raw)
        this._data = opts._data || null
        this._dtype = dtypeNameForUop(this._ctx, raw, opts._dtype)
      } else {
        // User construction from data. Resolve dtype and flatten to a single
        // TypedArray. Then call UOp.fromHost (one FFI call) which creates
        // the BUFFER UOp, registers a PolyBuffer wrapping the TypedArray's
        // bytes in ctx->buffers, and wraps in RESHAPE when ndim > 1.
        // JS also keeps a strong owner entry keyed by the C-side PolyBuffer*
        // address value, not by the BUFFER UOp.
        let dt, flat, shape
        if (data instanceof Float64Array) {
          dt = 'float64'; flat = new Float64Array(data); shape = [data.length]
        } else if (data instanceof Float32Array && (!opts.dtype || opts.dtype === 'float32')) {
          dt = 'float32'; flat = new Float32Array(data); shape = [data.length]
        } else if (ArrayBuffer.isView(data) && !(data instanceof DataView)) {
          dt = (opts && opts.dtype) || 'float32'
          const ArrayType = TA_BY_DTYPE[dt] || Float32Array
          flat = new ArrayType(data)
          shape = [data.length]
        } else {
          dt = (opts && opts.dtype) || 'float32'
          const r = flattenArray(data, dt)
          flat = r.data; shape = r.shape
        }
        this._dtype = dt
        this._data = flat
        const dtypeId = DTYPE_ID[dt] || 12
        const dims = shape.length > 1 ? shape : null
        currentUop = UOp.fromHost(this._ctx, core.ffi, flat, dtypeId, dims)
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
      if (!this._tensor && currentUop) {
        this._tensor = this._coreCreate(currentUop.raw, POLY_TENSOR_VALUE, this._device)
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
    static async _coreRealizeBatch(ctx, targets) {
      const realized = await ffi.poly_realize_tensors(ctx, targets.map(t => t._tensor))
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
      const root = this._graphUopRaw()
      const targets = []
      /* tinygrad discovers backward targets from all_tensors, not from a saved
       * input list: tensors whose logical UOp appears in the loss graph and
       * require gradients receive gradients. */
      for (const t of liveTensorSnapshot()) {
        if (!t || t._ctx !== this._ctx || !t._tensor || !t._requiresGrad) continue
        const target = t._graphUopRaw()
        if (target && ffi.poly_uop_reachable(this._ctx, root, target)) targets.push(t)
      }
      return targets
    }

    async realize(...lst) {
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
      const oldRoots = targets.map(t => t._currentUopRaw())
      const deviceIds = targets.map(t => tensorDevice(t._tensor))
      const realized = await Tensor._coreRealizeBatch(ctx, targets)
      const replacements = []
      for (let i = 0; i < targets.length; i++) {
        replacements.push({
          oldRaw: oldRoots[i],
          newRaw: tensorUop(realized[i]),
          deviceId: deviceIds[i],
          realized: realized[i]
        })
      }
      applyMapToTensors(ctx, replacements)
      return this
    }

    async _readBufferBytes() {
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
        raw = await ffi.poly_buffer_read(ctx, bufRaw, nbytes)
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

    async toArray() {
      const numel = this.numel()
      if (numel === 0) {
        const AT = TA_BY_DTYPE[this._dtype] || Float32Array
        return new AT(0)
      }
      let t = this
      if (this._dtype === 'float16' || this._dtype === 'bfloat16') t = t.cast('float32')
      if (!t.uop.hasBufferIdentity()) t = t.contiguous()
      await t.realize()
      return await t._readBufferBytes()
    }

    async toTypedArray() {
      return await this.toArray()
    }

    static async toTypedArrays(...tensors) {
      if (tensors.length === 1 && Array.isArray(tensors[0])) tensors = tensors[0]
      if (!tensors.length) return []
      for (const t of tensors) {
        if (!(t instanceof Tensor)) throw new TypeError('Tensor.toTypedArrays expects Tensor arguments')
      }
      const prepared = tensors.map(t => {
        if (t.numel() === 0) return t
        let out = t
        if (out._dtype === 'float16' || out._dtype === 'bfloat16') out = out.cast('float32')
        if (!out.uop.hasBufferIdentity()) out = out.contiguous()
        return out
      })
      const targets = prepared.filter(t => t.numel() !== 0)
      if (targets.length) await targets[0].realize(...targets.slice(1))
      const out = []
      for (const t of prepared) {
        if (t.numel() === 0) {
          const AT = TA_BY_DTYPE[t._dtype] || Float32Array
          out.push(new AT(0))
        } else {
          out.push(await t._readBufferBytes())
        }
      }
      return out
    }

    async item() {
      const arr = await this.toArray()
      if (arr.length !== 1) {
        throw new Error(`item() requires scalar tensor, got shape [${this.shape}]`)
      }
      return arr[0]
    }

    async tolist() {
      return _buildNested(await this.toArray(), this.shape, 0, 0).value
    }

    async detach() {
      return new Tensor(await this.toArray(), { dtype: this._dtype })
    }

    async clone() {
      const t = new Tensor(await this.toArray(), { dtype: this._dtype })
      t.requiresGrad = this._requiresGrad
      return t
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
      if (!ffi.poly_buffer_write) throw new Error('poly_buffer_write is required for Tensor.copyFrom')
      const buf = this.uop && this.uop.buffer ? this.uop.buffer.raw : null
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
      ffi.poly_buffer_write(this._ctx, buf, view)
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
      return t
    }

    cpu() { return this.to('cpu') }
    cuda() { return this.to('cuda') }

    contiguous() {
      const { ffi } = this._rt._core
      const uop = ffi.poly_contiguous(this._ctx, this._graphUopRaw())
      return this._makeResult(uop, [this])
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
      if (!curShape.length && this._graphBufferRaw() === null) return uop
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

    _binop(other, opName) {
      const { ffi, ops } = this._rt._core
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const xUop = this._broadcastUop(outShape)
      const yUop = other._broadcastUop(outShape)
      const uop = ffi.poly_alu2(this._ctx, ops[opName], xUop, yUop)
      return this._makeResult(uop, [this, other])
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
      const uop = ffi.poly_alu1(this._ctx, ops.NEG, this._graphUopRaw())
      return this._makeResult(uop, [this])
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
      const { ffi } = this._rt._core
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const uop = ffi.poly_maximum(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      return this._makeResult(uop, [this, other])
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
      const uop = ffi.poly_alu1(this._ctx, ops.EXP2, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    log2() {
      const { ffi, ops } = this._rt._core
      const uop = ffi.poly_alu1(this._ctx, ops.LOG2, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    sqrt() {
      const { ffi, ops } = this._rt._core
      const uop = ffi.poly_alu1(this._ctx, ops.SQRT, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    reciprocal() {
      const { ffi, ops } = this._rt._core
      const uop = ffi.poly_alu1(this._ctx, ops.RECIPROCAL, this._graphUopRaw())
      return this._makeResult(uop, [this])
    }

    trunc() {
      const { ffi, ops } = this._rt._core
      const uop = ffi.poly_alu1(this._ctx, ops.TRUNC, this._graphUopRaw())
      return this._makeResult(uop, [this])
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
      if (shape.indexOf(-1) !== -1) {
        const total = this.numel()
        const negIdx = shape.indexOf(-1)
        let known = 1
        for (let i = 0; i < shape.length; i++) {
          if (i !== negIdx) known *= shape[i]
        }
        shape = shape.map((s, i) => i === negIdx ? Math.floor(total / known) : s)
      }
      const uop = this._rt._core.ffi.poly_reshape(this._ctx, this._graphUopRaw(), shape, shape.length)
      return this._makeResult(uop, [this])
    }

    permute(...order) {
      if (order.length === 1 && Array.isArray(order[0])) order = order[0]
      const uop = this._rt._core.ffi.poly_permute(this._ctx, this._graphUopRaw(), order, order.length)
      const newShape = order.map(i => this.shape[i])
      return this._makeResult(uop, [this])
    }

    expand(...shape) {
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      const uop = this._rt._core.ffi.poly_expand(this._ctx, this._graphUopRaw(), shape, shape.length)
      return this._makeResult(uop, [this])
    }

    shrink(arg) {
      const flat = []
      for (let i = 0; i < arg.length; i++) {
        flat.push(arg[i][0], arg[i][1])
      }
      const uop = this._rt._core.ffi.poly_shrink(this._ctx, this._graphUopRaw(), flat, arg.length)
      const newShape = arg.map(([s, e]) => e - s)
      return this._makeResult(uop, [this])
    }

    pad(arg) {
      const flat = []
      for (let i = 0; i < arg.length; i++) {
        flat.push(arg[i][0], arg[i][1])
      }
      const uop = this._rt._core.ffi.poly_pad(this._ctx, this._graphUopRaw(), flat, arg.length)
      const newShape = this.shape.map((s, i) => s + arg[i][0] + arg[i][1])
      return this._makeResult(uop, [this])
    }

    flip(axis) {
      if (typeof axis === 'number') axis = [axis]
      const uop = this._rt._core.ffi.poly_flip(this._ctx, this._graphUopRaw(), axis, axis.length)
      return this._makeResult(uop, [this])
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

    async backward() {
      const { ffi } = this._rt._core
      const gradLeaves = this._liveGradTargets()
      if (!gradLeaves.length) {
        throw new Error('No leaf tensors require grad')
      }

      const targetUops = gradLeaves.map(leaf => leaf._graphUopRaw())
      // tinygrad computes all target gradients in one reverse pass. Keeping
      // JS backward lazy also avoids realize-time live-retargeting while the
      // gradient set is still being built.
      const gradUops = ffi.poly_grad_many(this._ctx, this._graphUopRaw(), 0, targetUops)
      if (!gradUops || gradUops.length !== gradLeaves.length) {
        throw new Error('poly_grad_many failed')
      }

      for (let i = 0; i < gradLeaves.length; i++) {
        const leaf = gradLeaves[i]
        const gradUop = gradUops[i]
        if (!gradUop) throw new Error('poly_grad_many returned NULL for a leaf tensor')
        const gradTensor = new Tensor(null, {
          _ctx: this._ctx,
          _uop: gradUop,
          _dtype: leaf._dtype,
          _device: leaf._device
        })
        if (leaf._grad) {
          leaf._grad = leaf._grad.add(gradTensor)
        } else {
          leaf._grad = leaf.shape.length > 1 ? gradTensor.reshape(...leaf.shape) : gradTensor
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
      const AT = Tensor._resolveArrayType(opts)
      const numel = shape.reduce((a, b) => a * b, 1)
      const t = new Tensor(new AT(numel).fill(0), opts)
      if (shape.length > 1) return t.reshape(...shape)
      return t
    }

    static ones(...args) {
      let shape = args, opts
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !(args[args.length - 1] instanceof Array)) {
        opts = args[args.length - 1]; shape = args.slice(0, -1)
      }
      const AT = Tensor._resolveArrayType(opts)
      const numel = shape.reduce((a, b) => a * b, 1)
      const t = new Tensor(new AT(numel).fill(1), opts)
      if (shape.length > 1) return t.reshape(...shape)
      return t
    }

    static full(shape, fillValue, opts) {
      if (typeof shape === 'number') shape = [shape]
      const AT = Tensor._resolveArrayType(opts)
      const numel = shape.reduce((a, b) => a * b, 1)
      const t = new Tensor(new AT(numel).fill(fillValue), opts)
      if (shape.length > 1) return t.reshape(...shape)
      return t
    }

    static arange(stop, start, step, opts) {
      if (start === undefined) start = 0
      if (step === undefined) step = 1
      const AT = Tensor._resolveArrayType(opts)
      const arr = []
      for (let i = start; i < stop; i += step) arr.push(i)
      return new Tensor(new AT(arr), opts)
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

    static randint(low, high, shape, opts) {
      if (high === undefined) { high = low; low = 0 }
      if (typeof shape === 'number') shape = [shape]
      if (!shape) shape = [1]
      const AT = Tensor._resolveArrayType(opts)
      const numel = shape.reduce((a, b) => a * b, 1)
      const data = new AT(numel)
      for (let i = 0; i < numel; i++) data[i] = Math.floor(Math.random() * (high - low)) + low
      const t = new Tensor(data, opts)
      if (shape.length > 1) return t.reshape(...shape)
      return t
    }

    static linspace(start, stop, steps, opts) {
      const AT = Tensor._resolveArrayType(opts)
      const data = new AT(steps)
      for (let i = 0; i < steps; i++) {
        data[i] = start + (stop - start) * i / (steps - 1)
      }
      return new Tensor(data, opts)
    }

    static eye(n, opts) {
      const AT = Tensor._resolveArrayType(opts)
      const data = new AT(n * n)
      for (let i = 0; i < n; i++) data[i * n + i] = 1
      return new Tensor(data, opts).reshape(n, n)
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
      const numel = shape.reduce((a, b) => a * b, 1)
      let uop = ffi.poly_buffer_by_id(ctx, dtypeId, numel)
      if (!uop) throw new Error('poly_buffer_by_id failed')
      if (shape.length !== 1 || (shape.length === 1 && shape[0] !== numel)) {
        uop = ffi.poly_reshape(ctx, uop, shape, shape.length)
        if (!uop) throw new Error('poly_reshape failed')
      }
      return new Tensor(null, {
        _ctx: ctx,
        _uop: new UOp(ctx, ffi, uop),
        _dtype: dtype,
        _device: opts && (opts._device || opts.device)
      })
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
