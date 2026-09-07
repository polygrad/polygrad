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

const FLOAT64_BITS_BUFFER = new ArrayBuffer(8)
const FLOAT64_BITS_VIEW = new DataView(FLOAT64_BITS_BUFFER)
const customKernelGradKeys = new WeakMap()
let nextCustomKernelGradKey = 1

function customKernelGradKey(gradFxn) {
  if (!gradFxn) return 0
  let key = customKernelGradKeys.get(gradFxn)
  if (key === undefined) {
    if (nextCustomKernelGradKey > 0xFFFFFFFF) {
      throw new RangeError('custom kernel gradient key space exhausted')
    }
    key = nextCustomKernelGradKey++
    customKernelGradKeys.set(gradFxn, key)
  }
  return key
}

function normalizeLogicalPolicy(value) {
  if (value === undefined || value === null) return null
  if (value === false) return 0
  if (value === true) return 1
  if (Number.isInteger(value) && value >= 0 && value <= 2) return value
  if (typeof value === 'string') {
    const policies = { never: 0, always: 1, until_realize: 2 }
    if (Object.prototype.hasOwnProperty.call(policies, value)) return policies[value]
  }
  throw new TypeError(
    'logical policy must be never, always, until_realize, false/0, true/1, 2, or null'
  )
}

function logicalPolicyName(value) {
  const names = ['never', 'always', 'until_realize']
  if (!Number.isInteger(value) || !names[value]) throw new Error(`unknown logical policy ${value}`)
  return names[value]
}

function logicalStateName(value) {
  const names = ['available', 'never_constructed', 'retired', 'unsupported_resource']
  if (!Number.isInteger(value) || !names[value]) throw new Error(`unknown logical state ${value}`)
  return names[value]
}

function roundShiftRightEven(value, shift) {
  if (shift <= 0) return value << BigInt(-shift)
  const s = BigInt(shift)
  const quotient = value >> s
  const remainder = value - (quotient << s)
  const halfway = 1n << (s - 1n)
  return remainder > halfway || (remainder === halfway && (quotient & 1n))
    ? quotient + 1n
    : quotient
}

// JavaScript has no baseline Float16Array. This is the IEEE-754 binary64 to
// binary16, round-to-nearest-even equivalent of pinned UOp._frompy's
// struct.pack('e', value) path (tinygrad/uop/ops.py:752-764).
function encodeFloat16Bits(value) {
  FLOAT64_BITS_VIEW.setFloat64(0, Number(value), false)
  const bits = FLOAT64_BITS_VIEW.getBigUint64(0, false)
  const sign = Number((bits >> 48n) & 0x8000n)
  const exponent = Number((bits >> 52n) & 0x7FFn)
  const fraction = bits & ((1n << 52n) - 1n)
  if (exponent === 0x7FF) return sign | (fraction === 0n ? 0x7C00 : 0x7E00)

  const significand = exponent === 0 ? fraction : (1n << 52n) | fraction
  if (significand === 0n) return sign
  const exponent2 = exponent === 0 ? -1074 : exponent - 1023 - 52
  const topBit = significand.toString(2).length - 1
  let unbiased = topBit + exponent2

  if (unbiased < -14) {
    const scale = exponent2 + 24
    const mantissa = scale >= 0
      ? significand << BigInt(scale)
      : roundShiftRightEven(significand, -scale)
    if (mantissa === 0n) return sign
    if (mantissa >= 1024n) return sign | 0x0400
    return sign | Number(mantissa)
  }

  let rounded = roundShiftRightEven(significand, topBit - 10)
  if (rounded === 2048n) {
    rounded = 1024n
    unbiased++
  }
  if (unbiased > 15) return sign | 0x7C00
  return sign | ((unbiased + 15) << 10) | Number(rounded - 1024n)
}

function numericTypedArray(values, dtype) {
  if (dtype !== 'float16') {
    const ArrayType = TA_BY_DTYPE[dtype] || Float32Array
    // Pinned _frompy converts to integer before wrapping to the storage width.
    // Keep BigInts exact; Number(BigInt) would lose the low bits before wrapping.
    if (dtype === 'int64' || dtype === 'uint64') {
      return ArrayType.from(values, v => typeof v === 'bigint' ? v : BigInt(Math.trunc(Number(v))))
    }
    // Lists are narrowed while flattening; TypedArrays are homogeneous.
    if (isIntegerDtype(dtype) && typeof values[0] === 'bigint') {
      const bits = ArrayType.BYTES_PER_ELEMENT * 8
      return ArrayType.from(values, v => typeof v === 'bigint' ? Number(BigInt.asUintN(bits, v)) : v)
    }
    return new ArrayType(values)
  }
  const out = new Uint16Array(values.length)
  for (let i = 0; i < values.length; i++) out[i] = encodeFloat16Bits(values[i])
  return out
}

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
  fp8e4m3: Uint8Array,
  fp8e5m2: Uint8Array,
  fp8e4m3fnuz: Uint8Array,
  fp8e5m2fnuz: Uint8Array,
  float32: Float32Array,
  float64: Float64Array
}

// --- Utility helpers ---

function flattenArray(arr, dtype) {
  if (typeof arr === 'number') {
    const v = dtype === 'bool' ? (arr ? 1 : 0) : arr
    return { data: numericTypedArray([v], dtype), shape: [1] }
  }
  if (arr instanceof Float32Array || arr instanceof Float64Array) {
    return { data: numericTypedArray(arr, dtype), shape: [arr.length] }
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
  const integerStorage = isIntegerDtype(dtype)
  const narrowBits = integerStorage && dtype !== 'int64' && dtype !== 'uint64'
    ? TA_BY_DTYPE[dtype].BYTES_PER_ELEMENT * 8 : 0
  const recurse = (a, depth) => {
    if (depth < shape.length) {
      if (!Array.isArray(a) || a.length !== shape[depth]) throw new TypeError('inhomogeneous shape')
      for (const el of a) recurse(el, depth + 1)
    } else {
      if (Array.isArray(a)) throw new TypeError('inhomogeneous shape')
      // Typed-array casts have their own conversion rules; list construction
      // follows _frompy's int conversion, which rejects NaN and infinities.
      if (integerStorage && typeof a === 'number' && !Number.isFinite(a)) {
        throw new RangeError('cannot convert nonfinite value to integer storage')
      }
      if (narrowBits && typeof a === 'bigint') a = Number(BigInt.asUintN(narrowBits, a))
      flat.push(dtype === 'bool' ? (a ? 1 : 0) : a)
    }
  }
  recurse(arr, 0)

  return { data: numericTypedArray(flat, dtype), shape }
}

// Pinned Tensor.__init__ infers list/tuple inputs as bool, default_int, or
// default_float from their flattened values (tensor.py:96-100).
function inferArrayDtype(arr) {
  let sawValue = false
  let allBool = true
  let allInt = true
  const visit = (value) => {
    if (Array.isArray(value)) {
      for (const item of value) visit(item)
      return
    }
    if (typeof value !== 'boolean' && typeof value !== 'number') {
      throw new Error(`Cannot infer dtype from value of type ${typeof value}`)
    }
    sawValue = true
    allBool = allBool && typeof value === 'boolean'
    allInt = allInt && (typeof value === 'boolean' || Number.isInteger(value))
  }
  visit(arr)
  if (!sawValue) return 'float32'
  if (allBool) return 'bool'
  return allInt ? 'int32' : 'float32'
}

function typedArrayDtype(data) {
  if (data instanceof Int8Array) return 'int8'
  if (data instanceof Uint8Array || data instanceof Uint8ClampedArray) return 'uint8'
  if (data instanceof Int16Array) return 'int16'
  if (data instanceof Uint16Array) return 'uint16'
  if (data instanceof Int32Array) return 'int32'
  if (data instanceof Uint32Array) return 'uint32'
  if (data instanceof BigInt64Array) return 'int64'
  if (data instanceof BigUint64Array) return 'uint64'
  if (data instanceof Float32Array) return 'float32'
  if (data instanceof Float64Array) return 'float64'
  return null
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
  return ['weakint', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'].includes(dtype)
}

function isFloatDtype(dtype) {
  return ['weakfloat', 'fp8e4m3', 'fp8e5m2', 'fp8e4m3fnuz', 'fp8e5m2fnuz',
    'float16', 'bfloat16', 'float32', 'float64'].includes(dtype)
}

function isFp8Dtype(dtype) {
  return ['fp8e4m3', 'fp8e5m2', 'fp8e4m3fnuz', 'fp8e5m2fnuz'].includes(dtype)
}

// Pinned tinygrad dtype.py:274-278. This is the induced sum_acc_dtype
// lattice over Polygrad's currently supported scalar dtypes.
function sumAccumulatorDtype(dtype) {
  if (dtype === 'uint64') return 'uint64'
  if (['uint8', 'uint16', 'uint32'].includes(dtype)) return 'uint32'
  if (dtype === 'int64') return 'int64'
  if (dtype === 'bool' || ['int8', 'int16', 'int32'].includes(dtype)) return 'int32'
  if (dtype === 'float64') return 'float64'
  return 'float32'
}

function createBoundTensorClass(runtime) {
  const _runtime = runtime
  const lifetime = _runtime._lifetime
  const DTYPE_ID = _runtime._core.dtypeIds
  if (!DTYPE_ID) throw new Error('polygrad: core missing dtypeIds')
  const DTYPE_NAME_BY_ID = new Map(
    Object.entries(DTYPE_ID).map(([name, id]) => [Number(id), name])
  )
  const ffi = _runtime._core.ffi
  const ops = _runtime._core.ops || {}
  const liveCore = () => {
    if (!lifetime.alive || !_runtime._core) {
      throw new Error('polygrad runtime has been disposed')
    }
    return _runtime._core
  }
  const POLY_TENSOR_VALUE = 0
  const normalizeDevice = (device) => String(device || _runtime.device || 'cpu').toLowerCase()
  const deviceId = (device) => ffi.poly_device_by_name(normalizeDevice(device))
  const rejectRequiresGrad = (opts) => {
    if (opts && (Object.prototype.hasOwnProperty.call(opts, 'requiresGrad') ||
                 Object.prototype.hasOwnProperty.call(opts, 'requires_grad'))) {
      throw new TypeError('Tensor does not accept requiresGrad; use is_param_ for optimizer selection')
    }
  }
  const tensorCreateWithRoots = (ctx, logical, physical, role, device) =>
    ffi.poly_tensor_create_with_roots(
      ctx, rawUop(logical), physical ? rawUop(physical) : null, role, deviceId(device)
    )
  const tensorCreateResultLike = (ctx, input, logical, physical, role, device) =>
    ffi.poly_tensor_create_result_like(
      ctx, input, logical ? rawUop(logical) : null, rawUop(physical), role, deviceId(device)
    )
  const tensorUop = (tensor) => ffi.poly_tensor_uop(tensor)
  const tensorUopPhysical = (tensor) => ffi.poly_tensor_uop_physical(tensor)
  const tensorUopLogical = (tensor) => ffi.poly_tensor_uop_logical(tensor)
  const tensorDevice = (tensor) => ffi.poly_tensor_device(tensor)
  const tensorFinalizer = typeof FinalizationRegistry === 'undefined'
    ? null
    : new FinalizationRegistry((owner) => {
        const pending = releaseTensorOwner(owner)
        if (pending && typeof pending.then === 'function') pending.catch(() => {})
      })
  function releaseTensorOwner(owner, token = null) {
    if (!owner || !owner.active) return undefined
    owner.active = false
    if (tensorFinalizer && token) tensorFinalizer.unregister(token)
    if (!owner.state.alive) return undefined
    const release = () => owner.ffi.poly_tensor_release(owner.tensor)
    if (owner.state.asyncHost && owner.state.core && owner.state.core.enqueueAsync) {
      return owner.state.core.enqueueAsync(release)
    }
    release()
    return undefined
  }
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
  const customKernelGradFxns = new Map()
  const customGradRecordsFor = (ctx, root) => {
    const rootRaw = rawUop(root)
    if (!rootRaw) return []
    const topo = []
    const seen = new Set()
    const stack = [{ node: rootRaw, expanded: false }]
    while (stack.length) {
      const { node, expanded } = stack.pop()
      const key = uopKey(node)
      if (expanded) {
        topo.push(node)
        continue
      }
      if (!node || seen.has(key)) continue
      seen.add(key)
      stack.push({ node, expanded: true })
      for (let i = Number(ffi.poly_uop_n_src(node)) - 1; i >= 0; i--) {
        const src = ffi.poly_uop_src(node, i)
        if (src) stack.push({ node: src, expanded: false })
      }
    }

    const byCall = new Map()
    for (const after of topo) {
      if (Number(ffi.poly_uop_op(after)) !== ops.AFTER || Number(ffi.poly_uop_n_src(after)) !== 2) {
        continue
      }
      const data = ffi.poly_uop_src(after, 0)
      const call = ffi.poly_uop_src(after, 1)
      if (!call || Number(ffi.poly_uop_op(call)) !== ops.CALL) continue
      const body = ffi.poly_uop_src(call, 0)
      if (!body || Number(ffi.poly_uop_op(body)) !== ops.SINK) continue
      let arg = null
      for (let i = 1; i < Number(ffi.poly_uop_n_src(call)); i++) {
        const candidate = ffi.poly_uop_src(call, i)
        if (uopKey(candidate) === uopKey(data)) {
          arg = candidate
          break
        }
      }
      if (!arg) continue
      const callKey = uopKey(call)
      let entry = byCall.get(callKey)
      if (!entry) {
        const gradKey = Number(ffi.poly_uop_call_grad_fxn_key(call)) >>> 0
        const gradFxn = gradKey ? customKernelGradFxns.get(gradKey) : null
        if (gradKey && !gradFxn) throw new Error('customKernel gradient callback is unavailable')
        entry = { rec: { gradFxn }, active: [] }
        byCall.set(callKey, entry)
      }
      if (!entry.active.some(existing => uopKey(existing.after) === uopKey(after))) {
        entry.active.push({ after, arg, call })
      }
    }
    return Array.from(byCall.values())
  }
  const gradResultRaw = (grad) => {
    if (!grad) return null
    if (grad instanceof Tensor) return grad._currentUopRaw()
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
      if (!lifetime.alive || !_runtime._core) {
        throw new Error('polygrad runtime has been disposed')
      }
      const options = opts ? { ...opts } : {}
      const policy = normalizeLogicalPolicy(options.logical)
      delete options.logical
      if (policy === null) {
        this._initialize(data, options)
        return
      }
      if (!ffi.poly_ctx_set_logical_policy || !ffi.poly_ctx_get_logical_policy) {
        throw new Error('logical policy requires current core support')
      }
      const oldPolicy = ffi.poly_ctx_get_logical_policy(options._ctx || _runtime._core.ctx)
      if (ffi.poly_ctx_set_logical_policy(options._ctx || _runtime._core.ctx, policy) !== 0) {
        throw new TypeError(`invalid logical policy ${String(opts.logical)}`)
      }
      try {
        this._initialize(data, options)
      } finally {
        if (ffi.poly_ctx_set_logical_policy(options._ctx || _runtime._core.ctx, oldPolicy) !== 0) {
          throw new Error('failed to restore logical policy')
        }
      }
    }

    _initialize(data, opts) {
      if (!lifetime.alive || !_runtime._core) {
        throw new Error('polygrad runtime has been disposed')
      }
      if (!opts) opts = {}
      rejectRequiresGrad(opts)
      const core = _runtime._core
      if (data instanceof UOp && !opts._uop) {
        if (data.ctx !== core.ctx || data.ffi !== core.ffi) {
          throw new Error('Tensor UOp must belong to the same Polygrad context')
        }
        if (opts._ctx && opts._ctx !== data.ctx) {
          throw new Error('Tensor UOp must belong to the same Polygrad context')
        }
        opts = Object.assign({}, opts, { _uop: data, _ctx: data.ctx })
        data = null
      }
      this._rt = _runtime
      this._ctx = opts._ctx || core.ctx
      this._grad = null
      this._isParam = opts.isParam ?? opts.is_param ?? opts._isParam ?? true
      this._isParam = Boolean(this._isParam)
      this._device = normalizeDevice(opts._device || opts.device || _runtime.device || 'cpu')
      this._tensor = null
      this._tensorOwner = null
      if (opts._tensor) this._adoptCoreTensor(opts._tensor)
      let currentUop = null
      let importedTensorFromHost = false
      const optUop = opts._uop || (this._tensor ? tensorUop(this._tensor) : null)

      if (optUop) {
        // Internal construction from ops -- shape is on the UOp.
        // Accept either a UOp wrapper instance or a raw handle.
        const raw = rawUop(optUop)
        currentUop = optUop instanceof UOp ? optUop : new UOp(this._ctx, core.ffi, raw, false)
        this._data = opts._data || null
        this._dtype = dtypeNameForUop(this._ctx, raw, opts._dtype)
      } else {
        const noneData = data === null
        if (noneData) data = 0
        const scalarData = typeof data === 'number' || typeof data === 'boolean' || typeof data === 'bigint'
        if (!scalarData && (opts.dtype === 'weakint' || opts.dtype === 'weakfloat')) {
          throw new Error(`cannot create storage for weak dtype ${opts.dtype}`)
        }
        // User construction from data. Resolve dtype and flatten to a single
        // TypedArray. Then call UOp.fromHost (one FFI call) which creates
        // the BUFFER UOp, registers a PolyBuffer wrapping the TypedArray's
        // bytes in ctx->buffers, and wraps in RESHAPE when ndim > 1.
        // JS also keeps a strong owner entry keyed by the C-side PolyBuffer*
        // address value, not by the BUFFER UOp.
        let dt, flat, shape
        if (scalarData) {
          dt = opts.dtype || (
            noneData ? 'weakfloat' : typeof data === 'boolean' ? 'bool' : (typeof data === 'bigint' || Number.isInteger(data)) ? 'weakint' : 'weakfloat'
          )
          const dtypeId = DTYPE_ID[dt]
          if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dt}`)
          const targetDeviceId = deviceId(this._device)
          if (dt === 'bool' || isIntegerDtype(dt)) {
            const value = dt === 'bool' ? (data ? 1 : 0) : typeof data === 'bigint' ? data : Math.trunc(Number(data))
            if (typeof value === 'bigint' && (value < -(1n << 63n) || value >= (1n << 64n))) throw new RangeError('integer literal out of 64-bit range')
            const factory = typeof value === 'bigint' && value >= (1n << 63n) ? ffi.poly_tensor_const_uint_by_id : ffi.poly_tensor_const_int_by_id
            this._adoptCoreTensor(factory(
              this._ctx, value, dtypeId, targetDeviceId
            ))
          } else {
            this._adoptCoreTensor(ffi.poly_tensor_const_float_by_id(
              this._ctx, Number(data), dtypeId, targetDeviceId
            ))
          }
          if (!this._tensor) throw new Error('C-owned scalar Tensor construction failed')
          const physical = tensorUopPhysical(this._tensor)
          if (!physical) throw new Error('scalar Tensor has no physical root')
          currentUop = new UOp(this._ctx, core.ffi, physical, false)
          this._dtype = dt
          this._data = null
        } else if (ArrayBuffer.isView(data) && !(data instanceof DataView)) {
          dt = (opts && opts.dtype) || typedArrayDtype(data) || 'float32'
          // Pinned UOp._frompy stages numeric BF16 values as float32 bytes and
          // then casts the Tensor graph (uop/ops.py:752-764). Uint16Array here
          // would truncate the numeric values before the graph-level cast.
          flat = numericTypedArray(data, dt === 'bfloat16' || isFp8Dtype(dt) ? 'float32' : dt)
          shape = [data.length]
        } else {
          dt = (opts && opts.dtype) || inferArrayDtype(data)
          const r = flattenArray(data, dt === 'bfloat16' || isFp8Dtype(dt) ? 'float32' : dt)
          flat = r.data; shape = r.shape
        }
        if (!scalarData) {
          this._dtype = dt
          const postCastDtype = dt === 'bfloat16' || isFp8Dtype(dt) ? dt : null
          this._data = flat
          const importDtype = postCastDtype ? 'float32' : dt
          const dtypeId = DTYPE_ID[importDtype]
          if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dt}`)
          // Preserve one-dimensional zero shapes. Without the explicit [0],
          // poly_buffer_from_host cannot distinguish an empty vector from an
          // unspecified/scalar host buffer and applies its scalar numel fallback.
          const dims = shape.length ? shape : null
          let ownerUop = null
          this._adoptCoreTensor(ffi.poly_tensor_from_host_by_id(
            this._ctx, flat, flat.byteLength, dtypeId, dims, dims ? dims.length : 0
          ))
          if (!this._tensor) throw new Error('poly_tensor_from_host_by_id failed')
          const physical = tensorUopPhysical(this._tensor)
          if (!physical) throw new Error('host Tensor source has no physical root')
          currentUop = new UOp(this._ctx, core.ffi, physical, false)
          ownerUop = currentUop
          if (postCastDtype) {
            const castTensor = ffi.poly_tensor_cast_by_id(
              this._ctx, this._tensor, DTYPE_ID[postCastDtype]
            )
            if (!castTensor) {
              throw new Error(`poly_tensor_cast_by_id failed for dtype ${postCastDtype}`)
            }
            this._adoptCoreTensor(castTensor)
            const castPhysical = tensorUopPhysical(this._tensor)
            if (!castPhysical) throw new Error('cast host Tensor has no physical root')
            currentUop = new UOp(this._ctx, core.ffi, castPhysical, false)
          }
          importedTensorFromHost = true
          const buffer = ownerUop
            ? core.ffi.poly_uop_buffer(this._ctx, ownerUop.raw)
            : null
          const needsFrontendHostOwner =
            Boolean(buffer && core.ffi.poly_buffer_get_key) &&
            (!core.caps || core.caps.core !== 'wasm')
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
          const movedTensor = ffi.poly_tensor_to_device(this._ctx, this._tensor, targetDeviceId)
          if (!movedTensor) throw new Error(`poly_tensor_to_device failed for ${this._device}`)
          this._adoptCoreTensor(movedTensor)
        }
      } else if (!this._tensor && currentUop) {
        this._adoptCoreTensor(
          this._coreCreate(currentUop.raw, POLY_TENSOR_VALUE, this._device)
        )
      }
      registerTensor(this)
    }

    _adoptCoreTensor(tensor) {
      if (!tensor) return null
      // Tensor-producing C APIs return one owned reference, including identity
      // returns. Releasing the prior owner first transfers that reference here.
      if (this._tensorOwner) {
        const pending = releaseTensorOwner(this._tensorOwner, this)
        if (pending && typeof pending.then === 'function') pending.catch(() => {})
      }
      this._tensor = tensor
      this._tensorOwner = { state: lifetime, ffi, tensor, active: true }
      if (tensorFinalizer) tensorFinalizer.register(this, this._tensorOwner, this)
      return tensor
    }

    _takeCoreTensor() {
      const tensor = this._tensor
      if (this._tensorOwner) {
        this._tensorOwner.active = false
        if (tensorFinalizer) tensorFinalizer.unregister(this)
      }
      this._tensorOwner = null
      this._tensor = null
      return tensor
    }

    dispose() {
      const owner = this._tensorOwner
      this._tensorOwner = null
      this._tensor = null
      return releaseTensorOwner(owner, this)
    }

    _coreCreate(uop, role, device) {
      // Pinned Tensor.__init__ stores a supplied UOp directly
      // (tensor.py:76-121). The C boundary records that exact current root.
      return tensorCreateWithRoots(
        this._ctx, uop, uop, role, device || this._device
      )
    }

    _coreCreateWithRoots(logical, physical, role, device) {
      return tensorCreateWithRoots(this._ctx, logical, physical, role, device || this._device)
    }

    _requireLive() {
      if (!lifetime.alive || !_runtime._core) {
        throw new Error('polygrad runtime has been disposed')
      }
    }

    _currentUopRaw() {
      this._requireLive()
      return this._tensor ? tensorUop(this._tensor) : null
    }

    _logicalUopRaw() {
      this._requireLive()
      return this._tensor ? tensorUopLogical(this._tensor) : null
    }

    _physicalUopRaw() {
      this._requireLive()
      return this._tensor ? tensorUopPhysical(this._tensor) : null
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
    get _buffer() {
      const raw = this._currentUopRaw()
      return raw ? ffi.poly_uop_buffer(this._ctx, raw) : null
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
    get logicalPolicy() {
      return logicalPolicyName(ffi.poly_tensor_logical_policy(this._tensor))
    }
    get logicalState() {
      return logicalStateName(ffi.poly_tensor_logical_state(this._tensor))
    }
    setLogicalPolicy(policy) {
      return ffi.poly_tensor_set_logical_policy(
        this._ctx, this._tensor, normalizeLogicalPolicy(policy)
      ) === 0
    }
    preserveLogical() {
      if (!this.setLogicalPolicy('always')) {
        throw new Error('logical producer is no longer available')
      }
      return this
    }

    get shape() {
      const { ffi } = this._rt._core
      if (!this._uop) return []
      return ffi.poly_uop_max_shape_dims(this._ctx, this._uop)
    }
    get dtype() { return this._dtype }

    elementSize() {
      // Pinned DTypeMixin.element_size rejects weak types before reading width.
      if (this.dtype === 'weakint' || this.dtype === 'weakfloat') {
        throw new Error(`elementSize requires a concrete dtype, got ${this.dtype}`)
      }
      return TA_BY_DTYPE[this.dtype].BYTES_PER_ELEMENT
    }

    isFloatingPoint() {
      return this.dtype === 'weakfloat' || this.dtype === 'bfloat16' ||
        this.dtype.startsWith('float') || isFp8Dtype(this.dtype)
    }
    get device() {
      if (!this._tensor) throw new Error('Tensor has no core PolyTensor')
      /* Public device is a placement label. The WASM core maps public CPU onto
       * the WASM execution backend internally, but user-facing Tensor.device
       * should stay CPU for parity with Python/tinygrad-style APIs. */
      return this._device.toUpperCase()
    }
    get ndim() { return this._rt._core.ffi.poly_uop_ndim(this._ctx, this._uop) || 0 }
    get isParam() { return this._isParam }
    set isParam(v) { this._isParam = Boolean(v) }
    get is_param() { return this._isParam }
    set is_param(v) { this._isParam = Boolean(v) }
    is_param_(isParam = true) {
      this._isParam = Boolean(isParam)
      return this
    }
    get grad() { return this._grad }
    // Tensor.grad owns another Tensor, whose C handle owns its graph roots.
    set grad(value) { this._grad = value }
    get T() { return this.transpose() }

    numel() {
      return this.shape.reduce((a, b) => a * b, 1)
    }

    // JS shape already exposes the core's maximum extents.
    get maxShape() { return this.shape }
    maxNumel() { return this.maxShape.reduce((a, b) => a * b, 1) }

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
      /* Pinned tensor.py:527-543 discovers targets only through the current
       * Tensor.uop. Polygrad's corresponding root is mandatory physical. */
      for (const t of liveTensorSnapshot()) {
        if (!t || t._ctx !== this._ctx || !t._tensor || !isFloatDtype(t.dtype)) continue
        const current = t._currentUopRaw()
        if (current && Number(ffi.poly_uop_device(current)) !== deviceId('auto') &&
            ffi.poly_uop_reachable(this._ctx, root, current)) {
          targets.push({ tensor: t, root: current })
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

    async _realizeAsyncUnleased(...lst) {
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

    realizeAsync(...lst) {
      return this._rt._withAsync(() => this._realizeAsyncUnleased(...lst))
    }

    _readBufferBytesWith(readBuffer) {
      const { ffi, ctx } = this._rt._core
      const numel = this.numel()
      const AT = TA_BY_DTYPE[this._dtype] || Float32Array
      const itemsize = AT.BYTES_PER_ELEMENT
      const current = this._currentUopRaw()
      const bufRaw = current ? ffi.poly_uop_buffer(this._ctx, current) : null
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
      const current = this._currentUopRaw()
      const bufRaw = current ? ffi.poly_uop_buffer(this._ctx, current) : null
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
      if (this._dtype === 'weakint') t = t.cast('int32')
      if (this._dtype === 'weakfloat' || this._dtype === 'float16' ||
          this._dtype === 'bfloat16' || isFp8Dtype(this._dtype)) t = t.cast('float32')
      t = t.contiguous()
      // Pinned tensor.py:259-266 clones a device-free source to CPU for
      // readback. Wrapper device metadata is not executable UOp placement.
      if (Number(ffi.poly_uop_device(this._currentUopRaw())) === deviceId('auto')) {
        t = t.clone('cpu')
      }
      t.realize()
      return t._readBufferBytes()
    }

    async _toArrayAsyncUnleased() {
      const numel = this.numel()
      if (numel === 0) {
        const AT = TA_BY_DTYPE[this._dtype] || Float32Array
        return new AT(0)
      }
      let t = this
      if (this._dtype === 'weakint') t = t.cast('int32')
      if (this._dtype === 'weakfloat' || this._dtype === 'float16' ||
          this._dtype === 'bfloat16' || isFp8Dtype(this._dtype)) t = t.cast('float32')
      t = t.contiguous()
      if (Number(ffi.poly_uop_device(this._currentUopRaw())) === deviceId('auto')) {
        t = t.clone('cpu')
      }
      await t._realizeAsyncUnleased()
      return await t._readBufferBytesAsync()
    }

    toArrayAsync() {
      return this._rt._withAsync(() => this._toArrayAsyncUnleased())
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
        if (out._dtype === 'float16' || out._dtype === 'bfloat16' || isFp8Dtype(out._dtype))
          out = out.cast('float32')
        out = out.contiguous()
        if (Number(ffi.poly_uop_device(t._currentUopRaw())) === deviceId('auto')) {
          out = out.clone('cpu')
        }
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
      const rt = tensors[0]._rt
      for (const t of tensors) {
        if (t._rt !== rt) throw new Error('Tensor.toTypedArraysAsync tensors must share a runtime')
      }
      return rt._withAsync(async () => {
        const prepared = tensors.map(t => {
          if (t.numel() === 0) return t
          let out = t
          if (out._dtype === 'float16' || out._dtype === 'bfloat16' || isFp8Dtype(out._dtype))
            out = out.cast('float32')
          out = out.contiguous()
          if (Number(ffi.poly_uop_device(t._currentUopRaw())) === deviceId('auto')) {
            out = out.clone('cpu')
          }
          return out
        })
        const targets = prepared.filter(t => t.numel() !== 0)
        if (targets.length) await targets[0]._realizeAsyncUnleased(...targets.slice(1))
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
      })
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
      // Pinned mixin/elementwise.py:33-37 is one DETACH Tensor ALU.
      const { ffi } = this._rt._core
      const core = ffi.poly_tensor_detach(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    contiguousBackward() {
      // Pinned mixin/elementwise.py:51-55 is one CONTIGUOUS_BACKWARD UOp;
      // C owns both retained and executable Tensor roots.
      const { ffi } = this._rt._core
      const core = ffi.poly_tensor_contiguous_backward(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    contiguous_backward() { return this.contiguousBackward() }

    detachAsync() {
      return this._rt._withAsync(() => this.detach())
    }

    clone(device) {
      const dev = normalizeDevice(device == null ? this._device : device)
      const cloned = ffi.poly_tensor_clone(this._ctx, this._tensor, deviceId(dev))
      if (!cloned) throw new Error('poly_tensor_clone failed')
      const t = new Tensor(undefined, {
        _ctx: this._ctx, _tensor: cloned, _dtype: this._dtype, _device: dev
      })
      t._isParam = this._isParam
      if (this._grad) t._grad = this._grad.clone(dev)
      return t
    }

    cloneAsync(device) {
      return this._rt._withAsync(() => this.clone(device))
    }

    assign(x) {
      if (!(x instanceof Tensor)) x = new Tensor(x, { dtype: this._dtype, device: this._device })
      if (!arraysEqual(this.shape, x.shape)) {
        const outShape = _broadcastShapes(x.shape, this.shape)
        if (!arraysEqual(outShape, this.shape)) {
          throw new Error(`assign shape mismatch [${this.shape}] != [${x.shape}]`)
        }
      }
      if (this._device !== x._device) {
        throw new Error(`assign device mismatch ${this.device} != ${x.device}`)
      }
      if (this._dtype !== x._dtype) {
        throw new Error(`assign dtype mismatch ${this._dtype} != ${x._dtype}`)
      }
      const assigned = ffi.poly_tensor_assign(this._ctx, this._tensor, x._tensor)
      if (!assigned) throw new Error('poly_tensor_assign failed')
      this._data = null
      return this
    }

    _copyFromData(data) {
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
      return view
    }

    _writeCurrentBuffer(view) {
      const physical = this._physicalUopRaw()
      const writeBuf = physical ? ffi.poly_uop_buffer(this._ctx, physical) : null
      if (!writeBuf) throw new Error('copyFrom requires a tensor backed by a BUFFER UOp')
      const targetDevice = deviceId(this._device)
      ffi.poly_buffer_ensure_device_allocated(this._ctx, writeBuf, targetDevice)
      ffi.poly_buffer_write(this._ctx, writeBuf, view)
      this._data = null
      return this
    }

    copyFrom(data) {
      const view = this._copyFromData(data)
      const physical = this._physicalUopRaw()
      if (!physical) throw new Error('copyFrom requires a physical Tensor root')
      // Pinned Tensor._buffer -> Buffer.copy_from finishes pending effects.
      // A recursive buffer lookup also crosses AFTER and is not proof that
      // those effects ran. Direct writes must never republish Tensor roots.
      if (!ffi.poly_uop_has_buffer_identity(physical)) {
        requireSyncHostBridge('copyFrom()', 'copyFromAsync()')
        this.realize()
      }
      return this._writeCurrentBuffer(view)
    }

    copyFromAsync(data) {
      return this._rt._withAsync(async () => {
        // Realization may suspend; do not borrow caller bytes across the await.
        const view = this._copyFromData(data).slice()
        await this._realizeAsyncUnleased()
        return this._writeCurrentBuffer(view)
      })
    }

    updateFrom(data) {
      return this.copyFrom(data)
    }

    to(device) {
      const dev = normalizeDevice(device)
      if (dev === this._device) return this
      if (Number(ffi.poly_uop_device(this._currentUopRaw())) === deviceId('auto')) return this
      const coreTensor = this._coreToDevice(dev)
      const t = new Tensor(null, {
        _ctx: this._ctx,
        _tensor: coreTensor,
        _data: this._data,
        _dtype: this._dtype,
        _device: dev
      })
      t._grad = this._grad ? this._grad.to(dev) : null
      t._isParam = this._isParam
      return t
    }

    to_(device) {
      const moved = this.to(device)
      if (moved === this) return this
      this._adoptCoreTensor(moved._takeCoreTensor())
      this._data = moved._data
      this._dtype = moved._dtype
      this._device = moved._device
      this._grad = moved._grad
      this._isParam = moved._isParam
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
      /* Pinned Tensor.contiguous -> UOp.contiguous (tensor.py:742-746,
       * uop/ops.py:587-591). C owns both retained/current roots. */
      const core = ffi.poly_tensor_contiguous(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    // --- Internal helpers ---

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
        const graph = t._currentUopRaw()
        return graph && ffi.poly_uop_op && ffi.poly_uop_op(graph) === ops.AFTER ? t : t.contiguous()
      })
      const placeholders = contig.map(
        (t, i) => UOp.placeholderLike(new UOp(t._ctx, ffi, t._currentUopRaw(), false), i)
      )
      const body = fxn(...placeholders)
      if (!(body instanceof UOp)) throw new TypeError('customKernel function must return a UOp SINK body')
      const gradFxnKey = customKernelGradKey(gradFxn)
      if (gradFxnKey) customKernelGradFxns.set(gradFxnKey, gradFxn)
      const cores = ffi.poly_tensor_custom_kernel(
        this._ctx, body.raw, contig.map(t => t._tensor), gradFxnKey
      )
      if (!Array.isArray(cores) || cores.length !== contig.length || cores.some(x => !x)) {
        throw new Error('poly_tensor_custom_kernel failed')
      }
      const physicalAfters = []
      const outs = contig.map((t, i) => {
        const physical = tensorUopPhysical(cores[i])
        physicalAfters.push(physical)
        return new Tensor(null, {
          _ctx: t._ctx,
          _tensor: cores[i],
          _dtype: t._dtype,
          _device: t._device
        })
      })
      const call = ffi.poly_uop_src(physicalAfters[0], 1)
      if (!call || Number(ffi.poly_uop_op(call)) !== ops.CALL) {
        throw new Error('customKernel physical output is not AFTER(data, CALL)')
      }
      if ((Number(ffi.poly_uop_call_grad_fxn_key(call)) >>> 0) !== gradFxnKey) {
        throw new Error('customKernel CALL lost its gradient identity')
      }
      return outs
    }

    custom_kernel(...args) { return this.customKernel(...args) }

    _ensureTensor(other) {
      if (other instanceof Tensor) return other
      if (typeof other === 'bigint') return new Tensor(other, { _ctx: this._ctx, _device: this._device })
      if (typeof other === 'number' || typeof other === 'boolean') {
        const dtype = typeof other === 'boolean' ? 'bool'
          : Number.isInteger(other) ? 'weakint' : 'weakfloat'
        const dtypeId = DTYPE_ID[dtype]
        if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dtype}`)
        const tensor = dtype === 'bool' || isIntegerDtype(dtype)
          ? ffi.poly_tensor_const_int_by_id(
              this._ctx, dtype === 'bool' ? (other ? 1 : 0) : Math.trunc(Number(other)),
              dtypeId, deviceId(this._device)
            )
          : ffi.poly_tensor_const_float_by_id(
              this._ctx, Number(other), dtypeId, deviceId(this._device)
            )
        if (!tensor) throw new Error('C-owned internal scalar Tensor construction failed')
        return new Tensor(null, {
          _ctx: this._ctx, _tensor: tensor, _dtype: dtype, _device: this._device
        })
      }
      throw new TypeError(`Cannot convert ${typeof other} to Tensor`)
    }

    constLike(value) {
      // Pinned uop/ops.py:581-583: one typed scalar CONST broadcast to this
      // Tensor's exact shape. C owns the shared Python/JS graph construction.
      if (typeof value !== 'number' && typeof value !== 'boolean') {
        throw new TypeError(`constLike value must be numeric, got ${typeof value}`)
      }
      const core = typeof value === 'boolean' || Number.isInteger(value)
        ? this._rt._core.ffi.poly_tensor_const_like_int(
            this._ctx, this._tensor, value === true ? 1 : value === false ? 0 : value
          )
        : this._rt._core.ffi.poly_tensor_const_like_float(
            this._ctx, this._tensor, Number(value)
          )
      return this._makeResultFromCore(core, [this])
    }

    const_like(value) { return this.constLike(value) }

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

    _broadcastTensor(targetShape) {
      return this.expand(targetShape)
    }

    _binop(other, opName, reverse = false) {
      const { ffi, ops } = this._rt._core
      other = this._ensureTensor(other)
      let x = reverse ? other : this
      let y = reverse ? this : other
      const core = ffi.poly_tensor_alu2(this._ctx, ops[opName], x._tensor, y._tensor)
      return this._makeResultFromCore(core, [x, y])
    }

    // --- Element-wise arithmetic ---

    // Pinned named add/mul preserve the optional reverse operand order
    // (mixin/elementwise.py:72-88,110-126).
    add(other, reverse = false) { return this._binop(other, 'ADD', reverse) }
    sub(other) {
      // C owns tinygrad's `a + (-b)` topology for every frontend
      // (mixin/elementwise.py:90-109).
      return this._binop(other, 'SUB')
    }
    mul(other, reverse = false) { return this._binop(other, 'MUL', reverse) }
    floorDiv(other) { return this.div(other, 'floor') }
    bitwiseNot() {
      const core = this._rt._core.ffi.poly_tensor_bitwise_not(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }
    mod(other) {
      const b = this._ensureTensor(other)
      if (isIntegerDtype(this._dtype) && isIntegerDtype(b._dtype)) return this._binop(b, 'FLOORMOD')
      return this.sub(this.div(b, 'floor').mul(b))
    }
    fmod(other) {
      const b = this._ensureTensor(other)
      if (isIntegerDtype(this._dtype) && isIntegerDtype(b._dtype)) return this._binop(b, 'CMOD')
      return this.sub(this.div(b, 'trunc').mul(b))
    }
    maskedFill(mask, value) { return this._ensureTensor(mask).where(value, this) }
    div(other, roundingMode = null) {
      // Pinned mixin/elementwise.py:219-247 selects integer CDIV/FLOORDIV
      // after promotion; floating rounding composes over true division.
      const rhs = this._ensureTensor(other)
      const rounding = [null, 'trunc', 'floor'].indexOf(roundingMode)
      if (rounding < 0) throw new Error(`rounding_mode='${roundingMode}' is not supported`)
      const core = this._rt._core.ffi.poly_tensor_div(
        this._ctx, this._tensor, rhs._tensor, rounding
      )
      return this._makeResultFromCore(core, [this, rhs])
    }
    pow(other, reverse = false) {
      // Tinygrad 2026-08-22/a9069c177a9d mixin/elementwise.py:545-564
      // validates scalar integer POW after promotion and preserves ordering.
      const scalar = !(other instanceof Tensor)
      const result = this._binop(other, 'POW', reverse)
      const nonnegativeInteger = typeof other === 'boolean' ||
        (typeof other === 'number' && Number.isInteger(other) && other >= 0)
      if (!isFloatDtype(result.dtype) && scalar && !nonnegativeInteger) {
        throw new Error('base needs to be float')
      }
      return result
    }
    lt(other) { return this._binop(other, 'CMPLT') }

    neg() {
      if (this._dtype === 'bool') return this.cast('bool').ne(true)
      return this.mul(-1)
    }

    // --- Comparisons (C core) ---

    eq(other) {
      // Pinned mixin/elementwise.py:315-322: promoted CMPNE + logical_not.
      return this._binop(other, 'CMPNE').ne(true)
    }

    ne(other) {
      return this._binop(other, 'CMPNE')
    }

    gt(other) {
      // Pinned mixin/elementwise.py:312-313: reversed promoted CMPLT.
      return this._binop(other, 'CMPLT', true)
    }

    ge(other) {
      return this.lt(other).ne(true)
    }

    le(other) {
      return this.gt(other).ne(true)
    }

    where(x, y) {
      const { ffi, ops } = this._rt._core
      // Current ElementwiseMixin.where uses one branch Tensor only to wrap
      // host scalars. C owns promotion and broadcast shape inference.
      const ref = x instanceof Tensor ? x : y instanceof Tensor ? y : this
      if (!(x instanceof Tensor)) x = ref._ensureTensor(x)
      if (!(y instanceof Tensor)) y = ref._ensureTensor(y)
      const core = ffi.poly_tensor_alu3(this._ctx, ops.WHERE, this._tensor, x._tensor, y._tensor)
      return this._makeResultFromCore(core, [this, x, y])
    }

    maximum(other) {
      return this._binop(other, 'MAX')
    }

    minimum(other) {
      const { ffi } = this._rt._core
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const x = this._broadcastTensor(outShape)
      const y = other._broadcastTensor(outShape)
      const core = ffi.poly_tensor_minimum(this._ctx, x._tensor, y._tensor)
      return this._makeResultFromCore(core, [x, y])
    }

    clamp(lo, hi) {
      if (lo === undefined && hi === undefined) {
        throw new Error("at least one of 'lo' or 'hi' must not be undefined")
      }
      // Pinned clamp conditionally composes comparison/WHERE Tensor
      // operations; an omitted bound is not a finite sentinel
      // (mixin/elementwise.py:569-580).
      const ret = lo !== undefined ? this.lt(lo).where(lo, this) : this
      return hi !== undefined ? ret.gt(hi).where(hi, ret) : ret
    }

    // Pinned mixin/elementwise.py:582-584: clip is the public clamp alias.
    clip(lo, hi) { return this.clamp(lo, hi) }

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
      const core = ffi.poly_tensor_cast_by_id(this._ctx, this._tensor, id)
      if (!core) throw new Error(`poly_tensor_cast_by_id failed for dtype ${dtype}`)
      return this._makeResultFromCore(core, [this], dtype)
    }

    bitcast(dtype) {
      dtype = String(dtype).toLowerCase()
      if (['weakint', 'weakfloat'].includes(this._dtype) || ['weakint', 'weakfloat'].includes(dtype)) {
        throw new Error(`bitcast requires concrete dtypes, got ${this._dtype} -> ${dtype}`)
      }
      if (dtype === this._dtype) return this
      const id = DTYPE_ID[dtype]
      if (id === undefined) throw new Error(`unsupported bitcast target dtype: ${dtype}`)
      /* Current DTypeMixin.bitcast emits one raw BITCAST; C owns concrete
       * dtype validation and UOp._shape last-axis scaling
       * (mixin/dtype.py:35-50, uop/ops.py:404-411). */
      const core = this._rt._core.ffi.poly_tensor_bitcast_by_id(
        this._ctx, this._tensor, id
      )
      if (!core) throw new Error('unsupported size in bitcast')
      return this._makeResultFromCore(core, [this], dtype)
    }

    half() { return this.cast('float16') }
    double() { return this.cast('float64') }

    // --- Triu/Tril ---

    static _tri(r, c, diagonal = 0, device) {
      // Pinned mixin/__init__.py:310-311. Polygrad's optional device is
      // wrapper placement metadata only; arange remains a deviceless UOp.
      const opts = device === undefined ? {} : { device }
      return Tensor.arange(r, opts).unsqueeze(-1).add(diagonal).le(Tensor.arange(c, opts))
    }

    triu(diagonal = 0) {
      // Pinned mixin/__init__.py:313-334.
      const [r, c] = this.shape.slice(-2)
      return Tensor._tri(r, c, diagonal, this._device).where(this, this.constLike(0))
    }

    tril(diagonal = 0) {
      // Pinned mixin/__init__.py:336-349.
      const [r, c] = this.shape.slice(-2)
      return Tensor._tri(r, c, diagonal + 1, this._device).where(this.constLike(0), this)
    }

    // --- Unary math (C core composed ops) ---

    exp2() {
      const { ffi, ops } = this._rt._core
      // Current Tinygrad emits the raw ALU op; C owns least_upper_float.
      const core = ffi.poly_tensor_alu1(this._ctx, ops.EXP2, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    log2() {
      const { ffi, ops } = this._rt._core
      // Current Tinygrad emits the raw ALU op; C owns the result dtype.
      const core = ffi.poly_tensor_alu1(this._ctx, ops.LOG2, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    sqrt() {
      const { ffi, ops } = this._rt._core
      // Current Tinygrad emits the raw ALU op; C owns the result dtype.
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
      const core = this._rt._core.ffi.poly_tensor_exp(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    log() {
      const core = this._rt._core.ffi.poly_tensor_log(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    log10() {
      const core = this._rt._core.ffi.poly_tensor_log10(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    atanh() {
      const core = this._rt._core.ffi.poly_tensor_atanh(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    asinh() {
      const core = this._rt._core.ffi.poly_tensor_asinh(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    acosh() {
      const core = this._rt._core.ffi.poly_tensor_acosh(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    asin() {
      const core = this._rt._core.ffi.poly_tensor_asin(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    acos() {
      const core = this._rt._core.ffi.poly_tensor_acos(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    atan() {
      const core = this._rt._core.ffi.poly_tensor_atan(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    logsigmoid() {
      const core = this._rt._core.ffi.poly_tensor_logsigmoid(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    sinh() {
      const core = this._rt._core.ffi.poly_tensor_sinh(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    cosh() {
      const core = this._rt._core.ffi.poly_tensor_cosh(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    erf() {
      const core = this._rt._core.ffi.poly_tensor_erf(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    softsign() {
      const core = this._rt._core.ffi.poly_tensor_softsign(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    isfinite() {
      const core = this._rt._core.ffi.poly_tensor_isfinite(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    celu(alpha) {
      // The pin's default is floating 1.0; JS Number alone loses that distinction.
      alpha = alpha === undefined
        ? new Tensor(1, {dtype: 'weakfloat', _ctx: this._ctx, device: this._device})
        : this._ensureTensor(alpha)
      const core = this._rt._core.ffi.poly_tensor_celu(this._ctx, this._tensor, alpha._tensor)
      return this._makeResultFromCore(core, [this, alpha])
    }

    selu(alpha = 1.67326, gamma = 1.0507) {
      alpha = this._ensureTensor(alpha)
      gamma = this._ensureTensor(gamma)
      const core = this._rt._core.ffi.poly_tensor_selu(this._ctx, this._tensor, alpha._tensor, gamma._tensor)
      return this._makeResultFromCore(core, [this, alpha, gamma])
    }

    isclose(other, {rtol = 1e-5, atol = 1e-8, equalNan = false} = {}) {
      other = this._ensureTensor(other)
      rtol = this._ensureTensor(rtol)
      atol = this._ensureTensor(atol)
      const core = this._rt._core.ffi.poly_tensor_isclose(this._ctx, this._tensor, other._tensor, rtol._tensor, atol._tensor, !!equalNan)
      return this._makeResultFromCore(core, [this, other, rtol, atol])
    }

    copysign(other) {
      other = this._ensureTensor(other)
      const core = this._rt._core.ffi.poly_tensor_copysign(this._ctx, this._tensor, other._tensor)
      return this._makeResultFromCore(core, [this, other])
    }

    lerp(end, weight) {
      const scalarWeight = !(weight instanceof Tensor)
      end = this._ensureTensor(end)
      weight = this._ensureTensor(weight)
      const core = this._rt._core.ffi.poly_tensor_lerp(this._ctx, this._tensor, end._tensor, weight._tensor, scalarWeight)
      return this._makeResultFromCore(core, [this, end, weight])
    }

    _lossReductionId(reduction) {
      const id = ['none', 'sum', 'mean'].indexOf(reduction)
      if (id < 0) throw new RangeError("reduction must be 'none', 'sum', or 'mean'")
      return id
    }

    binaryCrossEntropyLogits(target, {reduction = 'mean', posWeight = null} = {}) {
      const id = this._lossReductionId(reduction)
      target = this._ensureTensor(target)
      const weight = posWeight === null ? null : this._ensureTensor(posWeight)
      const core = this._rt._core.ffi.poly_tensor_binary_crossentropy_logits(
        this._ctx, this._tensor, target._tensor, weight ? weight._tensor : null, id)
      return this._makeResultFromCore(core, [this, target, ...(weight ? [weight] : [])])
    }

    nllLoss(target, {weight = null, ignoreIndex = null, reduction = 'mean'} = {}) {
      const id = this._lossReductionId(reduction)
      target = this._ensureTensor(target)
      weight = weight === null ? null : this._ensureTensor(weight)
      const ignore = ignoreIndex === null ? null : this._ensureTensor(ignoreIndex)
      const core = this._rt._core.ffi.poly_tensor_nll_loss(
        this._ctx, this._tensor, target._tensor, weight ? weight._tensor : null, ignore ? ignore._tensor : null, id)
      return this._makeResultFromCore(core, [this, target, ...[weight, ignore].filter(Boolean)])
    }

    log1p() {
      const core = this._rt._core.ffi.poly_tensor_log1p(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    expm1() {
      const core = this._rt._core.ffi.poly_tensor_expm1(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    sin() {
      const { ffi, ops } = this._rt._core
      // Current Tinygrad emits SIN over the exact Tensor occurrence; C owns
      // its least_upper_float result (mixin/elementwise.py:468-478).
      const core = ffi.poly_tensor_alu1(this._ctx, ops.SIN, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    cos() {
      // Current least_upper_float/float32 composition is shared in C.
      const core = this._rt._core.ffi.poly_tensor_cos(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    tan() {
      // Current self.sin()/self.cos() composition is shared in C.
      const core = this._rt._core.ffi.poly_tensor_tan(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    sigmoid() {
      // Pinned mixin/elementwise.py:667-677.
      return this.mul(-1 / Math.log(2)).exp2().add(1, true).reciprocal()
    }

    tanh() {
      // Pinned mixin/elementwise.py:739-749.
      return this.mul(2, true).sigmoid().mul(2, true).sub(1)
    }

    abs() {
      // Pinned mixin/elementwise.py:892-900.
      return this.mul(this.sign())
    }

    sign() {
      // Pinned mixin/elementwise.py:882-890.
      return this.ne(0).where(
        this.lt(0).where(this.constLike(-1), this.constLike(1)),
        this.constLike(0)
      )
    }

    square() {
      // Pinned mixin/elementwise.py:558-566.
      return this.mul(this)
    }

    rsqrt() {
      // Pinned mixin/elementwise.py:802-810.
      return this.sqrt().reciprocal()
    }

    ceil() {
      // Pinned mixin/elementwise.py:636-644.
      const b = this.trunc()
      return this.gt(b).where(b.add(1), b)
    }

    floor() {
      // Pinned mixin/elementwise.py:646-654.
      const b = this.trunc()
      return this.lt(b).where(b.sub(1), b)
    }

    round() {
      // Pinned mixin/elementwise.py:872-880 implements round-half-to-even
      // through ordinary Tensor primitives and their scalar broadcast rules.
      // JavaScript Number erases the source spelling distinction between 2
      // and 2.0. Pinned UOp.ufix keeps it in the receiver dtype for floating
      // inputs and uses from_py(float)==float32 for integer/bool inputs
      // (uop/ops.py:504-508).
      const two = new Tensor(2.0, {
        _ctx: this._ctx,
        dtype: isFloatDtype(this._dtype) ? this._dtype : 'float32',
        device: this._device
      })
      const b = this.trunc().div(two)
      return this.gt(0).eq(b.trunc().eq(b)).where(
        this.sub(0.5).ceil(), this.add(0.5).floor()
      )
    }

    isinf(detectPositive = true, detectNegative = true) {
      // Pinned mixin/elementwise.py:596-604 independently gates positive
      // and negative infinity before adding the boolean results.
      return this.eq(Infinity).mul(detectPositive)
        .add(this.eq(-Infinity).mul(detectNegative))
    }

    isnan() {
      // Pinned mixin/elementwise.py:586-594.
      return this.ne(this)
    }

    // --- Activations (C core composed ops) ---

    relu() {
      // Pinned mixin/elementwise.py:656-666. Tensor comparison/where keeps
      // scalar-zero broadcasting and exact ordered physical occurrences.
      return this.gt(0).where(this, 0)
    }

    relu6() {
      // Pinned mixin/elementwise.py:679-689.
      return this.relu().sub(this.sub(6).relu())
    }

    leakyRelu(negSlope) {
      if (negSlope === undefined) negSlope = 0.01
      // Pinned mixin/elementwise.py:726-737.
      return this.lt(0).where(this.mul(negSlope, true), this)
    }

    gelu(approximate = 'tanh') {
      // Pinned mixin/elementwise.py:761-776. C owns retained/current roots.
      if (!['tanh', 'none'].includes(approximate)) throw new Error(`unknown GELU approximation: ${approximate}`)
      const fn = approximate === 'tanh' ? this._rt._core.ffi.poly_tensor_gelu : this._rt._core.ffi.poly_tensor_gelu_exact
      const core = fn(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    quickGelu() {
      // Pinned mixin/elementwise.py:751-759 builds the broadcasted scalar
      // formula from the one current Tensor.uop. C owns both retained/current roots.
      const core = this._rt._core.ffi.poly_tensor_quick_gelu(this._ctx, this._tensor)
      return this._makeResultFromCore(core, [this])
    }

    silu() {
      // Pinned mixin/elementwise.py:790-800.
      return this.swish()
    }

    swish() {
      // Pinned mixin/elementwise.py:778-788.
      return this.mul(this.sigmoid())
    }

    elu(alpha) {
      if (alpha === undefined) alpha = 1.0
      // Pinned mixin/elementwise.py:945-955.
      return this.relu().sub(this.exp()._binop(1, 'SUB', true).relu().mul(alpha, true))
    }

    logaddexp(other) {
      // Pinned mixin/elementwise.py:403-410.
      let b = this._ensureTensor(other)
      const outShape = this._broadcastShape(b.shape)
      const a = this._broadcastTensor(outShape)
      b = b._broadcastTensor(outShape)
      const m = a.maximum(b)
      return a.sub(m).exp().add(b.sub(m).exp()).log().add(m)
    }

    softplus(beta) {
      if (beta === undefined) beta = 1.0
      // Pinned mixin/elementwise.py:981-989.
      return this.mul(beta).logaddexp(0.0).mul(1 / beta, true)
    }

    mish() {
      // Pinned mixin/elementwise.py:991-1001.
      return this.mul(this.softplus().tanh())
    }

    hardtanh(minVal, maxVal) {
      if (minVal === undefined) minVal = -1
      if (maxVal === undefined) maxVal = 1
      // Pinned mixin/elementwise.py:716-724.
      return this.clamp(minVal, maxVal)
    }

    hardswish() {
      // Pinned mixin/elementwise.py:691-701.
      return this.mul(this.add(3).relu6()).mul(1 / 6)
    }

    hardsigmoid(alpha = 1 / 6, beta = 0.5) {
      // Pinned mixin/elementwise.py:703-714.
      const y = this.mul(alpha, true).add(beta)
      return y.relu().sub(y.sub(1).relu())
    }

    // --- Softmax ---

    softmax(axis) {
      if (axis === undefined) axis = -1
      this._resolveDim(axis)
      const core = this._rt._core.ffi.poly_tensor_softmax(this._ctx, this._tensor, axis)
      return this._makeResultFromCore(core, [this])
    }

    logSoftmax(axis) {
      if (axis === undefined) axis = -1
      this._resolveDim(axis)
      const core = this._rt._core.ffi.poly_tensor_log_softmax(this._ctx, this._tensor, axis)
      return this._makeResultFromCore(core, [this])
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
      order = order.map(axis => axis < 0 ? axis + this.shape.length : axis)
      if (order.length !== this.shape.length ||
          [...order].sort((a, b) => a - b).some((axis, index) => axis !== index)) {
        throw new Error(`order is not a valid permutation, getting (${order})`)
      }
      if (order.every((axis, index) => axis === index)) return this
      const core = this._rt._core.ffi.poly_tensor_permute(
        this._ctx, this._tensor, order, order.length
      )
      return this._makeResultFromCore(core, [this])
    }

    expand(...shape) {
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      shape = normalizeExpandShape(this.shape, shape)
      if (arraysEqual(this.shape, shape)) return this
      const core = this._rt._core.ffi.poly_tensor_expand(
        this._ctx, this._tensor, shape, shape.length
      )
      return this._makeResultFromCore(core, [this])
    }

    shrink(arg) {
      const shape = this.shape
      if (arg.length !== shape.length) throw new Error(`ndim=${shape.length} != arg.length=${arg.length}`)
      arg = arg.map((pair, i) => pair === null ? [0, shape[i]] : pair)
      if (arg.every(([start, end], i) => start === 0 && end === shape[i])) return this
      const flat = []
      for (let i = 0; i < arg.length; i++) {
        flat.push(arg[i][0], arg[i][1])
      }
      const core = this._rt._core.ffi.poly_tensor_shrink(
        this._ctx, this._tensor, flat, arg.length
      )
      return this._makeResultFromCore(core, [this])
    }

    shrinkTo(...shape) {
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      return this.shrink(shape.map(s => s === null ? null : [0, s]))
    }

    padTo(...shape) {
      let value = 0
      const last = shape[shape.length - 1]
      if (last && typeof last === 'object' && !Array.isArray(last)) {
        value = shape.pop().value ?? 0
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      const current = this.shape
      if (shape.length !== current.length) throw new Error(`ndim=${current.length} != shape.length=${shape.length}`)
      shape = shape.map((s, i) => s === null ? current[i] : s)
      if (arraysEqual(shape, current)) return this
      // Unlike pad(), padTo cannot crop with negative padding.
      if (shape.some((s, i) => !Number.isSafeInteger(s) || s < current[i])) {
        throw new Error(`invalid padTo (${shape}) for (${current})`)
      }
      return this.pad(shape.map((s, i) => [0, s - current[i]]), 'constant', value)
    }

    pad(arg, mode = 'constant', value = 0.0) {
      arg = normalizePadArg(arg, this.shape.length)
      const flat = []
      for (let i = 0; i < arg.length; i++) {
        flat.push(arg[i][0], arg[i][1])
      }
      if (mode !== 'constant') {
        const tag = { circular: 1, reflect: 2, replicate: 3 }[mode]
        if (!tag) throw new Error(`mode=${mode} is not supported`)
        const core = this._rt._core.ffi.poly_tensor_pad_mode(this._ctx, this._tensor, flat, arg.length, tag)
        if (!core) throw new RangeError(`invalid ${mode} padding`)
        return this._makeResultFromCore(core, [this])
      }
      // Pinned _pad_constant shrinks negative pads before emitting a
      // non-negative PAD (mixin/__init__.py:359-368). The shared C boundary
      // owns that policy for zero and nonzero fill values alike.
      let fn
      if (typeof value === 'boolean') fn = this._rt._core.ffi.poly_tensor_pad_value_bool
      else if (typeof value === 'number' && Number.isInteger(value)) {
        fn = this._rt._core.ffi.poly_tensor_pad_value_int
      } else if (typeof value === 'number') fn = this._rt._core.ffi.poly_tensor_pad_value_float
      else throw new TypeError(`pad value must be boolean or number, got ${typeof value}`)
      const core = fn(this._ctx, this._tensor, flat, arg.length, value)
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
      if (!axes.length) return this
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
        dim = this._resolveDim(dim)
        if (!this.ndim || this.shape[dim] !== 1) return this
        const newShape = this.shape.filter((_, i) => i !== dim)
        return this.reshape(newShape)
      }
      const newShape = this.shape.filter(s => s !== 1)
      if (arraysEqual(newShape, this.shape)) return this
      return this.reshape(newShape)
    }

    unsqueeze(dim) {
      dim = this._resolveDim(dim, true)
      const newShape = [...this.shape]
      newShape.splice(dim, 0, 1)
      return this.reshape(newShape)
    }

    flatten(startDim, endDim) {
      if (startDim === undefined) startDim = 0
      if (endDim === undefined) endDim = -1
      const lo = -Math.max(1, this.shape.length)
      const hi = Math.max(1, this.shape.length) - 1
      if (startDim < lo || startDim > hi) throw new RangeError(`dim=${startDim} out of range [${lo}, ${hi}]`)
      if (endDim < lo || endDim > hi) throw new RangeError(`dim=${endDim} out of range [${lo}, ${hi}]`)
      if (startDim < 0) startDim += this.shape.length
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

    repeatInterleave(repeats, dim = null) {
      // Direct port of pinned mixin/movement.py:520-534.
      let x = this
      if (dim === null || dim === undefined) {
        x = this.flatten(); dim = 0
      } else if (dim < 0) dim += this.shape.length
      if (dim < 0 || dim >= x.shape.length) throw new RangeError(`dim ${dim} out of range`)
      const shape = x.shape
      x = x.reshape([...shape.slice(0, dim + 1), 1, ...shape.slice(dim + 1)])
      x = x.expand([...shape.slice(0, dim + 1), Number(repeats), ...shape.slice(dim + 1)])
      return x.reshape([...shape.slice(0, dim), shape[dim] * Number(repeats), ...shape.slice(dim + 1)])
    }

    repeat_interleave(repeats, dim = null) { return this.repeatInterleave(repeats, dim) }

    // --- Reduction ops ---

    sum(axis, keepdim, dtype) {
      if (keepdim === undefined) keepdim = false
      if (axis === undefined || axis === null) {
        axis = this.shape.map((_, i) => i)
      } else if (typeof axis === 'number') {
        axis = [axis]
      }
      axis = axis.map(a => this._resolveDim(a))
      const { ffi } = this._rt._core
      let core
      if (dtype === undefined || dtype === null) {
        core = ffi.poly_tensor_sum(
          this._ctx, this._tensor, axis, axis.length, Boolean(keepdim)
        )
      } else {
        const dtypeId = DTYPE_ID[dtype]
        if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dtype}`)
        core = ffi.poly_tensor_sum_dtype_by_id(
          this._ctx, this._tensor, axis, axis.length, Boolean(keepdim), dtypeId
        )
      }
      return this._makeResultFromCore(core, [this])
    }

    max(opts, keepdim = false) {
      return this._extremum('poly_tensor_max', opts, keepdim)
    }

    all(opts, keepdim = false) { return this._extremum('poly_tensor_all', opts, keepdim) }
    any(opts, keepdim = false) { return this._extremum('poly_tensor_any', opts, keepdim) }

    _scan(operation, axis, pair = false) {
      const rank = Math.max(1, this.shape.length)
      if (!Number.isInteger(axis) || axis < -rank || axis >= rank) throw new RangeError('invalid scan axis')
      const result = this._rt._core.ffi[operation](this._ctx, this._tensor, axis)
      if (!pair) return this._makeResultFromCore(result, [this])
      if (!result || result.length !== 2 || !result[0] || !result[1]) throw new Error('core cumulative extremum failed')
      return result.map(core => this._makeResultFromCore(core, [this]))
    }

    cumsum(axis = 0) { return this._scan('poly_tensor_cumsum', axis) }
    cumprod(axis) { return this._scan('poly_tensor_cumprod', axis) }
    cummax(axis = 0) { return this._scan('poly_tensor_cummax', axis, true) }
    cummin(axis = 0) { return this._scan('poly_tensor_cummin', axis, true) }

    _extremum(operation, opts, positionalKeepdim) {
      if (opts === undefined || opts === null) opts = {}
      let axis, keepdim
      if (typeof opts === 'object' && !Array.isArray(opts)) {
        axis = opts.axis
        keepdim = opts.keepdim || false
      } else {
        axis = opts
        keepdim = positionalKeepdim
      }

      const { ffi } = this._rt._core
      const rawAxes = axis === undefined || axis === null
        ? this.shape.map((_, i) => i)
        : (Array.isArray(axis) ? axis : [axis])
      const axes = rawAxes.map(a => {
        a = Number(a)
        return a < 0 ? a + Math.max(1, this.shape.length) : a
      })
      for (const a of axes) {
        if (!Number.isInteger(a) || a < 0 || a >= Math.max(1, this.shape.length)) {
          throw new RangeError(`axis ${a} out of range for ndim ${this.shape.length}`)
        }
      }
      const core = ffi[operation](
        this._ctx, this._tensor, axes, axes.length, Boolean(keepdim)
      )
      return this._makeResultFromCore(core, [this])
    }

    _resolveDim(dim, extra = false) {
      const total = this.ndim + Number(extra), bound = Math.max(1, total)
      if (!Number.isInteger(dim) || dim < -bound || dim >= bound) throw new RangeError(`dim=${dim} out of range`)
      return dim < 0 ? dim + total : dim
    }

    prod(axis = null, keepdim = false, dtype = null) {
      if (axis && typeof axis === 'object' && !Array.isArray(axis)) {
        const opts = axis; axis = opts.axis; keepdim = opts.keepdim || false; dtype = opts.dtype
      }
      const x = dtype == null ? this : this.cast(dtype)
      return x._extremum('poly_tensor_prod', axis, keepdim)
    }

    logsumexp(axis = null, keepdim = false) { return this._extremum('poly_tensor_logsumexp', axis, keepdim) }
    logcumsumexp(axis = 0) { return this._scan('poly_tensor_logcumsumexp', axis) }

    normalize({ p = 2, dim = 1, eps = 1e-12 } = {}) {
      const core = this._rt._core.ffi.poly_tensor_normalize(this._ctx, this._tensor, p, this._resolveDim(dim), eps)
      return this._makeResultFromCore(core, [this])
    }

    softmin(axis = -1, dtype = null) {
      const x = this.neg()
      return (dtype == null ? x : x.cast(dtype)).softmax(axis)
    }

    stdMean(axis, keepdim = false, correction = 1) { return [this.std(axis, keepdim, correction), this.mean(axis, keepdim)] }

    argmin(axis = null, keepdim = false) {
      if (axis == null) return this.flatten().argmin(0)
      const core = this._rt._core.ffi.poly_tensor_argmin(this._ctx, this._tensor, this._resolveDim(axis), Boolean(keepdim))
      return this._makeResultFromCore(core, [this])
    }

    diag() {
      if (this.ndim !== 1) throw new Error('diag requires a vector')
      return this._makeResultFromCore(this._rt._core.ffi.poly_tensor_diag(this._ctx, this._tensor), [this])
    }

    diagonal(offset = 0, dim1 = 0, dim2 = 1) {
      dim1 = this._resolveDim(dim1); dim2 = this._resolveDim(dim2)
      if (dim1 === dim2) throw new Error('diagonal dimensions must differ')
      const core = this._rt._core.ffi.poly_tensor_diagonal(this._ctx, this._tensor, offset, dim1, dim2)
      return this._makeResultFromCore(core, [this])
    }

    unfold(dim, size, step) {
      dim = this._resolveDim(dim)
      if (!Number.isInteger(size) || !Number.isInteger(step) || size < 0 || step <= 0 || size > this.shape[dim]) throw new Error('invalid unfold size or step')
      const core = this._rt._core.ffi.poly_tensor_unfold(this._ctx, this._tensor, dim, size, step)
      return this._makeResultFromCore(core, [this])
    }

    meshgrid(...args) { return Tensor.meshgrid(this, ...args) }
    static meshgrid(...tensors) {
      let indexing = 'ij'
      if (tensors.length && !(tensors.at(-1) instanceof Tensor)) indexing = tensors.pop().indexing || 'ij'
      if (!['ij', 'xy'].includes(indexing)) throw new Error('indexing must be ij or xy')
      if (tensors.length === 1) return tensors
      const basis = tensors.map((_, i) => i)
      if (indexing === 'xy') [basis[0], basis[1]] = [1, 0]
      const reshaped = tensors.map((t, i) => t.reshape([-1, ...Array(tensors.length - 1 - basis[i]).fill(1)]))
      const shape = reshaped.reduce((s, t) => _broadcastShapes(s, t.shape), [])
      return reshaped.map(t => t.expand(shape))
    }

    roll(shifts, dims = null) {
      if (dims == null) return this.flatten().roll(shifts, 0).reshape(this.shape)
      dims = (Array.isArray(dims) ? dims : [dims]).map(d => this._resolveDim(d))
      shifts = Array.isArray(shifts) ? shifts : [shifts]
      if (dims.length !== shifts.length) throw new Error('shifts and dims length mismatch')
      if (this.shape.includes(0)) return this
      const crop = this.shape.map(s => [0, s])
      dims.forEach((d, i) => { const n = this.shape[d], delta = n - ((shifts[i] % n) + n) % n; crop[d] = [delta, delta+n] })
      return this.repeat(this.shape.map((_, i) => dims.includes(i) ? 2 : 1)).shrink(crop)
    }

    argmax(axis, keepdim) {
      if (keepdim === undefined) keepdim = false
      if (axis === undefined || axis === null) return this.flatten().argmax(0, false)
      if (axis < 0) axis += this.shape.length
      const { ffi } = this._rt._core
      if (!ffi.poly_tensor_argmax) throw new Error('poly_tensor_argmax is required for Tensor.argmax')
      const core = ffi.poly_tensor_argmax(this._ctx, this._tensor, axis, Boolean(keepdim))
      return this._makeResultFromCore(core, [this], 'int32')
    }

    sort(dim, descending) {
      if (dim === undefined) dim = -1
      if (descending === undefined) descending = false
      if (dim < 0) dim += this.shape.length
      const { ffi } = this._rt._core
      if (!ffi.poly_tensor_sort) throw new Error('poly_tensor_sort is required for Tensor.sort')
      const pair = ffi.poly_tensor_sort(this._ctx, this._tensor, dim, descending ? 1 : 0)
      if (!pair || pair.length !== 2 || !pair[0] || !pair[1]) {
        throw new Error('poly_tensor_sort failed')
      }
      const values = this._makeResultFromCore(pair[0], [this])
      const indices = this._makeResultFromCore(pair[1], [this], 'int32')
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
      if (!ffi.poly_tensor_topk) throw new Error('poly_tensor_topk is required for Tensor.topk')
      const pair = ffi.poly_tensor_topk(
        this._ctx, this._tensor, k, dim, largest ? 1 : 0, sorted_ ? 1 : 0
      )
      if (!pair || pair.length !== 2 || !pair[0] || !pair[1]) {
        throw new Error('poly_tensor_topk failed')
      }
      const values = this._makeResultFromCore(pair[0], [this])
      const indices = this._makeResultFromCore(pair[1], [this], 'int32')
      return [values, indices]
    }

    min(opts, keepdim = false) {
      return this._extremum('poly_tensor_min', opts, keepdim)
    }

    mean(axis, keepdim) {
      if (keepdim === undefined) keepdim = false
      let axes
      if (axis === undefined || axis === null) {
        axes = this.shape.map((_, i) => i)
      } else {
        axes = (Array.isArray(axis) ? axis : [axis]).map(a => {
          a = Number(a)
          return a < 0 ? a + this.shape.length : a
        })
      }
      const numerator = this.cast(sumAccumulatorDtype(this._dtype)).sum(axes, keepdim)
      const denominator = product(axes.map(a => this.shape[a]))
      const outputDtype = isFloatDtype(this._dtype) ? this._dtype : 'float32'
      return numerator.div(denominator).cast(outputDtype)
    }

    dropout(p = 0.5) {
      // Direct port of pinned tensor.py:809-829.
      p = Number(p)
      if (!(p >= 0 && p <= 1)) throw new RangeError(`p=${p} is out of range [0, 1]`)
      if (!Tensor.training || p === 0) return this
      if (p === 1) return this.constLike(0)
      return Tensor.randLike(this, { dtype: 'float32', contiguous: false })
        .ge(p).contiguous().where(this, 0).div(1.0 - p)
    }

    scaledDotProductAttention(key, value, opts = {}) {
      // Direct port of pinned tensor.py:831-858. With float32 included in
      // the accumulation lattice, supported JS dtypes choose float64 iff
      // either Q or K is float64, otherwise float32.
      opts = opts || {}
      const attnMask = opts.attnMask === undefined ? (opts.attn_mask || null) : opts.attnMask
      const dropoutP = opts.dropoutP === undefined ? Number(opts.dropout_p || 0) : Number(opts.dropoutP)
      const isCausal = Boolean(opts.isCausal === undefined ? opts.is_causal : opts.isCausal)
      const enableGqa = Boolean(opts.enableGqa === undefined ? opts.enable_gqa : opts.enableGqa)
      if (enableGqa) {
        key = key.repeatInterleave(Math.trunc(this.shape.at(-3) / key.shape.at(-3)), -3)
        value = value.repeatInterleave(Math.trunc(this.shape.at(-3) / value.shape.at(-3)), -3)
      }
      const accDtype = this.dtype === 'float64' || key.dtype === 'float64' ? 'float64' : 'float32'
      let qk = this.matmul(key.transpose(-2, -1), false, accDtype).div(Math.sqrt(this.shape.at(-1)))
      let mask = attnMask
      if (isCausal) {
        if (mask !== null) throw new Error('cannot set attn_mask when is_causal=True')
        mask = qk.constLike(1).cast('bool').tril()
      }
      if (mask !== null) {
        if (mask.dtype === 'bool') mask = mask.where(0, -Infinity)
        qk = qk.add(mask)
      }
      return qk.cast(this.dtype).softmax(-1).dropout(dropoutP).matmul(value)
    }

    scaled_dot_product_attention(key, value, attnMask = null, dropoutP = 0, isCausal = false, enableGqa = false) {
      return this.scaledDotProductAttention(key, value, {
        attnMask, dropoutP, isCausal, enableGqa
      })
    }

    var(axis, keepdim, correction) {
      if (keepdim === undefined) keepdim = false
      if (correction === undefined) correction = 1
      // Pinned Tensor.var is this exact lazy Tensor expression
      // (mixin/__init__.py:608-635), including tuple axes and RELU on the
      // denominator. Every constituent operation is C-owned.
      const squares = this.sub(this.mean(axis, true)).square()
      const reducedShape = squares.sum(axis, true).shape
      const n = product(this.shape.filter((si, i) => Number(si) !== Number(reducedShape[i])))
      const reduced = squares.sum(axis, keepdim)
      const denominator = reduced.constLike(n).sub(correction)
      return reduced.div(denominator.relu())
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

      const core = this._rt._core.ffi.poly_tensor_gather_dim(
        this._ctx, this._tensor, dim, index._tensor
      )
      if (!core) throw new Error('poly_tensor_gather_dim failed')
      return this._makeResultFromCore(core, [this, index])
    }

    takeAlongAxis(index, axis) {
      return this.gather(axis, index)
    }

    oneHot(numClasses) {
      const core = this._rt._core.ffi.poly_tensor_one_hot(
        this._ctx, this._tensor, Number(numClasses)
      )
      if (!core) throw new Error('poly_tensor_one_hot failed')
      return this._makeResultFromCore(core, [this])
    }

    one_hot(numClasses) { return this.oneHot(numClasses) }

    _oneHotAlongDim(numClasses, dim = -1) {
      // Pinned tinygrad compares the integer index directly with a
      // right-aligned arange (mixin/__init__.py:1086-1091).
      if (!isIntegerDtype(this._dtype)) {
        throw new Error(`_one_hot_along_dim expects int index tensor, getting ${this._dtype}`)
      }
      if (dim < 0) dim += this.ndim
      if (dim < 0 || dim >= this.ndim) throw new Error(`dim=${dim} out of range`)
      const offset = this.ndim - dim - 1
      const dtype = Number(numClasses) > 0x7fffffff ? 'int64' : 'int32'
      const shape = [Number(numClasses), ...Array(offset).fill(1)]
      const classes = Tensor.arange(Number(numClasses), {
        dtype, _ctx: this._ctx, _device: this._device
      }).reshape(shape)
      return this.eq(classes)
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
      const core = ffi.poly_tensor_scatter_reduce(
        this._ctx, this._tensor, p.dim, p.index._tensor, p.src._tensor,
        reduce, includeSelf ? 1 : 0
      )
      if (!core) throw new Error('poly_tensor_scatter_reduce failed')
      return this._makeResultFromCore(core, [this, p.index, p.src])
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
      const core = ffi.poly_tensor_scatter(
        this._ctx, this._tensor, p.dim, p.index._tensor, p.src._tensor, reduce || ''
      )
      if (!core) throw new Error('poly_tensor_scatter failed')
      return this._makeResultFromCore(core, [this, p.index, p.src])
    }

    // --- Matmul (C core dot) ---

    dot(w, dtype) {
      if (!(w instanceof Tensor)) {
        throw new TypeError(`Expected Tensor, got ${typeof w}`)
      }
      const { ffi } = this._rt._core
      let core
      if (dtype === undefined || dtype === null) {
        core = ffi.poly_tensor_dot(this._ctx, this._tensor, w._tensor)
      } else {
        const dtypeId = DTYPE_ID[dtype]
        if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dtype}`)
        core = ffi.poly_tensor_dot_dtype_by_id(this._ctx, this._tensor, w._tensor, dtypeId)
      }
      if (!core) {
        throw new Error(`cannot dot ${JSON.stringify(this.shape)} and ${JSON.stringify(w.shape)}`)
      }
      return this._makeResultFromCore(core, [this, w])
    }

    matmul(other, reverse = false, dtype) {
      return reverse ? other.dot(this, dtype) : this.dot(other, dtype)
    }

    qr(opts) {
      let mode = 'complete'
      if (typeof opts === 'string') mode = opts
      else if (opts && typeof opts.mode === 'string') mode = opts.mode
      const modeId = mode === 'complete' ? 0 : mode === 'reduced' ? 1 : mode === 'r' ? 2 : -1
      if (modeId < 0) throw new Error("qr mode must be 'complete', 'reduced', or 'r'")
      const { ffi } = this._rt._core
      if (!ffi.poly_tensor_qr_ex) throw new Error('poly_tensor_qr_ex is required for Tensor.qr')
      const pair = ffi.poly_tensor_qr_ex(this._ctx, this._tensor, modeId)
      if (!pair || pair.length !== 2 || !pair[1] || (modeId !== 2 && !pair[0])) throw new Error('poly_tensor_qr_ex failed')
      if (modeId === 2) return this._makeResultFromCore(pair[1], [this])
      return [
        this._makeResultFromCore(pair[0], [this]),
        this._makeResultFromCore(pair[1], [this])
      ]
    }

    triangularSolve(b, opts) {
      if (!(b instanceof Tensor)) b = new Tensor(b)
      opts = opts || {}
      const { ffi } = this._rt._core
      if (!ffi.poly_tensor_triangular_solve) {
        throw new Error('poly_tensor_triangular_solve is required for Tensor.triangularSolve')
      }
      const core = ffi.poly_tensor_triangular_solve(
        this._ctx, this._tensor, b._tensor,
        opts.upper ? 1 : 0,
        opts.transposeA || opts.transpose_a ? 1 : 0,
        opts.unitDiagonal || opts.unit_diagonal ? 1 : 0
      )
      if (!core) {
        throw new Error(`cannot triangularSolve A.shape=${JSON.stringify(this.shape)} and b.shape=${JSON.stringify(b.shape)}`)
      }
      return this._makeResultFromCore(core, [this, b])
    }

    triangular_solve(b, upper, transpose_a, unit_diagonal) {
      return this.triangularSolve(b, { upper, transpose_a, unit_diagonal })
    }

    solveTriangular(b, opts) { return this.triangularSolve(b, opts) }

    cholesky(opts) {
      opts = opts || {}
      const { ffi } = this._rt._core
      if (!ffi.poly_tensor_cholesky) throw new Error('poly_tensor_cholesky is required for Tensor.cholesky')
      const core = ffi.poly_tensor_cholesky(this._ctx, this._tensor, opts.upper ? 1 : 0)
      if (!core) throw new Error(`cannot cholesky shape=${JSON.stringify(this.shape)}`)
      return this._makeResultFromCore(core, [this])
    }

    choleskySolve(b, opts) {
      if (!(b instanceof Tensor)) b = new Tensor(b)
      opts = opts || {}
      const { ffi } = this._rt._core
      if (!ffi.poly_tensor_cholesky_solve) {
        throw new Error('poly_tensor_cholesky_solve is required for Tensor.choleskySolve')
      }
      const core = ffi.poly_tensor_cholesky_solve(
        this._ctx, this._tensor, b._tensor, opts.upper ? 1 : 0
      )
      if (!core) {
        throw new Error(`cannot choleskySolve factor.shape=${JSON.stringify(this.shape)} and b.shape=${JSON.stringify(b.shape)}`)
      }
      return this._makeResultFromCore(core, [this, b])
    }

    cholesky_solve(b, upper) { return this.choleskySolve(b, { upper }) }

    solve(b) {
      if (!(b instanceof Tensor)) b = new Tensor(b)
      const { ffi } = this._rt._core
      if (!ffi.poly_tensor_solve) throw new Error('poly_tensor_solve is required for Tensor.solve')
      const core = ffi.poly_tensor_solve(this._ctx, this._tensor, b._tensor)
      if (!core) throw new Error(`cannot solve A.shape=${JSON.stringify(this.shape)} and b.shape=${JSON.stringify(b.shape)}`)
      return this._makeResultFromCore(core, [this, b])
    }

    lstsq(b) {
      if (!(b instanceof Tensor)) b = new Tensor(b)
      const { ffi } = this._rt._core
      if (!ffi.poly_tensor_lstsq) throw new Error('poly_tensor_lstsq is required for Tensor.lstsq')
      const core = ffi.poly_tensor_lstsq(this._ctx, this._tensor, b._tensor)
      if (!core) throw new Error(`cannot lstsq A.shape=${JSON.stringify(this.shape)} and b.shape=${JSON.stringify(b.shape)}`)
      return this._makeResultFromCore(core, [this, b])
    }

    linear(weight, bias = null, dtype = null) {
      // Direct port of pinned mixin/__init__.py:1335-1350.
      if (dtype !== null && dtype !== undefined) {
        return this.cast(dtype).linear(
          weight.cast(dtype), bias === null ? null : bias.cast(dtype)
        )
      }
      const result = weight.shape.length === 1 ? this.mul(weight) : this.dot(weight)
      return bias === null ? result : result.add(bias)
    }

    sequential(list) {
      let result = this
      for (const fn of list) result = fn(result)
      return result
    }

    _pool(kernelSize, stride = 1, dilation = 1) {
      const k = makeTuple(kernelSize, Array.isArray(kernelSize) ? kernelSize.length : 2)
      const strideTuple = makeTuple(stride, k.length)
      const dilationTuple = makeTuple(dilation, k.length)
      const core = this._rt._core.ffi.poly_tensor_pool(
        this._ctx, this._tensor, k, k.length, strideTuple, dilationTuple
      )
      if (!core) {
        throw new Error(
          `poly_pool failed for shape=${JSON.stringify(this.shape)}, ` +
          `kernel=${JSON.stringify(k)}, stride=${JSON.stringify(strideTuple)}, ` +
          `dilation=${JSON.stringify(dilationTuple)}`
        )
      }
      return this._makeResultFromCore(core, [this])
    }

    _pool2dArgs(kernelSize, opts) {
      const k = makeTuple(kernelSize, 2)
      const stride = opts.stride == null ? k : makeTuple(opts.stride, k.length)
      const dilation = opts.dilation == null ? makeTuple(1, k.length) : makeTuple(opts.dilation, k.length)
      if (stride.length !== k.length || dilation.length !== k.length) throw new Error('stride/dilation mismatch')
      const padding = resolvePoolPads(opts.padding == null ? 0 : opts.padding, k.length)
      return [this._ctx, this._tensor, k, k.length, stride, dilation, padding, padding.length]
    }

    maxPool2d(kernelSize = [2, 2], opts = {}) {
      if (typeof opts !== 'object' || Array.isArray(opts)) opts = { stride: opts }
      const indices = Boolean(opts.returnIndices || opts.return_indices)
      const core = this._rt._core.ffi.poly_tensor_max_pool2d(
        ...this._pool2dArgs(kernelSize, opts), Boolean(opts.ceilMode || opts.ceil_mode), indices)
      if (!core) throw new Error('poly_max_pool2d failed')
      if (indices) return core.map(ptr => this._makeResultFromCore(ptr, [this]))
      return this._makeResultFromCore(core, [this])
    }

    avgPool2d(kernelSize = [2, 2], opts = {}) {
      if (typeof opts !== 'object' || Array.isArray(opts)) opts = { stride: opts }
      const core = this._rt._core.ffi.poly_tensor_avg_pool2d(
        ...this._pool2dArgs(kernelSize, opts), Boolean(opts.ceilMode || opts.ceil_mode),
        Boolean(opts.countIncludePad ?? opts.count_include_pad ?? true))
      if (!core) throw new Error('poly_avg_pool2d failed')
      return this._makeResultFromCore(core, [this])
    }

    avg_pool2d(kernelSize = [2, 2], stride = null, dilation = 1, padding = 0, ceilMode = false, countIncludePad = true) {
      return this.avgPool2d(kernelSize, { stride, dilation, padding, ceilMode, countIncludePad })
    }

    maxUnpool2d(indices, kernelSize = [2, 2], opts = {}) {
      const [ctx, tensor, k, nk, s, d, p, np] = this._pool2dArgs(kernelSize, opts)
      const output = opts.outputSize ?? opts.output_size ?? []
      const core = this._rt._core.ffi.poly_tensor_max_unpool2d(ctx, tensor, indices._tensor, k, nk, s, d, p, np, output, output.length)
      if (!core) throw new Error('poly_max_unpool2d failed')
      return this._makeResultFromCore(core, [this, indices])
    }

    max_unpool2d(indices, kernelSize = [2, 2], stride = null, dilation = 1, padding = 0, outputSize = null) {
      return this.maxUnpool2d(indices, kernelSize, { stride, dilation, padding, outputSize })
    }

    interpolate(size, opts = {}) {
      if (typeof opts === 'string') opts = { mode: opts }
      const mode = opts.mode ?? 'linear', align = Boolean(opts.alignCorners ?? opts.align_corners)
      if (!Array.isArray(size) || size.length < 1 || size.length > this.ndim || !size.every(Number.isSafeInteger)) {
        throw new Error('invalid interpolate size')
      }
      if (!['linear', 'nearest', 'nearest-exact'].includes(mode) || (align && mode !== 'linear')) {
        throw new Error('interpolate supports linear, nearest, nearest-exact; alignCorners requires linear')
      }
      const core = this._rt._core.ffi.poly_tensor_interpolate(this._ctx, this._tensor, size, size.length, mode, align)
      if (!core) throw new Error('poly_interpolate failed')
      return this._makeResultFromCore(core, [this])
    }

    convTranspose2d(weight, bias = null, opts = {}) {
      if (!(weight instanceof Tensor)) weight = this._ensureTensor(weight)
      if (bias !== null && !(bias instanceof Tensor)) bias = this._ensureTensor(bias)
      const n = weight.ndim - 2
      let stride = makeTuple(opts.stride ?? 1, n)
      const dilation = makeTuple(opts.dilation ?? 1, n)
      if (dilation.length !== n) throw new Error('stride/dilation mismatch')
      // Only inserting strides are consumed by the pin; normalize unused
      // short/extra tuples before passing fixed-length C arrays.
      if (stride.some(s => s > 1)) {
        if (stride.length !== n) throw new Error('stride length mismatch')
      } else stride = makeTuple(1, n)
      const padding = resolvePoolPads(opts.padding ?? 0, n)
      const op = makeTuple(opts.outputPadding ?? opts.output_padding ?? 0, n).slice(0, n)
      if (!op.length) throw new Error('output_padding must not be empty')
      const core = this._rt._core.ffi.poly_tensor_conv_transpose2d(
        this._ctx, this._tensor, weight._tensor, bias ? bias._tensor : null, opts.groups ?? 1,
        stride, dilation, padding, padding.length, op, op.length)
      if (!core) throw new Error('poly_conv_transpose2d failed')
      return this._makeResultFromCore(core, bias ? [this, weight, bias] : [this, weight])
    }

    conv_transpose2d(weight, bias = null, groups = 1, stride = 1, dilation = 1, padding = 0, outputPadding = 0) {
      return this.convTranspose2d(weight, bias, { groups, stride, dilation, padding, outputPadding })
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
      const args = [
        this._ctx, this._tensor, weight._tensor, bias ? bias._tensor : null,
        groups, strideTuple, dilationTuple, paddingTuple, paddingTuple.length
      ]
      let core
      if (opts.dtype === undefined || opts.dtype === null) {
        core = this._rt._core.ffi.poly_tensor_conv2d(...args)
      } else {
        const dtypeId = DTYPE_ID[opts.dtype]
        if (dtypeId === undefined) throw new Error(`unsupported dtype: ${opts.dtype}`)
        core = this._rt._core.ffi.poly_tensor_conv2d_dtype_by_id(...args, dtypeId)
      }
      if (!core) throw new Error('poly_conv2d failed')
      const inputs = bias ? [this, weight, bias] : [this, weight]
      return this._makeResultFromCore(core, inputs)
    }

    batchnorm(weight, bias, mean, invstd, axis = 1) {
      const axes = (Array.isArray(axis) ? axis : [axis]).map(a => {
        a = Number(a)
        return a < 0 ? a + this.shape.length : a
      })
      const core = this._rt._core.ffi.poly_tensor_batchnorm(
        this._ctx, this._tensor,
        weight ? weight._tensor : null,
        bias ? bias._tensor : null,
        mean._tensor, invstd._tensor, axes, axes.length
      )
      if (!core) throw new Error('poly_batchnorm failed')
      const inputs = [this, mean, invstd]
      if (weight) inputs.push(weight)
      if (bias) inputs.push(bias)
      return this._makeResultFromCore(core, inputs)
    }

    // --- Loss functions ---

    crossEntropy(target, reduction = 'mean', labelSmoothing = 0.0, axis) {
      if (labelSmoothing < 0.0 || labelSmoothing > 1.0) {
        throw new Error('label_smoothing must be in [0.0, 1.0]')
      }
      if (!(target instanceof Tensor)) target = new Tensor(target)
      let classesDim = axis === undefined
        ? (this.shape.length === 1 ? 0 : 1)
        : Number(axis)
      if (classesDim < 0) classesDim += this.ndim
      if (classesDim < 0 || classesDim >= this.ndim) {
        throw new Error(`axis=${axis} out of range`)
      }
      if (!arraysEqual(this.shape, target.shape)) {
        const expected = this.shape.filter((_, i) => i !== classesDim)
        if (!arraysEqual(expected, target.shape)) {
          throw new Error(`shape mismatch: self.shape=${JSON.stringify(this.shape)}, target.shape=${JSON.stringify(target.shape)}`)
        }
        target = target.unsqueeze(classesDim)._oneHotAlongDim(
          this.shape[classesDim], classesDim
        )
      }
      target = target.mul(1 - labelSmoothing).add(
        labelSmoothing / Number(target.shape[classesDim])
      )
      const reduced = this.logSoftmax(classesDim).mul(target).sum(classesDim)
      if (reduction === 'none') return reduced.neg()
      if (reduction === 'sum') return reduced.sum().neg()
      if (reduction === 'mean') return reduced.mean().neg()
      throw new Error(
        `reduction=${JSON.stringify(reduction)} must be one of ('none', 'sum', 'mean')`
      )
    }

    sparseCategoricalCrossentropy(target, { ignoreIndex = -1, labelSmoothing = 0, reduction = 'mean' } = {}) {
      if (!(labelSmoothing >= 0 && labelSmoothing <= 1)) throw new RangeError('labelSmoothing must be in [0, 1]')
      target = this._ensureTensor(target)
      if (target.device !== this.device) throw new Error('loss inputs must be on the same device')
      if (!Number.isSafeInteger(ignoreIndex)) throw new RangeError('ignoreIndex must be a safe integer')
      const core = this._rt._core.ffi.poly_tensor_sparse_categorical_crossentropy(this._ctx, this._tensor, target._tensor, ignoreIndex, labelSmoothing, this._lossReductionId(reduction))
      return this._makeResultFromCore(core, [this, target])
    }

    binaryCrossEntropy(target, reduction = 'mean') {
      const id = this._lossReductionId(reduction)
      target = this._ensureTensor(target)
      const core = this._rt._core.ffi.poly_tensor_binary_crossentropy(this._ctx, this._tensor, target._tensor, id)
      return this._makeResultFromCore(core, [this, target])
    }

    layernorm(axis, eps) {
      if (axis === undefined) axis = -1
      if (eps === undefined) eps = 1e-5
      // Pinned mixin/__init__.py:1548-1564. Reuse the centered value and keep
      // the exact mul/rsqrt spelling as one ordinary lazy graph.
      const y = this.sub(this.mean(axis, true))
      return y.mul(y.mul(y).mean(axis, true).add(eps).rsqrt())
    }

    // --- Indexing ---

    _indexArgs(idx) {
      const ell = idx.map((i, j) => i === '...' ? j : -1).filter(j => j >= 0)
      if (ell.length > 1) throw new RangeError('indices can only have a single ellipsis')
      const real = idx.length - ell.length - idx.filter(i => i == null).length
      if (real > this.ndim) throw new RangeError('too many indices for tensor')
      idx = [...idx]
      idx.splice(ell.length ? ell[0] : idx.length, ell.length ? 1 : 0,
        ...Array.from({length: this.ndim - real}, () => ({step: 1})))
      const kinds = [], starts = [], sizes = [], steps = [], tensors = [], owners = []
      const integer = (v, what) => {
        if (!Number.isSafeInteger(v)) throw new TypeError(what + ' must be a safe integer')
        return v
      }
      let dim = 0
      for (const index of idx) {
        if (index == null) {
          kinds.push(0); starts.push(null); sizes.push(null); steps.push(1); tensors.push(null)
          continue
        }
        const size = this.shape[dim++]
        let start = 0, stop = size, step = 1, kind = 2, tensor = null
        if (typeof index === 'number') {
          integer(index, 'index')
          start = index < 0 ? index + size : index
          if (start < 0 || start >= size) throw new RangeError('index=' + index + ' is out of bounds with size=' + size)
          stop = start + 1
          kind = 1
        } else if (index instanceof Tensor) {
          if (!isIntegerDtype(index.dtype)) throw new TypeError('index dtype ' + index.dtype + ' is not supported')
          if (index._ctx !== this._ctx || index._device !== this._device) {
            throw new Error('expected index and self on the same device and context')
          }
          kind = 3
          tensor = index
          owners.push(index)
        } else {
          // Preserve the existing JS [start, stop] slice syntax. Tensor
          // objects express advanced indices; arrays at the call boundary
          // remain lists of axis specifications, not Python-style index lists.
          const slice = Array.isArray(index) && index.length === 2
            ? {start: index[0], stop: index[1], step: 1} : index
          if (!slice || typeof slice !== 'object' ||
              !['start', 'stop', 'step'].some(k => k in slice)) throw new TypeError('unsupported index type')
          step = slice.step == null ? 1 : integer(slice.step, 'slice step')
          if (step === 0) throw new RangeError('slice step cannot be zero')
          const lo = step < 0 ? -1 : 0, hi = step < 0 ? size-1 : size
          const clamp = v => Math.min(hi, Math.max(lo, v < 0 ? v+size : v))
          start = slice.start == null ? (step < 0 ? size-1 : 0) : clamp(integer(slice.start, 'slice start'))
          stop = slice.stop == null ? (step < 0 ? -1 : size) : clamp(integer(slice.stop, 'slice stop'))
          if ((step < 0 && stop > start) || (step > 0 && stop < start)) start = stop = 0
          else if (step < 0) [start, stop] = [stop+1, start+1]
        }
        kinds.push(kind)
        starts.push(ffi.poly_const_int_by_id(this._ctx, start, DTYPE_ID.weakint))
        sizes.push(ffi.poly_const_int_by_id(this._ctx, stop-start, DTYPE_ID.weakint))
        steps.push(step)
        tensors.push(tensor ? tensor._tensor : null)
      }
      return {args: [kinds, starts, sizes, steps, tensors, kinds.length], owners}
    }

    getitem(...idx) {
      if (idx.length === 1 && Array.isArray(idx[0])) idx = idx[0]
      const {args, owners} = this._indexArgs(idx)
      const core = ffi.poly_tensor_getitem(this._ctx, this._tensor, ...args)
      if (!core) throw new RangeError('cannot broadcast indices or unsupported indexing shape')
      if (uopKey(tensorUop(core)) === uopKey(this._currentUopRaw())) {
        ffi.poly_tensor_release(core)
        return this
      }
      return this._makeResultFromCore(core, [this, ...owners])
    }

    setitem(indices, value) {
      if (!(value instanceof Tensor)) value = new Tensor(value, {dtype: this.dtype, device: this._device})
      const {args, owners} = this._indexArgs(Array.isArray(indices) ? indices : [indices])
      const rc = ffi.poly_tensor_setitem(this._ctx, this._tensor, ...args, value._tensor)
      if (rc === -6) throw new RangeError('cannot broadcast indices')
      if (rc) throw new Error({
        '-2': "can't setitem on a tensor with other uses",
        '-3': 'setitem dtype mismatch',
        '-4': 'cannot setitem into a weak tensor; it has no storage',
        '-5': 'advanced setitem is not supported for DISK tensors'
      }[rc] || 'cannot broadcast assigned value or unsupported indexing shape')
      this._data = null
      return this
    }

    // --- Einsum (C core) ---

    static einsum(formula, ...operands) {
      if (operands.length === 1 && Array.isArray(operands[0])) operands = operands[0]
      if (!operands.length) throw new Error('einsum requires at least one operand')
      if (operands.some(t => !t || !t._tensor || !t._ctx)) {
        throw new TypeError('einsum operands must be Tensors')
      }
      const { ffi } = liveCore()
      const ctx = operands[0]._ctx
      if (operands.some(t => t._ctx !== ctx)) {
        throw new Error('einsum operands must belong to the same Polygrad context')
      }
      const t0 = operands[0]
      const core = ffi.poly_tensor_einsum(
        ctx, formula, operands.map(t => t._tensor)
      )
      if (!core) throw new Error(`poly_einsum failed for formula: ${formula}`)
      return t0._makeResultFromCore(core, operands)
    }

    // --- Rearrange (C core, einops-style) ---

    rearrange(formula, kwargs) {
      if (!kwargs) kwargs = {}
      const { ffi } = this._rt._core
      const core = ffi.poly_tensor_rearrange(this._ctx, formula, this._tensor, kwargs)
      if (!core) throw new Error(`poly_rearrange failed for formula: ${formula}`)
      return this._makeResultFromCore(core, [this])
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
            .map(raw => new UOp(this._ctx, ffi, raw, false))
          if (!upstreams.length) continue
          const call = new UOp(this._ctx, ffi, activeCall, false)
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
        const gradHandle = tensorCreateResultLike(
          this._ctx, leaf._tensor, tensorUopLogical(leaf._tensor) ? gradUop : null,
          gradUop, POLY_TENSOR_VALUE, leaf._device
        )
        if (!gradHandle) throw new Error('failed to store backward gradient roots')
        let gradTensor = new Tensor(null, {
          _ctx: this._ctx,
          _tensor: gradHandle,
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
      /* Tinygrad 2026-08-22/a9069c177a9d CreationMixin.zeros passes Python
       * 0.0 to full, preserving a weakfloat initializer (creation.py:106-120). */
      return Tensor.full(shape, 0, { ...(opts || {}), _inferredDtype: 'weakfloat' })
    }

    static ones(...args) {
      let shape = args, opts
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !(args[args.length - 1] instanceof Array)) {
        opts = args[args.length - 1]; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      shape = shape.map(x => Number(x))
      return Tensor.full(shape, 1, { ...(opts || {}), _inferredDtype: 'weakfloat' })
    }

    fullLike(value, opts = {}) {
      return Tensor.full(this.shape, value, { ...opts, dtype: opts.dtype || this._dtype, device: opts.device || this.device, _ctx: this._ctx })
    }
    zerosLike(opts = {}) { return this.fullLike(0, opts) }
    onesLike(opts = {}) { return this.fullLike(1, opts) }
    static fullLike(x, value, opts = {}) { return x.fullLike(value, opts) }
    static zerosLike(x, opts = {}) { return x.zerosLike(opts) }
    static onesLike(x, opts = {}) { return x.onesLike(opts) }

    static full(shape, fillValue, opts) {
      if (typeof shape === 'number') shape = [shape]
      shape = Array.from(shape, Number)
      opts = opts ? { ...opts } : {}
      rejectRequiresGrad(opts)
      const ctx = opts._ctx || liveCore().ctx
      const device = normalizeDevice(opts._device || opts.device || _runtime.device || 'cpu')
      const dtypeExplicit = Object.prototype.hasOwnProperty.call(opts, 'dtype')
      const buffer = opts.buffer !== false
      const dtype = opts.dtype || opts._inferredDtype || (
        typeof fillValue === 'boolean' ? 'bool' : (typeof fillValue === 'bigint' || Number.isInteger(fillValue)) ? 'weakint' : 'weakfloat'
      )
      const dtypeId = DTYPE_ID[dtype]
      if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dtype}`)
      const targetDevice = deviceId(device)
      const integer = typeof fillValue === 'boolean' ? Number(fillValue) : typeof fillValue === 'bigint' ? fillValue : Math.trunc(Number(fillValue))
      if (typeof integer === 'bigint' && (integer < -(1n << 63n) || integer >= (1n << 64n))) throw new RangeError('integer literal out of 64-bit range')
      const factory = typeof integer === 'bigint' && integer >= 0 ? ffi.poly_tensor_full_uint_by_id : ffi.poly_tensor_full_int_by_id
      const tensor = (dtype === 'bool' || isIntegerDtype(dtype))
        ? factory(
            ctx, shape, shape.length,
            integer,
            dtypeId, targetDevice, dtypeExplicit, buffer
          )
        : ffi.poly_tensor_full_float_by_id(
            ctx, shape, shape.length, Number(fillValue), dtypeId, targetDevice,
            dtypeExplicit, buffer
          )
      if (!tensor) throw new Error('C-owned Tensor.full construction failed')
      return new Tensor(null, {_ctx: ctx, _tensor: tensor, _dtype: dtype, _device: device})
    }

    static arange(start, stop, step, opts) {
      if (typeof stop === 'object' && stop !== null) { opts = stop; stop = undefined; step = undefined }
      if (typeof step === 'object' && step !== null) { opts = step; step = undefined }
      if (stop === undefined) { stop = start; start = 0 }
      if (step === undefined) step = 1
      if (step === 0) throw new Error('Tensor.arange step must not be zero')
      opts = opts ? { ...opts } : {}
      rejectRequiresGrad(opts)
      const ctx = opts._ctx || liveCore().ctx
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
      return new Tensor(null, {_ctx: ctx, _tensor: tensor, _dtype: dtype, _device: device})
    }

    static manual_seed(seed = 0) {
      // Pinned tinygrad tensor.py:475-504 resets all per-device RNG versions.
      ffi.poly_tensor_manual_seed(liveCore().ctx, Math.trunc(Number(seed)))
    }

    static rand(...args) {
      let shape = args, opts
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !(args[args.length - 1] instanceof Array)) {
        opts = args[args.length - 1]; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      opts = opts ? { ...opts } : {}
      rejectRequiresGrad(opts)
      shape = shape.map(Number)
      if (shape.some(dim => !Number.isInteger(dim) || dim < 0)) {
        throw new Error(`invalid input shape=${JSON.stringify(shape)}`)
      }
      const dtype = opts.dtype || 'float32'
      if (!isFloatDtype(dtype)) throw new Error(`rand only supports float dtypes, got ${dtype}`)
      const device = normalizeDevice(opts.device || _runtime.device || 'cpu')
      const dtypeId = DTYPE_ID[dtype]
      const ctx = liveCore().ctx
      const tensor = ffi.poly_tensor_rand_by_id(
        ctx, shape, shape.length, dtypeId, deviceId(device),
        opts.contiguous === false ? 0 : 1
      )
      if (!tensor) throw new Error('poly_tensor_rand_by_id failed')
      return new Tensor(null, {_ctx: ctx, _tensor: tensor, _dtype: dtype, _device: device})
    }

    static randLike(source, opts = {}) {
      // Direct single-device port of pinned mixin/rand.py:70-86.
      opts = { ...opts }
      const device = opts.device || source.device
      const dtype = opts.dtype || source.dtype
      let out = Tensor.rand(...source.shape, { ...opts, dtype })
      if (out.dtype !== dtype) out = out.cast(dtype)
      if (normalizeDevice(device) !== normalizeDevice(out.device)) out = out.to(device)
      return out
    }

    static rand_like(source, opts = {}) { return Tensor.randLike(source, opts) }

    randLike(opts = {}) { return Tensor.randLike(this, opts) }
    rand_like(opts = {}) { return Tensor.randLike(this, opts) }

    static randn(...args) {
      let shape = args, opts
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !(args[args.length - 1] instanceof Array)) {
        opts = args[args.length - 1]; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      opts = opts ? { ...opts } : {}
      rejectRequiresGrad(opts)
      const dtype = opts.dtype || 'float32'
      const device = normalizeDevice(opts.device || _runtime.device || 'cpu')
      shape = shape.map(Number)
      if (shape.some(dim => !Number.isInteger(dim) || dim < 0)) {
        throw new Error(`invalid input shape=${JSON.stringify(shape)}`)
      }
      if (!isFloatDtype(dtype)) throw new Error(`randn only supports float dtypes, got ${dtype}`)
      const ctx = liveCore().ctx
      const tensor = ffi.poly_tensor_randn_by_id(
        ctx, shape, shape.length, DTYPE_ID[dtype], deviceId(device)
      )
      if (!tensor) throw new Error('poly_tensor_randn_by_id failed')
      return new Tensor(null, {_ctx: ctx, _tensor: tensor, _dtype: dtype, _device: device})
    }

    static normal(...args) {
      let opts = {}
      if (args.length && typeof args.at(-1) === 'object' && !Array.isArray(args.at(-1))) opts = { ...args.pop() }
      const { mean = 0, std = 1 } = opts
      if (std < 0) throw new RangeError('std must be nonnegative')
      delete opts.mean; delete opts.std
      return Tensor.randn(...args, opts).mul(std, true).add(mean)
    }

    static kaimingNormal(...args) {
      let opts = {}
      if (args.length && typeof args.at(-1) === 'object' && !Array.isArray(args.at(-1))) opts = { ...args.pop() }
      const shape = args.length === 1 && Array.isArray(args[0]) ? args[0] : args
      const a = opts.a == null ? 0.01 : opts.a
      delete opts.a
      const fanIn = shape.slice(1).reduce((a, b) => a*b, 1)
      return Tensor.normal(shape, { ...opts, mean: 0, std: Math.sqrt(2 / (1+a*a) / fanIn) })
    }

    static uniform(...args) {
      let shape = args, opts = {}
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !Array.isArray(args[args.length - 1])) {
        opts = { ...args[args.length - 1] }; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      shape = shape.map(Number)
      if (shape.some(dim => !Number.isInteger(dim) || dim < 0)) {
        throw new Error(`invalid input shape=${JSON.stringify(shape)}`)
      }
      const low = opts.low == null ? 0.0 : Number(opts.low)
      const high = opts.high == null ? 1.0 : Number(opts.high)
      if (!(low < high)) {
        throw new Error(`Tensor.uniform requires low < high, got low=${low}, high=${high}`)
      }
      const dtype = opts.dtype || 'float32'
      const randOpts = { ...opts, dtype }
      delete randOpts.low
      delete randOpts.high
      return Tensor.rand(...shape, randOpts).mul(high - low).cast(dtype).add(low)
    }

    static scaledUniform(...args) {
      let shape = args, opts = {}
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !Array.isArray(args[args.length - 1])) {
        opts = { ...args[args.length - 1] }; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      shape = shape.map(Number)
      const scale = shape.reduce((product, dim) => product * dim, 1) ** -0.5
      return Tensor.uniform(...shape, { ...opts, low: -1.0, high: 1.0 }).mul(scale)
    }

    static scaled_uniform(...args) { return Tensor.scaledUniform(...args) }

    static glorotUniform(...args) {
      let shape = args, opts = {}
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !Array.isArray(args[args.length - 1])) {
        opts = { ...args[args.length - 1] }; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      shape = shape.map(Number)
      const fanOut = shape.slice(1).reduce((product, dim) => product * dim, 1)
      const bound = Math.sqrt(6 / (shape[0] + fanOut))
      return Tensor.uniform(...shape, { ...opts, low: -bound, high: bound })
    }

    static glorot_uniform(...args) { return Tensor.glorotUniform(...args) }

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
      rejectRequiresGrad(opts)
      const ctx = opts._ctx || liveCore().ctx
      const device = normalizeDevice(opts._device || opts.device || _runtime.device || 'cpu')
      const dtype = opts.dtype || 'float32'
      const dtypeId = DTYPE_ID[dtype]
      if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dtype}`)
      const tensor = ffi.poly_tensor_linspace_by_id(
        ctx, Number(start), Number(stop), Number(steps), dtypeId, deviceId(device)
      )
      if (!tensor) throw new Error('C-owned Tensor.linspace construction failed')
      return new Tensor(null, {_ctx: ctx, _tensor: tensor, _dtype: dtype, _device: device})
    }

    static eye(n, m, opts) {
      if (typeof m === 'object' && m !== null) { opts = m; m = undefined }
      opts = opts ? { ...opts } : {}
      rejectRequiresGrad(opts)
      const ctx = opts._ctx || liveCore().ctx
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
      return new Tensor(null, {_ctx: ctx, _tensor: tensor, _dtype: dtype, _device: device})
    }

    static invalids(...args) {
      let opts = {}, shape = args
      if (args.length && typeof args[args.length - 1] === 'object' && !Array.isArray(args[args.length - 1])) {
        opts = args[args.length - 1]; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      rejectRequiresGrad(opts)
      const ctx = opts._ctx || liveCore().ctx
      const dtype = opts.dtype || 'bool', dtypeId = DTYPE_ID[dtype]
      if (dtypeId === undefined) throw new Error(`unsupported dtype: ${dtype}`)
      const device = normalizeDevice(opts._device || opts.device || _runtime.device || 'cpu')
      const tensor = ffi.poly_tensor_full_invalid_by_id(ctx, shape, shape.length, dtypeId, deviceId(device), true)
      if (!tensor) throw new Error('C-owned Tensor.invalids construction failed')
      return new Tensor(null, { _ctx: ctx, _tensor: tensor, _dtype: dtype, _device: device })
    }

    static empty(...args) {
      let shape = args, opts
      if (args.length > 0 && typeof args[args.length - 1] === 'object'
          && !(args[args.length - 1] instanceof Array)) {
        opts = args[args.length - 1]; shape = args.slice(0, -1)
      }
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      shape = shape.map(x => Number(x))
      rejectRequiresGrad(opts)
      if (shape.some(x => x < 0)) throw new Error(`negative dimensions are not allowed: ${shape}`)
      if (opts && Object.prototype.hasOwnProperty.call(opts, 'name')) {
        throw new TypeError('Tensor.empty does not accept name; pass names to Model.fromTensors')
      }
      const ctx = (opts && opts._ctx) || liveCore().ctx
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
        _device: tensorDevice
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
      if (!tensors.length || tensors.some(t => !(t instanceof Tensor))) throw new TypeError('stack expects tensors')
      dim = tensors[0]._resolveDim(dim, true)
      if (tensors.some(t => !arraysEqual(t.shape, tensors[0].shape))) throw new Error('stack shape mismatch')
      const core = tensors[0]._rt._core.ffi.poly_tensor_stack(tensors[0]._ctx, tensors.map(t => t._tensor), dim)
      return tensors[0]._makeResultFromCore(core, tensors)
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
      return `Tensor(shape=[${this.shape}], dtype=${this.dtype}, realized=${ffi.poly_uop_has_buffer_identity(this._currentUopRaw())})`
    }
  }

  Tensor.training = false
  Tensor._liveTensorSnapshot = liveTensorSnapshot
  Tensor._disposeAll = () => {
    const pending = []
    for (const tensor of liveTensorSnapshot()) {
      const result = tensor.dispose()
      if (result && typeof result.then === 'function') pending.push(result)
    }
    customKernelGradFxns.clear()
    return pending.length ? Promise.all(pending) : undefined
  }
  return Tensor
}

module.exports = {
  createBoundTensorClass, flattenArray, arraysEqual, _buildNested, normalizeLogicalPolicy
}
