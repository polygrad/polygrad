/**
 * tensor.js -- Runtime-bound Tensor class factory for polygrad.
 *
 * All graph construction ops are synchronous (use _runtime._backend).
 * All realize/data-extraction ops are async (backend already resolved by create()).
 *
 * The backend's ffi table normalizes int64 marshalling:
 *   - Functions taking shape/axis arrays accept plain JS number[]
 *   - Shapes are read from UOps via poly_uop_dims (shape-on-UOp)
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
  const DTYPE_ID = _runtime._backend.dtypeIds
  if (!DTYPE_ID) throw new Error('polygrad: backend missing dtypeIds')
  let _seed = 0
  const ffi = _runtime._backend.ffi
  if (ffi.poly_set_frontend_buffer_release && !_runtime._backend.__frontendBufferReleaseRegistered) {
    ffi.poly_set_frontend_buffer_release((bufferKey) => {
      const key = normalizeBufferKey(bufferKey)
      hostBuffers.delete(key)
      if (_runtime._backend.unregisterHostBuffer) {
        _runtime._backend.unregisterHostBuffer(key)
      }
    })
    _runtime._backend.__frontendBufferReleaseRegistered = true
  }

  class Tensor {
    /**
     * @param {number|number[]|Float32Array|Float64Array|null} data
     * @param {object} opts
     */
    constructor(data, opts) {
      if (!opts) opts = {}
      const backend = _runtime._backend
      this._rt = _runtime
      this._ctx = opts._ctx || backend.ctx
      this._requiresGrad = opts.requiresGrad || false
      this._grad = null

      if (opts._uop) {
        // Internal construction from ops -- shape is on the UOp.
        // Accept either a UOp wrapper instance or a raw handle.
        if (opts._uop instanceof UOp) {
          this.uop = opts._uop
          this._uop = opts._uop.raw
        } else {
          this._uop = opts._uop
          this.uop = new UOp(this._ctx, backend.ffi, opts._uop)
        }
        this._buffer = opts._buffer || null
        this._data = opts._data || null
        this._inputs = opts._inputs || []
        this._dtype = opts._dtype || 'float32'
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
        // `uop` is the wrapper instance (has .buffer / .hasBufferIdentity);
        // `_uop` is the raw handle that existing FFI call sites pass into C.
        this.uop = UOp.fromHost(this._ctx, backend.ffi, flat, dtypeId, dims)
        this._uop = this.uop.raw
        this._buffer = this.uop.buffer ? this.uop.buffer.raw : null
        if (this._buffer && backend.ffi.poly_buffer_get_key) {
          const bufferKey = backend.ffi.poly_buffer_get_key(this._ctx, this._buffer)
          if (bufferKey) {
            const key = normalizeBufferKey(bufferKey)
            hostBuffers.set(key, flat)
            if (backend.registerHostBuffer) backend.registerHostBuffer(key, flat)
          }
        }
        this._inputs = []
      }
    }

    get shape() {
      const { ffi } = this._rt._backend
      if (!this._uop) return []
      return ffi.poly_uop_dims(this._ctx, this._uop)
    }
    get dtype() { return this._dtype }
    get device() { return String(this._rt.device || 'cpu').toUpperCase() }
    get ndim() { return this._rt._backend.ffi.poly_uop_ndim(this._ctx, this._uop) || 0 }
    get requiresGrad() { return this._requiresGrad }
    set requiresGrad(v) { this._requiresGrad = v }
    get grad() { return this._grad }
    get T() { return this.transpose() }

    numel() {
      return this.shape.reduce((a, b) => a * b, 1) || 1
    }

    size(dim) {
      if (dim === undefined || dim === null) return [...this.shape]
      if (dim < 0) dim += this.shape.length
      return this.shape[dim]
    }

    _isLeaf() {
      return this._data !== null && this._buffer !== null
    }

    _collectLeaves(seen) {
      if (!seen) seen = new Set()
      const id = this._buffer || this._uop
      if (seen.has(id)) return []
      seen.add(id)

      if (this._isLeaf()) return [this]

      const leaves = []
      for (const inp of this._inputs) {
        leaves.push(...inp._collectLeaves(seen))
      }
      return leaves
    }

    _collectLeafMap(map) {
      if (!map) map = new Map()
      const id = this._buffer || this._uop
      if (map.has(id)) return map

      if (this._isLeaf()) {
        map.set(this._buffer, this._data)
        return map
      }
      for (const inp of this._inputs) {
        inp._collectLeafMap(map)
      }
      return map
    }

    // _realize() is a back-compat shim. realize() (below) is the new entry
    // point matching Python's Tensor.realize(*lst) signature.
    // Back-compat shim. realize() (below) is the new entry point matching
    // Python's Tensor.realize(*lst) signature.
    async _realize() { await this.realize() }

    async realize(...lst) {
      // Triggers the computation needed to create these Tensor(s). Filters
      // out tensors already with buffer identity, batches the rest into a
      // single graph-side realize call, and retargets each tensor's uop
      // to its realized buffer-identity UOp. Port of Python's
      // Tensor.realize(*lst).
      const { ffi, ctx } = this._rt._backend
      const tensors = [this, ...lst]
      const targets = tensors.filter(t => !t.uop.hasBufferIdentity())
      if (!targets.length) return this
      const out = await ffi.poly_realize(ctx, targets.map(t => t._uop))
      if (!out || out.length !== targets.length) {
        throw new Error('poly_realize failed')
      }
      for (let i = 0; i < targets.length; i++) {
        const t = targets[i]
        t._saved_uop = t._uop
        t._uop = out[i]
        t.uop = new UOp(t._ctx, ffi, out[i])
        t._buffer = t.uop.buffer ? t.uop.buffer.raw : null
        t._inputs = []
      }
      return this
    }

    async _readBufferBytes() {
      const { ffi, ctx } = this._rt._backend
      const numel = this.numel()
      const AT = TA_BY_DTYPE[this._dtype] || Float32Array
      const itemsize = AT.BYTES_PER_ELEMENT
      const bufRaw = this.uop && this.uop.buffer ? this.uop.buffer.raw : null
      if (!bufRaw) throw new Error('toArray: tensor has no buffer identity')
      let raw
      const bufferKey = ffi.poly_buffer_get_key ? ffi.poly_buffer_get_key(ctx, bufRaw) : 0
      const normKey = bufferKey ? normalizeBufferKey(bufferKey) : null
      if (normKey && hostBuffers.has(normKey)) {
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
      if (this._dtype === 'float16' || this._dtype === 'bfloat16') {
        const t = this.cast('float32')
        await t.realize()
        return await t._readBufferBytes()
      }
      await this.realize()
      return await this._readBufferBytes()
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
      t._requiresGrad = this._requiresGrad
      return t
    }

    contiguous() {
      const { ffi } = this._rt._backend
      const uop = ffi.poly_contiguous(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    // --- Internal helpers ---

    _makeResult(uop, inputs, forcedDtype) {
      let dt = forcedDtype || 'float32'
      if (!forcedDtype) {
        for (let i = 0; i < inputs.length; i++) {
          if (inputs[i]._dtype === 'float64') { dt = 'float64'; break }
        }
      }
      const t = new Tensor(null, {
        _ctx: this._ctx,
        _uop: uop,
        _inputs: inputs,
        _dtype: dt
      })
      for (let i = 0; i < inputs.length; i++) {
        if (inputs[i]._requiresGrad) { t._requiresGrad = true; break }
      }
      return t
    }

    _ensureTensor(other) {
      if (other instanceof Tensor) return other
      if (typeof other === 'number') {
        const { ffi } = this._rt._backend
        const c = this._dtype === 'float64'
          ? ffi.poly_const_double(this._ctx, other)
          : ffi.poly_const_float(this._ctx, other)
        return new Tensor(null, { _ctx: this._ctx, _uop: c, _inputs: [],
                                  _dtype: this._dtype })
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
      if (arraysEqual(this.shape, targetShape)) return this._uop
      const { ffi } = this._rt._backend
      let uop = this._uop
      let curShape = [...this.shape]
      if (!curShape.length && this._buffer === null) return uop
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
      const { ffi, ops } = this._rt._backend
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
    lt(other) {
      const { ffi } = this._rt._backend
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const cmp = ffi.poly_alu2(
        this._ctx, this._rt._backend.ops.CMPLT, this._broadcastUop(outShape), other._broadcastUop(outShape))
      const uop = ffi.poly_cast_by_id(this._ctx, cmp, DTYPE_ID.float32)
      return this._makeResult(uop, [this, other], 'float32')
    }

    neg() {
      const { ffi, ops } = this._rt._backend
      const uop = ffi.poly_alu1(this._ctx, ops.NEG, this._uop)
      return this._makeResult(uop, [this])
    }

    // --- Comparisons (C core) ---

    eq(other) {
      const { ffi } = this._rt._backend
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const cmp = ffi.poly_eq(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      const uop = ffi.poly_cast_by_id(this._ctx, cmp, DTYPE_ID.float32)
      return this._makeResult(uop, [this, other], 'float32')
    }

    ne(other) {
      const { ffi } = this._rt._backend
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const cmp = ffi.poly_ne(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      const uop = ffi.poly_cast_by_id(this._ctx, cmp, DTYPE_ID.float32)
      return this._makeResult(uop, [this, other], 'float32')
    }

    gt(other) {
      const { ffi } = this._rt._backend
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const cmp = ffi.poly_gt(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      const uop = ffi.poly_cast_by_id(this._ctx, cmp, DTYPE_ID.float32)
      return this._makeResult(uop, [this, other], 'float32')
    }

    ge(other) {
      const { ffi } = this._rt._backend
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const cmp = ffi.poly_ge(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      const uop = ffi.poly_cast_by_id(this._ctx, cmp, DTYPE_ID.float32)
      return this._makeResult(uop, [this, other], 'float32')
    }

    le(other) {
      const { ffi } = this._rt._backend
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const cmp = ffi.poly_le(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      const uop = ffi.poly_cast_by_id(this._ctx, cmp, DTYPE_ID.float32)
      return this._makeResult(uop, [this, other], 'float32')
    }

    where(x, y) {
      const { ffi } = this._rt._backend
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
      const { ffi } = this._rt._backend
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const uop = ffi.poly_maximum(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      return this._makeResult(uop, [this, other])
    }

    minimum(other) {
      const { ffi } = this._rt._backend
      other = this._ensureTensor(other)
      const outShape = this._broadcastShape(other.shape)
      const uop = ffi.poly_minimum(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
      return this._makeResult(uop, [this, other])
    }

    clamp(lo, hi) {
      if (lo === undefined && hi === undefined) {
        throw new Error("at least one of 'lo' or 'hi' must not be undefined")
      }
      const { ffi } = this._rt._backend
      lo = lo !== undefined ? lo : -1e38
      hi = hi !== undefined ? hi : 1e38
      const uop = ffi.poly_clamp(this._ctx, this._uop, lo, hi)
      return this._makeResult(uop, [this])
    }

    // --- Cast ---

    cast(dtype) {
      if (dtype === this._dtype) return this
      const id = DTYPE_ID[dtype]
      if (id === undefined) throw new Error(`unsupported cast target dtype: ${dtype}`)
      const { ffi } = this._rt._backend
      const uop = ffi.poly_cast_by_id(this._ctx, this._uop, id)
      if (!uop) throw new Error(`poly_cast_by_id failed for dtype ${dtype}`)
      return new Tensor(null, {
        _ctx: this._ctx, _uop: uop,
        _inputs: [this], _dtype: dtype
      })
    }

    half() { return this.cast('float16') }
    double() { return this.cast('float64') }

    // --- Triu/Tril ---

    triu(diagonal = 0) {
      const { ffi } = this._rt._backend
      const uop = ffi.poly_triu(this._ctx, this._uop, diagonal)
      return this._makeResult(uop, [this])
    }

    tril(diagonal = 0) {
      const { ffi } = this._rt._backend
      const uop = ffi.poly_tril(this._ctx, this._uop, diagonal)
      return this._makeResult(uop, [this])
    }

    // --- Unary math (C core composed ops) ---

    exp2() {
      const { ffi, ops } = this._rt._backend
      const uop = ffi.poly_alu1(this._ctx, ops.EXP2, this._uop)
      return this._makeResult(uop, [this])
    }

    log2() {
      const { ffi, ops } = this._rt._backend
      const uop = ffi.poly_alu1(this._ctx, ops.LOG2, this._uop)
      return this._makeResult(uop, [this])
    }

    sqrt() {
      const { ffi, ops } = this._rt._backend
      const uop = ffi.poly_alu1(this._ctx, ops.SQRT, this._uop)
      return this._makeResult(uop, [this])
    }

    reciprocal() {
      const { ffi, ops } = this._rt._backend
      const uop = ffi.poly_alu1(this._ctx, ops.RECIPROCAL, this._uop)
      return this._makeResult(uop, [this])
    }

    trunc() {
      const { ffi, ops } = this._rt._backend
      const uop = ffi.poly_alu1(this._ctx, ops.TRUNC, this._uop)
      return this._makeResult(uop, [this])
    }

    exp() {
      const uop = this._rt._backend.ffi.poly_exp(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    log() {
      const uop = this._rt._backend.ffi.poly_log(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    log1p() {
      const uop = this._rt._backend.ffi.poly_log1p(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    expm1() {
      const uop = this._rt._backend.ffi.poly_expm1(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    sin() {
      const uop = this._rt._backend.ffi.poly_sin(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    cos() {
      const uop = this._rt._backend.ffi.poly_cos(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    tan() {
      const uop = this._rt._backend.ffi.poly_tan(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    sigmoid() {
      const uop = this._rt._backend.ffi.poly_sigmoid(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    tanh() {
      const uop = this._rt._backend.ffi.poly_tanh_act(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    abs() {
      const uop = this._rt._backend.ffi.poly_abs(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    sign() {
      const uop = this._rt._backend.ffi.poly_sign(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    square() {
      const uop = this._rt._backend.ffi.poly_square(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    rsqrt() {
      const uop = this._rt._backend.ffi.poly_rsqrt(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    ceil() {
      const uop = this._rt._backend.ffi.poly_ceil(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    floor() {
      const uop = this._rt._backend.ffi.poly_floor(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    round() {
      const uop = this._rt._backend.ffi.poly_round_f(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    isinf() {
      const uop = this._rt._backend.ffi.poly_isinf(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    isnan() {
      const uop = this._rt._backend.ffi.poly_isnan(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    // --- Activations (C core composed ops) ---

    relu() {
      const uop = this._rt._backend.ffi.poly_relu(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    relu6() {
      const uop = this._rt._backend.ffi.poly_relu6(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    leakyRelu(negSlope) {
      if (negSlope === undefined) negSlope = 0.01
      const uop = this._rt._backend.ffi.poly_leaky_relu(this._ctx, this._uop, negSlope)
      return this._makeResult(uop, [this])
    }

    gelu() {
      const uop = this._rt._backend.ffi.poly_gelu(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    quickGelu() {
      const uop = this._rt._backend.ffi.poly_quick_gelu(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    silu() {
      const uop = this._rt._backend.ffi.poly_silu(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    swish() { return this.silu() }

    elu(alpha) {
      if (alpha === undefined) alpha = 1.0
      const uop = this._rt._backend.ffi.poly_elu(this._ctx, this._uop, alpha)
      return this._makeResult(uop, [this])
    }

    softplus(beta) {
      if (beta === undefined) beta = 1.0
      const uop = this._rt._backend.ffi.poly_softplus(this._ctx, this._uop, beta)
      return this._makeResult(uop, [this])
    }

    mish() {
      const uop = this._rt._backend.ffi.poly_mish(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    hardtanh(minVal, maxVal) {
      if (minVal === undefined) minVal = -1
      if (maxVal === undefined) maxVal = 1
      const uop = this._rt._backend.ffi.poly_hardtanh(this._ctx, this._uop, minVal, maxVal)
      return this._makeResult(uop, [this])
    }

    hardswish() {
      const uop = this._rt._backend.ffi.poly_hardswish(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    hardsigmoid() {
      const uop = this._rt._backend.ffi.poly_hardsigmoid(this._ctx, this._uop)
      return this._makeResult(uop, [this])
    }

    // --- Softmax ---

    async softmax(axis) {
      if (axis === undefined) axis = -1
      const m = await this.max({ axis, keepdim: true }).realize()
      const e = await this.sub(m).exp().realize()
      const s = await e.sum(axis, true).realize()
      return e.div(s)
    }

    async logSoftmax(axis) {
      if (axis === undefined) axis = -1
      const m = await this.max({ axis, keepdim: true }).realize()
      const shifted = await this.sub(m).realize()
      const e = await shifted.exp().realize()
      const s = await e.sum(axis, true).realize()
      return shifted.sub(s.log())
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
      const uop = this._rt._backend.ffi.poly_reshape(this._ctx, this._uop, shape, shape.length)
      return this._makeResult(uop, [this])
    }

    permute(...order) {
      if (order.length === 1 && Array.isArray(order[0])) order = order[0]
      const uop = this._rt._backend.ffi.poly_permute(this._ctx, this._uop, order, order.length)
      const newShape = order.map(i => this.shape[i])
      return this._makeResult(uop, [this])
    }

    expand(...shape) {
      if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
      const uop = this._rt._backend.ffi.poly_expand(this._ctx, this._uop, shape, shape.length)
      return this._makeResult(uop, [this])
    }

    shrink(arg) {
      const flat = []
      for (let i = 0; i < arg.length; i++) {
        flat.push(arg[i][0], arg[i][1])
      }
      const uop = this._rt._backend.ffi.poly_shrink(this._ctx, this._uop, flat, arg.length)
      const newShape = arg.map(([s, e]) => e - s)
      return this._makeResult(uop, [this])
    }

    pad(arg) {
      const flat = []
      for (let i = 0; i < arg.length; i++) {
        flat.push(arg[i][0], arg[i][1])
      }
      const uop = this._rt._backend.ffi.poly_pad(this._ctx, this._uop, flat, arg.length)
      const newShape = this.shape.map((s, i) => s + arg[i][0] + arg[i][1])
      return this._makeResult(uop, [this])
    }

    flip(axis) {
      if (typeof axis === 'number') axis = [axis]
      const uop = this._rt._backend.ffi.poly_flip(this._ctx, this._uop, axis, axis.length)
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

      const { ffi, ops } = this._rt._backend
      let uop = ffi.poly_reduce_axis(this._ctx, ops.ADD, this._uop, axis, axis.length)

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

      const { ffi } = this._rt._backend

      if (axis === undefined || axis === null) {
        let result = this
        for (let i = this.shape.length - 1; i >= 0; i--) {
          const uop = ffi.poly_max_reduce(this._ctx, result._uop, i, keepdim ? 1 : 0)
          result = this._makeResult(uop, [result])
        }
        return result
      }

      if (axis < 0) axis += this.shape.length
      const uop = ffi.poly_max_reduce(this._ctx, this._uop, axis, keepdim ? 1 : 0)
      return this._makeResult(uop, [this])
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
      const { ffi } = this._rt._backend
      const uop = ffi.poly_mean_reduce(this._ctx, this._uop, axis, keepdim ? 1 : 0)
      return this._makeResult(uop, [this])
    }

    async var(axis, keepdim, correction) {
      if (keepdim === undefined) keepdim = false
      if (correction === undefined) correction = 1
      if (axis === undefined || axis === null) {
        const m = this.mean()
        const diff = this.sub(m)
        const sq = diff.mul(diff)
        return sq.sum().div(this.numel() - correction)
      }
      if (axis < 0) axis += this.shape.length
      const m = await this.mean(axis, true).realize()
      const diff = this.sub(m)
      const sq = diff.mul(diff)
      const dimSize = this.shape[axis]
      return sq.sum(axis, keepdim).div(dimSize - correction)
    }

    async std(axis, keepdim, correction) {
      if (keepdim === undefined) keepdim = false
      if (correction === undefined) correction = 1
      return (await this.var(axis, keepdim, correction)).sqrt()
    }

    // --- Matmul (C core dot) ---

    dot(w) {
      if (!(w instanceof Tensor)) {
        throw new TypeError(`Expected Tensor, got ${typeof w}`)
      }
      const { ffi } = this._rt._backend
      const uop = ffi.poly_dot(this._ctx, this._uop, w._uop)
      if (!uop) {
        throw new Error(`cannot dot ${JSON.stringify(this.shape)} and ${JSON.stringify(w.shape)}`)
      }
      return this._makeResult(uop, [this, w])
    }

    matmul(other) { return this.dot(other) }

    linear(weight, bias) {
      let result = this.dot(weight.transpose(-1, -2))
      if (bias) result = result.add(bias)
      return result
    }

    // --- Loss functions ---

    async crossEntropy(target, axis) {
      if (axis === undefined) axis = this.shape.length === 1 ? 0 : 1
      if (!(target instanceof Tensor)) target = new Tensor(target)
      const { ffi } = this._rt._backend
      const uop = ffi.poly_cross_entropy(this._ctx,
        this._uop, target._uop, axis)
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

    async layernorm(axis, eps) {
      if (axis === undefined) axis = -1
      if (eps === undefined) eps = 1e-5
      const m = this.mean(axis, true)
      const v = await this.var(axis, true, 0)
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
      const { ffi } = _runtime._backend
      const ctx = operands[0]._ctx
      const r = ffi.poly_einsum(ctx, formula, operands)
      if (!r.uop) throw new Error(`poly_einsum failed for formula: ${formula}`)
      return new Tensor(null, { _ctx: ctx, _uop: r.uop, _inputs: [...operands] })
    }

    // --- Rearrange (C core, einops-style) ---

    rearrange(formula, kwargs) {
      if (!kwargs) kwargs = {}
      const { ffi } = this._rt._backend
      const r = ffi.poly_rearrange(this._ctx, formula, this._uop, this.shape, kwargs)
      if (!r.uop) throw new Error(`poly_rearrange failed for formula: ${formula}`)
      return this._makeResult(r.uop, [this])
    }

    // --- Autograd ---

    async backward() {
      const backend = this._rt._backend
      const { ffi } = backend
      const leafMap = this._collectLeafMap()
      const allLeaves = this._collectLeaves()
      const gradLeaves = allLeaves.filter(t => t._requiresGrad)
      if (!gradLeaves.length) {
        throw new Error('No leaf tensors require grad')
      }

      for (const leaf of gradLeaves) {
        const gradUop = ffi.poly_grad(this._ctx, this._uop, leaf._buffer)
        if (!gradUop) {
          throw new Error('poly_grad returned NULL for a leaf tensor')
        }

        const numel = leaf.shape.reduce((a, b) => a * b, 1) || 1
        const isF64 = leaf._dtype === 'float64'
        const gradBuf = isF64
          ? ffi.poly_buffer_f64(this._ctx, numel)
          : ffi.poly_buffer_f32(this._ctx, numel)
        const store = ffi.poly_store_val(this._ctx, gradBuf, gradUop)
        const sink = ffi.poly_sink1(this._ctx, store)

        leafMap.set(gradBuf, isF64 ? new Float64Array(numel) : new Float32Array(numel))

        const gradResult = await backend.realize(this._ctx, sink, numel, leafMap, isF64)

        const gradTensor = new Tensor(gradResult, { _ctx: this._ctx, dtype: leaf._dtype })
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
      const { ffi, ctx } = _runtime._backend
      const seed = _seed++
      const uop = ffi.poly_rand(ctx, shape, shape.length, seed)
      if (!uop) throw new Error('poly_rand failed')
      return new Tensor(null, {
        _ctx: ctx, _uop: uop, _inputs: [],
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
      const { ffi, ctx } = _runtime._backend
      const seed = _seed++
      const uop = ffi.poly_randn(ctx, shape, shape.length, seed)
      if (!uop) throw new Error('poly_randn failed')
      return new Tensor(null, {
        _ctx: ctx, _uop: uop, _inputs: [],
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
      const AT = Tensor._resolveArrayType(opts)
      const numel = shape.reduce((a, b) => a * b, 1)
      const t = new Tensor(new AT(numel), opts)
      if (shape.length > 1) return t.reshape(...shape)
      return t
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
      return `Tensor(shape=[${this.shape}], dtype=${this.dtype}, realized=${this._isLeaf()})`
    }
  }

  return Tensor
}

module.exports = { createBoundTensorClass, flattenArray, arraysEqual, _buildNested }
