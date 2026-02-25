/**
 * polygrad-browser — Browser/WASM frontend for the polygrad tensor compiler.
 *
 * Architecture:
 * - Emscripten-compiled polygrad core (build/polygrad.js + polygrad.wasm) handles
 *   graph construction, scheduling, linearization, and WASM kernel rendering.
 * - Each realize() call gets a WASM binary from the core, then instantiates
 *   and executes it via WebAssembly.instantiate().
 */

'use strict'

const path = require('path')

// Load the Emscripten module
const PolygradModuleFactory = require(
  path.resolve(__dirname, '..', '..', 'build', 'polygrad.js')
)

let Module = null
let _ffi = null

// Pre-allocated scratch pointers (initialized in init())
let _scratchLenPtr = 0
let _scratchNBufsPtr = 0
// Pre-allocated scratch buffer for int64 axis arrays (up to 8 dims)
let _scratchAxisPtr = 0
// Pre-allocated scratch for output shape/ndim (8 dims × 8 bytes + 4 bytes)
let _scratchOutShapePtr = 0
let _scratchOutNdimPtr = 0

// Execution context cache: hash(wasmBytes) → full execution context
// Caches instance + memory + buffer ordering + offsets to eliminate
// poly_kernel_buf FFI calls and Map allocations on cache hit
const _execCache = new Map()

function hashBytes(bytes) {
  // FNV-1a 32-bit — fast, good distribution for small buffers
  let h = 0x811c9dc5
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i]
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/**
 * Math imports for WASM kernels.
 * The kernel WASM module imports these from the "math" namespace.
 */
const mathImports = {
  exp2f: (x) => Math.pow(2, x),
  log2f: (x) => Math.log2(x),
  sinf: (x) => Math.sin(x),
  powf: (a, b) => Math.pow(a, b)
}

/**
 * Core kernel execution: render → cache lookup → copy data → execute → copy result.
 * Shared by _realize() and backward(). Returns Float32Array with output data.
 *
 * bufferDataFn(bufUop) should return the Float32Array data for a given buffer UOp.
 */
function _renderAndExec(ctx, sink, outBuf, numel, bufferDataFn) {
  // 1. Render kernel (C-side cache hit after first call)
  const wasmPtr = Module._poly_render_kernel_wasm(ctx, sink, _scratchLenPtr, _scratchNBufsPtr)
  // Direct HEAP32 access — faster than Module.getValue(ptr, 'i32')
  const wasmLen = Module.HEAP32[_scratchLenPtr >> 2]
  const nBufs = Module.HEAP32[_scratchNBufsPtr >> 2]

  if (!wasmPtr || wasmLen === 0) {
    throw new Error('poly_render_kernel_wasm failed')
  }

  // 2. Get current buffer UOps and their data (must be done every call
  //    since buffer UOp pointers change between training steps)
  const bufData = new Array(nBufs)
  for (let i = 0; i < nBufs; i++) {
    const bufUop = Module._poly_kernel_buf(ctx, i)
    const data = bufferDataFn(bufUop)
    if (!data) throw new Error(`No data binding for kernel buffer param ${i}`)
    bufData[i] = data
  }

  // 3. Cache lookup: instance + memory + offsets (stable for same kernel)
  const wasmBytes = new Uint8Array(Module.HEAPU8.buffer, wasmPtr, wasmLen)
  const key = hashBytes(wasmBytes)
  let exec = _execCache.get(key)

  if (!exec) {
    // Cache miss: compute offsets, compile, instantiate
    const offsets = new Array(nBufs)
    let totalBytes = 0
    for (let i = 0; i < nBufs; i++) {
      offsets[i] = totalBytes
      totalBytes += bufData[i].length * 4
    }

    const neededPages = Math.max(1, Math.ceil(totalBytes / 65536))
    const memory = new WebAssembly.Memory({ initial: neededPages })
    const module = new WebAssembly.Module(wasmBytes)
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
      math: mathImports
    })

    exec = { instance, memory, memPages: neededPages, nBufs, offsets, totalBytes }
    _execCache.set(key, exec)
  }

  Module._free(wasmPtr)

  // 4. Copy input data into persistent WASM memory
  const memView = new Float32Array(exec.memory.buffer)
  for (let i = 0; i < nBufs; i++) {
    memView.set(bufData[i], exec.offsets[i] / 4)
  }

  // 5. Execute kernel
  exec.instance.exports.kernel(...exec.offsets)

  // 6. Copy output data back
  const outView = new Float32Array(exec.memory.buffer)
  const result = new Float32Array(numel)
  result.set(outView.subarray(exec.offsets[0] / 4, exec.offsets[0] / 4 + numel))

  return result
}

/**
 * Initialize the polygrad WASM module. Must be called before using Tensor.
 * Returns a promise that resolves when the module is ready.
 */
async function init() {
  if (Module) return
  Module = await PolygradModuleFactory()

  // Hot-path FFI: direct Module._ calls (skip cwrap overhead)
  // Only use cwrap for functions needing string marshalling or infrequent use
  const cwrapName = Module.cwrap('poly_op_name', 'string', ['number'])
  const cwrapReshape = Module.cwrap('poly_reshape', 'number', ['number', 'number', 'number', 'number'])
  const cwrapExpand = Module.cwrap('poly_expand', 'number', ['number', 'number', 'number', 'number'])
  const cwrapPermute = Module.cwrap('poly_permute', 'number', ['number', 'number', 'number', 'number'])
  const cwrapShrink = Module.cwrap('poly_shrink', 'number', ['number', 'number', 'number', 'number'])
  const cwrapFlip = Module.cwrap('poly_flip', 'number', ['number', 'number', 'number', 'number'])
  const cwrapPad = Module.cwrap('poly_pad', 'number', ['number', 'number', 'number', 'number'])

  _ffi = {
    poly_ctx_new: Module._poly_ctx_new,
    poly_ctx_destroy: Module._poly_ctx_destroy,
    poly_const_float: Module._poly_const_float,
    poly_const_int: (ctx, val) => Module._poly_const_int(ctx, BigInt(val)),
    poly_alu1: Module._poly_alu1,
    poly_alu2: Module._poly_alu2,
    poly_alu3: Module._poly_alu3,
    poly_store_val: Module._poly_store_val,
    poly_sink1: Module._poly_sink1,
    poly_buffer_f32: (ctx, size) => Module._poly_buffer_f32(ctx, BigInt(size)),
    poly_reshape: cwrapReshape,
    poly_expand: cwrapExpand,
    poly_reduce_axis: Module._poly_reduce_axis,
    poly_permute: cwrapPermute,
    poly_shrink: cwrapShrink,
    poly_flip: cwrapFlip,
    poly_pad: cwrapPad,
    poly_grad: Module._poly_grad,
    poly_render_kernel_wasm: Module._poly_render_kernel_wasm,
    poly_kernel_buf: Module._poly_kernel_buf,
    // Composed elementwise ops
    poly_exp: Module._poly_exp,
    poly_log: Module._poly_log,
    poly_sin: Module._poly_sin,
    poly_cos: Module._poly_cos,
    poly_tan: Module._poly_tan,
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
    // Shape-aware ops
    poly_max_reduce: Module._poly_max_reduce,
    poly_mean_reduce: Module._poly_mean_reduce,
    poly_dot: Module._poly_dot,
    poly_einsum: Module._poly_einsum,
    poly_rearrange: Module._poly_rearrange,
    OPS: {}
  }

  // Build op name -> int mapping
  const opCount = Module._poly_op_count()
  for (let i = 0; i < opCount; i++) {
    const name = cwrapName(i)
    if (name) _ffi.OPS[name] = i
  }

  // Pre-allocate scratch pointers for render calls
  _scratchLenPtr = Module._malloc(4)
  _scratchNBufsPtr = Module._malloc(4)
  // Scratch buffer for int64 axis arrays (8 dims × 8 bytes = 64 bytes)
  _scratchAxisPtr = Module._malloc(64)
  // Scratch for output shape (8 × 8 bytes) and ndim (4 bytes)
  _scratchOutShapePtr = Module._malloc(64)
  _scratchOutNdimPtr = Module._malloc(4)

  // Create default context
  Tensor._defaultCtx = _ffi.poly_ctx_new()
}

/**
 * Write an int64 array into the Emscripten heap.
 * Emscripten WASM is 32-bit, so int64_t is stored as two i32s (little-endian).
 * Returns the heap pointer (must be freed with Module._free).
 */
function writeInt64Array(arr) {
  const ptr = Module._malloc(arr.length * 8)
  const h = Module.HEAP32
  for (let i = 0; i < arr.length; i++) {
    const base = (ptr >> 2) + i * 2
    const val = arr[i]
    h[base] = val & 0xFFFFFFFF
    h[base + 1] = val < 0 ? -1 : 0
  }
  return ptr
}

/**
 * Write an int64 array into the pre-allocated scratch buffer.
 * Returns scratch pointer (DO NOT free). Only for small arrays (<=8 elements).
 */
function writeInt64Scratch(arr) {
  const h = Module.HEAP32
  for (let i = 0; i < arr.length; i++) {
    const base = (_scratchAxisPtr >> 2) + i * 2
    const val = arr[i]
    h[base] = val & 0xFFFFFFFF
    h[base + 1] = val < 0 ? -1 : 0
  }
  return _scratchAxisPtr
}

/**
 * Read output shape from the scratch out_shape/out_ndim pointers.
 */
function readOutShape() {
  const ndim = Module.HEAP32[_scratchOutNdimPtr >> 2]
  const shape = []
  for (let i = 0; i < ndim; i++) {
    const base = (_scratchOutShapePtr >> 2) + i * 2
    const lo = Module.HEAP32[base] >>> 0
    const hi = Module.HEAP32[base + 1]
    shape.push(hi >= 0 ? (hi * 0x100000000 + lo) : -(~hi * 0x100000000 + (~lo >>> 0) + 1))
  }
  return shape
}

/**
 * Write shape to scratch out buffer and return the pointer.
 */
function writeShapeToScratch(shape) {
  const h = Module.HEAP32
  for (let i = 0; i < shape.length; i++) {
    const base = (_scratchOutShapePtr >> 2) + i * 2
    const val = shape[i]
    h[base] = val & 0xFFFFFFFF
    h[base + 1] = val < 0 ? -1 : 0
  }
  return _scratchOutShapePtr
}

/**
 * Flatten a nested JS array and return { data: Float32Array, shape: number[] }.
 */
function flattenArray(arr) {
  if (typeof arr === 'number') {
    return { data: new Float32Array([arr]), shape: [1] }
  }
  if (arr instanceof Float32Array) {
    return { data: new Float32Array(arr), shape: [arr.length] }
  }
  if (!Array.isArray(arr)) {
    throw new Error('Expected number, array, or Float32Array')
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
      flat.push(a)
    }
  }
  recurse(arr)

  return { data: new Float32Array(flat), shape }
}

/**
 * Call a polygrad C function that takes an int64_t* array parameter.
 * Uses scratch buffer for small arrays (<=8), heap alloc for larger.
 */
function callWithInt64Array(fn, ctx, uop, arr, ...extra) {
  if (arr.length <= 8) {
    const ptr = writeInt64Scratch(arr)
    return fn(ctx, uop, ptr, ...extra)
  }
  const ptr = writeInt64Array(arr)
  const result = fn(ctx, uop, ptr, ...extra)
  Module._free(ptr)
  return result
}

// ── Tensor class ─────────────────────────────────────────────────────

class Tensor {
  constructor(data, opts = {}) {
    this._ctx = opts._ctx || Tensor._defaultCtx
    this._requiresGrad = opts.requiresGrad || false
    this._grad = null

    if (opts._uop) {
      this._uop = opts._uop
      this._buffer = opts._buffer || null
      this._data = opts._data || null
      this._shape = opts._shape ? [...opts._shape] : []
      this._inputs = opts._inputs || []
    } else if (data instanceof Float32Array) {
      // Fast path: Float32Array input (most common in training loops)
      this._shape = [data.length]
      this._data = new Float32Array(data)
      this._buffer = _ffi.poly_buffer_f32(this._ctx, data.length)
      this._uop = this._buffer
      this._inputs = []
    } else {
      const { data: flat, shape } = flattenArray(data)
      this._shape = shape
      this._data = flat
      this._buffer = _ffi.poly_buffer_f32(this._ctx, flat.length)
      if (shape.length > 1) {
        this._uop = callWithInt64Array(_ffi.poly_reshape, this._ctx, this._buffer, shape, shape.length)
      } else {
        this._uop = this._buffer
      }
      this._inputs = []
    }
  }

  get shape() { return [...this._shape] }
  get dtype() { return 'float32' }
  get device() { return 'CPU' }
  get ndim() { return this._shape.length }
  get requiresGrad() { return this._requiresGrad }
  set requiresGrad(v) { this._requiresGrad = v }
  get grad() { return this._grad }
  get T() { return this.transpose() }

  numel() {
    return this._shape.reduce((a, b) => a * b, 1) || 1
  }

  size(dim) {
    if (dim === undefined || dim === null) return [...this._shape]
    if (dim < 0) dim += this._shape.length
    return this._shape[dim]
  }

  _isLeaf() {
    return this._data !== null && this._buffer !== null
  }

  // Collect leaf tensors, building a buffer→data lookup table
  _collectLeafMap(map) {
    if (!map) map = new Map()
    const id = this._buffer || this._uop
    if (map.has(id)) return map
    if (this._isLeaf()) {
      map.set(this._buffer, this._data)
      return map
    }
    for (let i = 0; i < this._inputs.length; i++) {
      this._inputs[i]._collectLeafMap(map)
    }
    return map
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

  /**
   * Realize: render WASM kernel, execute via cached instance.
   */
  _realize() {
    if (this._isLeaf()) return

    const numel = this._shape.reduce((a, b) => a * b, 1) || 1
    const outBuf = _ffi.poly_buffer_f32(this._ctx, numel)
    const store = _ffi.poly_store_val(this._ctx, outBuf, this._uop)
    const sink = _ffi.poly_sink1(this._ctx, store)

    // Build buffer→data map (single traversal)
    const leafMap = this._collectLeafMap()
    const outData = new Float32Array(numel)
    leafMap.set(outBuf, outData)

    const result = _renderAndExec(this._ctx, sink, outBuf, numel, (bufUop) => leafMap.get(bufUop))

    this._data = result
    this._buffer = outBuf
    // Preserve shape in UOp graph so subsequent ops see correct dimensions
    if (this._shape.length > 1) {
      this._uop = callWithInt64Array(_ffi.poly_reshape, this._ctx, outBuf, this._shape, this._shape.length)
    } else {
      this._uop = outBuf
    }
    this._inputs = []
  }

  realize() {
    this._realize()
    return this
  }

  toArray() {
    this._realize()
    return new Float32Array(this._data)
  }

  item() {
    this._realize()
    if (this._data.length !== 1) {
      throw new Error(`item() requires scalar tensor, got shape [${this._shape}]`)
    }
    return this._data[0]
  }

  tolist() {
    this._realize()
    return _buildNested(this._data, this._shape, 0, 0).value
  }

  detach() {
    return new Tensor(this.toArray())
  }

  clone() {
    const t = new Tensor(this.toArray())
    t._requiresGrad = this._requiresGrad
    return t
  }

  contiguous() { return this }

  // --- Internal helpers ---

  _makeResult(uop, shape, inputs) {
    const t = new Tensor(null, {
      _ctx: this._ctx,
      _uop: uop,
      _shape: shape,
      _inputs: inputs
    })
    for (let i = 0; i < inputs.length; i++) {
      if (inputs[i]._requiresGrad) { t._requiresGrad = true; break }
    }
    return t
  }

  _ensureTensor(other) {
    if (other instanceof Tensor) return other
    if (typeof other === 'number') {
      const c = _ffi.poly_const_float(this._ctx, other)
      return new Tensor(null, { _ctx: this._ctx, _uop: c, _shape: [], _inputs: [] })
    }
    throw new TypeError(`Cannot convert ${typeof other} to Tensor`)
  }

  _broadcastShape(otherShape) {
    const a = this._shape
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
    if (arraysEqual(this._shape, targetShape)) return this._uop
    let uop = this._uop
    let curShape = [...this._shape]
    if (!curShape.length && this._buffer === null) return uop
    const targetNd = targetShape.length
    if (curShape.length < targetNd) {
      curShape = new Array(targetNd - curShape.length).fill(1).concat(curShape)
      uop = callWithInt64Array(_ffi.poly_reshape, this._ctx, uop, curShape, curShape.length)
    }
    if (!arraysEqual(curShape, targetShape)) {
      uop = callWithInt64Array(_ffi.poly_expand, this._ctx, uop, targetShape, targetShape.length)
    }
    return uop
  }

  _binop(other, opName) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const xUop = this._broadcastUop(outShape)
    const yUop = other._broadcastUop(outShape)
    const uop = _ffi.poly_alu2(this._ctx, _ffi.OPS[opName], xUop, yUop)
    return this._makeResult(uop, outShape, [this, other])
  }

  // --- Element-wise arithmetic ---

  add(other) { return this._binop(other, 'ADD') }
  sub(other) { return this._binop(other, 'SUB') }
  mul(other) { return this._binop(other, 'MUL') }
  div(other) { return this._binop(other, 'FDIV') }
  pow(other) { return this._binop(other, 'POW') }
  lt(other) { return this._binop(other, 'CMPLT') }

  neg() {
    const uop = _ffi.poly_alu1(this._ctx, _ffi.OPS.NEG, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  // --- Comparisons (C core) ---

  eq(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = _ffi.poly_eq(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  ne(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = _ffi.poly_ne(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  gt(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = _ffi.poly_gt(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  ge(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = _ffi.poly_ge(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  le(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = _ffi.poly_le(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  where(x, y) {
    x = this._ensureTensor(x)
    y = this._ensureTensor(y)
    let outShape = this._broadcastShape(x._shape)
    const tmp = new Tensor(null, { _ctx: this._ctx, _uop: this._uop, _shape: outShape, _inputs: [] })
    outShape = tmp._broadcastShape(y._shape)
    const cUop = this._broadcastUop(outShape)
    const xUop = x._broadcastUop(outShape)
    const yUop = y._broadcastUop(outShape)
    const uop = _ffi.poly_where_op(this._ctx, cUop, xUop, yUop)
    return this._makeResult(uop, outShape, [this, x, y])
  }

  maximum(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = _ffi.poly_maximum(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  minimum(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = _ffi.poly_minimum(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  clamp(lo, hi) {
    if (lo === undefined && hi === undefined) {
      throw new Error("at least one of 'lo' or 'hi' must not be undefined")
    }
    lo = lo !== undefined ? lo : -1e38
    hi = hi !== undefined ? hi : 1e38
    const uop = _ffi.poly_clamp(this._ctx, this._uop, lo, hi)
    return this._makeResult(uop, [...this._shape], [this])
  }

  // --- Unary math (C core composed ops) ---

  exp2() {
    const uop = _ffi.poly_alu1(this._ctx, _ffi.OPS.EXP2, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  log2() {
    const uop = _ffi.poly_alu1(this._ctx, _ffi.OPS.LOG2, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  sqrt() {
    const uop = _ffi.poly_alu1(this._ctx, _ffi.OPS.SQRT, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  reciprocal() {
    const uop = _ffi.poly_alu1(this._ctx, _ffi.OPS.RECIPROCAL, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  trunc() {
    const uop = _ffi.poly_alu1(this._ctx, _ffi.OPS.TRUNC, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  exp() {
    const uop = _ffi.poly_exp(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  log() {
    const uop = _ffi.poly_log(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  sin() {
    const uop = _ffi.poly_sin(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  cos() {
    const uop = _ffi.poly_cos(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  tan() {
    const uop = _ffi.poly_tan(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  sigmoid() {
    const uop = _ffi.poly_sigmoid(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  tanh() {
    const uop = _ffi.poly_tanh_act(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  abs() {
    const uop = _ffi.poly_abs(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  sign() {
    const uop = _ffi.poly_sign(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  square() {
    const uop = _ffi.poly_square(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  rsqrt() {
    const uop = _ffi.poly_rsqrt(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  ceil() {
    const uop = _ffi.poly_ceil(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  floor() {
    const uop = _ffi.poly_floor(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  round() {
    const uop = _ffi.poly_round_f(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  isinf() {
    const uop = _ffi.poly_isinf(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  isnan() {
    const uop = _ffi.poly_isnan(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  // --- Activations (C core composed ops) ---

  relu() {
    const uop = _ffi.poly_relu(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  relu6() {
    const uop = _ffi.poly_relu6(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  leakyRelu(negSlope = 0.01) {
    const uop = _ffi.poly_leaky_relu(this._ctx, this._uop, negSlope)
    return this._makeResult(uop, [...this._shape], [this])
  }

  gelu() {
    const uop = _ffi.poly_gelu(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  quickGelu() {
    const uop = _ffi.poly_quick_gelu(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  silu() {
    const uop = _ffi.poly_silu(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  swish() { return this.silu() }

  elu(alpha = 1.0) {
    const uop = _ffi.poly_elu(this._ctx, this._uop, alpha)
    return this._makeResult(uop, [...this._shape], [this])
  }

  softplus(beta = 1.0) {
    const uop = _ffi.poly_softplus(this._ctx, this._uop, beta)
    return this._makeResult(uop, [...this._shape], [this])
  }

  mish() {
    const uop = _ffi.poly_mish(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  hardtanh(minVal = -1, maxVal = 1) {
    const uop = _ffi.poly_hardtanh(this._ctx, this._uop, minVal, maxVal)
    return this._makeResult(uop, [...this._shape], [this])
  }

  hardswish() {
    const uop = _ffi.poly_hardswish(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  hardsigmoid() {
    const uop = _ffi.poly_hardsigmoid(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  // --- Softmax (Tensor-level composition with realize boundaries) ---

  softmax(axis = -1) {
    const m = this.max({ axis, keepdim: true }).realize()
    const e = this.sub(m).exp().realize()
    const s = e.sum(axis, true).realize()
    return e.div(s)
  }

  logSoftmax(axis = -1) {
    const m = this.max({ axis, keepdim: true }).realize()
    const shifted = this.sub(m).realize()
    const e = shifted.exp().realize()
    const s = e.sum(axis, true).realize()
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
    const uop = callWithInt64Array(_ffi.poly_reshape, this._ctx, this._uop, shape, shape.length)
    return this._makeResult(uop, shape, [this])
  }

  permute(...order) {
    if (order.length === 1 && Array.isArray(order[0])) order = order[0]
    const uop = callWithInt64Array(_ffi.poly_permute, this._ctx, this._uop, order, order.length)
    const newShape = order.map(i => this._shape[i])
    return this._makeResult(uop, newShape, [this])
  }

  expand(...shape) {
    if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
    const uop = callWithInt64Array(_ffi.poly_expand, this._ctx, this._uop, shape, shape.length)
    return this._makeResult(uop, shape, [this])
  }

  shrink(arg) {
    const flat = []
    for (let i = 0; i < arg.length; i++) {
      flat.push(arg[i][0], arg[i][1])
    }
    const ptr = writeInt64Array(flat)
    const uop = _ffi.poly_shrink(this._ctx, this._uop, ptr, arg.length)
    Module._free(ptr)
    const newShape = arg.map(([s, e]) => e - s)
    return this._makeResult(uop, newShape, [this])
  }

  pad(arg) {
    const flat = []
    for (let i = 0; i < arg.length; i++) {
      flat.push(arg[i][0], arg[i][1])
    }
    const ptr = writeInt64Array(flat)
    const uop = _ffi.poly_pad(this._ctx, this._uop, ptr, arg.length)
    Module._free(ptr)
    const newShape = this._shape.map((s, i) => s + arg[i][0] + arg[i][1])
    return this._makeResult(uop, newShape, [this])
  }

  flip(axis) {
    if (typeof axis === 'number') axis = [axis]
    const uop = callWithInt64Array(_ffi.poly_flip, this._ctx, this._uop, axis, axis.length)
    return this._makeResult(uop, [...this._shape], [this])
  }

  transpose(dim0 = -2, dim1 = -1) {
    const nd = this._shape.length
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
      if (dim < 0) dim += this._shape.length
      if (this._shape[dim] !== 1) return this
      const newShape = this._shape.filter((_, i) => i !== dim)
      if (!newShape.length) return this.reshape(1)
      return this.reshape(newShape)
    }
    const newShape = this._shape.filter(s => s !== 1)
    if (!newShape.length) return this.reshape(1)
    if (arraysEqual(newShape, this._shape)) return this
    return this.reshape(newShape)
  }

  unsqueeze(dim) {
    if (dim < 0) dim += this._shape.length + 1
    const newShape = [...this._shape]
    newShape.splice(dim, 0, 1)
    return this.reshape(newShape)
  }

  flatten(startDim = 0, endDim = -1) {
    if (endDim < 0) endDim += this._shape.length
    const before = this._shape.slice(0, startDim)
    let flatDim = 1
    for (let i = startDim; i <= endDim; i++) flatDim *= this._shape[i]
    const after = this._shape.slice(endDim + 1)
    return this.reshape([...before, flatDim, ...after])
  }

  unflatten(dim, sizes) {
    if (dim < 0) dim += this._shape.length
    const before = this._shape.slice(0, dim)
    const after = this._shape.slice(dim + 1)
    return this.reshape([...before, ...sizes, ...after])
  }

  view(...shape) { return this.reshape(...shape) }

  repeat(...repeats) {
    if (repeats.length === 1 && Array.isArray(repeats[0])) repeats = repeats[0]
    const nd = Math.max(this._shape.length, repeats.length)
    const shape = new Array(nd - this._shape.length).fill(1).concat(this._shape)
    repeats = new Array(nd - repeats.length).fill(1).concat(repeats)
    const newShape = []
    const expShape = []
    for (let i = 0; i < nd; i++) {
      newShape.push(shape[i], 1)
      expShape.push(shape[i], repeats[i])
    }
    const finalShape = shape.map((s, i) => s * repeats[i])
    return this.reshape(newShape).expand(expShape).reshape(finalShape)
  }

  // --- Reduction ops ---

  sum(axis, keepdim = false) {
    if (axis === undefined || axis === null) {
      axis = this._shape.map((_, i) => i)
    } else if (typeof axis === 'number') {
      axis = [axis]
    }
    const nd = this._shape.length
    axis = axis.map(a => a < 0 ? a + nd : a)

    const ptr = axis.length <= 8 ? writeInt64Scratch(axis) : writeInt64Array(axis)
    const uop = _ffi.poly_reduce_axis(this._ctx, _ffi.OPS.ADD, this._uop, ptr, axis.length)
    if (axis.length > 8) Module._free(ptr)

    const axisSet = new Set(axis)
    const newShape = []
    for (let i = 0; i < this._shape.length; i++) {
      if (axisSet.has(i)) {
        if (keepdim) newShape.push(1)
      } else {
        newShape.push(this._shape[i])
      }
    }

    return this._makeResult(uop, newShape, [this])
  }

  max(opts = {}) {
    let axis, keepdim
    if (typeof opts === 'object' && !Array.isArray(opts)) {
      axis = opts.axis
      keepdim = opts.keepdim || false
    } else {
      axis = opts
      keepdim = arguments[1] || false
    }

    if (axis === undefined || axis === null) {
      let result = this
      for (let i = this._shape.length - 1; i >= 0; i--) {
        const shPtr = writeInt64Array(result._shape)
        Module.HEAP32[_scratchOutNdimPtr >> 2] = 0
        const uop = _ffi.poly_max_reduce(this._ctx, result._uop,
          shPtr, result._shape.length, i, keepdim ? 1 : 0,
          _scratchOutShapePtr, _scratchOutNdimPtr)
        Module._free(shPtr)
        result = this._makeResult(uop, readOutShape(), [result])
      }
      return result
    }

    if (axis < 0) axis += this._shape.length
    const shPtr = writeInt64Array(this._shape)
    Module.HEAP32[_scratchOutNdimPtr >> 2] = 0
    const uop = _ffi.poly_max_reduce(this._ctx, this._uop,
      shPtr, this._shape.length, axis, keepdim ? 1 : 0,
      _scratchOutShapePtr, _scratchOutNdimPtr)
    Module._free(shPtr)
    return this._makeResult(uop, readOutShape(), [this])
  }

  min(opts = {}) {
    return this.neg().max(opts).neg()
  }

  mean(axis, keepdim = false) {
    if (axis === undefined || axis === null) {
      return this.sum(null, keepdim).div(this.numel())
    }
    if (axis < 0) axis += this._shape.length
    const shPtr = writeInt64Array(this._shape)
    Module.HEAP32[_scratchOutNdimPtr >> 2] = 0
    const uop = _ffi.poly_mean_reduce(this._ctx, this._uop,
      shPtr, this._shape.length, axis, keepdim ? 1 : 0,
      _scratchOutShapePtr, _scratchOutNdimPtr)
    Module._free(shPtr)
    return this._makeResult(uop, readOutShape(), [this])
  }

  var(axis, keepdim = false, correction = 1) {
    if (axis === undefined || axis === null) {
      const m = this.mean()
      const diff = this.sub(m)
      const sq = diff.mul(diff)
      return sq.sum().div(this.numel() - correction)
    }
    if (axis < 0) axis += this._shape.length
    const m = this.mean(axis, true).realize()
    const diff = this.sub(m)
    const sq = diff.mul(diff)
    const dimSize = this._shape[axis]
    return sq.sum(axis, keepdim).div(dimSize - correction)
  }

  std(axis, keepdim = false, correction = 1) {
    return this.var(axis, keepdim, correction).sqrt()
  }

  // --- Matmul (C core dot) ---

  dot(w) {
    if (!(w instanceof Tensor)) {
      throw new TypeError(`Expected Tensor, got ${typeof w}`)
    }
    const xShPtr = writeInt64Array(this._shape)
    const wShPtr = writeInt64Array(w._shape)
    Module.HEAP32[_scratchOutNdimPtr >> 2] = 0
    const uop = _ffi.poly_dot(this._ctx,
      this._uop, xShPtr, this._shape.length,
      w._uop, wShPtr, w._shape.length,
      _scratchOutShapePtr, _scratchOutNdimPtr)
    Module._free(xShPtr)
    Module._free(wShPtr)
    return this._makeResult(uop, readOutShape(), [this, w])
  }

  matmul(other) { return this.dot(other) }

  linear(weight, bias) {
    let result = this.dot(weight.transpose(-1, -2))
    if (bias) result = result.add(bias)
    return result
  }

  // --- Loss functions ---

  crossEntropy(target, axis = -1) {
    const logProbs = this.logSoftmax(axis)
    return target.mul(logProbs).neg().sum(axis).mean()
  }

  binaryCrossEntropy(target) {
    const t1 = target.mul(this.log())
    const t2 = this._ensureTensor(1.0).sub(target).mul(this._ensureTensor(1.0).sub(this).log())
    return t1.add(t2).neg().mean()
  }

  layernorm(axis = -1, eps = 1e-5) {
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
        if (ii < 0) ii += result._shape[dim]
        const arg = result._shape.map((s, d) => d === dim ? [ii, ii + 1] : [0, s])
        result = result.shrink(arg)
        result = result.squeeze(dim)
      } else if (Array.isArray(i) && i.length === 2) {
        const [start, stop] = i
        const arg = result._shape.map((s, d) => d === dim ? [start, stop] : [0, s])
        result = result.shrink(arg)
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
    const ctx = operands[0]._ctx
    const n = operands.length

    // Allocate arrays in WASM heap
    const tensorPtrs = Module._malloc(n * 4)
    for (let i = 0; i < n; i++) {
      Module.HEAP32[(tensorPtrs >> 2) + i] = operands[i]._uop
    }

    const shapePtrs = Module._malloc(n * 4)
    const shapeArrays = []
    for (let i = 0; i < n; i++) {
      const shPtr = writeInt64Array(operands[i]._shape)
      shapeArrays.push(shPtr)
      Module.HEAP32[(shapePtrs >> 2) + i] = shPtr
    }

    const ndimPtr = Module._malloc(n * 4)
    for (let i = 0; i < n; i++) {
      Module.HEAP32[(ndimPtr >> 2) + i] = operands[i]._shape.length
    }

    // Allocate formula string
    const formulaBytes = new TextEncoder().encode(formula + '\0')
    const formulaPtr = Module._malloc(formulaBytes.length)
    Module.HEAPU8.set(formulaBytes, formulaPtr)

    Module.HEAP32[_scratchOutNdimPtr >> 2] = 0
    const uop = _ffi.poly_einsum(ctx, formulaPtr,
      tensorPtrs, shapePtrs, ndimPtr, n,
      _scratchOutShapePtr, _scratchOutNdimPtr)

    // Free allocations
    Module._free(formulaPtr)
    Module._free(ndimPtr)
    for (const ptr of shapeArrays) Module._free(ptr)
    Module._free(shapePtrs)
    Module._free(tensorPtrs)

    if (!uop) throw new Error(`poly_einsum failed for formula: ${formula}`)
    return new Tensor(null, { _ctx: ctx, _uop: uop, _shape: readOutShape(), _inputs: [...operands] })
  }

  // --- Rearrange (C core, einops-style) ---

  rearrange(formula, kwargs = {}) {
    const names = Object.keys(kwargs)
    const values = names.map(k => kwargs[k])
    const n = names.length

    // Allocate formula string
    const formulaBytes = new TextEncoder().encode(formula + '\0')
    const formulaPtr = Module._malloc(formulaBytes.length)
    Module.HEAPU8.set(formulaBytes, formulaPtr)

    // Shape
    const shPtr = writeInt64Array(this._shape)

    // Axis names
    let namesPtr = 0
    let valuesPtr = 0
    if (n > 0) {
      const namesStr = names.join(' ') + '\0'
      const namesBytes = new TextEncoder().encode(namesStr)
      namesPtr = Module._malloc(namesBytes.length)
      Module.HEAPU8.set(namesBytes, namesPtr)
      valuesPtr = writeInt64Array(values)
    }

    Module.HEAP32[_scratchOutNdimPtr >> 2] = 0
    const uop = _ffi.poly_rearrange(this._ctx, formulaPtr,
      this._uop, shPtr, this._shape.length,
      namesPtr, valuesPtr, n,
      _scratchOutShapePtr, _scratchOutNdimPtr)

    // Free
    Module._free(formulaPtr)
    Module._free(shPtr)
    if (namesPtr) Module._free(namesPtr)
    if (valuesPtr) Module._free(valuesPtr)

    if (!uop) throw new Error(`poly_rearrange failed for formula: ${formula}`)
    return this._makeResult(uop, readOutShape(), [this])
  }

  // --- Autograd ---

  backward() {
    const leafMap = this._collectLeafMap()
    const allLeaves = this._collectLeaves()
    const gradLeaves = allLeaves.filter(t => t._requiresGrad)
    if (!gradLeaves.length) {
      throw new Error('No leaf tensors require grad')
    }

    for (const leaf of gradLeaves) {
      const gradUop = _ffi.poly_grad(this._ctx, this._uop, leaf._buffer)
      if (!gradUop) {
        throw new Error('poly_grad returned NULL for a leaf tensor')
      }

      const numel = leaf._shape.reduce((a, b) => a * b, 1) || 1
      const gradBuf = _ffi.poly_buffer_f32(this._ctx, numel)
      const store = _ffi.poly_store_val(this._ctx, gradBuf, gradUop)
      const sink = _ffi.poly_sink1(this._ctx, store)

      leafMap.set(gradBuf, new Float32Array(numel))

      const gradResult = _renderAndExec(this._ctx, sink, gradBuf, numel, (bufUop) => leafMap.get(bufUop))

      const gradTensor = new Tensor(gradResult, { _ctx: this._ctx })
      if (leaf._grad) {
        leaf._grad = leaf._grad.add(gradTensor)
      } else {
        leaf._grad = gradTensor
      }
    }
  }

  // --- Static constructors ---

  static zeros(...shape) {
    const numel = shape.reduce((a, b) => a * b, 1)
    const t = new Tensor(new Float32Array(numel).fill(0))
    if (shape.length > 1) return t.reshape(...shape)
    return t
  }

  static ones(...shape) {
    const numel = shape.reduce((a, b) => a * b, 1)
    const t = new Tensor(new Float32Array(numel).fill(1))
    if (shape.length > 1) return t.reshape(...shape)
    return t
  }

  static full(shape, fillValue) {
    if (typeof shape === 'number') shape = [shape]
    const numel = shape.reduce((a, b) => a * b, 1)
    const t = new Tensor(new Float32Array(numel).fill(fillValue))
    if (shape.length > 1) return t.reshape(...shape)
    return t
  }

  static arange(stop, start = 0, step = 1) {
    const arr = []
    for (let i = start; i < stop; i += step) arr.push(i)
    return new Tensor(new Float32Array(arr))
  }

  static rand(...shape) {
    if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
    const numel = shape.reduce((a, b) => a * b, 1)
    const data = new Float32Array(numel)
    for (let i = 0; i < numel; i++) data[i] = Math.random()
    const t = new Tensor(data)
    if (shape.length > 1) return t.reshape(...shape)
    return t
  }

  static randn(...shape) {
    if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
    const numel = shape.reduce((a, b) => a * b, 1)
    const data = new Float32Array(numel)
    for (let i = 0; i < numel; i += 2) {
      const u1 = Math.random() || 1e-10
      const u2 = Math.random()
      const r = Math.sqrt(-2 * Math.log(u1))
      data[i] = r * Math.cos(2 * Math.PI * u2)
      if (i + 1 < numel) data[i + 1] = r * Math.sin(2 * Math.PI * u2)
    }
    const t = new Tensor(data)
    if (shape.length > 1) return t.reshape(...shape)
    return t
  }

  static randint(low, high, shape) {
    if (high === undefined) { high = low; low = 0 }
    if (typeof shape === 'number') shape = [shape]
    if (!shape) shape = [1]
    const numel = shape.reduce((a, b) => a * b, 1)
    const data = new Float32Array(numel)
    for (let i = 0; i < numel; i++) data[i] = Math.floor(Math.random() * (high - low)) + low
    const t = new Tensor(data)
    if (shape.length > 1) return t.reshape(...shape)
    return t
  }

  static linspace(start, stop, steps) {
    const data = new Float32Array(steps)
    for (let i = 0; i < steps; i++) {
      data[i] = start + (stop - start) * i / (steps - 1)
    }
    return new Tensor(data)
  }

  static eye(n) {
    const data = new Float32Array(n * n)
    for (let i = 0; i < n; i++) data[i * n + i] = 1
    return new Tensor(data).reshape(n, n)
  }

  static empty(...shape) {
    if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
    const numel = shape.reduce((a, b) => a * b, 1)
    const t = new Tensor(new Float32Array(numel))
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

    const ndim = tensors[0]._shape.length
    if (dim < 0) dim += ndim
    const outShape = [...tensors[0]._shape]
    outShape[dim] = tensors.reduce((acc, t) => acc + t._shape[dim], 0)

    let offset = 0
    let result = null
    for (const t of tensors) {
      const padBefore = new Array(ndim).fill(0)
      const padAfter = new Array(ndim).fill(0)
      padBefore[dim] = offset
      padAfter[dim] = outShape[dim] - offset - t._shape[dim]
      const padArg = Array.from({ length: ndim }, (_, i) => [padBefore[i], padAfter[i]])
      const padded = t.pad(padArg)
      result = result ? result.add(padded) : padded
      offset += t._shape[dim]
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

  split(sizes, dim = 0) {
    if (dim < 0) dim += this._shape.length
    if (typeof sizes === 'number') {
      const total = this._shape[dim]
      const chunkSize = sizes
      sizes = []
      for (let i = 0; i < total; i += chunkSize) {
        sizes.push(Math.min(chunkSize, total - i))
      }
    }
    const results = []
    let offset = 0
    for (const sz of sizes) {
      const arg = this._shape.map((s, d) => d === dim ? [offset, offset + sz] : [0, s])
      results.push(this.shrink(arg))
      offset += sz
    }
    return results
  }

  chunk(n, dim = 0) {
    if (dim < 0) dim += this._shape.length
    const total = this._shape[dim]
    const chunkSize = Math.ceil(total / n)
    return this.split(chunkSize, dim)
  }

  toString() {
    return `Tensor(shape=[${this._shape}], dtype=${this.dtype}, realized=${this._isLeaf()})`
  }
}

// --- Utility helpers ---

function arraysEqual(a, b) {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
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

Tensor._defaultCtx = null

module.exports = { init, Tensor }
