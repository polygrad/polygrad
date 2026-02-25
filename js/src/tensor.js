/**
 * tensor.js — Lazy Tensor class backed by polygrad's C11 compiler core.
 * CPU-only, float32-only for v0.
 */

'use strict'

const ffi = require('./ffi')

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

class Tensor {
  /**
   * @param {number|number[]|Float32Array|null} data
   * @param {object} opts
   */
  constructor(data, opts = {}) {
    this._ctx = opts._ctx || Tensor._defaultCtx
    this._requiresGrad = opts.requiresGrad || false
    this._grad = null

    if (opts._uop) {
      // Internal construction from ops
      this._uop = opts._uop
      this._buffer = opts._buffer || null
      this._data = opts._data || null
      this._shape = opts._shape ? [...opts._shape] : []
      this._inputs = opts._inputs || []
    } else {
      // User construction from data
      const { data: flat, shape } = flattenArray(data)
      this._shape = shape
      this._data = flat
      this._buffer = ffi.poly_buffer_f32(this._ctx, flat.length)
      // For multi-dimensional tensors, insert RESHAPE so scheduler knows shape
      if (shape.length > 1) {
        this._uop = ffi.poly_reshape(this._ctx, this._buffer, shape, shape.length)
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

  _collectLeaves(seen) {
    if (!seen) seen = new Set()
    // Use buffer pointer as identity for dedup
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

  _realize() {
    if (this._isLeaf()) return

    const leaves = this._collectLeaves()
    const numel = this._shape.reduce((a, b) => a * b, 1) || 1
    const outBuf = ffi.poly_buffer_f32(this._ctx, numel)
    const outData = new Float32Array(numel)

    const store = ffi.poly_store_val(this._ctx, outBuf, this._uop)
    const sink = ffi.poly_sink1(this._ctx, store)

    // Bind all buffers via stateful builder
    ffi.poly_realize_begin(this._ctx)
    for (const leaf of leaves) {
      ffi.poly_realize_bind(this._ctx, leaf._buffer, leaf._data)
    }
    ffi.poly_realize_bind(this._ctx, outBuf, outData)

    const ret = ffi.poly_realize_exec(this._ctx, sink)
    if (ret !== 0) {
      throw new Error('poly_realize failed')
    }

    this._data = outData
    this._buffer = outBuf
    // Preserve shape in UOp graph so subsequent ops see correct dimensions
    if (this._shape.length > 1) {
      this._uop = ffi.poly_reshape(this._ctx, outBuf, this._shape, this._shape.length)
    } else {
      this._uop = outBuf
    }
    this._inputs = []
  }

  /**
   * Realize the computation (in-place). Returns self for chaining.
   */
  realize() {
    this._realize()
    return this
  }

  /**
   * Realize and return a Float32Array copy.
   */
  toArray() {
    this._realize()
    return new Float32Array(this._data)
  }

  /**
   * Return scalar value.
   */
  item() {
    const arr = this.toArray()
    if (arr.length !== 1) {
      throw new Error(`item() requires scalar tensor, got shape [${this._shape}]`)
    }
    return arr[0]
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

  contiguous() {
    return this  // no-op for now
  }

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
      const c = ffi.poly_const_float(this._ctx, other)
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

  /**
   * Broadcast this tensor's UOp to target_shape via RESHAPE+EXPAND.
   * Matches tinygrad's _broadcast_to.
   */
  _broadcastUop(targetShape) {
    if (arraysEqual(this._shape, targetShape)) return this._uop
    let uop = this._uop
    let curShape = [...this._shape]
    // CONST scalars auto-broadcast — no EXPAND needed
    if (!curShape.length && this._buffer === null) return uop
    // Pad left with 1s if needed
    const targetNd = targetShape.length
    if (curShape.length < targetNd) {
      curShape = new Array(targetNd - curShape.length).fill(1).concat(curShape)
      uop = ffi.poly_reshape(this._ctx, uop, curShape, curShape.length)
    }
    // Expand where size 1 → target size
    if (!arraysEqual(curShape, targetShape)) {
      uop = ffi.poly_expand(this._ctx, uop, targetShape, targetShape.length)
    }
    return uop
  }

  /**
   * Binary op with explicit EXPAND broadcasting (matches tinygrad _broadcasted).
   */
  _binop(other, opName) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const xUop = this._broadcastUop(outShape)
    const yUop = other._broadcastUop(outShape)
    const uop = ffi.poly_alu2(this._ctx, ffi.OPS[opName], xUop, yUop)
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
    const uop = ffi.poly_alu1(this._ctx, ffi.OPS.NEG, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  // --- Comparisons (C core) ---

  eq(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = ffi.poly_eq(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  ne(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = ffi.poly_ne(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  gt(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = ffi.poly_gt(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  ge(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = ffi.poly_ge(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  le(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = ffi.poly_le(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
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
    const uop = ffi.poly_where_op(this._ctx, cUop, xUop, yUop)
    return this._makeResult(uop, outShape, [this, x, y])
  }

  maximum(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = ffi.poly_maximum(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  minimum(other) {
    other = this._ensureTensor(other)
    const outShape = this._broadcastShape(other._shape)
    const uop = ffi.poly_minimum(this._ctx, this._broadcastUop(outShape), other._broadcastUop(outShape))
    return this._makeResult(uop, outShape, [this, other])
  }

  clamp(lo, hi) {
    if (lo === undefined && hi === undefined) {
      throw new Error("at least one of 'lo' or 'hi' must not be undefined")
    }
    lo = lo !== undefined ? lo : -1e38
    hi = hi !== undefined ? hi : 1e38
    const uop = ffi.poly_clamp(this._ctx, this._uop, lo, hi)
    return this._makeResult(uop, [...this._shape], [this])
  }

  // --- Unary math (C core composed ops) ---

  exp2() {
    const uop = ffi.poly_alu1(this._ctx, ffi.OPS.EXP2, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  log2() {
    const uop = ffi.poly_alu1(this._ctx, ffi.OPS.LOG2, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  sqrt() {
    const uop = ffi.poly_alu1(this._ctx, ffi.OPS.SQRT, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  reciprocal() {
    const uop = ffi.poly_alu1(this._ctx, ffi.OPS.RECIPROCAL, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  trunc() {
    const uop = ffi.poly_alu1(this._ctx, ffi.OPS.TRUNC, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  exp() {
    const uop = ffi.poly_exp(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  log() {
    const uop = ffi.poly_log(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  sin() {
    const uop = ffi.poly_sin(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  cos() {
    const uop = ffi.poly_cos(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  tan() {
    const uop = ffi.poly_tan(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  sigmoid() {
    const uop = ffi.poly_sigmoid(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  tanh() {
    const uop = ffi.poly_tanh_act(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  abs() {
    const uop = ffi.poly_abs(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  sign() {
    const uop = ffi.poly_sign(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  square() {
    const uop = ffi.poly_square(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  rsqrt() {
    const uop = ffi.poly_rsqrt(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  ceil() {
    const uop = ffi.poly_ceil(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  floor() {
    const uop = ffi.poly_floor(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  round() {
    const uop = ffi.poly_round_f(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  isinf() {
    const uop = ffi.poly_isinf(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  isnan() {
    const uop = ffi.poly_isnan(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  // --- Activations (C core composed ops) ---

  relu() {
    const uop = ffi.poly_relu(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  relu6() {
    const uop = ffi.poly_relu6(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  leakyRelu(negSlope = 0.01) {
    const uop = ffi.poly_leaky_relu(this._ctx, this._uop, negSlope)
    return this._makeResult(uop, [...this._shape], [this])
  }

  gelu() {
    const uop = ffi.poly_gelu(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  quickGelu() {
    const uop = ffi.poly_quick_gelu(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  silu() {
    const uop = ffi.poly_silu(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  swish() { return this.silu() }

  elu(alpha = 1.0) {
    const uop = ffi.poly_elu(this._ctx, this._uop, alpha)
    return this._makeResult(uop, [...this._shape], [this])
  }

  softplus(beta = 1.0) {
    const uop = ffi.poly_softplus(this._ctx, this._uop, beta)
    return this._makeResult(uop, [...this._shape], [this])
  }

  mish() {
    const uop = ffi.poly_mish(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  hardtanh(minVal = -1, maxVal = 1) {
    const uop = ffi.poly_hardtanh(this._ctx, this._uop, minVal, maxVal)
    return this._makeResult(uop, [...this._shape], [this])
  }

  hardswish() {
    const uop = ffi.poly_hardswish(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  hardsigmoid() {
    const uop = ffi.poly_hardsigmoid(this._ctx, this._uop)
    return this._makeResult(uop, [...this._shape], [this])
  }

  // --- Softmax (Tensor-level composition with realize boundaries) ---
  // Composed at Tensor level (not C core) because the reduce→expand→alu
  // pattern requires separate kernels. Each .realize() forces a kernel boundary.

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
    // Support -1 dimension inference
    if (shape.indexOf(-1) !== -1) {
      const total = this.numel()
      const negIdx = shape.indexOf(-1)
      let known = 1
      for (let i = 0; i < shape.length; i++) {
        if (i !== negIdx) known *= shape[i]
      }
      shape = shape.map((s, i) => i === negIdx ? Math.floor(total / known) : s)
    }
    const uop = ffi.poly_reshape(this._ctx, this._uop, shape, shape.length)
    return this._makeResult(uop, shape, [this])
  }

  permute(...order) {
    if (order.length === 1 && Array.isArray(order[0])) order = order[0]
    const uop = ffi.poly_permute(this._ctx, this._uop, order, order.length)
    const newShape = order.map(i => this._shape[i])
    return this._makeResult(uop, newShape, [this])
  }

  expand(...shape) {
    if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0]
    const uop = ffi.poly_expand(this._ctx, this._uop, shape, shape.length)
    return this._makeResult(uop, shape, [this])
  }

  shrink(arg) {
    // arg: [[s0,e0], [s1,e1], ...]
    const flat = new BigInt64Array(arg.length * 2)
    for (let i = 0; i < arg.length; i++) {
      flat[i * 2] = BigInt(arg[i][0])
      flat[i * 2 + 1] = BigInt(arg[i][1])
    }
    const uop = ffi.poly_shrink(this._ctx, this._uop, flat, arg.length)
    const newShape = arg.map(([s, e]) => e - s)
    return this._makeResult(uop, newShape, [this])
  }

  pad(arg) {
    const flat = new BigInt64Array(arg.length * 2)
    for (let i = 0; i < arg.length; i++) {
      flat[i * 2] = BigInt(arg[i][0])
      flat[i * 2 + 1] = BigInt(arg[i][1])
    }
    const uop = ffi.poly_pad(this._ctx, this._uop, flat, arg.length)
    const newShape = this._shape.map((s, i) => s + arg[i][0] + arg[i][1])
    return this._makeResult(uop, newShape, [this])
  }

  flip(axis) {
    if (typeof axis === 'number') axis = [axis]
    const uop = ffi.poly_flip(this._ctx, this._uop, axis, axis.length)
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
    // Normalize negative axes
    const nd = this._shape.length
    axis = axis.map(a => a < 0 ? a + nd : a)

    const uop = ffi.poly_reduce_axis(this._ctx, ffi.OPS.ADD, this._uop, axis, axis.length)

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
      // Legacy: max(axis, keepdim)
      axis = opts
      keepdim = arguments[1] || false
    }

    if (axis === undefined || axis === null) {
      let result = this
      for (let i = this._shape.length - 1; i >= 0; i--) {
        const outShape = new BigInt64Array(8)
        const outNdim = new Int32Array(1)
        const uop = ffi.poly_max_reduce(this._ctx, result._uop,
          result._shape, result._shape.length, i, keepdim ? 1 : 0,
          outShape, outNdim)
        const shape = readShape(outShape, outNdim[0])
        result = this._makeResult(uop, shape, [result])
      }
      return result
    }

    if (axis < 0) axis += this._shape.length
    const outShape = new BigInt64Array(8)
    const outNdim = new Int32Array(1)
    const uop = ffi.poly_max_reduce(this._ctx, this._uop,
      this._shape, this._shape.length, axis, keepdim ? 1 : 0,
      outShape, outNdim)
    return this._makeResult(uop, readShape(outShape, outNdim[0]), [this])
  }

  min(opts = {}) {
    return this.neg().max(opts).neg()
  }

  mean(axis, keepdim = false) {
    if (axis === undefined || axis === null) {
      return this.sum(null, keepdim).div(this.numel())
    }
    if (axis < 0) axis += this._shape.length
    const outShape = new BigInt64Array(8)
    const outNdim = new Int32Array(1)
    const uop = ffi.poly_mean_reduce(this._ctx, this._uop,
      this._shape, this._shape.length, axis, keepdim ? 1 : 0,
      outShape, outNdim)
    return this._makeResult(uop, readShape(outShape, outNdim[0]), [this])
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
    const outShape = new BigInt64Array(8)
    const outNdim = new Int32Array(1)
    const uop = ffi.poly_dot(this._ctx,
      this._uop, this._shape, this._shape.length,
      w._uop, w._shape, w._shape.length,
      outShape, outNdim)
    return this._makeResult(uop, readShape(outShape, outNdim[0]), [this, w])
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
        // None → unsqueeze
        result = result.unsqueeze(dim)
        dim += 1
      } else if (typeof i === 'number') {
        let ii = i
        if (ii < 0) ii += result._shape[dim]
        const arg = result._shape.map((s, d) => d === dim ? [ii, ii + 1] : [0, s])
        result = result.shrink(arg)
        result = result.squeeze(dim)
      } else if (Array.isArray(i) && i.length === 2) {
        // [start, stop] slice
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

    const tensorPtrs = operands.map(t => t._uop)
    const shapePtrs = operands.map(t =>
      ffi.koffi.as(new BigInt64Array(t._shape.map(BigInt)), 'int64_t *'))
    const ndims = new Int32Array(operands.map(t => t._shape.length))

    const outShape = new BigInt64Array(8)
    const outNdim = new Int32Array(1)

    const uop = ffi.poly_einsum(ctx, formula,
      tensorPtrs, shapePtrs, ndims, n,
      outShape, outNdim)
    if (!uop) throw new Error(`poly_einsum failed for formula: ${formula}`)
    const shape = Array.from(outShape.slice(0, outNdim[0])).map(Number)
    return new Tensor(null, { _ctx: ctx, _uop: uop, _shape: shape, _inputs: [...operands] })
  }

  // --- Rearrange (C core, einops-style) ---

  rearrange(formula, kwargs = {}) {
    const names = Object.keys(kwargs)
    const values = names.map(k => kwargs[k])
    const n = names.length
    const axisNames = names.length ? names.join(' ') : null
    const axisValues = n > 0 ? new BigInt64Array(values.map(BigInt)) : null

    const outShape = new BigInt64Array(8)
    const outNdim = new Int32Array(1)

    const uop = ffi.poly_rearrange(this._ctx, formula,
      this._uop, this._shape, this._shape.length,
      axisNames, axisValues, n,
      outShape, outNdim)
    if (!uop) throw new Error(`poly_rearrange failed for formula: ${formula}`)
    const shape = Array.from(outShape.slice(0, outNdim[0])).map(Number)
    return this._makeResult(uop, shape, [this])
  }

  // --- Autograd ---

  backward() {
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

      const numel = leaf._shape.reduce((a, b) => a * b, 1) || 1
      const gradBuf = ffi.poly_buffer_f32(this._ctx, numel)
      const gradData = new Float32Array(numel)

      const store = ffi.poly_store_val(this._ctx, gradBuf, gradUop)
      const sink = ffi.poly_sink1(this._ctx, store)

      // Bind all buffers via stateful builder
      ffi.poly_realize_begin(this._ctx)
      for (const l of allLeaves) {
        ffi.poly_realize_bind(this._ctx, l._buffer, l._data)
      }
      ffi.poly_realize_bind(this._ctx, gradBuf, gradData)

      const ret = ffi.poly_realize_exec(this._ctx, sink)
      if (ret !== 0) {
        throw new Error('Failed to realize gradient')
      }

      if (leaf._grad) {
        for (let i = 0; i < numel; i++) {
          leaf._grad._data[i] += gradData[i]
        }
      } else {
        const t = new Tensor(gradData, { _ctx: this._ctx })
        leaf._grad = leaf._shape.length > 1 ? t.reshape(...leaf._shape) : t
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
    // Box-Muller transform
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
    // Check if last arg is options object
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

function readShape(outShape, ndim) {
  const shape = []
  for (let i = 0; i < ndim; i++) shape.push(Number(outShape[i]))
  return shape
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

// Static initializer
Tensor._defaultCtx = null

module.exports = { Tensor }
