'use strict'

/**
 * Thin JS wrapper around polygrad's C UOp nodes. Mirrors py/polygrad/uop/ops.py.
 *
 * The raw handle (`raw`) is whatever ffi returns -- a native External for the
 * Node addon, a number for the WASM runtime. Since hash-consing is done in C,
 * two UOp instances with the same raw handle represent the same UOp.
 *
 * `ffi` is the core's ffi object (native binding or wasm wrappers).
 */

const AxisType = Object.freeze({
  DEVICE: 0,
  GLOBAL: 1,
  WARP: 2,
  LOCAL: 3,
  WEAK: 4,
  GROUP_REDUCE: 5,
  REDUCE: 6,
  UPCAST: 7,
  UNROLL: 8,
  THREAD: 9,
  PLACEHOLDER: 10,
  LOOP: 11
})

class KernelInfo {
  constructor(opts = {}) {
    if (typeof opts === 'string') opts = { name: opts }
    this.name = opts.name || 'test'
    this.opts_to_apply = opts.opts_to_apply === undefined ? null : opts.opts_to_apply
  }
}

class UOp {
  constructor(ctx, ffi, raw) {
    this.ctx = ctx
    this.ffi = ffi
    this.raw = raw || null
  }

  toString() { return `UOp(${this.raw})` }

  get key() {
    if (!this.raw) return '0'
    if (this.ffi.poly_uop_key) return String(this.ffi.poly_uop_key(this.raw))
    return String(this.raw)
  }

  get op() {
    if (!this.raw || !this.ffi.poly_uop_op) return 0
    return Number(this.ffi.poly_uop_op(this.raw))
  }

  get src() {
    if (!this.raw || !this.ffi.poly_uop_n_src || !this.ffi.poly_uop_src) return []
    const n = Number(this.ffi.poly_uop_n_src(this.raw))
    const out = []
    for (let i = 0; i < n; i++) {
      const raw = this.ffi.poly_uop_src(this.raw, i)
      out.push(new UOp(this.ctx, this.ffi, raw))
    }
    return out
  }

  get base() {
    const ops = this.ffi.__polygradOps || {}
    const op = this.op
    if (op === ops.RESHAPE || op === ops.EXPAND || op === ops.PERMUTE ||
        op === ops.PAD || op === ops.SHRINK || op === ops.FLIP ||
        op === ops.MULTI || op === ops.DETACH) {
      const sources = this.src
      if (sources.length) return sources[0].base
    }
    return this
  }

  hasBufferIdentity() {
    if (!this.raw) return false
    return !!this.ffi.poly_uop_has_buffer_identity(this.raw)
  }

  get buffer() {
    if (!this.raw) return null
    const r = this.ffi.poly_uop_buffer
      ? this.ffi.poly_uop_buffer(this.ctx, this.raw)
      : this.ffi.poly_uop_get_buffer_identity(this.raw)
    return r ? new UOp(this.ctx, this.ffi, r) : null
  }

  get realized() {
    const ops = this.ffi.__polygradOps || {}
    if (this.op !== ops.BUFFER && this.op !== ops.BUFFER_VIEW) return null
    const buf = this.buffer
    if (!buf || !this.ffi.poly_buffer_is_allocated) return null
    return this.ffi.poly_buffer_is_allocated(this.ctx, buf.raw) ? buf : null
  }

  get isRealized() {
    // tinygrad/uop/ops.py:881-891: movement views are realized when their
    // recursive base buffer is allocated.
    return this.base.realized !== null
  }

  get is_realized() { return this.isRealized }

  // --- Factories ---

  /**
   * Polygrad equivalent of tinygrad's _fromnp: create a BUFFER UOp, attach
   * the JS TypedArray's bytes as the PolyBuffer in ctx->buffers (borrowing
   * the pointer), and wrap in RESHAPE when ndim > 1. Caller must keep the
   * TypedArray alive for as long as this UOp is reachable -- the host owner
   * registry on the Tensor side does that.
   */
  static fromHost(ctx, ffi, typedArray, dtypeId, dims) {
    const raw = ffi.poly_buffer_from_host(
      ctx, typedArray, typedArray.byteLength, dtypeId, dims || null, dims ? dims.length : 0
    )
    return raw ? new UOp(ctx, ffi, raw) : null
  }

  static placeholderLike(uop, slot = 0) {
    const raw = uop.ffi.poly_uop_placeholder_like(uop.ctx, rawUop(uop), Number(slot))
    return raw ? new UOp(uop.ctx, uop.ffi, raw) : null
  }

  static range(ctx, ffi, bound, axisId = 0, axisType = AxisType.WEAK) {
    const raw = ffi.poly_uop_range(ctx, Number(bound), Number(axisId), Number(axisType))
    return raw ? new UOp(ctx, ffi, raw) : null
  }

  numel() {
    const n = this.ffi.poly_uop_numel(this.ctx, this.raw)
    if (n < 0) throw new Error('poly_uop_numel failed')
    return Number(n)
  }

  flatten() {
    const raw = this.ffi.poly_uop_flatten(this.ctx, this.raw)
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  index(...idx) {
    if (idx.length === 1 && Array.isArray(idx[0])) idx = idx[0]
    const rawIdx = idx.map(x => {
      if (x instanceof UOp) return x.raw
      if (Number.isInteger(x)) return this.ffi.poly_const_int(this.ctx, x)
      throw new TypeError(`unsupported index type ${typeof x}`)
    })
    const raw = this.ffi.poly_uop_index(this.ctx, this.raw, rawIdx)
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  load() {
    const raw = this.ffi.poly_uop_load(this.ctx, this.raw)
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  store(value) {
    const v = this._coerce(value)
    const raw = this.ffi.poly_uop_store(this.ctx, this.raw, v.raw)
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  set(value, ...ranges) {
    if (ranges.length === 1 && Array.isArray(ranges[0])) ranges = ranges[0]
    const v = this._coerce(value)
    const raw = this.ffi.poly_uop_set(this.ctx, this.raw, v.raw, ranges.map(rawUop))
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  group(...srcs) {
    const raw = this.ffi.poly_uop_group(this.ctx, [this, ...srcs].filter(Boolean).map(rawUop))
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  end(...ranges) {
    const raw = this.ffi.poly_uop_end(this.ctx, this.raw, ranges.map(rawUop))
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  sink(...srcs) {
    let arg = null
    if (srcs.length && srcs[srcs.length - 1] instanceof KernelInfo) {
      arg = srcs.pop()
    } else if (srcs.length && srcs[srcs.length - 1] && srcs[srcs.length - 1].arg instanceof KernelInfo) {
      arg = srcs.pop().arg
    }
    const rawSrcs = [this, ...srcs].filter(Boolean).map(rawUop)
    const raw = arg
      ? this.ffi.poly_uop_sink_ex(
        this.ctx, rawSrcs, arg.name || null,
        !(Array.isArray(arg.opts_to_apply) && arg.opts_to_apply.length === 0)
      )
      : this.ffi.poly_uop_sink(this.ctx, rawSrcs)
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  call(...srcs) {
    const raw = this.ffi.poly_uop_call(this.ctx, this.raw, srcs.map(rawUop))
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  after(...effects) {
    let out = this
    for (const effect of effects) {
      const raw = this.ffi.poly_uop_after(this.ctx, out.raw, rawUop(effect))
      out = raw ? new UOp(this.ctx, this.ffi, raw) : null
      if (!out) return null
    }
    return out
  }

  _coerce(value) {
    if (value instanceof UOp) return value
    if (typeof value === 'boolean') {
      const dtypeId = (this.ffi.__polygradDtypeIds || {}).bool
      return new UOp(this.ctx, this.ffi, this.ffi.poly_const_int_by_id(this.ctx, value ? 1 : 0, dtypeId))
    }
    if (Number.isInteger(value)) return new UOp(this.ctx, this.ffi, this.ffi.poly_const_int(this.ctx, value))
    if (typeof value === 'number') return new UOp(this.ctx, this.ffi, this.ffi.poly_const_float(this.ctx, value))
    throw new TypeError(`cannot convert ${typeof value} to UOp`)
  }

  _dtypeName() {
    if (!this.raw || !this.ffi.poly_uop_dtype_id) return null
    const id = Number(this.ffi.poly_uop_dtype_id(this.ctx, this.raw))
    const names = this.ffi.__polygradDtypeNameById || {}
    return names[id] || null
  }

  _dtypeId() {
    if (!this.raw || !this.ffi.poly_uop_dtype_id) return -1
    return Number(this.ffi.poly_uop_dtype_id(this.ctx, this.raw))
  }

  _coerceLike(value, ref) {
    if (value instanceof UOp) return value
    if (typeof value !== 'number') throw new TypeError(`cannot convert ${typeof value} to UOp`)
    const dtypeId = ref instanceof UOp ? ref._dtypeId() : -1
    if (dtypeId >= 0 && this.ffi.poly_const_float_by_id) {
      const raw = this.ffi.poly_const_float_by_id(this.ctx, value, dtypeId)
      if (raw) return new UOp(this.ctx, this.ffi, raw)
    }
    if (dtypeId >= 0 && Number.isInteger(value) && this.ffi.poly_const_int_by_id) {
      const raw = this.ffi.poly_const_int_by_id(this.ctx, value, dtypeId)
      if (raw) return new UOp(this.ctx, this.ffi, raw)
    }
    const dtype = ref instanceof UOp ? ref._dtypeName() : null
    if (dtype === 'float64') return new UOp(this.ctx, this.ffi, this.ffi.poly_const_double(this.ctx, value))
    if (dtype && (dtype.startsWith('float') || dtype.startsWith('fp8') || dtype === 'bfloat16')) {
      return new UOp(this.ctx, this.ffi, this.ffi.poly_const_float(this.ctx, value))
    }
    if (Number.isInteger(value)) return new UOp(this.ctx, this.ffi, this.ffi.poly_const_int(this.ctx, value))
    return new UOp(this.ctx, this.ffi, this.ffi.poly_const_float(this.ctx, value))
  }

  _op(name) {
    const ops = this.ffi.__polygradOps || {}
    const op = ops[name]
    if (op === undefined) throw new Error(`polygrad: missing op ${name}`)
    return op
  }

  _alu1(name) {
    const raw = this.ffi.poly_alu1(this.ctx, this._op(name), this.raw)
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  _alu2(name, other) {
    const b = this._coerce(other)
    const raw = this.ffi.poly_binop(this.ctx, this._op(name), this.raw, b.raw)
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  _alu3(name, b, c) {
    b = this._coerceLike(b, this)
    c = this._coerceLike(c, this)
    const raw = this.ffi.poly_alu3(this.ctx, this._op(name), this.raw, b.raw, c.raw)
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  add(other) { return this._alu2('ADD', other) }
  sub(other) { return this._alu2('SUB', other) }
  mul(other) { return this._alu2('MUL', other) }
  div(other) { return this._alu2('FDIV', other) }
  cdiv(other) { return this._alu2('CDIV', other) }
  cmod(other) { return this._alu2('CMOD', other) }
  floordiv(other) { return this._alu2('FLOORDIV', other) }
  floormod(other) { return this._alu2('FLOORMOD', other) }
  mod(other) { return this.floormod(other) }
  max(other) { return this._alu2('MAX', other) }
  and(other) { return this._alu2('AND', other) }
  or(other) { return this._alu2('OR', other) }
  xor(other) { return this._alu2('XOR', other) }
  shl(other) { return this._alu2('SHL', other) }
  shr(other) { return this._alu2('SHR', other) }
  pow(other) { return this._alu2('POW', other) }
  lt(other) { return this._alu2('CMPLT', other) }
  eq(other) { return this._alu2('CMPEQ', other) }
  ne(other) { return this._alu2('CMPNE', other) }
  cmplt(other) { return this.lt(other) }
  cmpeq(other) { return this.eq(other) }
  cmpne(other) { return this.ne(other) }
  neg() { return this._alu1('NEG') }
  sqrt() { return this._alu1('SQRT') }
  exp2() { return this._alu1('EXP2') }
  log2() { return this._alu1('LOG2') }
  sin() { return this._alu1('SIN') }
  reciprocal() { return this._alu1('RECIPROCAL') }
  trunc() { return this._alu1('TRUNC') }
  cast(dtype) {
    // Pinned tinygrad uop/ops.py:509-513: UOp.cast constructs CAST unless the
    // dtype already matches. The C primitive owns that same canonicalization.
    const dtypeIds = this.ffi.__polygradDtypeIds || {}
    const dtypeId = dtypeIds[String(dtype)]
    if (dtypeId === undefined || dtypeId < 0) throw new TypeError(`unknown dtype ${dtype}`)
    const raw = this.ffi.poly_cast_by_id(this.ctx, this.raw, dtypeId)
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }
  where(yes, no) {
    const y = this._coerce(yes)
    const n = this._coerce(no)
    const raw = this.ffi.poly_where_op(this.ctx, this.raw, y.raw, n.raw)
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }
  mulacc(mul, acc) { return this._alu3('MULACC', mul, acc) }

  reduce(op, ...ranges) {
    if (typeof op !== 'string') throw new TypeError('UOp.reduce expects an op name')
    const raw = this.ffi.poly_uop_reduce(this.ctx, this._op(op), this.raw, ranges.map(rawUop))
    return raw ? new UOp(this.ctx, this.ffi, raw) : null
  }

  sum(...ranges) { return this.reduce('ADD', ...ranges) }
  maxReduce(...ranges) { return this.reduce('MAX', ...ranges) }
}

function rawUop(value) {
  if (value instanceof UOp) return value.raw
  if (value && Object.prototype.hasOwnProperty.call(value, 'raw')) return value.raw
  return value || null
}

function createBoundUopNamespace(runtime) {
  const { ffi, ctx, dtypeIds } = runtime._core
  ffi.__polygradOps = runtime._core.ops || {}
  const dtypeNameById = {}
  for (const [name, id] of Object.entries(dtypeIds || {})) dtypeNameById[Number(id)] = name
  ffi.__polygradDtypeNameById = dtypeNameById
  ffi.__polygradDtypeIds = dtypeIds || {}

  function wrap(value) {
    if (value instanceof UOp) return value
    return new UOp(ctx, ffi, value)
  }

  return {
    UOp,
    AxisType,
    KernelInfo,
    wrap,
    range(bound, axisId = 0, axisType = AxisType.WEAK) {
      return UOp.range(ctx, ffi, bound, axisId, axisType)
    },
    constant(value, dtype = null) {
      if (dtype !== null && dtype !== undefined) {
        const dtypeId = dtypeIds[String(dtype)]
        if (dtypeId === undefined || dtypeId < 0) throw new TypeError(`unknown dtype ${dtype}`)
        const raw = (typeof value === 'boolean' || Number.isInteger(value))
          ? ffi.poly_const_int_by_id(ctx, typeof value === 'boolean' ? (value ? 1 : 0) : value, dtypeId)
          : ffi.poly_const_float_by_id(ctx, value, dtypeId)
        return raw ? new UOp(ctx, ffi, raw) : null
      }
      if (typeof value === 'boolean') {
        return new UOp(ctx, ffi, ffi.poly_const_int_by_id(ctx, value ? 1 : 0, dtypeIds.bool))
      }
      if (Number.isInteger(value)) return new UOp(ctx, ffi, ffi.poly_const_int(ctx, value))
      if (typeof value === 'number') return new UOp(ctx, ffi, ffi.poly_const_float(ctx, value))
      throw new TypeError(`cannot convert ${typeof value} to UOp`)
    },
    placeholderLike(value, slot = 0) {
      return UOp.placeholderLike(wrap(rawUop(value)), slot)
    },
    key(value) {
      return wrap(rawUop(value)).key
    },
    shape(value) {
      const raw = rawUop(value)
      if (!raw) return []
      return ffi.poly_uop_max_shape_dims(ctx, raw)
    },
    dtype(value) {
      const raw = rawUop(value)
      if (!raw) return null
      const id = Number(ffi.poly_uop_dtype_id(ctx, raw))
      return dtypeNameById[id] || String(id)
    },
    op(value) {
      return wrap(rawUop(value)).op
    },
    hasBufferIdentity(value) {
      return wrap(rawUop(value)).hasBufferIdentity()
    },
    buffer(value) {
      return wrap(rawUop(value)).buffer
    }
  }
}

UOp.AxisType = AxisType
UOp.KernelInfo = KernelInfo

module.exports = { UOp, AxisType, KernelInfo, createBoundUopNamespace }
