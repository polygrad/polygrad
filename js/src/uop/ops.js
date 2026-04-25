'use strict'

/**
 * Thin JS wrapper around polygrad's C UOp nodes. Mirrors py/polygrad/uop/ops.py.
 *
 * The raw handle (`raw`) is whatever ffi returns -- a native External for the
 * Node addon, a number for the WASM runtime. Since hash-consing is done in C,
 * two UOp instances with the same raw handle represent the same UOp.
 *
 * `ffi` is the backend's ffi object (native binding or wasm wrappers).
 */

class UOp {
  constructor(ctx, ffi, raw) {
    this.ctx = ctx
    this.ffi = ffi
    this.raw = raw || null
  }

  toString() { return `UOp(${this.raw})` }

  hasBufferIdentity() {
    if (!this.raw) return false
    return !!this.ffi.poly_uop_has_buffer_identity(this.raw)
  }

  get buffer() {
    if (!this.raw) return null
    const r = this.ffi.poly_uop_get_buffer_identity(this.raw)
    return r ? new UOp(this.ctx, this.ffi, r) : null
  }

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
}

module.exports = { UOp }
