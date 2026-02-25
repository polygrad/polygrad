/**
 * polygrad — JS frontend for the polygrad tensor compiler.
 * CPU-only, float32-only for v0.
 */

'use strict'

const ffi = require('./ffi')
const { Tensor } = require('./tensor')

// Create default context
Tensor._defaultCtx = ffi.poly_ctx_new()

// Cleanup on exit
process.on('exit', () => {
  if (Tensor._defaultCtx) {
    ffi.poly_ctx_destroy(Tensor._defaultCtx)
    Tensor._defaultCtx = null
  }
})

module.exports = { Tensor }
