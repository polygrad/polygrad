'use strict'

// JS Numbers outside the safe range have already lost information. Require
// BigInt there, and check before Wasm's i64 conversion silently wraps.
function integer64(value, unsigned = false) {
  if (typeof value !== 'bigint') {
    if (typeof value !== 'number' || !Number.isSafeInteger(value)) {
      throw new TypeError('expected a safe integer Number or BigInt')
    }
    value = BigInt(value)
  }
  const lower = unsigned ? 0n : -(1n << 63n)
  const upper = unsigned ? (1n << 64n) - 1n : (1n << 63n) - 1n
  if (value < lower || value > upper) throw new RangeError('integer outside 64-bit range')
  return value
}

module.exports = { integer64 }
