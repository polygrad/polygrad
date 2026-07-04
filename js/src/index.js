'use strict'

const { createRuntime, normalizeOptions } = require('./runtime')

async function resolveNodeCore(name, opts) {
  if (name === 'wasm') {
    const { createWasmCore } = require('./core/wasm')
    return createWasmCore(opts.device)
  }

  if (name === 'native') {
    /* Native core validates and routes cpu/x86/cuda/hip/interp through the C
     * context. POLY_DEVICE is folded into opts.device above. */
    const { createNativeCore } = require('./core/native')
    return createNativeCore(opts.device)
  }

  if (name !== 'auto') {
    throw new Error(`polygrad: unknown core '${name}'`)
  }

  try {
    const { createNativeCore } = require('./core/native')
    return createNativeCore(opts.device)
  } catch (e) { /* fall through to WASM */ }

  const { createWasmCore } = require('./core/wasm')
  return createWasmCore(opts.device)
}

async function create(opts) {
  const options = normalizeOptions(opts)
  if (options.core === 'auto' && typeof process !== 'undefined' && process.env) {
    if (process.env.POLY_CORE) {
      options.core = process.env.POLY_CORE
    }
  }
  if (options.device === 'auto' && typeof process !== 'undefined' && process.env && process.env.POLY_DEVICE) {
    options.device = process.env.POLY_DEVICE
  }
  return createRuntime(options, resolveNodeCore)
}

module.exports = { create }
