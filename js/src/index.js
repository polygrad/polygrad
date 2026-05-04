'use strict'

const { createRuntime, normalizeOptions } = require('./runtime')

async function resolveNodeCore(name, opts) {
  if (name === 'wasm') {
    const { createWasmCore } = require('./core/wasm')
    return createWasmCore(opts.device)
  }

  if (name === 'native') {
    /* Native core delegates device selection to the C library via POLY_DEVICE
     * env var. The C runtime validates and routes to cpu/x64/cuda/hip/interp. */
    const { createNativeCore } = require('./core/native')
    return createNativeCore()
  }

  if (name !== 'auto') {
    throw new Error(`polygrad: unknown core '${name}'`)
  }

  try {
    const { createNativeCore } = require('./core/native')
    return createNativeCore()
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
