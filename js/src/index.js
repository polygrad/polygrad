'use strict'

const { createRuntime, normalizeOptions } = require('./runtime')

async function resolveNodeTarget(name, opts) {
  if (name === 'wasm') {
    const { createWasmBackend } = require('./exec_wasm')
    return createWasmBackend(opts.device)
  }

  if (name === 'native') {
    /* Native target delegates device selection to the C library via POLY_DEVICE
     * env var. The C backend validates and routes to cpu/x64/cuda/hip/interp. */
    const { createNativeBackend } = require('./native')
    return createNativeBackend()
  }

  if (name !== 'auto') {
    throw new Error(`polygrad: unknown target '${name}'`)
  }

  try {
    const { createNativeBackend } = require('./native')
    return createNativeBackend()
  } catch (e) { /* fall through to WASM */ }

  const { createWasmBackend } = require('./exec_wasm')
  return createWasmBackend(opts.device)
}

async function create(opts) {
  const options = normalizeOptions(opts)
  if (options.target === 'auto' && typeof process !== 'undefined' && process.env) {
    if (process.env.POLY_TARGET) {
      options.target = process.env.POLY_TARGET
    } else if (process.env.POLY_BACKEND) {
      options.target = process.env.POLY_BACKEND
    }
  }
  if (options.device === 'auto' && typeof process !== 'undefined' && process.env && process.env.POLY_DEVICE) {
    options.device = process.env.POLY_DEVICE
  }
  return createRuntime(options, resolveNodeTarget)
}

module.exports = { create }
