'use strict'

const { createRuntime, normalizeOptions } = require('./runtime')

async function resolveNodeTarget(name, opts) {
  if (opts.device !== 'auto' && opts.device !== 'cpu') {
    throw new Error(`polygrad: only device='cpu' is supported today (got ${opts.device})`)
  }

  if (name === 'wasm') {
    const { createWasmBackend } = require('./wasm')
    return createWasmBackend()
  }

  if (name === 'native') {
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

  const { createWasmBackend } = require('./wasm')
  return createWasmBackend()
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
