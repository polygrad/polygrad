'use strict'

const { createRuntime } = require('./runtime')

async function resolveNodeBackend(name) {
  if (name === 'wasm') {
    const { createWasmBackend } = require('./wasm')
    return createWasmBackend()
  }

  if (name === 'native') {
    const { createNativeBackend } = require('./native')
    return createNativeBackend()
  }

  if (name !== 'auto') {
    throw new Error(`polygrad: unknown backend '${name}'`)
  }

  try {
    const { createNativeBackend } = require('./native')
    return createNativeBackend()
  } catch (e) { /* fall through to WASM */ }

  const { createWasmBackend } = require('./wasm')
  return createWasmBackend()
}

async function create(opts) {
  const options = opts ? { ...opts } : {}
  if (!options.backend && typeof process !== 'undefined' && process.env && process.env.POLY_BACKEND) {
    options.backend = process.env.POLY_BACKEND
  }
  return createRuntime(options, resolveNodeBackend)
}

module.exports = { create }
