'use strict'

const { createRuntime } = require('./runtime')

async function resolveBrowserBackend(name) {
  if (name === 'native') {
    throw new Error('polygrad: browser bundle does not support the native backend')
  }
  if (name !== 'auto' && name !== 'wasm') {
    throw new Error(`polygrad: browser bundle only supports the wasm backend (got ${name})`)
  }
  const { createWasmBackend } = require('./wasm')
  return createWasmBackend()
}

async function create(opts) {
  return createRuntime(opts, resolveBrowserBackend)
}

module.exports = { create }
