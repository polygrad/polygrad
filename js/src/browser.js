'use strict'

const { createRuntime, normalizeOptions } = require('./runtime')

async function resolveBrowserTarget(name, opts) {
  if (opts.device !== 'auto' && opts.device !== 'cpu') {
    throw new Error(`polygrad: browser bundle only supports device='cpu' today (got ${opts.device})`)
  }
  if (name === 'native') {
    throw new Error('polygrad: browser bundle does not support target=\'native\'')
  }
  if (name !== 'auto' && name !== 'wasm') {
    throw new Error(`polygrad: browser bundle only supports target='wasm' (got ${name})`)
  }
  const { createWasmBackend } = require('./wasm')
  return createWasmBackend()
}

async function create(opts) {
  return createRuntime(normalizeOptions(opts), resolveBrowserTarget)
}

module.exports = { create }
