'use strict'

const { createRuntime, normalizeOptions } = require('./runtime')

const BROWSER_DEVICES = new Set(['auto', 'cpu', 'interp'])

async function resolveBrowserTarget(name, opts) {
  if (!BROWSER_DEVICES.has(opts.device)) {
    throw new Error(`polygrad: browser supports device=${[...BROWSER_DEVICES].join('/')} (got ${opts.device})`)
  }
  if (name === 'native') {
    throw new Error('polygrad: browser bundle does not support target=\'native\'')
  }
  if (name !== 'auto' && name !== 'wasm') {
    throw new Error(`polygrad: browser bundle only supports target='wasm' (got ${name})`)
  }
  const { createWasmBackend } = require('./wasm')
  return createWasmBackend(opts.device)
}

async function create(opts) {
  return createRuntime(normalizeOptions(opts), resolveBrowserTarget)
}

module.exports = { create }
