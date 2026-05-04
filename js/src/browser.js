'use strict'

const { createRuntime, normalizeOptions } = require('./runtime')

const BROWSER_DEVICES = new Set(['auto', 'cpu', 'interp', 'webgpu'])

async function resolveBrowserCore(name, opts) {
  if (!BROWSER_DEVICES.has(opts.device)) {
    throw new Error(`polygrad: browser supports device=${[...BROWSER_DEVICES].join('/')} (got ${opts.device})`)
  }
  if (name === 'native') {
    throw new Error('polygrad: browser bundle does not support core=\'native\'')
  }
  if (name !== 'auto' && name !== 'wasm') {
    throw new Error(`polygrad: browser bundle only supports core='wasm' (got ${name})`)
  }
  const { createWasmCore } = require('./core/wasm')
  return createWasmCore(opts.device)
}

async function create(opts) {
  return createRuntime(normalizeOptions(opts), resolveBrowserCore)
}

module.exports = { create }
