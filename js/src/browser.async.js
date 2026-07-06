'use strict'

const { createRuntimeAsync, normalizeOptions } = require('./runtime')
const { PolyAsyncRequired, PolyWasmSyncUnsupported } = require('./errors')
const { createWasmCoreAsync } = require('./core/wasm_async')

const BROWSER_DEVICES = new Set(['auto', 'cpu', 'interp', 'webgpu'])

async function resolveBrowserCoreAsync(name, opts) {
  if (!BROWSER_DEVICES.has(opts.device)) {
    throw new Error(`polygrad: browser supports device=${[...BROWSER_DEVICES].join('/')} (got ${opts.device})`)
  }
  if (name === 'native') {
    throw new Error('polygrad: browser bundle does not support core=\'native\'')
  }
  if (name !== 'auto' && name !== 'wasm') {
    throw new Error(`polygrad: browser bundle only supports core='wasm' (got ${name})`)
  }
  return createWasmCoreAsync(opts.device)
}

function create() {
  throw new PolyWasmSyncUnsupported(
    "polygrad: this browser bundle is async-only; use await createAsync(...)"
  )
}

async function createAsync(opts) {
  return createRuntimeAsync(normalizeOptions(opts), resolveBrowserCoreAsync)
}

function getDefaultRuntime() {
  throw new PolyWasmSyncUnsupported(
    "polygrad: async browser bundle has no implicit default runtime; use await createAsync(...)"
  )
}

function disposeDefault() {}

const api = {
  create,
  createAsync,
  getDefaultRuntime,
  disposeDefault,
  PolyAsyncRequired,
  PolyWasmSyncUnsupported
}

module.exports = api
