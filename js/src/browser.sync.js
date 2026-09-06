'use strict'

const { createRuntime, normalizeOptions } = require('./runtime')
const { PolyAsyncRequired, PolyWasmSyncUnsupported } = require('./errors')
const { createWasmCoreSync } = require('./core/wasm_sync')

const BROWSER_DEVICES = new Set(['auto', 'cpu', 'interp', 'webgpu'])

function resolveBrowserCore(name, opts) {
  if (!BROWSER_DEVICES.has(opts.device)) {
    throw new Error(`polygrad: browser supports device=${[...BROWSER_DEVICES].join('/')} (got ${opts.device})`)
  }
  if (name === 'native') {
    throw new Error('polygrad: browser bundle does not support core=\'native\'')
  }
  if (name !== 'auto' && name !== 'wasm') {
    throw new Error(`polygrad: browser bundle only supports core='wasm' (got ${name})`)
  }
  return createWasmCoreSync(opts.device)
}

function create(opts) {
  return createRuntime(normalizeOptions(opts), resolveBrowserCore)
}

async function createAsync() {
  throw new PolyWasmSyncUnsupported(
    "polygrad: this browser bundle is sync-only; import 'polygrad/async' or 'polygrad/browser/async' for async WASM startup"
  )
}

let defaultRuntime = null

function getDefaultRuntime() {
  if (!defaultRuntime) defaultRuntime = create({ core: 'wasm', device: 'auto' })
  return defaultRuntime
}

function disposeDefault() {
  if (defaultRuntime) defaultRuntime.dispose()
  defaultRuntime = null
}

function defaultGetter(name) {
  return {
    enumerable: true,
    get() { return getDefaultRuntime()[name] }
  }
}

const api = {
  create,
  createAsync,
  getDefaultRuntime,
  disposeDefault,
  PolyAsyncRequired,
  PolyWasmSyncUnsupported
}

Object.defineProperties(api, {
  Tensor: defaultGetter('Tensor'),
  uop: defaultGetter('uop'),
  jit: defaultGetter('jit'),
  jitAsync: defaultGetter('jitAsync'),
  compile: defaultGetter('compile'),
  compileAsync: defaultGetter('compileAsync'),
  Model: defaultGetter('Model'),
  models: defaultGetter('models'),
  Tokenizer: defaultGetter('Tokenizer'),
  nn: defaultGetter('nn')
})

api.stats = () => getDefaultRuntime().stats()
api.canRun = (query) => getDefaultRuntime().canRun(query)

module.exports = api
