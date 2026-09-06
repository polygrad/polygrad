'use strict'

const { createRuntime, createRuntimeAsync, normalizeOptions } = require('./runtime')
const { PolyAsyncRequired, PolyWasmSyncUnsupported } = require('./errors')

function applyNodeEnv(options) {
  if (options.core === 'auto' && typeof process !== 'undefined' && process.env) {
    if (process.env.POLY_CORE) options.core = process.env.POLY_CORE
  }
  if (options.device === 'auto' && typeof process !== 'undefined' && process.env && process.env.POLY_DEVICE) {
    options.device = process.env.POLY_DEVICE
  }
  return options
}

function resolveNodeCore(name, opts) {
  if (name === 'wasm') {
    const { createWasmCoreSync } = require('./core/wasm')
    return createWasmCoreSync(opts.device)
  }

  if (name === 'native') {
    /* Native core validates and routes cpu/x86/cuda/hip/interp through the C
     * context. POLY_DEVICE is folded into opts.device above. */
    const { createNativeCore } = require('./core/native')
    return createNativeCore(opts.device)
  }

  if (name !== 'auto') {
    throw new Error(`polygrad: unknown core '${name}'`)
  }

  try {
    const { createNativeCore } = require('./core/native')
    return createNativeCore(opts.device)
  } catch (e) { /* fall through to WASM */ }

  const { createWasmCoreSync } = require('./core/wasm')
  return createWasmCoreSync(opts.device)
}

async function resolveNodeCoreAsync(name, opts) {
  if (name === 'wasm') {
    const { createWasmCoreAsync } = require('./core/wasm')
    return createWasmCoreAsync(opts.device)
  }

  if (name === 'native') {
    const { createNativeCore } = require('./core/native')
    return createNativeCore(opts.device)
  }

  if (name !== 'auto') {
    throw new Error(`polygrad: unknown core '${name}'`)
  }

  try {
    const { createNativeCore } = require('./core/native')
    return createNativeCore(opts.device)
  } catch (e) { /* fall through to WASM */ }

  const { createWasmCoreAsync } = require('./core/wasm')
  return createWasmCoreAsync(opts.device)
}

function create(opts) {
  return createRuntime(applyNodeEnv(normalizeOptions(opts)), resolveNodeCore)
}

async function createAsync(opts) {
  return createRuntimeAsync(applyNodeEnv(normalizeOptions(opts)), resolveNodeCoreAsync)
}

let defaultRuntime = null

function getDefaultRuntime() {
  if (!defaultRuntime) defaultRuntime = create({ core: 'auto', device: 'auto' })
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
  nn: defaultGetter('nn'),
  ROLE_PARAM: defaultGetter('ROLE_PARAM'),
  ROLE_INPUT: defaultGetter('ROLE_INPUT'),
  ROLE_TARGET: defaultGetter('ROLE_TARGET'),
  ROLE_OUTPUT: defaultGetter('ROLE_OUTPUT'),
  ROLE_AUX: defaultGetter('ROLE_AUX'),
  OPTIM_NONE: defaultGetter('OPTIM_NONE'),
  OPTIM_SGD: defaultGetter('OPTIM_SGD'),
  OPTIM_ADAM: defaultGetter('OPTIM_ADAM'),
  OPTIM_ADAMW: defaultGetter('OPTIM_ADAMW')
})

api.stats = () => getDefaultRuntime().stats()
api.canRun = (query) => getDefaultRuntime().canRun(query)

module.exports = api
