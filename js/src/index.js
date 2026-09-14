'use strict'

const { createRuntime, createRuntimeAsync, normalizeOptions } = require('./runtime')
const { PolyAsyncRequired, PolyWasmSyncUnsupported } = require('./errors')

function applyNodeEnv(options) {
  if (typeof process !== 'undefined' && process.env) {
    for (const key of ['BEAM', 'NOOPT']) {
      if (!process.env[key]) continue
      const n = Number(process.env[key])
      if (!/^\s*[+-]?\d+\s*$/.test(process.env[key]) || !Number.isInteger(n) || n < -2147483648 || n > 2147483647) {
        throw new Error(`${key} must be a decimal int32`)
      }
    }
  }
  if (options.core === 'auto' && typeof process !== 'undefined' && process.env) {
    if (process.env.POLY_CORE) options.core = process.env.POLY_CORE
  }
  if (options.device === 'auto' && typeof process !== 'undefined' && process.env) {
    const target = process.env.POLY_DEV || process.env.DEV
    if (target) {
      const name = target.toLowerCase()
      // DEV is Target syntax, not a Tensor device's ordinal suffix.
      if (!['auto', 'host', 'cpu', 'interp', 'x86', 'cuda', 'hip', 'wasm', 'webgpu', 'cpu:x86'].includes(name)) {
        throw new Error(`Unsupported Polygrad device target: ${target} (POLY_DEV/DEV)`)
      }
      options.device = name === 'cpu:x86' ? 'x86' : name
    }
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
     * context. POLY_DEV is folded into opts.device above. */
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
  const runtime = createRuntime(applyNodeEnv(normalizeOptions(opts)), resolveNodeCore)
  runtime._modelFiles = require('node:fs')
  return runtime
}

async function createAsync(opts) {
  const runtime = await createRuntimeAsync(applyNodeEnv(normalizeOptions(opts)), resolveNodeCoreAsync)
  runtime._modelFiles = require('node:fs')
  return runtime
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
