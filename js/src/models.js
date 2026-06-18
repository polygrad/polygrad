'use strict'

function normalizeSpec(spec) {
  if (typeof spec === 'string') return spec
  if (spec && typeof spec === 'object') return JSON.stringify(spec)
  throw new TypeError('polygrad: model spec must be an object or JSON string')
}

function createBoundModels(runtime) {
  const _runtime = runtime

  function ensureApi() {
    const api = _runtime._core.instance
    if (!api) throw new Error('polygrad: model runtime unavailable for this core')
    return api
  }

  function wrap(handle, family) {
    if (!handle) throw new Error(`polygrad: failed to create ${family} instance`)
    return new _runtime.Instance(handle)
  }

  function MLP(spec) {
    const api = ensureApi()
    return wrap(api.mlp(normalizeSpec(spec)), 'MLP')
  }

  function TabM(spec) {
    const api = ensureApi()
    return wrap(api.tabm(normalizeSpec(spec)), 'TabM')
  }

  function NAM(spec) {
    const api = ensureApi()
    return wrap(api.nam(normalizeSpec(spec)), 'NAM')
  }

  return { MLP, TabM, NAM }
}

module.exports = { createBoundModels }
