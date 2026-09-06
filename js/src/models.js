'use strict'

const { PolyAsyncRequired } = require('./errors')

function normalizeSpec(spec) {
  if (typeof spec === 'string') return spec
  if (spec && typeof spec === 'object') return JSON.stringify(spec)
  throw new TypeError('polygrad: model spec must be an object or JSON string')
}

function createBoundModels(runtime) {
  const _runtime = runtime

  function ensureApi() {
    if (_runtime._closing || !_runtime._core) throw new Error('polygrad runtime has been disposed')
    const api = _runtime._core.model
    if (!api) throw new Error('polygrad: model runtime unavailable for this core')
    return api
  }

  function wrap(handle, family) {
    if (!handle) throw new Error(`polygrad: failed to create ${family} model`)
    return new _runtime.Model(handle)
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

  function compose(family, spec, async = false) {
    const api = ensureApi()
    const json = normalizeSpec(spec)
    if (_runtime._usesAsyncHostBridge()) {
      if (!async) throw new PolyAsyncRequired(`models.${family}()`, `models.${family}Async()`)
      // Register ownership before releasing async admission, so Runtime disposal
      // also covers a factory suspended inside C construction or initialization.
      return _runtime._withAsync(() => _runtime._core.enqueueAsync(async () =>
        wrap(await api.composeAsync(_runtime._core.ctx, json, family), family)))
    }
    return wrap(api.compose(_runtime._core.ctx, json, family), family)
  }

  function Sequential(spec) { return compose('Sequential', spec) }
  function Graph(spec) { return compose('Graph', spec) }
  async function SequentialAsync(spec) { return compose('Sequential', spec, true) }
  async function GraphAsync(spec) { return compose('Graph', spec, true) }

  return { MLP, TabM, NAM, Sequential, Graph, SequentialAsync, GraphAsync }
}

module.exports = { createBoundModels }
