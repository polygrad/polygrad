'use strict'

const { PolyAsyncRequired } = require('./errors')

function normalizeSpec(spec) {
  if (typeof spec === 'string') return spec
  if (spec && typeof spec === 'object') return JSON.stringify(spec)
  throw new TypeError('polygrad: model spec must be an object or JSON string')
}

function buildModel(runtime, family, spec, async = false) {
  if (runtime._closing || !runtime._core) throw new Error('polygrad runtime has been disposed')
  if (runtime._activeAsync > 0) throw new Error('Model construction requires an idle Runtime')
  const api = runtime._core.model
  if (!api) throw new Error('polygrad: model runtime unavailable for this core')
  const json = normalizeSpec(spec)
  // Published tabular factories support synchronous host construction followed
  // by deferred WebGPU placement. Keep that API; new families use async capture.
  if (!async && runtime._usesAsyncHostBridge() && !['MLP', 'TabM', 'NAM'].includes(family))
    throw new PolyAsyncRequired('Model construction', `models.${family || 'Graph'}Async()`)
  const wrap = handle => {
    if (!handle) throw new Error('polygrad: Model construction failed')
    return runtime.Model._fromHandle(handle)
  }
  if (async && runtime._usesAsyncHostBridge()) {
    // Register ownership before releasing async admission.
    return runtime._withAsync(() => runtime._core.enqueueAsync(async () =>
      wrap(await api.fromConfigAsync(runtime._core.ctx, json, family))))
  }
  return wrap(api.fromConfig(runtime._core.ctx, json, family))
}

function createBoundModels(runtime) {
  const result = {}
  const api = runtime._core.model
  if (!api) return result
  const types = []
  for (let i = 0, name; (name = api.typeName(i)); i++) {
    const flags = api.typeCapabilities(i)
    types.push({ name, constructible: Boolean(flags & 1), hf: Boolean(flags & 2), gguf: Boolean(flags & 4) })
    if (flags & 1) {
      result[name] = spec => buildModel(runtime, name, spec)
      result[name + 'Async'] = async spec => buildModel(runtime, name, spec, true)
    } else {
      const loaders = [flags & 2 ? 'Model.fromHF' : null, flags & 4 ? 'Model.fromGGUF' : null].filter(Boolean)
      const reject = () => { throw new Error(`${name} is import-only; use ${loaders.join(' or ')}`) }
      result[name] = reject
      result[name + 'Async'] = async () => reject()
    }
  }
  result.list = () => types.map(type => ({ ...type }))
  return result
}

module.exports = { createBoundModels, buildModel }
