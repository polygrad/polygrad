'use strict'

// Method signatures/ownership come from the SDK manifest, not frontend lists.
function validateMethods(spec) {
  if (!Object.hasOwn(spec, 'methods')) {
    if (spec.construction_abi === 2) throw Error('unsupported extension manifest: ABI 2 requires methods')
    return
  }
  if (spec.construction_abi !== 2 || !Array.isArray(spec.methods) || !spec.methods.length || spec.methods.length > 256)
    throw Error('unsupported extension method manifest')
  const names = new Set(), objects = new Set()
  for (const m of spec.methods) {
    if (!m || typeof m.name !== 'string' || !m.name || names.has(m.name) || !Array.isArray(m.args) || m.args.length > 64 ||
        !m.result || !['object','tensor','string','value','status','void'].includes(m.result.kind))
      throw Error('invalid extension method declaration')
    names.add(m.name)
    if (m.result.kind === 'object') {
      if (typeof m.result.object !== 'string' || !m.result.object || typeof m.result.release !== 'string')
        throw Error('invalid extension object ownership')
      objects.add(m.result.object)
    }
    if (m.result.kind === 'string' && !['borrowed','owned'].includes(m.result.ownership))
      throw Error('invalid extension string ownership')
  }
  for (const m of spec.methods) for (const a of m.args) {
    if (!a || !['context','object','tensor','model','string','int','double'].includes(a.kind) ||
        (a.kind === 'object' && !objects.has(a.object))) throw Error('invalid extension method argument')
  }
}

function attachMethods(rt, spec, invoke, release, wrap, idle) {
  const objects = new Map(), wrappers = new WeakMap(), pending = new Set()
  let closed = false, generation = 0, disposal
  const registry = rt._extensions || (rt._extensions = new Set())
  function disposeObject(owner, token) {
    if (!owner.active) return undefined
    owner.active = false
    objects.delete(owner.generation)
    if (token && finalizer) finalizer.unregister(token)
    const run = () => { if (rt._lifetime.alive) release(owner.method, owner.handle) }
    const state = rt._lifetime
    // Match Tensor/UOp finalizers: never reenter a suspended Asyncify stack.
    const result = state.asyncHost && state.core && state.core.enqueueAsync ? state.core.enqueueAsync(run) : run()
    if (result && typeof result.then === 'function') {
      pending.add(result)
      result.then(() => pending.delete(result), () => pending.delete(result))
    }
    return result
  }
  const finalizer = typeof FinalizationRegistry === 'undefined' ? null : new FinalizationRegistry(owner => {
    const result = disposeObject(owner)
    if (result && typeof result.catch === 'function') result.catch(() => {})
  })
  const state = {dispose() {
    if (closed) return disposal
    closed = true
    const errors = []
    for (const owner of [...objects.values()].reverse()) {
      try { disposeObject(owner) } catch (e) { errors.push(e) }
    }
    if (pending.size) disposal = Promise.allSettled([...pending]).then(results => {
      const error = errors[0] || results.find(r => r.status === 'rejected')?.reason
      if (error) throw error
    }).finally(() => registry.delete(state))
    else {
      registry.delete(state)
      if (errors.length) throw errors[0]
    }
    return disposal
  }}
  registry.add(state)
  const methods = Object.create(null)
  spec.methods.forEach((m, method) => {
    methods[m.name] = (...values) => {
      idle()
      if (closed) throw Error('extension is disposed')
      if (values.length !== m.args.filter(a => a.kind !== 'context').length) throw TypeError('extension method argument count mismatch')
      const handles = [], strings = [], scalars = []; let v = 0
      m.args.forEach(a => {
        const x = a.kind === 'context' ? null : values[v++]
        handles.push(null); strings.push(null); scalars.push(0)
        const i = handles.length - 1
        if (a.kind === 'object') {
          const owner = wrappers.get(x)
          if (!owner || !owner.active || owner.type !== a.object || objects.get(owner.generation) !== owner)
            throw Error('extension method requires a live object from this extension instance')
          handles[i] = owner.handle
        } else if (a.kind === 'tensor') {
          if (!x || x._rt !== rt || !x._tensor) throw Error('extension method requires a live Tensor from this runtime')
          handles[i] = x._tensor
        } else if (a.kind === 'model') {
          if (!(x instanceof rt.Model) || x._rt !== rt || !x._handle || x._closing)
            throw Error('extension method requires a live Model from this runtime')
          handles[i] = x._handle
        } else if (a.kind === 'string') {
          if (typeof x !== 'string' || x.includes('\0') || /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(x))
            throw TypeError('extension string must be valid UTF-8 without NUL')
          strings[i] = x
        } else if (a.kind !== 'context') {
          if (typeof x !== 'number' || !Number.isFinite(x) ||
              (a.kind === 'int' && (!Number.isInteger(x) || x < -2147483648 || x > 2147483647)))
            throw TypeError('invalid extension scalar')
          scalars[i] = x
        }
      })
      const result = invoke(method, handles, strings, scalars)
      if (m.result.kind === 'tensor') {
        try { return wrap(result.handle) } catch (e) { release(method, result.handle); throw e }
      }
      if (m.result.kind === 'object') {
        const owner = {handle:result.handle, method, type:m.result.object, generation:++generation, active:true}
        const object = Object.freeze({dispose() { return disposeObject(owner, object) }})
        objects.set(owner.generation, owner); wrappers.set(object, owner)
        if (finalizer) finalizer.register(object, owner, object)
        return object
      }
      if (m.result.kind === 'string') return result.text
      if (m.result.kind === 'value') return result.value
      return undefined
    }
  })
  return {methods:Object.freeze(methods), dispose:state.dispose}
}
module.exports = {validateMethods, attachMethods}
