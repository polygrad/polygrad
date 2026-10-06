'use strict'

const { UOp } = require('./uop/ops')

// C owns FUNCTION parameterization and both physical/logical graph rewrites.
function createBoundFunction(runtime) {
  return function capture(fn, {allowImplicit = false, precompile = false, precompileBackward = false, gradFxn = null} = {}) {
    if (typeof fn !== 'function') throw new TypeError('function expects a callable')
    if (gradFxn !== null) throw new Error('function gradFxn callbacks are not yet implemented')
    for (const flag of [allowImplicit, precompile, precompileBackward]) {
      if (typeof flag !== 'boolean') throw new TypeError('function options must be boolean')
    }
    return function (...args) {
      if (!runtime._lifetime.alive || runtime._closing) throw new Error('polygrad runtime has been disposed')
      if (runtime._activeAsync) throw new Error('polygrad runtime is busy')
      const {ffi, ctx} = runtime._core
      const logical = [], physical = [], retained = [], active = new Set()
      const visit = value => {
        let l, p
        if (value instanceof UOp) {
          if (value.ctx !== ctx || value.ffi !== ffi) throw new Error('function inputs must share a runtime')
          l = p = value.raw
        } else if (value && typeof value._physicalUopRaw === 'function') {
          if (value._rt !== runtime) throw new Error('function inputs must share a runtime')
          l = value._logicalUopRaw(); p = value._physicalUopRaw()
        } else {
          if (!value || typeof value !== 'object' || active.has(value)) return
          active.add(value)
          for (const item of Object.values(value)) visit(item)
          active.delete(value)
          return
        }
        if (!p) throw new Error('function input has no physical root')
        // Snapshot and retain before invoking the body: assignment/collection
        // must not retarget or invalidate the explicit input occurrences.
        logical.push(l); physical.push(p)
        for (const root of [l,p]) if (root) retained.push(new UOp(ctx, ffi, root))
      }
      try {
        visit([this === runtime ? null : this, args])
        const ret = fn.apply(this, args)
        const results = Array.isArray(ret) ? ret : [ret]
        if (!results.length || results.some(t => !(t instanceof runtime.Tensor) || !t._tensor)) {
          throw new TypeError('function must return a live Tensor or a nonempty Tensor array from this runtime')
        }
        const cores = ffi.poly_tensor_function(ctx, results.map(t => t._tensor), logical, physical,
          fn.name || 'function', allowImplicit, precompile, precompileBackward)
        const outputs = []
        try {
          for (let i = 0; i < cores.length; i++) outputs.push(results[i]._makeResultFromCore(cores[i]))
        } catch (error) {
          for (const output of outputs) output.dispose()
          for (let i = outputs.length; i < cores.length; i++) ffi.poly_tensor_release(cores[i])
          throw error
        }
        return Array.isArray(ret) ? outputs : outputs[0]
      } finally {
        for (const root of retained) root.dispose()
      }
    }
  }
}

module.exports = { createBoundFunction }
