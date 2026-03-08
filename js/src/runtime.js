'use strict'

const { createBoundTensorClass } = require('./tensor')

class PolyRuntime {
  constructor(backend) {
    this._backend = backend
    this.Tensor = createBoundTensorClass(this)
  }

  get backend() { return this._backend.caps.backend }

  async dispose() {
    if (this._backend && this._backend.destroy) this._backend.destroy()
    this._backend = null
  }
}

async function createRuntime(opts, resolveBackend) {
  if (typeof resolveBackend !== 'function') {
    throw new TypeError('polygrad: createRuntime requires a backend resolver')
  }
  if (!opts) opts = {}
  const backendName = opts.backend || 'auto'
  const backend = await resolveBackend(backendName, opts)
  return new PolyRuntime(backend)
}

module.exports = { PolyRuntime, createRuntime }
