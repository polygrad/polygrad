'use strict'

const { createBoundTensorClass } = require('./tensor')

function normalizeOptions(opts) {
  const options = opts ? { ...opts } : {}

  if (options.target == null && options.backend != null) {
    options.target = options.backend
  }
  if (options.target == null) {
    options.target = 'auto'
  }
  if (options.device == null) {
    options.device = 'auto'
  }

  return options
}

class PolyRuntime {
  constructor(binding) {
    this._backend = binding
    this.Tensor = createBoundTensorClass(this)
  }

  get target() { return this._backend.caps.target }

  get device() { return this._backend.caps.device }

  // Compatibility alias for the old public shape.
  get backend() { return this.target }

  async dispose() {
    if (this._backend && this._backend.destroy) this._backend.destroy()
    this._backend = null
  }
}

async function createRuntime(opts, resolveTarget) {
  if (typeof resolveTarget !== 'function') {
    throw new TypeError('polygrad: createRuntime requires a target resolver')
  }
  const options = normalizeOptions(opts)
  const binding = await resolveTarget(options.target, options)
  return new PolyRuntime(binding)
}

module.exports = { PolyRuntime, createRuntime, normalizeOptions }
