'use strict'

const { createBoundInstanceClass } = require('./instance')
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
    this.supportsInstance = Boolean(binding.instance)
    this.Tensor = createBoundTensorClass(this)
    this.Instance = createBoundInstanceClass(this)
    this.ROLE_PARAM = this.Instance.ROLE_PARAM
    this.ROLE_INPUT = this.Instance.ROLE_INPUT
    this.ROLE_TARGET = this.Instance.ROLE_TARGET
    this.ROLE_OUTPUT = this.Instance.ROLE_OUTPUT
    this.ROLE_AUX = this.Instance.ROLE_AUX
    this.OPTIM_NONE = this.Instance.OPTIM_NONE
    this.OPTIM_SGD = this.Instance.OPTIM_SGD
    this.OPTIM_ADAM = this.Instance.OPTIM_ADAM
    this.OPTIM_ADAMW = this.Instance.OPTIM_ADAMW
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
