'use strict'

const { createBoundInstanceClass } = require('./instance')
const { createBoundModules } = require('./nn/modules')
const { createBoundOptim } = require('./nn/optim')
const { createBoundModel } = require('./nn/model')
const { createBoundTensorClass } = require('./tensor')
const { createBoundTokenizerClass } = require('./tokenizer')

function normalizeOptions(opts) {
  const options = opts ? { ...opts } : {}

  if (options.core == null) {
    options.core = 'auto'
  }
  if (options.device == null) {
    options.device = 'auto'
  }

  return options
}

class PolyRuntime {
  constructor(binding) {
    this._core = binding
    this.supportsInstance = Boolean(binding.instance)
    this.Tensor = createBoundTensorClass(this)
    this.Instance = createBoundInstanceClass(this)
    this.Tokenizer = createBoundTokenizerClass(this)
    this.ROLE_PARAM = this.Instance.ROLE_PARAM
    this.ROLE_INPUT = this.Instance.ROLE_INPUT
    this.ROLE_TARGET = this.Instance.ROLE_TARGET
    this.ROLE_OUTPUT = this.Instance.ROLE_OUTPUT
    this.ROLE_AUX = this.Instance.ROLE_AUX
    this.OPTIM_NONE = this.Instance.OPTIM_NONE
    this.OPTIM_SGD = this.Instance.OPTIM_SGD
    this.OPTIM_ADAM = this.Instance.OPTIM_ADAM
    this.OPTIM_ADAMW = this.Instance.OPTIM_ADAMW
    const modules = createBoundModules(this)
    const optim = createBoundOptim(this)
    const model = createBoundModel(this)
    this.nn = {
      Linear: modules.Linear,
      Input: model.Input,
      Target: model.Target,
      Model: model.Model,
      trace: model.trace,
      export: model.export,
      getParameters: model.getParameters,
      getStateDict: model.getStateDict,
      optim,
      Optimizer: optim.Optimizer,
      OptimizerGroup: optim.OptimizerGroup,
      SGD: optim.SGD,
      Adam: optim.Adam,
      AdamW: optim.AdamW
    }
  }

  get core() { return this._core.caps.core }

  get device() { return this._core.caps.device }

  get caps() {
    // Public capability surface for tests and callers to avoid dispatching
    // unsupported dtype/backend combinations, mirroring tinygrad's checks.
    return { ...this._core.caps }
  }

  async dispose() {
    if (this._core && this._core.destroy) this._core.destroy()
    this._core = null
  }
}

async function createRuntime(opts, resolveCore) {
  if (typeof resolveCore !== 'function') {
    throw new TypeError('polygrad: createRuntime requires a core resolver')
  }
  const options = normalizeOptions(opts)
  const binding = await resolveCore(options.core, options)
  return new PolyRuntime(binding)
}

module.exports = { PolyRuntime, createRuntime, normalizeOptions }
