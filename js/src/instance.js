'use strict'

const ROLE_PARAM = 0
const ROLE_INPUT = 1
const ROLE_TARGET = 2
const ROLE_OUTPUT = 3
const ROLE_AUX = 4

const OPTIM_NONE = 0
const OPTIM_SGD = 1
const OPTIM_ADAM = 2
const OPTIM_ADAMW = 3

function normalizeSpec(spec) {
  if (typeof spec === 'string') return spec
  if (spec && typeof spec === 'object') return JSON.stringify(spec)
  throw new TypeError('polygrad: model spec must be an object or JSON string')
}

function normalizeBytes(bytes, name) {
  if (bytes == null) return null
  if (bytes instanceof Uint8Array) return bytes
  if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes)
  throw new TypeError(`polygrad: ${name} must be a Uint8Array or ArrayBuffer`)
}

function normalizeBindings(io) {
  if (!io || typeof io !== 'object' || Array.isArray(io)) {
    throw new TypeError('polygrad: bindings must be an object of name -> Float32 data')
  }

  const names = []
  const arrays = []
  for (const [name, value] of Object.entries(io)) {
    let arr
    if (value instanceof Float32Array) {
      arr = value
    } else if (value instanceof Float64Array) {
      arr = new Float32Array(value)
    } else if (Array.isArray(value)) {
      arr = Float32Array.from(value)
    } else if (typeof value === 'number') {
      arr = new Float32Array([value])
    } else {
      throw new TypeError(`polygrad: binding '${name}' must be number, array, or Float32Array`)
    }
    names.push(name)
    arrays.push(arr)
  }
  return { names, arrays }
}

function createBoundInstanceClass(runtime) {
  const _runtime = runtime

  class Instance {
    constructor(handle) {
      if (!handle) throw new Error('polygrad: failed to create PolyInstance')
      this._rt = _runtime
      this._handle = handle
    }

    static fromIR(irBytes, weightsBytes) {
      const api = _runtime._backend.instance
      if (!api) throw new Error('polygrad: model runtime unavailable for this target')
      const inst = api.fromIR(
        normalizeBytes(irBytes, 'irBytes'),
        normalizeBytes(weightsBytes, 'weightsBytes')
      )
      if (!inst) throw new Error('polygrad: failed to create PolyInstance from IR')
      return new Instance(inst)
    }

    static mlp(spec) {
      const api = _runtime._backend.instance
      if (!api) throw new Error('polygrad: model runtime unavailable for this target')
      const inst = api.mlp(normalizeSpec(spec))
      if (!inst) throw new Error('polygrad: failed to create MLP instance')
      return new Instance(inst)
    }

    static tabm(spec) {
      const api = _runtime._backend.instance
      if (!api) throw new Error('polygrad: model runtime unavailable for this target')
      const inst = api.tabm(normalizeSpec(spec))
      if (!inst) throw new Error('polygrad: failed to create TabM instance')
      return new Instance(inst)
    }

    static nam(spec) {
      const api = _runtime._backend.instance
      if (!api) throw new Error('polygrad: model runtime unavailable for this target')
      const inst = api.nam(normalizeSpec(spec))
      if (!inst) throw new Error('polygrad: failed to create NAM instance')
      return new Instance(inst)
    }

    dispose() {
      if (this._handle) {
        this._rt._backend.instance.free(this._handle)
        this._handle = null
      }
    }

    free() {
      this.dispose()
    }

    get paramCount() {
      return this._rt._backend.instance.paramCount(this._handle)
    }

    paramName(i) {
      return this._rt._backend.instance.paramName(this._handle, i)
    }

    paramShape(i) {
      return this._rt._backend.instance.paramShape(this._handle, i)
    }

    paramData(i) {
      return this._rt._backend.instance.paramData(this._handle, i)
    }

    params() {
      const items = []
      for (let i = 0; i < this.paramCount; i++) {
        items.push([this.paramName(i), this.paramShape(i), this.paramData(i)])
      }
      return items
    }

    get bufCount() {
      return this._rt._backend.instance.bufCount(this._handle)
    }

    bufName(i) {
      return this._rt._backend.instance.bufName(this._handle, i)
    }

    bufRole(i) {
      return this._rt._backend.instance.bufRole(this._handle, i)
    }

    bufShape(i) {
      return this._rt._backend.instance.bufShape(this._handle, i)
    }

    bufData(i) {
      return this._rt._backend.instance.bufData(this._handle, i)
    }

    findBuf(name) {
      for (let i = 0; i < this.bufCount; i++) {
        if (this.bufName(i) === name) return i
      }
      return -1
    }

    exportWeights() {
      return this._rt._backend.instance.exportWeights(this._handle)
    }

    importWeights(bytes) {
      const rc = this._rt._backend.instance.importWeights(
        this._handle,
        normalizeBytes(bytes, 'weights')
      )
      if (rc !== 0) throw new Error(`polygrad: importWeights failed (rc=${rc})`)
    }

    exportIR() {
      return this._rt._backend.instance.exportIR(this._handle)
    }

    saveBundle() {
      return this._rt._backend.instance.saveBundle(this._handle)
    }

    static fromBundle(bytes) {
      const api = _runtime._backend.instance
      if (!api) throw new Error('polygrad: model runtime unavailable for this target')
      const handle = api.fromBundle(bytes)
      if (!handle) throw new Error('polygrad: fromBundle failed')
      return new Instance(handle)
    }

    setOptimizer(kind, lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.0) {
      const rc = this._rt._backend.instance.setOptimizer(
        this._handle, kind, lr, beta1, beta2, eps, weightDecay
      )
      if (rc !== 0) throw new Error(`polygrad: setOptimizer failed (rc=${rc})`)
      return this
    }

    forward(io) {
      const { names, arrays } = normalizeBindings(io)
      const rc = this._rt._backend.instance.forward(this._handle, names, arrays)
      if (rc !== 0) throw new Error(`polygrad: forward failed (rc=${rc})`)
      return this._collectOutputs()
    }

    trainStep(io) {
      const { names, arrays } = normalizeBindings(io)
      const loss = this._rt._backend.instance.trainStep(this._handle, names, arrays)
      if (loss == null || Number.isNaN(loss)) {
        throw new Error('polygrad: trainStep failed')
      }
      return loss
    }

    _collectOutputs() {
      const outputs = {}
      for (let i = 0; i < this.bufCount; i++) {
        if (this.bufRole(i) === ROLE_OUTPUT) {
          outputs[this.bufName(i)] = this.bufData(i)
        }
      }
      return outputs
    }
  }

  Instance.ROLE_PARAM = ROLE_PARAM
  Instance.ROLE_INPUT = ROLE_INPUT
  Instance.ROLE_TARGET = ROLE_TARGET
  Instance.ROLE_OUTPUT = ROLE_OUTPUT
  Instance.ROLE_AUX = ROLE_AUX
  Instance.OPTIM_NONE = OPTIM_NONE
  Instance.OPTIM_SGD = OPTIM_SGD
  Instance.OPTIM_ADAM = OPTIM_ADAM
  Instance.OPTIM_ADAMW = OPTIM_ADAMW

  return Instance
}

module.exports = { createBoundInstanceClass }
