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

function optimizerKind(kind) {
  if (typeof kind === 'string') {
    const k = kind.toLowerCase()
    if (k === 'sgd') return OPTIM_SGD
    if (k === 'adam') return OPTIM_ADAM
    if (k === 'adamw') return OPTIM_ADAMW
  }
  return Number(kind)
}

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

function isPromiseLike(v) {
  return v && typeof v.then === 'function'
}

function createBoundInstanceClass(runtime) {
  const _runtime = runtime

  class Instance {
    constructor(handle) {
      if (!handle) throw new Error('polygrad: failed to create PolyInstance')
      this._rt = _runtime
      this._handle = handle
      this._asyncTail = Promise.resolve()
    }

    _usesAsyncHostBridge() {
      const caps = this._rt && this._rt._core && this._rt._core.caps
      return caps && caps.core === 'wasm' && caps.device === 'webgpu'
    }

    _enqueueAsync(fn) {
      const run = this._asyncTail.then(fn, fn)
      this._asyncTail = run.catch(() => {})
      return run
    }

    _paramDataRaw(i) {
      return this._rt._core.instance.paramData(this._handle, i)
    }

    _bufDataRaw(i) {
      return this._rt._core.instance.bufData(this._handle, i)
    }

    static fromIR(irBytes, weightsBytes) {
      const api = _runtime._core.instance
      if (!api) throw new Error('polygrad: model runtime unavailable for this core')
      const inst = api.fromIR(
        normalizeBytes(irBytes, 'irBytes'),
        normalizeBytes(weightsBytes, 'weightsBytes')
      )
      if (!inst) throw new Error('polygrad: failed to create PolyInstance from IR')
      return new Instance(inst)
    }

    static mlp(spec) {
      const api = _runtime._core.instance
      if (!api) throw new Error('polygrad: model runtime unavailable for this core')
      const inst = api.mlp(normalizeSpec(spec))
      if (!inst) throw new Error('polygrad: failed to create MLP instance')
      return new Instance(inst)
    }

    static tabm(spec) {
      const api = _runtime._core.instance
      if (!api) throw new Error('polygrad: model runtime unavailable for this core')
      const inst = api.tabm(normalizeSpec(spec))
      if (!inst) throw new Error('polygrad: failed to create TabM instance')
      return new Instance(inst)
    }

    static nam(spec) {
      const api = _runtime._core.instance
      if (!api) throw new Error('polygrad: model runtime unavailable for this core')
      const inst = api.nam(normalizeSpec(spec))
      if (!inst) throw new Error('polygrad: failed to create NAM instance')
      return new Instance(inst)
    }

    dispose() {
      if (this._handle) {
        this._rt._core.instance.free(this._handle)
        this._handle = null
      }
    }

    free() {
      this.dispose()
    }

    get paramCount() {
      return this._rt._core.instance.paramCount(this._handle)
    }

    paramName(i) {
      return this._rt._core.instance.paramName(this._handle, i)
    }

    paramShape(i) {
      return this._rt._core.instance.paramShape(this._handle, i)
    }

    paramData(i) {
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(() => this._paramDataRaw(i))
      return this._paramDataRaw(i)
    }

    paramTrainable(i) {
      return this._rt._core.instance.paramTrainable(this._handle, i)
    }

    setParamTrainable(i, trainable) {
      const rc = this._rt._core.instance.setParamTrainable(this._handle, i, Boolean(trainable))
      if (rc !== 0) throw new Error(`polygrad: setParamTrainable failed (rc=${rc})`)
      return this
    }

    params() {
      if (this._usesAsyncHostBridge()) {
        return this._enqueueAsync(async () => {
          const items = []
          for (let i = 0; i < this.paramCount; i++) {
            items.push([this.paramName(i), this.paramShape(i), await this._paramDataRaw(i)])
          }
          return items
        })
      }

      const items = []
      for (let i = 0; i < this.paramCount; i++) {
        items.push([this.paramName(i), this.paramShape(i), this._paramDataRaw(i)])
      }
      return items
    }

    get bufCount() {
      return this._rt._core.instance.bufCount(this._handle)
    }

    bufName(i) {
      return this._rt._core.instance.bufName(this._handle, i)
    }

    bufRole(i) {
      return this._rt._core.instance.bufRole(this._handle, i)
    }

    bufTrainable(i) {
      return this._rt._core.instance.bufTrainable(this._handle, i)
    }

    setBufTrainable(i, trainable) {
      const rc = this._rt._core.instance.setBufTrainable(this._handle, i, Boolean(trainable))
      if (rc !== 0) throw new Error(`polygrad: setBufTrainable failed (rc=${rc})`)
      return this
    }

    bufShape(i) {
      return this._rt._core.instance.bufShape(this._handle, i)
    }

    bufData(i) {
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(() => this._bufDataRaw(i))
      return this._bufDataRaw(i)
    }

    findBuf(name) {
      for (let i = 0; i < this.bufCount; i++) {
        if (this.bufName(i) === name) return i
      }
      return -1
    }

    exportWeights() {
      const run = () => this._rt._core.instance.exportWeights(this._handle)
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(run)
      return run()
    }

    importWeights(bytes) {
      const rc = this._rt._core.instance.importWeights(
        this._handle,
        normalizeBytes(bytes, 'weights')
      )
      if (rc !== 0) throw new Error(`polygrad: importWeights failed (rc=${rc})`)
    }

    exportIR() {
      return this._rt._core.instance.exportIR(this._handle)
    }

    saveBundle() {
      const run = () => this._rt._core.instance.saveBundle(this._handle)
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(run)
      return run()
    }

    static fromBundle(bytes) {
      const api = _runtime._core.instance
      if (!api) throw new Error('polygrad: model runtime unavailable for this core')
      const handle = api.fromBundle(bytes)
      if (!handle) throw new Error('polygrad: fromBundle failed')
      return new Instance(handle)
    }

    static loadHF(configBytes, weightFiles, opts = {}) {
      const api = _runtime._core.instance
      const cfg = normalizeBytes(configBytes, 'config')
      const wf = weightFiles.map((f, i) => normalizeBytes(f, `weight file ${i}`))
      const handle = api.loadHF(cfg, wf, opts.maxBatch, opts.maxSeqLen)
      if (!handle) {
        const err = api.importLastError && api.importLastError()
        throw new Error('polygrad: loadHF failed' + (err ? ': ' + err.message : ''))
      }
      return new Instance(handle)
    }

    static loadGGUF(ggufBytes, opts = {}) {
      const api = _runtime._core.instance
      const bytes = normalizeBytes(ggufBytes, 'gguf')
      const handle = api.loadGGUF(bytes, opts.maxBatch, opts.maxSeqLen)
      if (!handle) {
        const err = api.importLastError && api.importLastError()
        throw new Error('polygrad: loadGGUF failed' + (err ? ': ' + err.message : ''))
      }
      return new Instance(handle)
    }

    setOptimizer(kind, lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.0) {
      const rc = this._rt._core.instance.setOptimizer(
        this._handle, kind, lr, beta1, beta2, eps, weightDecay
      )
      if (rc !== 0) throw new Error(`polygrad: setOptimizer failed (rc=${rc})`)
      return this
    }

    forward(io) {
      const { names, arrays } = normalizeBindings(io)
      const run = () => {
        const rc = this._rt._core.instance.forward(this._handle, names, arrays)
        if (isPromiseLike(rc)) {
          return rc.then(v => {
            if (v !== 0) throw new Error(`polygrad: forward failed (rc=${v})`)
            return this._collectOutputsRaw()
          })
        }
        if (rc !== 0) throw new Error(`polygrad: forward failed (rc=${rc})`)
        return this._collectOutputsRaw()
      }
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(run)
      return run()
    }

    trainStep(io) {
      const { names, arrays } = normalizeBindings(io)
      const run = () => {
        const loss = this._rt._core.instance.trainStep(this._handle, names, arrays)
        if (isPromiseLike(loss)) {
          return loss.then(v => {
            if (v == null || Number.isNaN(v)) throw new Error('polygrad: trainStep failed')
            return v
          })
        }
        if (loss == null || Number.isNaN(loss)) {
          throw new Error('polygrad: trainStep failed')
        }
        return loss
      }
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(run)
      return run()
    }

    _collectOutputsRaw() {
      if (this._usesAsyncHostBridge()) {
        return (async () => {
          const outputs = {}
          for (let i = 0; i < this.bufCount; i++) {
            if (this.bufRole(i) === ROLE_OUTPUT) {
              outputs[this.bufName(i)] = await this._bufDataRaw(i)
            }
          }
          return outputs
        })()
      }

      const outputs = {}
      for (let i = 0; i < this.bufCount; i++) {
        if (this.bufRole(i) === ROLE_OUTPUT) {
          outputs[this.bufName(i)] = this._bufDataRaw(i)
        }
      }
      return outputs
    }

    _collectOutputs() {
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(() => this._collectOutputsRaw())
      return this._collectOutputsRaw()
    }

    fit(io, opts = {}) {
      /* This mirrors the Python convenience wrapper: keep loop ownership in the
       * frontend while all optimizer math and scheduling stay in the C core. */
      const epochs = opts.epochs == null ? 1 : Number(opts.epochs)
      if (opts.optimizer != null) {
        this.setOptimizer(
          optimizerKind(opts.optimizer),
          opts.lr == null ? 0.01 : opts.lr,
          opts.beta1 == null ? 0.9 : opts.beta1,
          opts.beta2 == null ? 0.999 : opts.beta2,
          opts.eps == null ? 1e-8 : opts.eps,
          opts.weightDecay == null ? 0.0 : opts.weightDecay
        )
      }
      const losses = []
      let asyncChain = null
      const runStep = (step) => {
        const loss = this.trainStep(io)
        if (isPromiseLike(loss)) {
          return loss.then(v => {
            losses.push(v)
            if (opts.onStep) opts.onStep(step, v)
          })
        }
        losses.push(loss)
        if (opts.onStep) opts.onStep(step, loss)
        return null
      }
      for (let step = 0; step < epochs; step++) {
        if (asyncChain) asyncChain = asyncChain.then(() => runStep(step))
        else {
          const r = runStep(step)
          if (isPromiseLike(r)) asyncChain = r
        }
      }
      if (asyncChain) return asyncChain.then(() => losses)
      return losses
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
