'use strict'

const { UOp } = require('../uop/ops')

function normalizeShape(shape) {
  if (shape == null) throw new Error('shape is required for exported ABI tensors')
  if (typeof shape === 'number') return [shape]
  return Array.from(shape, x => Number(x))
}

function normalizeNamed(value, defaultName) {
  if (!value) return {}
  if (value && value._tensor) return { [defaultName]: value }
  return { ...value }
}

function normalizeParams(params) {
  if (!params) return []
  if (Array.isArray(params)) return params.map((p, i) => [`param_${i}`, p])
  if (params instanceof Map) return Array.from(params.entries())
  return Object.entries(params)
}

function getParameters(obj) {
  const out = []
  const seen = new Set()
  const walk = (x) => {
    if (!x || seen.has(x)) return
    seen.add(x)
    if (x._tensor) {
      if (x.requiresGrad) out.push(x)
      return
    }
    if (Array.isArray(x)) {
      for (const v of x) walk(v)
      return
    }
    if (typeof x === 'object') {
      for (const [k, v] of Object.entries(x)) {
        if (!String(k).startsWith('_')) walk(v)
      }
    }
  }
  walk(obj)
  return out
}

function getStateDict(obj) {
  const state = {}
  const seen = new Set()
  const walk = (x, prefix) => {
    if (!x || seen.has(x)) return
    seen.add(x)
    if (x._tensor) {
      if (prefix) state[prefix] = x
      return
    }
    if (Array.isArray(x)) {
      x.forEach((v, i) => walk(v, prefix ? `${prefix}.${i}` : String(i)))
      return
    }
    if (typeof x === 'object') {
      for (const [k, v] of Object.entries(x)) {
        if (!String(k).startsWith('_')) walk(v, prefix ? `${prefix}.${k}` : k)
      }
    }
  }
  walk(obj, '')
  return state
}

function createBoundModel(runtime) {
  const Tensor = runtime.Tensor
  const ffi = runtime._core.ffi
  const ctx = runtime._core.ctx

  function registerTensor(name, shape, dtype, role, opts = {}) {
    const dims = normalizeShape(shape)
    const dtypeId = runtime._core.dtypeIds[dtype || 'float32']
    const raw = ffi.poly_register_buffer_by_id(ctx, role, dtypeId, dims, name)
    if (!raw) throw new Error(`polygrad: failed to register named buffer ${name}`)
    let uop = raw
    if (dims.length > 1) {
      uop = ffi.poly_reshape(ctx, raw, dims, dims.length)
      if (!uop) throw new Error(`polygrad: failed to shape named buffer ${name}`)
    }
    return new Tensor(null, {
      _ctx: ctx,
      _uop: new UOp(ctx, ffi, uop),
      _dtype: dtype || 'float32',
      _device: opts.device || runtime.device || 'cpu'
    })
  }

  function Input(name, opts = {}) {
    if (typeof name !== 'string') {
      opts = { ...opts, shape: name }
      name = 'input'
    }
    return registerTensor(name, opts.shape, opts.dtype || 'float32', runtime.ROLE_INPUT, opts)
  }

  function Target(name, opts = {}) {
    if (typeof name !== 'string') {
      opts = { ...opts, shape: name }
      name = 'target'
    }
    return registerTensor(name, opts.shape, opts.dtype || 'float32', runtime.ROLE_TARGET, opts)
  }

  class Model {
    constructor({ inputs, outputs, targets, losses, params, name } = {}) {
      this.name = name || null
      this.inputs = normalizeNamed(inputs, 'input')
      this.outputs = normalizeNamed(outputs, 'output')
      this.targets = normalizeNamed(targets, 'target')
      this.losses = normalizeNamed(losses, 'loss')
      this.params = params || null
      const tensors = [
        ...Object.values(this.inputs),
        ...Object.values(this.outputs),
        ...Object.values(this.targets),
        ...Object.values(this.losses)
      ]
      if (!tensors.length) throw new Error('nn.Model requires at least one endpoint tensor')
      for (const t of tensors) {
        if (!t || t._ctx !== tensors[0]._ctx) {
          throw new Error('all nn.Model tensors must belong to the same PolyCtx')
        }
      }
      this._ctx = tensors[0]._ctx
    }

    static trace(obj, { inputs, targets, outputs, loss, name } = {}) {
      const inps = normalizeNamed(inputs, 'input')
      const call = typeof obj === 'function' ? obj : obj && obj.call
      if (typeof call !== 'function') {
        throw new Error('nn.trace expects a function or an object with call(x, ...)')
      }
      const y = call.apply(obj, Object.values(inps))
      const outs = outputs || { output: y }
      let losses = null
      if (loss) {
        const tgts = normalizeNamed(targets, 'target')
        losses = { loss: loss(y, ...Object.values(tgts)) }
      }
      return new Model({
        inputs: inps,
        outputs: outs,
        targets,
        losses,
        params: getStateDict(obj),
        name
      })
    }

    async _registerParams() {
      for (const [name, tensor] of normalizeParams(this.params)) {
        if (!tensor || !tensor._tensor) continue
        if (!tensor.uop || !tensor.uop.hasBufferIdentity()) await tensor.realize()
        const buf = tensor.uop && tensor.uop.buffer
        if (!buf) throw new Error(`parameter ${name} has no buffer identity`)
        const raw = ffi.poly_register_existing_buffer(
          this._ctx, runtime.ROLE_PARAM, buf.raw, tensor.shape,
          String(name), Boolean(tensor.requiresGrad)
        )
        if (!raw) throw new Error(`polygrad: failed to register parameter ${name}`)
      }
    }

    _outputBuffer(name, tensor) {
      const dtypeId = runtime._core.dtypeIds[tensor.dtype || 'float32']
      const raw = ffi.poly_register_buffer_by_id(
        this._ctx, runtime.ROLE_OUTPUT, dtypeId, tensor.shape, name
      )
      if (!raw) throw new Error(`polygrad: failed to register output buffer ${name}`)
      return raw
    }

    _entrySink(tensors) {
      const stores = []
      for (const [name, tensor] of Object.entries(tensors)) {
        const out = this._outputBuffer(name, tensor)
        const store = ffi.poly_store_val(this._ctx, out, tensor._uop)
        if (!store) throw new Error(`polygrad: failed to build STORE for ${name}`)
        stores.push(store)
      }
      if (stores.length === 1) return ffi.poly_sink1(this._ctx, stores[0])
      return ffi.poly_sink_n(this._ctx, stores)
    }

    async export() {
      await this._registerParams()
      const names = []
      const sinks = []
      if (Object.keys(this.outputs).length) {
        names.push('forward')
        sinks.push(this._entrySink(this.outputs))
      }
      if (Object.keys(this.losses).length) {
        names.push('loss')
        sinks.push(this._entrySink(this.losses))
      }
      if (!sinks.length) throw new Error('nn.Model.export requires outputs or losses')
      const handle = runtime._core.instance.fromSinks(this._ctx, names, sinks)
      return new runtime.Instance(handle)
    }

    async fit(io, opts = {}) {
      /* Model is an authoring wrapper. Train through the exported Instance so
       * custom loops and model.fit share the same C runtime path. */
      const inst = await this.export()
      this.instance = inst
      return inst.fit(io, {
        optimizer: opts.optimizer || 'sgd',
        lr: opts.lr == null ? 0.01 : opts.lr,
        beta1: opts.beta1,
        beta2: opts.beta2,
        eps: opts.eps,
        weightDecay: opts.weightDecay,
        epochs: opts.epochs == null ? 1 : opts.epochs,
        onStep: opts.onStep
      })
    }

    async saveBundle() {
      const inst = await this.export()
      return inst.saveBundle()
    }
  }

  const trace = (obj, opts) => Model.trace(obj, opts)
  const exportModel = async (obj, opts) => (await trace(obj, opts).export())

  return { Input, Target, Model, trace, export: exportModel, getParameters, getStateDict }
}

module.exports = { createBoundModel }
