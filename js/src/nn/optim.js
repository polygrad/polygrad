'use strict'

function dedup(items) {
  const out = []
  const seen = new Set()
  for (const item of items) {
    if (seen.has(item)) continue
    seen.add(item)
    out.push(item)
  }
  return out
}

function normalizeParams(params) {
  if (!params) return []
  if (Array.isArray(params)) return params
  return Array.from(params)
}

function parseOptions(lrOrOpts, defaults) {
  if (lrOrOpts != null && typeof lrOrOpts === 'object' && !lrOrOpts._tensor) return { ...defaults, ...lrOrOpts }
  return { ...defaults, lr: lrOrOpts == null ? defaults.lr : lrOrOpts }
}

function createBoundOptim(runtime) {
  const Tensor = runtime.Tensor
  const ffi = runtime._core.ffi

  class Optimizer {
    constructor(params, lr = 0.001, opts = {}) {
      const lrTensor = lr instanceof Tensor ? lr : null
      if (!lrTensor && Number(lr) < 0) throw new Error(`Invalid learning rate: ${lr}`)
      const allParams = normalizeParams(params)
      this.params = dedup(allParams.filter(p => p && p.isParam))
      if (!this.params.length) throw new Error('optimizer must have at least one param')
      this.buffers = dedup(allParams.filter(p => p && !p.isParam))
      this.device = opts.device || this.params[0].device
      this._ctx = this.params[0]._ctx
      this._rt = this.params[0]._rt
      // Same supplied-storage contract as Python: never detach or copy a
      // caller-owned LR Tensor; the scheduled graph must see later writes.
      this.lr = lrTensor || new Tensor([Number(lr)], {
        dtype: runtime.defaultFloat === 'float64' ? 'float64' : 'float32',
        device: this.device, _ctx: this._ctx
      })
      this._validateLearningRate()
    }

    zeroGrad() {
      for (const p of this.params) p._grad = null
    }

    zero_grad() { this.zeroGrad() }

    _config() {
      throw new Error('optimizer subclass must implement _config()')
    }

    _stateArgs() {
      return { m: null, v: null, bc1: null, bc2: null }
    }

    _scheduledEffects() {
      return [...this.params]
    }

    _validateLearningRate() {
      if (!(this.lr instanceof Tensor) || this.lr._ctx !== this._ctx) {
        throw new Error('learning rate Tensor must share the optimizer context')
      }
      const lrShape = this.lr.shape
      if (this.lr.device !== String(this.device).toUpperCase() ||
          !((lrShape.length === 0) || (lrShape.length === 1 && lrShape[0] === 1))) {
        throw new Error('learning rate Tensor must be scalar or shape [1] on the optimizer device')
      }
      if (this.lr.dtype !== 'float32' && this.lr.dtype !== 'float64') {
        throw new Error('learning rate Tensor must have at least float32 precision')
      }
    }

    scheduleStep() {
      // nn.optim.Optimizer.schedule_step uses the same training context as
      // dropout and BatchNorm; reject before publishing assignment effects.
      if (!Tensor.training) throw new Error('TRAINING must be enabled to use the optimizer')
      const grads = []
      for (const p of this.params) {
        if (!p.grad) throw new Error('optimizer parameter has no gradient')
        grads.push(p.grad)
      }
      this._validateLearningRate()
      const state = this._stateArgs()
      const result = ffi.poly_optim_build_step(
        this._ctx,
        this._config(),
        this.lr._tensor,
        this.params.map(t => t._tensor),
        grads.map(t => t._tensor),
        state.m ? state.m.map(t => t._tensor) : null,
        state.v ? state.v.map(t => t._tensor) : null,
        state.bc1 ? state.bc1._tensor : null,
        state.bc2 ? state.bc2._tensor : null
      )
      if (!result) throw new Error('optimizer step graph build failed')

      const scheduled = this._scheduledEffects()
      if (result.length !== scheduled.length) {
        throw new Error(
          `optimizer step graph returned ${result.length} tensors, expected ${scheduled.length}`
        )
      }
      return scheduled.concat(this.buffers)
    }

    schedule_step() { return this.scheduleStep() }

    step() {
      const scheduled = this.scheduleStep()
      if (scheduled.length) scheduled[0].realize(...scheduled.slice(1))
    }

    async stepAsync() {
      const scheduled = this.scheduleStep()
      if (scheduled.length) await scheduled[0].realizeAsync(...scheduled.slice(1))
    }
  }

  class OptimizerGroup {
    constructor(...optimizers) {
      this.optimizers = optimizers
      this.params = optimizers.flatMap(o => o.params)
      this.buffers = optimizers.flatMap(o => o.buffers)
    }

    zeroGrad() {
      for (const opt of this.optimizers) opt.zeroGrad()
    }

    zero_grad() { this.zeroGrad() }

    scheduleStep() {
      return this.optimizers.flatMap(opt => opt.scheduleStep())
    }

    schedule_step() { return this.scheduleStep() }

    step() {
      const scheduled = this.scheduleStep()
      if (scheduled.length) scheduled[0].realize(...scheduled.slice(1))
    }

    async stepAsync() {
      const scheduled = this.scheduleStep()
      if (scheduled.length) await scheduled[0].realizeAsync(...scheduled.slice(1))
    }
  }

  class SGD extends Optimizer {
    constructor(params, lrOrOpts = 0.001) {
      const opts = parseOptions(lrOrOpts, {
        lr: 0.001,
        momentum: 0,
        weightDecay: 0,
        nesterov: false,
        classic: false,
        fused: false,
        device: null
      })
      if (opts.fused) throw new Error('fused optimizers are not implemented in Polygrad yet')
      if (opts.momentum < 0) throw new Error(`Invalid momentum value: ${opts.momentum}`)
      super(params, opts.lr, opts)
      this.momentum = Number(opts.momentum || 0)
      this.weightDecay = Number(opts.weight_decay != null ? opts.weight_decay : (opts.weightDecay || 0))
      this.nesterov = Boolean(opts.nesterov)
      this.classic = Boolean(opts.classic)
      this.b = this.momentum
        ? this.params.map(p => Tensor.zeros(
          p.shape, { dtype: 'float32', device: this.device, _ctx: this._ctx }
        ).is_param_(false))
        : []
      this.velocities = this.b
    }

    _config() {
      return {
        kind: runtime.OPTIM_SGD,
        beta1: 0,
        beta2: 0,
        eps: 0,
        weightDecay: this.weightDecay,
        momentum: this.momentum,
        nesterov: this.nesterov,
        classic: this.classic
      }
    }

    _stateArgs() {
      return { m: this.momentum ? this.b : null, v: null, bc1: null, bc2: null }
    }

    _scheduledEffects() {
      if (!this.momentum) return [...this.params]
      return [...this.b, ...this.params]
    }
  }

  class Adam extends Optimizer {
    constructor(params, lrOrOpts = 0.001) {
      const opts = parseOptions(lrOrOpts, {
        lr: 0.001,
        betas: null,
        b1: 0.9,
        b2: 0.999,
        eps: 1e-8,
        weightDecay: 0,
        fused: false,
        device: null
      })
      if (opts.fused) throw new Error('fused optimizers are not implemented in Polygrad yet')
      const weightDecay = opts.weight_decay != null ? opts.weight_decay : opts.weightDecay
      if (weightDecay) throw new Error('Adam weightDecay is not tinygrad-compatible; use AdamW')
      if (opts.betas) {
        opts.b1 = opts.betas[0]
        opts.b2 = opts.betas[1]
      }
      super(params, opts.lr, opts)
      this.b1 = Number(opts.b1)
      this.b2 = Number(opts.b2)
      this.eps = Number(opts.eps)
      this.weightDecay = 0
      this.m = this.params.map(p => Tensor.zeros(
        p.shape, { dtype: 'float32', device: this.device, _ctx: this._ctx }
      ).is_param_(false))
      this.v = this.params.map(p => Tensor.zeros(
        p.shape, { dtype: 'float32', device: this.device, _ctx: this._ctx }
      ).is_param_(false))
      this.b1_t = Tensor.ones(
        1, { dtype: 'float32', device: this.device, _ctx: this._ctx }
      ).is_param_(false)
      this.b2_t = Tensor.ones(
        1, { dtype: 'float32', device: this.device, _ctx: this._ctx }
      ).is_param_(false)
      this._bc1 = this.b1_t
      this._bc2 = this.b2_t
    }

    _kind() { return runtime.OPTIM_ADAM }

    _config() {
      return {
        kind: this._kind(),
        beta1: this.b1,
        beta2: this.b2,
        eps: this.eps,
        weightDecay: this.weightDecay,
        momentum: 0,
        nesterov: false,
        classic: false
      }
    }

    _stateArgs() {
      return { m: this.m, v: this.v, bc1: this.b1_t, bc2: this.b2_t }
    }

    _scheduledEffects() {
      return [this.b1_t, this.b2_t, ...this.m, ...this.v, ...this.params]
    }
  }

  class AdamW extends Adam {
    constructor(params, lrOrOpts = 0.001) {
      const opts = parseOptions(lrOrOpts, {
        lr: 0.001,
        betas: null,
        b1: 0.9,
        b2: 0.999,
        eps: 1e-8,
        weightDecay: 0.01,
        fused: false,
        device: null
      })
      if (opts.weight_decay != null) opts.weightDecay = opts.weight_decay
      if (opts.weightDecay < 0) throw new Error(`Invalid weightDecay value: ${opts.weightDecay}`)
      super(params, { ...opts, weightDecay: 0, weight_decay: 0 })
      this.weightDecay = Number(opts.weightDecay)
    }

    _kind() { return runtime.OPTIM_ADAMW }
  }

  class LARS extends SGD {
    constructor(params, lrOrOpts = .001) {
      const opts = parseOptions(lrOrOpts, {lr: .001, momentum: .9, weightDecay: 1e-4,
        nsSteps: 0, nsCoefficients: null, nesterov: false, classic: true, preWd: true, tcoef: .001})
      super(params, opts)
      this.nsSteps = opts.ns_steps == null ? opts.nsSteps : opts.ns_steps
      this.nsCoefficients = Array.from(opts.ns_coefficients || opts.nsCoefficients || [])
      this.preWd = opts.pre_wd == null ? opts.preWd : opts.pre_wd
      this.tcoef = opts.tcoef
    }
    _config() {
      return {...super._config(), kind: 4, nsSteps: this.nsSteps, nsCoefficients: this.nsCoefficients,
        preWd: this.preWd, tcoef: this.tcoef}
    }
  }

  function Muon(params, lrOrOpts = .001) {
    const opts = parseOptions(lrOrOpts, {lr: .001, momentum: .95, weightDecay: .1, nsSteps: 5,
      nsCoefficients: [3.4445, -4.775, 2.0315], nesterov: true})
    if (opts.fused) throw new Error('FUSE_OPTIM not allowed for Muon optimizer')
    return new LARS(params, {...opts, classic: false, preWd: false, pre_wd: false, tcoef: 0})
  }

  class LAMB extends AdamW {
    constructor(params, lrOrOpts = .001) {
      const opts = parseOptions(lrOrOpts, {lr: .001, eps: 1e-6, weightDecay: 0, adam: false})
      super(params, opts)
      this.adam = Boolean(opts.adam)
    }
    _kind() { return this.adam ? runtime.OPTIM_ADAMW : 5 }
  }

  return { Optimizer, OptimizerGroup, SGD, Adam, AdamW, LARS, LAMB, Muon }
}

module.exports = { createBoundOptim }
