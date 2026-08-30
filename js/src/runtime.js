'use strict'

const { createBoundInstanceClass } = require('./instance')
const { createBoundJit } = require('./jit')
const { createBoundModels } = require('./models')
const { createBoundModules } = require('./nn/modules')
const { createBoundOptim } = require('./nn/optim')
const { getParameters, getStateDict } = require('./nn/state')
const { createBoundTensorClass } = require('./tensor')
const { createBoundTokenizerClass } = require('./tokenizer')
const { createBoundUopNamespace } = require('./uop/ops')

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
    const caps = binding && binding.caps
    this._lifetime = {
      alive: true,
      core: binding,
      asyncHost: Boolean(caps && caps.core === 'wasm' && caps.device === 'webgpu')
    }
    this._closing = false
    this._disposePromise = null
    this._activeAsync = 0
    this._asyncDrain = []
    this.supportsInstance = Boolean(binding.instance)
    this.uop = createBoundUopNamespace(this)
    this.Tensor = createBoundTensorClass(this)
    this.jit = createBoundJit(this)
    this.compile = this.jit.compile
    this.jitAsync = this.jit.async
    this.compileAsync = this.jit.compileAsync
    this.Instance = createBoundInstanceClass(this)
    this.models = createBoundModels(this)
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
    this.nn = {
      Linear: modules.Linear,
      LayerNorm: modules.LayerNorm,
      LayerNorm2d: modules.LayerNorm2d,
      Conv2d: modules.Conv2d,
      GroupNorm: modules.GroupNorm,
      getParameters,
      getStateDict,
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

  stats() {
    if (this._activeAsync > 0) {
      throw new Error('polygrad runtime has active async work')
    }
    const coreStats = this._core && this._core.ffi && this._core.ffi.poly_ctx_stats
      ? this._core.ffi.poly_ctx_stats(this._core.ctx)
      : null
    return {
      core: this.core,
      device: this.device,
      caps: this.caps,
      coreStats,
      jit: this.jit && this.jit.stats ? this.jit.stats() : null
    }
  }

  resetCounters() {
    if (!this._core || !this._core.ffi || !this._core.ffi.poly_ctx_reset_counters) {
      throw new Error('resetCounters requires core counter support')
    }
    this._core.ffi.poly_ctx_reset_counters(this._core.ctx)
  }

  canRun(query) {
    const q = query || {}
    const caps = this.caps
    if (q.core && q.core !== 'auto' && q.core !== caps.core) return false
    if (q.device && q.device !== 'auto' && q.device !== caps.device) return false
    const dtype = normalizeCanRunDType(q.dtype || 'float32')
    if (dtype === 'float64' && caps.f64 === false) return false
    if (dtype === 'float16' && caps.f16 === false) return false
    if (q.shape != null && q.op == null) {
      throw new Error('canRun shape queries require an op')
    }
    if (q.op == null) return true
    if (!this._core || typeof this._core.canRunOp !== 'function') {
      throw new Error('canRun op/shape queries require core support')
    }
    const op = normalizeCanRunOp(q.op)
    const shape = normalizeCanRunShape(op, q.shape, q.shapes)
    const dtypeId = this._core.dtypeIds && this._core.dtypeIds[dtype]
    if (dtypeId == null || dtypeId < 0) return false
    const deviceId = canRunDeviceId(this._core, q.device || caps.device || 'auto')
    const rc = this._core.canRunOp(deviceId, op, dtypeId, shape)
    if (rc < 0) {
      throw new Error('canRun cannot prove this op/shape query')
    }
    return rc === 1
  }

  _usesAsyncHostBridge() {
    const caps = this._core && this._core.caps
    return Boolean(caps && caps.core === 'wasm' && caps.device === 'webgpu')
  }

  _beginAsync() {
    if (this._closing || !this._core) throw new Error('polygrad runtime has been disposed')
    if (!this._usesAsyncHostBridge()) return () => {}
    this._activeAsync++
    let released = false
    return () => {
      if (released) return
      released = true
      this._activeAsync--
      if (this._activeAsync === 0) {
        const waiters = this._asyncDrain.splice(0)
        for (const resolve of waiters) resolve()
      }
    }
  }

  _withAsync(fn) {
    let release
    try {
      release = this._beginAsync()
      return Promise.resolve(fn()).finally(release)
    } catch (err) {
      if (release) release()
      return Promise.reject(err)
    }
  }

  _waitForAsync() {
    if (this._activeAsync === 0) return Promise.resolve()
    return new Promise(resolve => this._asyncDrain.push(resolve))
  }

  dispose() {
    if (this._disposePromise) return this._disposePromise
    if (!this._core) return undefined

    const core = this._core
    const asyncHost = this._usesAsyncHostBridge()
    this._closing = true

    if (!asyncHost) {
      if (this.Instance && this.Instance._disposeAll) this.Instance._disposeAll()
      if (this.jit && this.jit.disposeAll) this.jit.disposeAll(true)
      if (this.Tensor && this.Tensor._disposeAll) this.Tensor._disposeAll()
      if (this.uop && this.uop._disposeAll) this.uop._disposeAll()
      if (this._lifetime) this._lifetime.alive = false
      if (core.destroy) core.destroy()
      this._core = null
      return undefined
    }

    this._disposePromise = this._waitForAsync().then(async () => {
      if (this.Instance && this.Instance._disposeAll) await this.Instance._disposeAll()
      if (this.jit && this.jit.disposeAll) this.jit.disposeAll(true)
      if (this.Tensor && this.Tensor._disposeAll) await this.Tensor._disposeAll()
      if (this.uop && this.uop._disposeAll) this.uop._disposeAll()
      if (this._lifetime) this._lifetime.alive = false
      return core.destroy ? core.destroy() : undefined
    }).finally(() => {
      this._core = null
    })
    return this._disposePromise
  }
}

function normalizeCanRunDType(dtype) {
  if (dtype == null) return 'float32'
  const d = String(dtype).toLowerCase()
  if (d === 'half') return 'float16'
  if (d === 'double') return 'float64'
  return d
}

function normalizeCanRunOp(op) {
  const raw = String(op)
  const s = raw.replace(/[A-Z]/g, c => '_' + c.toLowerCase()).replace(/-/g, '_').toLowerCase()
  if (s === 'reduce_sum') return 'reduce_sum'
  if (s === 'triangularsolve') return 'triangular_solve'
  return s
}

function normalizeShapeArray(shape, label) {
  if (!Array.isArray(shape)) throw new TypeError(`canRun ${label} must be an array`)
  return shape.map(x => {
    const v = Number(x)
    if (!Number.isSafeInteger(v) || v < 0) {
      throw new RangeError(`canRun ${label} contains invalid dimension ${x}`)
    }
    return v
  })
}

function normalizeCanRunShape(op, shape, shapes) {
  if (shapes != null) {
    if (!Array.isArray(shapes) || shapes.length === 0) {
      throw new TypeError('canRun shapes must be a non-empty array of shapes')
    }
    const ss = shapes.map((s, i) => normalizeShapeArray(s, `shapes[${i}]`))
    if (op === 'matmul' || op === 'dot') {
      if (ss.length < 2 || ss[0].length < 2 || ss[1].length < 2) {
        throw new Error('canRun matmul shapes must be [[m,k],[k,n]]')
      }
      const a = ss[0], b = ss[1]
      return [a[a.length - 2], a[a.length - 1], b[b.length - 1]]
    }
    if (op === 'triangular_solve' || op === 'solve' || op === 'lstsq') {
      if (ss.length < 2 || ss[0].length < 2 || ss[1].length < 1) {
        throw new Error(`canRun ${op} shapes must be [matrixShape, rhsShape]`)
      }
      const a = ss[0], b = ss[1]
      const rhs = b.length >= 2 ? b[b.length - 1] : null
      return rhs == null ? [a[a.length - 2], a[a.length - 1]] : [a[a.length - 2], a[a.length - 1], rhs]
    }
    return ss[0]
  }
  if (shape == null) throw new Error('canRun op queries require shape or shapes')
  return normalizeShapeArray(shape, 'shape')
}

function canRunDeviceId(core, device) {
  const name = device || 'auto'
  if (core.deviceIds && Object.prototype.hasOwnProperty.call(core.deviceIds, name)) {
    return core.deviceIds[name]
  }
  if (core.ffi && typeof core.ffi.poly_device_by_name === 'function') {
    return core.ffi.poly_device_by_name(name)
  }
  return 0
}

function createRuntime(opts, resolveCore) {
  if (typeof resolveCore !== 'function') {
    throw new TypeError('polygrad: createRuntime requires a core resolver')
  }
  const options = normalizeOptions(opts)
  const binding = resolveCore(options.core, options)
  return new PolyRuntime(binding)
}

async function createRuntimeAsync(opts, resolveCore) {
  if (typeof resolveCore !== 'function') {
    throw new TypeError('polygrad: createRuntimeAsync requires a core resolver')
  }
  const options = normalizeOptions(opts)
  const binding = await resolveCore(options.core, options)
  return new PolyRuntime(binding)
}

module.exports = { PolyRuntime, createRuntime, createRuntimeAsync, normalizeOptions }
