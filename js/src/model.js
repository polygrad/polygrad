'use strict'

const { PolyAsyncRequired } = require('./errors')

const ROLE_PARAM = 0
const ROLE_INPUT = 1
const ROLE_TARGET = 2
const ROLE_OUTPUT = 3
const ROLE_AUX = 4

const ROLE_IDS = {
  param: ROLE_PARAM,
  state: ROLE_PARAM,
  input: ROLE_INPUT,
  target: ROLE_TARGET,
  output: ROLE_OUTPUT,
  aux: ROLE_AUX
}

const OPTIM_NONE = 0
const OPTIM_SGD = 1
const OPTIM_ADAM = 2
const OPTIM_ADAMW = 3

const EXPORT_WEIGHTS_PARAMS = 1
const EXPORT_WEIGHTS_OPTIMIZER = 2
const EXPORT_WEIGHTS_DEFAULT = EXPORT_WEIGHTS_PARAMS | EXPORT_WEIGHTS_OPTIMIZER
const BIND_F_FROZEN = 1 << 2

function optimizerKind(kind) {
  if (typeof kind === 'string') {
    const k = kind.toLowerCase()
    if (k === 'sgd') return OPTIM_SGD
    if (k === 'adam') return OPTIM_ADAM
    if (k === 'adamw') return OPTIM_ADAMW
  }
  return Number(kind)
}

function weightExportFlags(options) {
  if (options == null) return EXPORT_WEIGHTS_DEFAULT
  if (typeof options === 'boolean') {
    return options ? EXPORT_WEIGHTS_DEFAULT : EXPORT_WEIGHTS_PARAMS
  }
  const includeOptimizer =
    options.includeOptimizer != null ? !!options.includeOptimizer :
      options.include_optimizer != null ? !!options.include_optimizer :
        true
  return EXPORT_WEIGHTS_PARAMS | (includeOptimizer ? EXPORT_WEIGHTS_OPTIMIZER : 0)
}

function modelDtypeName(core, dtypeId) {
  for (const [name, id] of Object.entries(core.dtypeIds || {})) {
    if (Number(id) === Number(dtypeId)) return name
  }
  throw new Error(`polygrad: unsupported Model storage dtype id ${dtypeId}`)
}


function normalizeBytes(bytes, name) {
  if (bytes == null) return null
  if (bytes instanceof Uint8Array) return bytes
  if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes)
  throw new TypeError(`polygrad: ${name} must be a Uint8Array or ArrayBuffer`)
}

function normalizeBindings(io) {
  if (!io || typeof io !== 'object' || Array.isArray(io)) {
    throw new TypeError('polygrad: bindings must be an object of name -> numeric data')
  }

  const names = []
  const arrays = []
  for (const [name, value] of Object.entries(io)) {
    let arr
    if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
      arr = value
    } else if (Array.isArray(value)) {
      arr = Float32Array.from(value)
    } else if (typeof value === 'number') {
      arr = new Float32Array([value])
    } else {
      throw new TypeError(`polygrad: binding '${name}' must be a number, array, or numeric TypedArray`)
    }
    names.push(name)
    arrays.push(arr)
  }
  return { names, arrays }
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

function roleId(role) {
  if (typeof role === 'string') {
    const key = role.toLowerCase()
    if (!Object.prototype.hasOwnProperty.call(ROLE_IDS, key)) {
      throw new Error(`polygrad: unknown Model binding role '${role}'`)
    }
    return ROLE_IDS[key]
  }
  return Number(role)
}

function entryNameList(names) {
  if (names == null) return []
  if (typeof names === 'string') return [names]
  return Array.from(names)
}

function bindingFields(binding) {
  if (Array.isArray(binding)) {
    if (binding.length === 3) {
      const [name, role, tensor] = binding
      return { name, role, tensor, flags: 0 }
    }
    if (binding.length === 4) {
      const [name, role, tensor, flags] = binding
      return { name, role, tensor, flags }
    }
  } else if (binding && typeof binding === 'object') {
    return {
      name: binding.name,
      role: binding.role,
      tensor: binding.tensor,
      flags: binding.flags || 0,
      trainable: binding.trainable
    }
  }
  throw new TypeError('polygrad: Model bindings must be objects or [name, role, tensor, flags] arrays')
}

function entryFields(entry) {
  if (Array.isArray(entry)) {
    if (entry.length === 3) {
      const [name, inputs, outputs] = entry
      return { name, inputs, outputs, objective: null, flags: 0 }
    }
    if (entry.length === 4) {
      const [name, inputs, outputs, objective] = entry
      return { name, inputs, outputs, objective, flags: 0 }
    }
    if (entry.length === 5) {
      const [name, inputs, outputs, objective, flags] = entry
      return { name, inputs, outputs, objective, flags }
    }
  } else if (entry && typeof entry === 'object') {
    return {
      name: entry.name,
      inputs: entry.inputs,
      outputs: entry.outputs,
      objective: entry.objective || null,
      flags: entry.flags || 0
    }
  }
  throw new TypeError('polygrad: Model entrypoints must be objects or [name, inputs, outputs] arrays')
}

function moduleFields(module) {
  if (Array.isArray(module) && module.length === 3) {
    const [name, inputs, output] = module
    return { name, inputs, output }
  }
  if (module && typeof module === 'object') {
    return { name: module.name, inputs: module.inputs, output: module.output }
  }
  throw new TypeError('polygrad: Model modules must be objects or [name, inputs, output] arrays')
}

function moduleInputs(inputs) {
  if (inputs == null) return []
  if (inputs && inputs._tensor) return [inputs]
  return Array.from(inputs)
}

function deviceMapEntries(deviceMap) {
  let rows
  if (deviceMap instanceof Map) rows = Array.from(deviceMap.entries())
  else if (Array.isArray(deviceMap)) rows = deviceMap
  else if (deviceMap && typeof deviceMap === 'object') rows = Object.entries(deviceMap)
  else rows = []
  return rows.map(row => {
    if (Array.isArray(row) && row.length === 2) {
      return { module: String(row[0]), device: String(row[1]) }
    }
    if (row && typeof row === 'object' && row.module != null && row.device != null) {
      return { module: String(row.module), device: String(row.device) }
    }
    throw new TypeError('polygrad: device-map entries must be [module, device] or {module, device}')
  })
}

function isPromiseLike(v) {
  return v && typeof v.then === 'function'
}

function isModelSpec(v) {
  return v && typeof v === 'object' && !Array.isArray(v) &&
    (v.inputs || v.targets || v.outputs || v.losses || v.params || v.state || v.entrypoints || v.modules)
}

function createBoundModelClass(runtime) {
  const _runtime = runtime
  const liveModelOwners = new Set()
  const modelFinalizer = typeof FinalizationRegistry === 'undefined'
    ? null
    : new FinalizationRegistry(owner => {
        const pending = releaseModelOwner(owner)
        if (pending && typeof pending.then === 'function') pending.catch(() => {})
      })

  function releaseModelOwner(owner, token = null) {
    if (!owner || !owner.active) return undefined
    owner.active = false
    liveModelOwners.delete(owner)
    if (modelFinalizer && token) modelFinalizer.unregister(token)
    const model = owner.ref && owner.ref.deref ? owner.ref.deref() : null
    if (model) {
      model._owner = null
      model._handle = null
      model._closing = true
    }
    if (!owner.state.alive || !owner.core || !owner.handle) return undefined
    const free = () => owner.core.model.free(owner.handle)
    if (owner.asyncHost && owner.core.enqueueAsync) return owner.core.enqueueAsync(free)
    free()
    return undefined
  }

  function requireTensor(name, tensor) {
    if (!tensor || !tensor._tensor) throw new Error(`${name} is not a Tensor`)
    return tensor
  }

  function requireStorageBinding(name, tensor) {
    if (!tensor.uop || !tensor.uop.hasBufferIdentity()) {
      throw new Error(`${name} has no buffer identity; inputs and targets require storage-backed Tensors`)
    }
  }

  function defineModulesOnHandle(handle, modules, expectedCtx = null) {
    if (modules == null) return
    const rows = Array.from(modules)
    if (!rows.length) throw new Error('Model.defineModules requires at least one module')
    const parsed = rows.map(module => {
      const m = moduleFields(module)
      if (m.name == null) throw new Error('Model module is missing a name')
      const inputs = moduleInputs(m.inputs).map((tensor, i) =>
        requireTensor(`${m.name}.inputs[${i}]`, tensor))
      const output = requireTensor(`${m.name}.output`, m.output)
      for (const tensor of inputs.concat([output])) {
        if (expectedCtx != null && tensor._ctx !== expectedCtx) {
          throw new Error(`Model module '${m.name}' contains a Tensor from another PolyCtx`)
        }
      }
      return {
        name: String(m.name),
        inputs: inputs.map(tensor => tensor._tensor),
        output: output._tensor
      }
    })
    const api = _runtime._core.model
    if (!api.defineModules) throw new Error('polygrad: module placement unavailable for this core')
    const rc = api.defineModules(handle, parsed)
    if (rc !== 0) throw new Error('polygrad: invalid or ambiguous Model module cuts')
  }

  function traceSpec(fn, { inputs, targets = {}, loss = null, state, params, entrypoints } = {}) {
    if (typeof fn !== 'function') throw new TypeError('Model.trace requires a callable')
    const result = fn(inputs)
    if (isPromiseLike(result)) throw new TypeError('Model.trace authoring must be synchronous')
    const losses = loss == null ? null : loss(result, targets)
    if (isPromiseLike(losses)) throw new TypeError('Model.trace loss must be synchronous')
    return { inputs, targets, outputs: result, losses, state, params, entrypoints }
  }

  function sealBindings(ctx, bindings, entries, modules, async = false) {
    const api = _runtime._core.model
    if (!api.fromBindings) throw new Error('polygrad: fromBindings unavailable for this core')
    const finish = handle => {
      if (!handle) throw new Error('polygrad: failed to create Model from tensor bindings')
      try {
        defineModulesOnHandle(handle, modules, ctx)
      } catch (err) {
        api.free(handle)
        throw err
      }
      return async ? new Model(handle) : handle
    }
    if (_runtime._usesAsyncHostBridge()) {
      if (!async) throw new PolyAsyncRequired('Model construction', 'Model.fromTensors()/traceAsync()/fromBindingsAsync()')
      // Queue construction and its marshalling before later Tensor releases.
      return _runtime._withAsync(() => _runtime._core.enqueueAsync(async () =>
        finish(await api.fromBindingsAsync(ctx, bindings, entries))))
    }
    return finish(api.fromBindings(ctx, bindings, entries))
  }

  function lowerTensorSpec(spec = {}, async = false) {
    const { inputs, outputs, targets, losses, entrypoints, modules } = spec
    let { params, state } = spec
    if (params != null && state != null) {
      throw new Error('Model accepts params or state, not both')
    }
    if (state != null) params = state

    const inps = normalizeNamed(inputs, 'input')
    const tgts = normalizeNamed(targets, 'target')
    const outs = normalizeNamed(outputs, 'output')
    const lossMap = normalizeNamed(losses, 'loss')
    const paramItems = normalizeParams(params)

    const namedTensors = []
    for (const group of [inps, tgts, outs, lossMap]) {
      for (const [name, tensor] of Object.entries(group)) {
        namedTensors.push([name, requireTensor(name, tensor)])
      }
    }
    for (const [name, tensor] of paramItems) namedTensors.push([name, requireTensor(name, tensor)])
    if (!namedTensors.length) throw new Error('Model requires at least one tensor binding')
    if (!Object.keys(outs).length && !Object.keys(lossMap).length) {
      throw new Error('Model requires outputs or losses')
    }

    const ctx = namedTensors[0][1]._ctx
    for (const [name, tensor] of namedTensors) {
      if (tensor._ctx !== ctx) throw new Error(`${name} belongs to another PolyCtx`)
    }
    for (const [name, tensor] of [...Object.entries(inps), ...Object.entries(tgts)]) {
      requireStorageBinding(name, tensor)
    }

    const bindings = []
    const addBinding = (name, role, tensor, flags = 0) => {
      bindings.push({ name: String(name), role, tensor: tensor._tensor, flags })
    }
    for (const [name, tensor] of Object.entries(inps)) addBinding(name, ROLE_INPUT, tensor)
    for (const [name, tensor] of Object.entries(tgts)) addBinding(name, ROLE_TARGET, tensor)
    for (const [name, tensor] of paramItems) {
      const role = state != null && !tensor.isParam ? ROLE_AUX : ROLE_PARAM
      addBinding(name, role, tensor, role === ROLE_PARAM && !tensor.isParam ? BIND_F_FROZEN : 0)
    }
    for (const [name, tensor] of Object.entries(outs)) addBinding(name, ROLE_OUTPUT, tensor)
    for (const [name, tensor] of Object.entries(lossMap)) addBinding(name, ROLE_OUTPUT, tensor)

    const entries = []
    if (entrypoints != null) {
      for (const entry of entrypoints) {
        const e = entryFields(entry)
        if (e.name == null) throw new Error('Model entrypoint is missing a name')
        entries.push({
          name: String(e.name),
          inputs: entryNameList(e.inputs),
          outputs: entryNameList(e.outputs),
          objective: e.objective == null ? null : String(e.objective),
          flags: Number(e.flags || 0)
        })
      }
    } else {
      const inputNames = Object.keys(inps)
      const targetNames = Object.keys(tgts)
      const outputNames = Object.keys(outs)
      const lossNames = Object.keys(lossMap)
      if (outputNames.length) entries.push({ name: 'forward', inputs: inputNames, outputs: outputNames })
      if (lossNames.length) {
        const objective = lossNames.length === 1 ? lossNames[0] : null
        entries.push({
          name: 'loss',
          inputs: inputNames.concat(targetNames),
          outputs: lossNames,
          objective
        })
      }
    }

    return sealBindings(ctx, bindings, entries, modules, async)
  }

  class Model {
    constructor(handle) {
      if (isModelSpec(handle)) handle = lowerTensorSpec(handle)
      if (!handle) throw new Error('polygrad: failed to create PolyModel')
      this._rt = _runtime
      this._handle = handle
      const caps = _runtime && _runtime._core && _runtime._core.caps
      this._asyncHostBridge = Boolean(caps && caps.core === 'wasm' && caps.device === 'webgpu')
      this._asyncTail = Promise.resolve()
      this._closing = false
      this._activeAsync = 0
      this._idleWaiters = []
      this._disposePromise = null
      this._owner = {
        state: _runtime._lifetime,
        core: _runtime._core,
        handle,
        asyncHost: this._asyncHostBridge,
        active: true,
        ref: typeof WeakRef === 'undefined' ? null : new WeakRef(this)
      }
      liveModelOwners.add(this._owner)
      if (modelFinalizer) modelFinalizer.register(this, this._owner, this)
    }

    _usesAsyncHostBridge() {
      return this._asyncHostBridge
    }

    _requireOpen() {
      if (this._closing || !this._handle) throw new Error('Model has been disposed')
    }

    _requireSync(method, asyncMethod) {
      this._requireOpen()
      if (this._usesAsyncHostBridge()) throw new PolyAsyncRequired(method, asyncMethod)
    }

    _enqueueAsync(fn) {
      if (this._closing || !this._handle) {
        return Promise.reject(new Error('Model has been disposed'))
      }
      this._activeAsync++
      const core = this._rt && this._rt._core
      const run = this._rt._withAsync(() => {
        if (core && core.enqueueAsync) return core.enqueueAsync(fn)
        const queued = this._asyncTail.then(fn, fn)
        this._asyncTail = queued.catch(() => {})
        return queued
      })
      return run.finally(() => {
        this._activeAsync--
        if (this._activeAsync === 0) {
          const waiters = this._idleWaiters.splice(0)
          for (const resolve of waiters) resolve()
        }
      })
    }

    _waitForAsync() {
      if (this._activeAsync === 0) return Promise.resolve()
      return new Promise(resolve => this._idleWaiters.push(resolve))
    }

    _paramDataRaw(i) {
      return this._rt._core.model.paramData(this._handle, i)
    }

    _bufDataRaw(i) {
      return this._rt._core.model.bufData(this._handle, i)
    }

    static fromIR(irBytes, weightsBytes) {
      const api = _runtime._core.model
      if (!api) throw new Error('polygrad: model runtime unavailable for this core')
      const inst = api.fromIR(
        normalizeBytes(irBytes, 'irBytes'),
        normalizeBytes(weightsBytes, 'weightsBytes')
      )
      if (!inst) throw new Error('polygrad: failed to create PolyModel from IR')
      return new Model(inst)
    }

    static fromProgram(programBytes, weightsBytes) {
      const api = _runtime._core.model
      if (!api) throw new Error('polygrad: model runtime unavailable for this core')
      if (_runtime._usesAsyncHostBridge()) {
        throw new PolyAsyncRequired('Model.fromProgram()', 'Model.fromProgramAsync()')
      }
      const inst = api.fromProgram(
        normalizeBytes(programBytes, 'programBytes'),
        normalizeBytes(weightsBytes, 'weightsBytes')
      )
      if (!inst) throw new Error('polygrad: failed to create PolyModel from program')
      return new Model(inst)
    }

    static async fromProgramAsync(programBytes, weightsBytes) {
      const api = _runtime._core.model
      if (!api) throw new Error('polygrad: model runtime unavailable for this core')
      const release = _runtime._beginAsync()
      try {
        const program = normalizeBytes(programBytes, 'programBytes')
        const weights = normalizeBytes(weightsBytes, 'weightsBytes')
        const inst = api.fromProgramAsync
          ? await api.fromProgramAsync(program, weights)
          : api.fromProgram(program, weights)
        if (!inst) throw new Error('polygrad: failed to create PolyModel from program')
        return new Model(inst)
      } finally {
        release()
      }
    }

    static _fromBindings(bindings, entrypoints, modules, async) {
      bindings = Array.from(bindings || [])
      entrypoints = Array.from(entrypoints || [])
      if (!bindings.length) throw new Error('Model.fromBindings requires at least one binding')
      if (!entrypoints.length) throw new Error('Model.fromBindings requires at least one entrypoint')

      const parsed = bindings.map(binding => {
        const b = bindingFields(binding)
        if (b.name == null) throw new Error('Model binding is missing a name')
        if (b.role == null) throw new Error(`Model binding '${b.name}' is missing a role`)
        const tensor = requireTensor(b.name, b.tensor)
        const role = roleId(b.role)
        let flags = Number(b.flags || 0)
        if (role === ROLE_PARAM && !(b.trainable == null ? tensor.isParam : b.trainable)) {
          flags |= BIND_F_FROZEN
        }
        return { name: String(b.name), role, tensor, flags }
      })

      const ctx = parsed[0].tensor._ctx
      for (const b of parsed) {
        if (b.tensor._ctx !== ctx) throw new Error(`${b.name} belongs to another PolyCtx`)
        if (b.role === ROLE_INPUT || b.role === ROLE_TARGET) {
          requireStorageBinding(b.name, b.tensor)
        }
      }

      const entries = entrypoints.map(entry => {
        const e = entryFields(entry)
        if (e.name == null) throw new Error('Model entrypoint is missing a name')
        return {
          name: String(e.name),
          inputs: entryNameList(e.inputs),
          outputs: entryNameList(e.outputs),
          objective: e.objective == null ? null : String(e.objective),
          flags: Number(e.flags || 0)
        }
      })

      return sealBindings(ctx, parsed.map(b => ({
        name: b.name, role: b.role, tensor: b.tensor._tensor, flags: b.flags
      })), entries, modules, async)
    }

    static fromBindings(bindings, entrypoints, modules = null) {
      return new Model(this._fromBindings(bindings, entrypoints, modules, false))
    }

    static async fromBindingsAsync(bindings, entrypoints, modules = null) {
      return this._fromBindings(bindings, entrypoints, modules, true)
    }

    static trace(fn, options) {
      if (_runtime._usesAsyncHostBridge()) throw new PolyAsyncRequired('Model.trace()', 'Model.traceAsync()')
      return new Model(traceSpec(fn, options))
    }

    static traceAsync(fn, options) {
      return this.fromTensors(traceSpec(fn, options))
    }

    static async fromTensors(spec = {}) {
      return lowerTensorSpec(spec, true)
    }

    static fromHF(configBytes, weightFiles, opts = {}) {
      const api = _runtime._core.model
      const cfg = normalizeBytes(configBytes, 'config')
      const wf = weightFiles.map((f, i) => normalizeBytes(f, `weight file ${i}`))
      const handle = api.loadHF(cfg, wf, opts.maxBatch, opts.maxSeqLen)
      if (!handle) {
        const err = api.importLastError && api.importLastError()
        throw new Error('polygrad: fromHF failed' + (err ? ': ' + err.message : ''))
      }
      return new Model(handle)
    }

    static fromGGUF(ggufBytes, opts = {}) {
      const api = _runtime._core.model
      const bytes = normalizeBytes(ggufBytes, 'gguf')
      const handle = api.loadGGUF(bytes, opts.maxBatch, opts.maxSeqLen)
      if (!handle) {
        const err = api.importLastError && api.importLastError()
        throw new Error('polygrad: fromGGUF failed' + (err ? ': ' + err.message : ''))
      }
      return new Model(handle)
    }

    dispose() {
      if (this._disposePromise) return this._disposePromise
      if (!this._handle) return undefined
      this._closing = true
      if (!this._rt || !this._rt._core) {
        if (this._owner) {
          this._owner.active = false
          liveModelOwners.delete(this._owner)
          if (modelFinalizer) modelFinalizer.unregister(this)
          this._owner = null
        }
        this._handle = null
        return undefined
      }
      if (!this._usesAsyncHostBridge()) {
        return releaseModelOwner(this._owner, this)
      }
      this._disposePromise = this._rt._withAsync(async () => {
        await this._waitForAsync()
        await releaseModelOwner(this._owner, this)
      })
      return this._disposePromise
    }

    free() {
      return this.dispose()
    }

    defineModules(modules) {
      defineModulesOnHandle(this._handle, modules)
      return this
    }

    setDeviceMap(deviceMap) {
      this._requireSync('setDeviceMap()', 'setDeviceMapAsync()')
      const entries = deviceMapEntries(deviceMap)
      if (!entries.length) throw new Error('Model.setDeviceMap requires at least one mapping')
      const rc = this._rt._core.model.setDeviceMap(this._handle, entries)
      if (isPromiseLike(rc)) throw new PolyAsyncRequired('setDeviceMap()', 'setDeviceMapAsync()')
      if (rc !== 0) throw new Error(`polygrad: invalid, incomplete, or unsupported device map (rc=${rc})`)
      return this
    }

    setDeviceMapAsync(deviceMap) {
      const entries = deviceMapEntries(deviceMap)
      if (!entries.length) {
        return Promise.reject(new Error('Model.setDeviceMap requires at least one mapping'))
      }
      const run = () => Promise.resolve(
        this._rt._core.model.setDeviceMap(this._handle, entries)
      ).then(rc => {
        if (rc !== 0) {
          throw new Error(`polygrad: invalid, incomplete, or unsupported device map (rc=${rc})`)
        }
        return this
      })
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(run)
      return run()
    }

    place(device) {
      this._requireSync('place()', 'placeAsync()')
      if (typeof device !== 'string') return this.setDeviceMap(device)
      if (this._rt._core.model.setDevice(this._handle, device) !== 0)
        throw new Error(`polygrad: Model placement failed for '${device}'`)
      return this
    }

    placeAsync(device) {
      this._requireOpen()
      if (typeof device !== 'string') return this.setDeviceMapAsync(device)
      const run = async () => {
        if (await this._rt._core.model.setDevice(this._handle, device) !== 0)
          throw new Error(`polygrad: Model placement failed for '${device}'`)
        return this
      }
      return this._usesAsyncHostBridge() ? this._enqueueAsync(run) : run()
    }

    get paramCount() {
      return this._rt._core.model.paramCount(this._handle)
    }

    paramName(i) {
      return this._rt._core.model.paramName(this._handle, i)
    }

    paramShape(i) {
      return this._rt._core.model.paramShape(this._handle, i)
    }

    paramDtype(i) {
      const core = this._rt._core
      return modelDtypeName(core, core.model.paramDtypeId(this._handle, i))
    }

    paramData(i) {
      this._requireSync('paramData()', 'paramDataAsync()')
      return this._paramDataRaw(i)
    }

    paramDataAsync(i) {
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(() => this._paramDataRaw(i))
      return Promise.resolve(this._paramDataRaw(i))
    }

    paramTrainable(i) {
      return this._rt._core.model.paramTrainable(this._handle, i)
    }

    setParamTrainable(i, trainable) {
      this._requireOpen()
      const rc = this._rt._core.model.setParamTrainable(this._handle, i, Boolean(trainable))
      if (rc !== 0) throw new Error(`polygrad: setParamTrainable failed (rc=${rc})`)
      return this
    }

    params() {
      this._requireSync('params()', 'paramsAsync()')

      const items = []
      for (let i = 0; i < this.paramCount; i++) {
        items.push([this.paramName(i), this.paramShape(i), this._paramDataRaw(i)])
      }
      return items
    }

    paramsAsync() {
      if (this._usesAsyncHostBridge()) {
        return this._enqueueAsync(async () => {
          const items = []
          for (let i = 0; i < this.paramCount; i++) {
            items.push([this.paramName(i), this.paramShape(i), await this._paramDataRaw(i)])
          }
          return items
        })
      }
      return Promise.resolve(this.params())
    }

    get bufCount() {
      return this._rt._core.model.bufCount(this._handle)
    }

    bufName(i) {
      return this._rt._core.model.bufName(this._handle, i)
    }

    bufRole(i) {
      return this._rt._core.model.bufRole(this._handle, i)
    }

    bufTrainable(i) {
      return this._rt._core.model.bufTrainable(this._handle, i)
    }

    setBufTrainable(i, trainable) {
      this._requireOpen()
      const rc = this._rt._core.model.setBufTrainable(this._handle, i, Boolean(trainable))
      if (rc !== 0) throw new Error(`polygrad: setBufTrainable failed (rc=${rc})`)
      return this
    }

    bufShape(i) {
      return this._rt._core.model.bufShape(this._handle, i)
    }

    bufDtype(i) {
      const core = this._rt._core
      return modelDtypeName(core, core.model.bufDtypeId(this._handle, i))
    }

    bufData(i) {
      this._requireSync('bufData()', 'bufDataAsync()')
      return this._bufDataRaw(i)
    }

    bufDataAsync(i) {
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(() => this._bufDataRaw(i))
      return Promise.resolve(this._bufDataRaw(i))
    }

    findBuf(name) {
      for (let i = 0; i < this.bufCount; i++) {
        if (this.bufName(i) === name) return i
      }
      return -1
    }

    _bufferIndex(name) {
      const index = this.findBuf(name)
      if (index < 0) throw new Error(`polygrad: no buffer '${name}'`)
      return index
    }

    bindings() {
      this._requireOpen()
      if (this._rt._activeAsync) throw new Error('Model metadata unavailable during active async work')
      return Array.from({ length: this.bufCount }, (_, i) => ({
        name: this.bufName(i), role: this.bufRole(i), dtype: this.bufDtype(i),
        shape: this.bufShape(i), trainable: this.bufTrainable(i)
      }))
    }

    entrypoints() {
      this._requireOpen()
      if (this._rt._activeAsync) throw new Error('Model metadata unavailable during active async work')
      return this._rt._core.model.entrypoints(this._handle)
    }

    setTrainable(name, trainable) {
      return this.setBufTrainable(this._bufferIndex(name), trainable)
    }

    readBuffer(name) {
      this._requireSync('readBuffer()', 'readBufferAsync()')
      return this._bufDataRaw(this._bufferIndex(name))
    }

    readBufferAsync(name) {
      this._requireOpen()
      const run = () => this._bufDataRaw(this._bufferIndex(name))
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(run)
      return Promise.resolve(run())
    }

    writeBuffer(name, array) {
      this._requireSync('writeBuffer()', 'writeBufferAsync()')
      const rc = this._rt._core.model.writeBuf(this._handle, this._bufferIndex(name), array)
      if (rc !== 0) throw new Error('polygrad: buffer write failed')
      return this
    }

    writeBufferAsync(name, array) {
      this._requireOpen()
      if (!ArrayBuffer.isView(array) || array instanceof DataView)
        return Promise.reject(new TypeError('polygrad: buffer write requires a TypedArray'))
      // Capture caller bytes before queueing; later mutation cannot change the write.
      const copy = new array.constructor(array)
      const run = async () => {
        const rc = await this._rt._core.model.writeBuf(this._handle, this._bufferIndex(name), copy)
        if (rc !== 0) throw new Error('polygrad: buffer write failed')
        return this
      }
      return this._usesAsyncHostBridge() ? this._enqueueAsync(run) : run()
    }

    exportWeights(options = null) {
      this._requireSync('exportWeights()', 'exportWeightsAsync()')
      const flags = weightExportFlags(options)
      return this._rt._core.model.exportWeights(this._handle, flags)
    }

    exportWeightsAsync(options = null) {
      const flags = weightExportFlags(options)
      const run = () => this._rt._core.model.exportWeights(this._handle, flags)
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(run)
      return Promise.resolve(run())
    }

    importWeights(bytes) {
      const rc = this._rt._core.model.importWeights(
        this._handle,
        normalizeBytes(bytes, 'weights')
      )
      if (rc !== 0) throw new Error(`polygrad: importWeights failed (rc=${rc})`)
    }

    exportIR() {
      return this._rt._core.model.exportIR(this._handle)
    }

    exportProgram() {
      this._requireSync('exportProgram()', 'exportProgramAsync()')
      return this._rt._core.model.exportProgram(this._handle)
    }

    exportProgramAsync() {
      const run = () => this._rt._core.model.exportProgram(this._handle)
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(run)
      return Promise.resolve(run())
    }

    saveBundle(options = null) {
      this._requireSync('saveBundle()', 'saveBundleAsync()')
      const flags = weightExportFlags(options)
      return this._rt._core.model.saveBundle(this._handle, flags)
    }

    saveBundleAsync(options = null) {
      const flags = weightExportFlags(options)
      const run = () => this._rt._core.model.saveBundle(this._handle, flags)
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(run)
      return Promise.resolve(run())
    }

    static fromBundle(bytes) {
      const api = _runtime._core.model
      if (!api) throw new Error('polygrad: model runtime unavailable for this core')
      const handle = api.fromBundle(bytes)
      if (!handle) throw new Error('polygrad: fromBundle failed')
      return new Model(handle)
    }


    setOptimizer(
      kind,
      lr = 0.01,
      beta1 = 0.9,
      beta2 = 0.999,
      eps = 1e-8,
      weightDecay = 0.0,
      momentum = 0.0,
      nesterov = false,
      classic = false
    ) {
      this._requireOpen()
      const rc = this._rt._core.model.setOptimizer(
        this._handle, kind, lr, beta1, beta2, eps, weightDecay, momentum, nesterov, classic
      )
      if (rc !== 0) throw new Error(`polygrad: setOptimizer failed (rc=${rc})`)
      return this
    }

    forward(io) {
      return this.call('forward', io)
    }

    call(entrypoint, io) {
      this._requireSync('call()', 'callAsync()')
      const { names, arrays } = normalizeBindings(io)
      const rc = this._rt._core.model.call(this._handle, String(entrypoint), names, arrays)
      if (isPromiseLike(rc)) throw new PolyAsyncRequired('call()', 'callAsync()')
      if (rc !== 0) throw new Error(`polygrad: call('${entrypoint}') failed (rc=${rc})`)
      return this._collectOutputsRaw(String(entrypoint))
    }

    forwardAsync(io) {
      return this.callAsync('forward', io)
    }

    callAsync(entrypoint, io) {
      entrypoint = String(entrypoint)
      const { names, arrays } = normalizeBindings(io)
      const run = () => {
        const rc = this._rt._core.model.call(this._handle, entrypoint, names, arrays)
        if (isPromiseLike(rc)) {
          return rc.then(v => {
            if (v !== 0) throw new Error(`polygrad: call('${entrypoint}') failed (rc=${v})`)
            return this._collectOutputsRawAsync(entrypoint)
          })
        }
        if (rc !== 0) throw new Error(`polygrad: call('${entrypoint}') failed (rc=${rc})`)
        return this._collectOutputsRaw(entrypoint)
      }
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(run)
      return Promise.resolve(run())
    }

    trainStep(io, entrypoint = null) {
      this._requireSync('trainStep()', 'trainStepAsync()')
      const { names, arrays } = normalizeBindings(io)
      const loss = this._rt._core.model.trainStep(this._handle, names, arrays, entrypoint)
      if (isPromiseLike(loss)) throw new PolyAsyncRequired('trainStep()', 'trainStepAsync()')
      if (loss == null || Number.isNaN(loss)) throw new Error('polygrad: trainStep failed')
      return loss
    }

    trainStepAsync(io, entrypoint = null) {
      const { names, arrays } = normalizeBindings(io)
      const run = () => {
        const loss = this._rt._core.model.trainStep(this._handle, names, arrays, entrypoint)
        if (isPromiseLike(loss)) {
          return loss.then(v => {
            if (v == null || Number.isNaN(v)) throw new Error('polygrad: trainStep failed')
            return v
          })
        }
        if (loss == null || Number.isNaN(loss)) throw new Error('polygrad: trainStep failed')
        return loss
      }
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(run)
      return Promise.resolve(run())
    }

    _collectOutputsRaw(entrypoint = 'forward') {
      const outputs = {}
      const api = this._rt._core.model
      const nOutputs = api.entrypointOutputCount(this._handle, entrypoint)
      if (nOutputs < 0) throw new Error(`polygrad: unknown Model entrypoint '${entrypoint}'`)
      for (let i = 0; i < nOutputs; i++) {
        const name = api.entrypointOutputName(this._handle, entrypoint, i)
        const bi = this.findBuf(name)
        if (name == null || bi < 0) {
          throw new Error(`polygrad: entrypoint '${entrypoint}' references a missing output`)
        }
        outputs[name] = this._bufDataRaw(bi)
      }
      return outputs
    }

    async _collectOutputsRawAsync(entrypoint = 'forward') {
      const outputs = {}
      const api = this._rt._core.model
      const nOutputs = api.entrypointOutputCount(this._handle, entrypoint)
      if (nOutputs < 0) throw new Error(`polygrad: unknown Model entrypoint '${entrypoint}'`)
      for (let i = 0; i < nOutputs; i++) {
        const name = api.entrypointOutputName(this._handle, entrypoint, i)
        const bi = this.findBuf(name)
        if (name == null || bi < 0) {
          throw new Error(`polygrad: entrypoint '${entrypoint}' references a missing output`)
        }
        outputs[name] = await this._bufDataRaw(bi)
      }
      return outputs
    }

    _collectOutputs() {
      this._requireSync('_collectOutputs()', '_collectOutputsAsync()')
      return this._collectOutputsRaw()
    }

    _collectOutputsAsync() {
      if (this._usesAsyncHostBridge()) return this._enqueueAsync(() => this._collectOutputsRawAsync())
      return Promise.resolve(this._collectOutputsRaw())
    }

    fit(io, opts = {}) {
      // Repeat one supplied batch; this is not a dataset/batching framework.
      this._requireSync('fit()', 'trainStepAsync()')
      const epochs = opts.epochs == null ? 1 : Number(opts.epochs)
      if (opts.optimizer != null) {
        this.setOptimizer(
          optimizerKind(opts.optimizer),
          opts.lr == null ? 0.01 : opts.lr,
          opts.beta1 == null ? 0.9 : opts.beta1,
          opts.beta2 == null ? 0.999 : opts.beta2,
          opts.eps == null ? 1e-8 : opts.eps,
          opts.weightDecay == null ? 0.0 : opts.weightDecay,
          opts.momentum == null ? 0.0 : opts.momentum,
          !!opts.nesterov,
          !!opts.classic
        )
      }
      const losses = []
      for (let step = 0; step < epochs; step++) {
        const loss = this.trainStep(io, opts.entrypoint)
        losses.push(loss)
        if (opts.onStep) opts.onStep(step, loss)
      }
      return losses
    }

  }

  Model._disposeAll = () => {
    const pending = []
    for (const owner of Array.from(liveModelOwners)) {
      const result = releaseModelOwner(owner)
      if (isPromiseLike(result)) pending.push(result)
    }
    return pending.length ? Promise.all(pending) : undefined
  }

  Model.ROLE_PARAM = ROLE_PARAM
  Model.ROLE_INPUT = ROLE_INPUT
  Model.ROLE_TARGET = ROLE_TARGET
  Model.ROLE_OUTPUT = ROLE_OUTPUT
  Model.ROLE_AUX = ROLE_AUX
  Model.OPTIM_NONE = OPTIM_NONE
  Model.OPTIM_SGD = OPTIM_SGD
  Model.OPTIM_ADAM = OPTIM_ADAM
  Model.OPTIM_ADAMW = OPTIM_ADAMW
  Model.EXPORT_WEIGHTS_PARAMS = EXPORT_WEIGHTS_PARAMS
  Model.EXPORT_WEIGHTS_OPTIMIZER = EXPORT_WEIGHTS_OPTIMIZER
  Model.EXPORT_WEIGHTS_DEFAULT = EXPORT_WEIGHTS_DEFAULT

  return Model
}

module.exports = { createBoundModelClass }
