'use strict'

function flattenTensors(value, Tensor, out) {
  if (value instanceof Tensor) {
    out.push(value)
    return
  }
  if (Array.isArray(value)) {
    for (const item of value) flattenTensors(item, Tensor, out)
    return
  }
  if (value && typeof value === 'object') {
    for (const key of Object.keys(value).sort()) flattenTensors(value[key], Tensor, out)
  }
}

function retTensors(value, Tensor, out) {
  if (value == null) return
  if (value instanceof Tensor) {
    out.push(value)
    return
  }
  if (Array.isArray(value)) {
    for (const item of value) retTensors(item, Tensor, out)
    return
  }
  if (value && typeof value === 'object') {
    for (const key of Object.keys(value).sort()) retTensors(value[key], Tensor, out)
    return
  }
  throw new Error(`jit return contains non-Tensor value of type ${typeof value}`)
}

function inputTensors(args, Tensor) {
  const inputs = []
  for (const arg of args) flattenTensors(arg, Tensor, inputs)
  return inputs
}

function inputSignature(inputs) {
  return JSON.stringify(inputs.map(t => ({
    shape: t.shape,
    dtype: t.dtype,
    device: t.device
  })))
}

async function resolveUserReturn(value) {
  if (value && typeof value.then === 'function') return await value
  return value
}

function checkDuplicateBuffers(inputs) {
  const seen = new Set()
  for (const t of inputs) {
    const u = t.uop
    const b = u && u.buffer
    const key = b ? b.key : '0'
    if (key === '0') throw new Error('jit inputs must be real buffers')
    if (seen.has(key)) throw new Error('duplicate inputs to jit')
    seen.add(key)
  }
}

function tensorHandles(inputs) {
  return inputs.map(t => t._tensor)
}

function nowMs() {
  if (typeof performance !== 'undefined' && performance && typeof performance.now === 'function') {
    return performance.now()
  }
  return Date.now()
}

async function realizeReturn(value, Tensor) {
  const outs = []
  retTensors(value, Tensor, outs)
  for (const t of outs) await t.realize()
  return outs
}

function createBoundJit(runtime) {
  const Tensor = runtime.Tensor
  const { ffi, ctx } = runtime._core
  const live = new Set()

  class Jit {
    constructor(fxn, opts) {
      if (typeof fxn !== 'function') throw new TypeError('jit requires a function')
      opts = opts || {}
      this.fxn = fxn
      this.prune = Boolean(opts.prune)
      this.cnt = 0
      this.captured = false
      this.ret = null
      this.signature = null
      this.inputCount = 0
      this._jit = 0
      this.disposed = false
      this.callCount = 0
      this.replayCount = 0
      this.lastCallMs = 0
      live.add(this)
    }

    _clear() {
      if (this._jit && ffi.poly_jit_free) ffi.poly_jit_free(this._jit)
      this._jit = 0
      this.cnt = 0
      this.captured = false
      this.ret = null
      this.signature = null
      this.inputCount = 0
      this.callCount = 0
      this.replayCount = 0
      this.lastCallMs = 0
    }

    reset() {
      if (this.disposed) throw new Error('jit has been disposed')
      this._clear()
    }

    dispose() {
      this._clear()
      this.disposed = true
      live.delete(this)
    }

    get scheduleCount() {
      return this._jit && ffi.poly_jit_schedule_count ? ffi.poly_jit_schedule_count(this._jit) : 0
    }

    get schedule_count() {
      return this.scheduleCount
    }

    stats() {
      return {
        captured: this.captured,
        disposed: this.disposed,
        prune: this.prune,
        callCount: this.callCount,
        replayCount: this.replayCount,
        lastCallMs: this.lastCallMs,
        scheduleCount: this.scheduleCount,
        schedule_count: this.schedule_count,
        inputCount: this.inputCount
      }
    }

    async call(...args) {
      if (this.disposed) throw new Error('jit has been disposed')
      if (!ffi.poly_jit_new) throw new Error('polygrad core does not expose poly_jit')
      const callStart = nowMs()
      let replayed = false

      const inputs = inputTensors(args, Tensor)
      if (inputs.length === 0) throw new Error('jit requires at least one Tensor input')
      if (this.cnt > 0 && inputs.some(t => t._ctx !== ctx)) throw new Error('jit inputs must share runtime context')
      for (const t of inputs) await t.realize()
      checkDuplicateBuffers(inputs)

      let ret
      if (this.cnt === 0) {
        ret = await resolveUserReturn(this.fxn(...args))
        await realizeReturn(ret, Tensor)
      } else if (this.cnt === 1) {
        const sig = inputSignature(inputs)
        this._jit = ffi.poly_jit_new(ctx)
        if (!this._jit) throw new Error('poly_jit_new failed')
        if (ffi.poly_jit_set_prune(this._jit, this.prune) !== 0) {
          this._clear()
          throw new Error('poly_jit_set_prune failed')
        }
        if (ffi.poly_jit_begin_capture(this._jit, tensorHandles(inputs)) !== 0) {
          this._clear()
          throw new Error('poly_jit_begin_capture failed')
        }
        try {
          ret = await resolveUserReturn(this.fxn(...args))
          await realizeReturn(ret, Tensor)
          if (ffi.poly_jit_end_capture(this._jit) !== 0) throw new Error("didn't jit anything")
        } catch (err) {
          if (this._jit && ffi.poly_jit_cancel_capture) ffi.poly_jit_cancel_capture(this._jit)
          throw err
        }
        this.ret = ret
        this.signature = sig
        this.inputCount = inputs.length
        this.captured = true
      } else {
        if (!this.captured || !this._jit) throw new Error('jit has not captured')
        if (inputs.length !== this.inputCount) {
          throw new Error(`args mismatch in jit: expected ${this.inputCount} inputs, got ${inputs.length}`)
        }
        const sig = inputSignature(inputs)
        if (sig !== this.signature) {
          throw new Error(`args mismatch in jit: expected ${this.signature}, got ${sig}`)
        }
        if (await ffi.poly_jit_run(this._jit, tensorHandles(inputs)) !== 0) {
          throw new Error('poly_jit_run failed')
        }
        ret = this.ret
        replayed = true
      }

      this.cnt++
      this.callCount++
      if (replayed) this.replayCount++
      this.lastCallMs = nowMs() - callStart
      return ret
    }
  }

  function jit(fxn, opts) {
    if (fxn == null) return (f) => jit(f, opts)
    if (typeof fxn === 'object' && typeof fxn !== 'function') return (f) => jit(f, fxn)
    const state = new Jit(fxn, opts)
    const wrapped = (...args) => state.call(...args)
    wrapped.dispose = () => state.dispose()
    wrapped.reset = () => state.reset()
    Object.defineProperty(wrapped, 'scheduleCount', {
      get() { return state.scheduleCount }
    })
    Object.defineProperty(wrapped, 'schedule_count', {
      get() { return state.schedule_count }
    })
    wrapped.stats = () => state.stats()
    return wrapped
  }

  async function compile(fxn, sampleInputs, opts) {
    if (typeof fxn !== 'function') throw new TypeError('compile requires a function')
    if (!Array.isArray(sampleInputs)) throw new TypeError('compile requires sample input array')

    const wrapped = jit(fxn, opts)
    const t0 = nowMs()
    await wrapped(...sampleInputs)
    await wrapped(...sampleInputs)
    const compileMs = nowMs() - t0
    if (wrapped.scheduleCount <= 0) {
      wrapped.dispose()
      throw new Error("didn't jit anything")
    }

    let disposed = false
    let runCount = 0
    let lastRunMs = 0

    const compiled = {
      async run(inputs) {
        if (disposed) throw new Error('compiled callable has been disposed')
        if (!Array.isArray(inputs)) throw new TypeError('compiled.run requires input array')
        const rt0 = nowMs()
        const ret = await wrapped(...inputs)
        lastRunMs = nowMs() - rt0
        runCount++
        return ret
      },
      async call(...inputs) {
        return this.run(inputs)
      },
      dispose() {
        if (disposed) return
        disposed = true
        wrapped.dispose()
      },
      stats() {
        return {
          captured: !disposed,
          captureRuns: 2,
          compileMs,
          lastRunMs,
          runCount,
          callCount: runCount,
          scheduleCount: wrapped.scheduleCount,
          schedule_count: wrapped.schedule_count,
          inputCount: sampleInputs.length
        }
      },
      get scheduleCount() { return wrapped.scheduleCount },
      get schedule_count() { return wrapped.schedule_count },
      get runCount() { return runCount },
      get lastRunMs() { return lastRunMs },
      get compileMs() { return compileMs }
    }
    return compiled
  }

  function disposeAll() {
    for (const state of Array.from(live)) state.dispose()
  }

  function stats() {
    return { liveCount: live.size }
  }

  jit.disposeAll = disposeAll
  jit.compile = compile
  jit.stats = stats
  return jit
}

module.exports = { createBoundJit }
