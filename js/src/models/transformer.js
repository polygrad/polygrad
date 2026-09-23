'use strict'

function createBoundTransformerClass(runtime) {
  return class Transformer extends runtime.Model {
    constructor(spec, {modelType = 'Llama'} = {}) {
      // Factory owns creation; specialization below preserves its finalizer.
      const { buildModel } = require('./index')
      const model = buildModel(runtime, modelType, spec, false, false)
      try { return Transformer.fromModel(model) }
      catch (error) { model.dispose(); throw error }
    }

    // Trusted producers only: admission does not prove causal, append-only caches.
    static fromModel(model) {
      if (!(model instanceof runtime.Model)) throw new TypeError('Transformer.fromModel requires this Runtime\'s Model')
      model._requireOpen()
      if (runtime._activeAsync > 0) throw new Error('Transformer adoption requires an idle Runtime')
      if (model instanceof Transformer) return model
      return Transformer._adoptBuilt(model)
    }

    static _adoptBuilt(model) {
      const api = runtime._core.transformer
      const handle = api.fromModel(model._handle)
      if (!handle) throw new Error('Model lacks a compatible dense causal Transformer contract')
      // In-place specialization keeps aliases and the existing GC/runtime owner.
      model._transformer = handle
      model._owner.resource = handle
      model._owner.release = api.free
      Object.setPrototypeOf(model, Transformer.prototype)
      return model
    }

    static load(source) {
      const model = runtime.Model.load(source)
      try { return Transformer.fromModel(model) }
      catch (error) { model.dispose(); throw error }
    }

    _transformerError(fallback) {
      return new Error(runtime._core.transformer.lastError(this._transformer).trim() || fallback)
    }

    get decodePosition() {
      this._requireOpen()
      if (this._rt._activeAsync > 0) throw new Error('await pending Runtime operations before reading decodePosition')
      return this._rt._core.transformer.decodePosition(this._transformer)
    }

    rewind(position) {
      this._requireOpen()
      if (this._rt._activeAsync > 0) throw new Error('await pending Runtime operations before rewinding')
      if (!Number.isInteger(position) || position < 0 || position > 2147483647)
        throw new RangeError('rewind position must fit a nonnegative int32')
      if (this._rt._core.transformer.rewind(this._transformer, position) !== 0) throw this._transformerError('decoder rewind failed')
      return this
    }

    reset() { return this.resetTransient() }
    resetAsync() { return this.resetTransientAsync() }

    appendTokens(tokens) { return this._decodeTokens(tokens, false) }
    prefillTokens(tokens) { return this._decodeTokens(tokens, true) }
    appendTokensAsync(tokens) { return this._decodeTokensAsync(tokens, false) }
    prefillTokensAsync(tokens) { return this._decodeTokensAsync(tokens, true) }

    async *generate(tokens, {temperature = 0, maxTokens = Infinity} = {}) {
      this._requireOpen()
      if (!(tokens instanceof Int32Array)) throw new TypeError('generate expects Int32Array tokens')
      if (!Number.isFinite(temperature) || temperature < 0)
        throw new RangeError('temperature must be finite and nonnegative')
      if (maxTokens !== Infinity && (!Number.isSafeInteger(maxTokens) || maxTokens < 0))
        throw new RangeError('maxTokens must be a nonnegative integer')
      if (maxTokens === 0) return
      tokens = tokens.slice()
      const api = this._rt._core.transformer
      const run = fn => this._usesAsyncHostBridge() ? this._enqueueAsync(fn) : fn()
      if (await run(() => api.start(this._transformer, tokens)) !== 0)
        throw this._transformerError('Transformer prefill failed')
      for (let i = 0; i < maxTokens; i++) {
        this._requireOpen() // Disposal between yields must not enter C.
        const {status, token} = await run(() => api.next(this._transformer, temperature))
        if (status === 1) return
        if (status !== 0) throw this._transformerError('Transformer sampling failed')
        yield token
      }
    }

    _decodeTokens(tokens, reuse) {
      const name = reuse ? 'prefillTokens' : 'appendTokens'
      this._requireSync(`${name}()`, `${name}Async()`)
      if (!(tokens instanceof Int32Array)) throw new TypeError('decoder expects Int32Array tokens')
      const out = this._rt._core.transformer.appendTokens(this._transformer, tokens, reuse)
      if (out == null) throw this._transformerError('decoder append failed; reset the Transformer before retrying')
      return out
    }

    _decodeTokensAsync(tokens, reuse) {
      this._requireOpen()
      if (!(tokens instanceof Int32Array)) throw new TypeError('decoder expects Int32Array tokens')
      // Queued calls own their token snapshot, not a mutable caller view.
      tokens = tokens.slice()
      const run = async () => {
        const out = await this._rt._core.transformer.appendTokens(this._transformer, tokens, reuse)
        if (out == null) throw this._transformerError('decoder append failed; reset the Transformer before retrying')
        return out
      }
      return this._usesAsyncHostBridge() ? this._enqueueAsync(run) : run()
    }

  }
}

module.exports = { createBoundTransformerClass }
