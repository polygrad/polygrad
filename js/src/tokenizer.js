'use strict'

/**
 * Tokenizer -- BPE tokenizer for LLM inference.
 *
 * Usage:
 *   const tok = polygrad.Tokenizer.fromGGUF(ggufBytes)
 *   const tok = polygrad.Tokenizer.fromJSON(tokenizerJsonBytes)
 *   const ids = tok.encode('Hello world')
 *   const text = tok.decode(ids)
 *   tok.free()
 *
 * Matches tinygrad SimpleTokenizer / HF AutoTokenizer API.
 */

function createBoundTokenizerClass(runtime) {
  const _runtime = runtime

  class Tokenizer {
    constructor(handle) {
      if (!handle) throw new Error('polygrad: null tokenizer handle')
      this._handle = handle
      this._rt = _runtime
    }

    static fromGGUF(ggufBytes) {
      const api = _runtime._core.instance
      if (!(ggufBytes instanceof Uint8Array))
        ggufBytes = new Uint8Array(ggufBytes)
      const handle = api.tokenizerFromGGUF(ggufBytes)
      if (!handle) throw new Error('polygrad: tokenizer fromGGUF failed')
      return new Tokenizer(handle)
    }

    static fromJSON(jsonBytes) {
      const api = _runtime._core.instance
      if (typeof jsonBytes === 'string')
        jsonBytes = new TextEncoder().encode(jsonBytes)
      if (!(jsonBytes instanceof Uint8Array))
        jsonBytes = new Uint8Array(jsonBytes)
      const handle = api.tokenizerFromJSON(jsonBytes)
      if (!handle) throw new Error('polygrad: tokenizer fromJSON failed')
      return new Tokenizer(handle)
    }

    encode(text) {
      const api = this._rt._core.instance
      return api.tokenize(this._handle, text)
    }

    decode(ids) {
      const api = this._rt._core.instance
      if (ids instanceof Int32Array) return api.detokenize(this._handle, ids)
      return api.detokenize(this._handle, Int32Array.from(ids))
    }

    get vocabSize() {
      return this._rt._core.instance.tokenizerVocabSize(this._handle)
    }

    get bosId() {
      return this._rt._core.instance.tokenizerBosId(this._handle)
    }

    get eosId() {
      return this._rt._core.instance.tokenizerEosId(this._handle)
    }

    free() {
      if (this._handle) {
        this._rt._core.instance.tokenizerFree(this._handle)
        this._handle = null
      }
    }
  }

  return Tokenizer
}

module.exports = { createBoundTokenizerClass }
