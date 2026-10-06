'use strict'

/**
 * Tokenizer -- BPE tokenizer for LLM inference.
 *
 * Usage:
 *   const tok = polygrad.Tokenizer.fromGGUF(ggufBytes)
 *   const ids = tok.encode('Hello world')
 *   const text = tok.decode(ids)
 *   tok.free()
 *
 * Vocabulary-ranked GGUF BPE following tinygrad SimpleTokenizer.
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
      const api = _runtime._core.model
      if (!(ggufBytes instanceof Uint8Array))
        ggufBytes = new Uint8Array(ggufBytes)
      const handle = api.tokenizerFromGGUF(ggufBytes)
      if (!handle) throw new Error('polygrad: tokenizer fromGGUF failed')
      return new Tokenizer(handle)
    }

    static fromJSON(jsonBytes, { strict = true } = {}) {
      if (typeof strict !== 'boolean') throw new TypeError('polygrad: tokenizer strict must be a boolean')
      const api = _runtime._core.model
      if (typeof jsonBytes === 'string')
        jsonBytes = new TextEncoder().encode(jsonBytes)
      if (!(jsonBytes instanceof Uint8Array))
        jsonBytes = new Uint8Array(jsonBytes)
      const { handle, warning } = api.tokenizerFromJSON(jsonBytes, strict)
      if (warning) console.warn('polygrad: ' + warning)
      return new Tokenizer(handle)
    }

    encode(text) {
      const api = this._rt._core.model
      return api.tokenize(this._handle, text)
    }

    decode(ids) {
      const api = this._rt._core.model
      if (ids instanceof Int32Array) return api.detokenize(this._handle, ids)
      return api.detokenize(this._handle, Int32Array.from(ids))
    }

    get vocabSize() {
      return this._rt._core.model.tokenizerVocabSize(this._handle)
    }

    get bosId() {
      return this._rt._core.model.tokenizerBosId(this._handle)
    }

    get eosId() {
      return this._rt._core.model.tokenizerEosId(this._handle)
    }

    free() {
      if (this._handle) {
        this._rt._core.model.tokenizerFree(this._handle)
        this._handle = null
      }
    }
  }

  return Tokenizer
}

module.exports = { createBoundTokenizerClass }
