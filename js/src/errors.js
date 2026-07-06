'use strict'

class PolyAsyncRequired extends Error {
  constructor(method, asyncMethod) {
    super(`${method} requires async backend work; use ${asyncMethod}`)
    this.name = 'PolyAsyncRequired'
    this.method = method
    this.asyncMethod = asyncMethod
  }
}

class PolyWasmSyncUnsupported extends Error {
  constructor(message) {
    super(message || 'synchronous WASM startup is not supported; use createAsync()')
    this.name = 'PolyWasmSyncUnsupported'
  }
}

module.exports = { PolyAsyncRequired, PolyWasmSyncUnsupported }
