'use strict'

const { PolyWasmSyncUnsupported } = require('../errors')
const { createWasmCoreFromModule } = require('./wasm_common')

let _syncModule = null

function getSyncModule() {
  if (_syncModule) return _syncModule
  try {
    _syncModule = require('../../wasm/core.sync.js')
  } catch (e) {
    if (typeof process !== 'undefined' && process.versions && process.versions.node) {
      // eslint-disable-next-line no-eval
      const nodeRequire = typeof __non_webpack_require__ !== 'undefined'
        ? __non_webpack_require__ : eval('require')
      const path = nodeRequire('path')
      _syncModule = nodeRequire(
        path.resolve(__dirname, '..', '..', '..', 'build', 'core.sync.js')
      )
    } else {
      throw new PolyWasmSyncUnsupported(
        'polygrad: sync WASM module not found. Use createAsync() or rebuild package artifacts.'
      )
    }
  }
  if (_syncModule && typeof _syncModule.then === 'function') {
    throw new PolyWasmSyncUnsupported('polygrad: sync WASM artifact returned a Promise; use createAsync()')
  }
  return _syncModule
}

function createWasmCoreSync(device) {
  return createWasmCoreFromModule(getSyncModule(), device)
}

module.exports = { createWasmCoreSync, createWasmCore: createWasmCoreSync }
