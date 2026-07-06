'use strict'

const { createWasmCoreFromModule } = require('./wasm_common')

let _asyncModuleFactory = null

function getAsyncModuleFactory() {
  if (_asyncModuleFactory) return _asyncModuleFactory
  try {
    _asyncModuleFactory = require('../../wasm/core.async.js')
  } catch (e) {
    if (typeof process !== 'undefined' && process.versions && process.versions.node) {
      // eslint-disable-next-line no-eval
      const nodeRequire = typeof __non_webpack_require__ !== 'undefined'
        ? __non_webpack_require__ : eval('require')
      const path = nodeRequire('path')
      _asyncModuleFactory = nodeRequire(
        path.resolve(__dirname, '..', '..', '..', 'build', 'core.async.js')
      )
    } else {
      throw new Error(
        'polygrad: async WASM module not found. The package may be installed incorrectly.'
      )
    }
  }
  return _asyncModuleFactory
}

async function createWasmCoreAsync(device) {
  const Module = await getAsyncModuleFactory()()
  return createWasmCoreFromModule(Module, device)
}

module.exports = { createWasmCoreAsync }
