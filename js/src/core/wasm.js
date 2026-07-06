'use strict'

const { createWasmCoreSync, createWasmCore } = require('./wasm_sync')
const { createWasmCoreAsync } = require('./wasm_async')

module.exports = { createWasmCoreSync, createWasmCoreAsync, createWasmCore }
