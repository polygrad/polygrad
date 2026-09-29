'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')

function source(file) {
  return fs.readFileSync(path.join(__dirname, '../../src', file), 'utf8')
}

// Execute the actual embedded bridges with sparse heaps: exercise the upper
// half of wasm32 without allocating a multi-gigabyte buffer in the test runner.
function bridge(file, name, args, env) {
  const text = source(file).split(name + ',')[1]?.split(/\nEM_(?:ASYNC_)?JS\(/)[0]
  assert(text, name)
  const end = text.search(/\n(?:    }\n\)|}\);)/)
  assert(end > 0, name + ' closing delimiter')
  const body = text.slice(text.indexOf('{') + 1, end)
  return vm.runInNewContext(`(function(${args}) {${body}\n})`, env)
}

async function checkWebgpu(address, timed) {
  const args = address, valuePtr = address + 64, elapsed = address + 128
  const HEAPU32 = {[args >>> 2]: 1, [(args >>> 2) + 1]: valuePtr >>> 0}
  const HEAP32 = {[valuePtr >>> 2]: -17}, HEAPF64 = {}
  let submits = 0, uniform
  const buffer = () => ({destroy() {}, unmap() {}, async mapAsync() {},
    getMappedRange() { return new BigUint64Array([1000n, 7000n]).buffer }})
  const output = buffer()
  const device = {
    features: new Set(['timestamp-query']),
    createBuffer: buffer, createQuerySet: buffer,
    createBindGroup({entries}) {
      assert.equal(entries[1].resource.buffer, output)
      return {}
    },
    createCommandEncoder() {
      return {beginComputePass() { return {setPipeline() {}, setBindGroup() {},
        dispatchWorkgroups() {}, end() {}} }, resolveQuerySet() {}, copyBufferToBuffer() {}, finish() { return {} }}
    },
    queue: {submit() { submits++ }, writeBuffer(buf, offset, values) { uniform = values[0] }},
  }
  const st = {device, pipelines: new Map([[1, {}]]), buffers: new Map([[1, output]]),
    bufferSizes: new Map([[1, 4]])}
  const dispatch = bridge('runtime_webgpu.c', 'js_webgpu_dispatch',
    'pipeline_id,args,n_args,n_params,gx,gy,gz,debug_level,elapsed_us', {
      Module: {__polygradWebGpuState: st}, HEAPU32, HEAP32, HEAPF64,
      Asyncify: {state: 0, State: {Rewinding: 2}, handleAsync: fn => fn()},
      GPUBufferUsage: {}, GPUMapMode: {READ: 1}, console,
    })
  assert.equal(await dispatch(1, args, 2, 1, 1, 1, 1, 0, timed ? elapsed : 0), 0)
  assert.equal(uniform, -17, 'integer values stay signed; only addresses are unsigned')
  assert.equal(submits, 1)
  if (timed) assert.equal(HEAPF64[elapsed >>> 3], 6)
}

function checkWasm(address, count) {
  let received
  const kernel = (...args) => { received = args }
  Object.defineProperty(kernel, 'length', {value: count})
  const HEAP32 = {}, expected = Array.from({length: count}, (_, i) => -i - 1)
  expected.forEach((value, i) => { HEAP32[(address >>> 2) + i] = value })
  const Module = {}
  const compile = bridge('runtime_wasm.c', 'js_compile_wasm_kernel', 'bytes,len', {
    Module, HEAP32, HEAPU8: new Uint8Array(), wasmMemory: {},
    WebAssembly: {Module: function() {}, Instance: function() { this.exports = {kernel} }},
  })
  assert.equal(compile(0, 0), 0)
  Module._polyKernelCache[0].launch(address)
  assert.deepEqual(received, expected)
}

;(async () => {
  for (const address of [1024, 0x80000400, 0x80000400 | 0]) {
    for (const timed of [false, true]) await checkWebgpu(address, timed)
    for (let count = 0; count <= 9; count++) checkWasm(address, count)
  }
  console.log('Wasm addresses: 36 passed')
})().catch(error => { console.error(error); process.exitCode = 1 })
