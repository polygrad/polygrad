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
  const end = text.search(/\n(?:    }\n\)|}\);?)/)
  assert(end > 0, name + ' closing delimiter')
  const body = text.slice(text.indexOf('{') + 1, end)
  return vm.runInNewContext(`(${name.startsWith('js_webgpu_read') ? 'async ' : ''}function(${args}) {${body}\n})`, env)
}

async function checkHostCopies(address) {
  const values = new Uint8Array([1, 2, 3, 4])
  const Module = {__polygradHostBuffers: new Map([[String(address >>> 0), values]])}
  const HEAPU8 = {
    set(bytes, offset) { assert.equal(offset, address >>> 0); assert.deepEqual(Array.from(bytes), [1, 2, 3, 4]) },
    subarray(start, end) { assert.equal(start, address >>> 0); assert.equal(end, (address >>> 0) + 4); return values },
  }
  const env = {Module, HEAPU8}
  assert.equal(bridge('runtime_wasm.c', 'js_host_copy_out_to_wasm', 'src_key,dst,nbytes', env)(address, address, 4), 0)
  assert.equal(bridge('runtime_wasm.c', 'js_host_copy_in_from_wasm', 'dst_key,src,nbytes', env)(address, address, 4), 0)
  const buffer = () => ({destroy() {}, unmap() {}, async mapAsync() {}, getMappedRange() { return values.buffer }})
  const st = {buffers: new Map([[1, buffer()]]), device: {
    createBuffer: buffer,
    createCommandEncoder() { return {copyBufferToBuffer() {}, finish() { return {} }} },
    queue: {submit() {}, writeBuffer(buf, offset, bytes) { assert.deepEqual(Array.from(bytes), [1, 2, 3, 4]) }},
  }}
  Module.__polygradWebGpuState = st
  Module.__polygradEnsureWebGPU = async () => st
  Object.assign(env, {GPUBufferUsage: {}, GPUMapMode: {READ: 1}})
  assert.equal(bridge('runtime_webgpu.c', 'js_webgpu_write_buffer_from_wasm', 'handle,src,nbytes', env)(1, address, 4), 0)
  assert.equal(bridge('runtime_webgpu.c', 'js_webgpu_write_buffer_from_hostkey', 'handle,src_key,nbytes', env)(1, address, 4), 0)
  assert.equal(await bridge('runtime_webgpu.c', 'js_webgpu_read_buffer', 'dst,handle,nbytes,host_key', env)(address, 1, 4, 1), 0)
  assert.equal(await bridge('runtime_webgpu.c', 'js_webgpu_read_buffer', 'dst,handle,nbytes,host_key', env)(address, 1, 4, 0), 0)
}

async function checkReadbackStaging() {
  const buffers = [], heap = new Uint8Array(1048584)
  let failure, growHeap = false
  const st = {buffers:new Map([[1,{}]]), bufferOffsets:new Map([[1,256]]), device:{
    createBuffer({size}) {
      if (failure==='allocate') throw Error('injected allocation failure')
      const b = {size, destroyed:false, destroy(){this.destroyed=true}, unmap(){},
        async mapAsync(mode,offset,length){
          assert.equal(offset,0); assert(length<=size && length%4===0)
          if (failure==='map') throw Error('injected mapping failure')
          if (growHeap) env.HEAPU8 = new Uint8Array(heap.length)
        }, getMappedRange(offset,length){
          if (failure==='range') throw Error('injected range failure')
          assert.equal(offset,0); assert(length<=size)
          return new Uint8Array(length).fill(7).buffer
        }}
      buffers.push(b)
      return b
    },
    createCommandEncoder(){return {copyBufferToBuffer(src,offset){assert.equal(offset,256)}, finish(){return {}}}},
    queue:{submit(){if(failure==='submit') throw Error('injected submission failure')}}
  }}
  const Module = {__polygradWebGpuState:st, __polygradEnsureWebGPU:async()=>st}
  const errors=[]
  const env = {Module,HEAPU8:heap,GPUBufferUsage:{},GPUMapMode:{READ:1},console:{error(...args){errors.push(args)}}}
  const read = bridge('runtime_webgpu.c','js_webgpu_read_buffer','dst,handle,nbytes,host_key',
    env)
  for (let i=0;i<5;i++) assert.equal(await read(0,1,4,0),0)
  assert.deepEqual(Array.from(heap.slice(0,4)),[7,7,7,7])
  assert.equal(buffers.length,1,'warm reads must reuse staging storage')
  assert.equal(buffers[0].destroyed,false)
  assert.equal(await read(0,1,8,0),0)
  assert.equal(buffers[0].destroyed,true,'growth releases the smaller allocation')
  const cache = st.readbackStaging
  heap.fill(0)
  assert.equal(await read(0,1,3,0),0)
  assert.deepEqual(Array.from(heap.slice(0,4)),[7,7,7,0],'alignment padding is not published')
  assert.equal(await read(0,1,1048580,0),0)
  assert.equal(st.readbackStaging,cache,'large one-off reads do not evict the small cache')
  assert.equal(buffers.at(-1).destroyed,true)
  for (let i=0;i<100;i++) assert.equal(await read(0,1,4,0),0)
  assert.equal(buffers.length,3,'repeated reads have bounded allocation count')
  for (failure of ['submit','map','range','allocate']) {
    if (failure==='allocate') {st.readbackStaging?.destroy();st.readbackStaging=null}
    heap.fill(0)
    assert.equal(await read(0,1,4,0),-1)
    assert.equal(st.readbackStaging,null,'failed staging cannot be reused')
    assert.equal(buffers.at(-1).destroyed,true)
    assert.equal(heap[0],0,'failure must not publish stale bytes')
  }
  assert.equal(errors.length,4)
  failure=null;growHeap=true
  assert.equal(await read(0,1,4,0),0)
  assert.equal(env.HEAPU8[0],7,'readback must use the grown heap')
  assert.equal(heap[0],0)
  st.readbackStaging.destroy()
}

async function checkWebgpu(address, timed, validationError = false, repeats = 1, sampleUs = 200, failRound = 1) {
  const args = address, valuePtr = address + 64, elapsed = address + 128
  const HEAPU32 = {[args >>> 2]: 1, [(args >>> 2) + 1]: valuePtr >>> 0}
  const HEAP32 = {[valuePtr >>> 2]: -17}, HEAPF64 = {}
  const adaptive = timed && repeats < 0
  let submits = 0, dispatches = 0, passDispatches = 0, uniform
  const allocated = [], groups = new Set()
  const buffer = () => {
    const b = {destroy() { this.destroyed = true }, unmap() {}, async mapAsync() {},
      getMappedRange() { return new BigUint64Array([1000n, 1000n + (adaptive ? BigInt(passDispatches * sampleUs * 1000) : 6000n)]).buffer }}
    allocated.push(b)
    return b
  }
  const output = buffer()
  const device = {
    limits: {maxComputeWorkgroupsPerDimension: 65535},
    pushErrorScope() {},
    async popErrorScope() { return validationError && submits === failRound ? new Error('injected dispatch validation') : null; },
    features: new Set(['timestamp-query']),
    createBuffer: buffer, createQuerySet: buffer,
    createBindGroup({entries}) {
      if (!adaptive) assert.equal(entries[1].resource.buffer, output)
      groups.add(entries[1].resource.buffer)
      return {}
    },
    createCommandEncoder() {
      return {beginComputePass() { passDispatches = 0; return {setPipeline() {}, setBindGroup() {},
        dispatchWorkgroups() { dispatches++; passDispatches++ }, end() {}} }, resolveQuerySet() {}, copyBufferToBuffer() {}, finish() { return {} }}
    },
    queue: {submit() { submits++ }, writeBuffer(buf, offset, values) { uniform = values[0] }},
  }
  const st = {device, pipelines: new Map([[1, {}]]), buffers: new Map([[1, output]]),
    bufferSizes: new Map([[1, 4]])}
  const dispatch = bridge('runtime_webgpu.c', 'js_webgpu_dispatch',
    'pipeline_id,args,n_args,n_params,gx,gy,gz,debug_level,elapsed_us,timing_repeats', {
      Module: {__polygradWebGpuState: st}, HEAPU32, HEAP32, HEAPF64,
      Asyncify: {state: 0, State: {Rewinding: 2}, handleAsync: fn => fn()},
      GPUBufferUsage: {}, GPUMapMode: {READ: 1}, console,
    })
  const failed = validationError || (adaptive && sampleUs === 0)
  assert.equal(await dispatch(1, args, 2, 1, 1, 1, 1, 0, timed ? elapsed : 0, repeats), failed ? -1 : 0)
  assert.equal(uniform, -17, 'integer values stay signed; only addresses are unsigned')
  const rounds = adaptive ? (failed ? (validationError ? failRound : 1) : 3) : 1
  assert.equal(submits, rounds)
  const calibrated = Math.max(1, Math.min(128, Math.ceil(4000 / sampleUs)))
  assert.equal(dispatches, adaptive ? 1 + (rounds - 1) * calibrated : timed ? repeats : 1, 'repeat only disposable timing calls')
  if (timed) assert.equal(HEAPF64[elapsed >>> 3], failed ? undefined : adaptive ? sampleUs : 6 / repeats)
  if (adaptive) assert.equal(groups.size, 16, 'rotate disposable storage rather than reusing one cache-hot buffer')
  assert(!output.destroyed, 'caller storage must survive timing')
  assert(allocated.slice(1).every(b => b.destroyed), 'timing scratch must be released on success and failure')
}

function checkWasm(address, count) {
  let received
  const kernel = (...args) => { received = args }
  Object.defineProperty(kernel, 'length', {value: count})
  const HEAP32 = {}, expected = Array.from({length: count}, (_, i) => -i - 1)
  expected.forEach((value, i) => { HEAP32[(address >>> 2) + i] = value })
  const Module = {}
  const compile = bridge('runtime_wasm.c', 'js_compile_wasm_kernel', 'bytes,len', {
    Module, HEAP32, HEAPU8: {subarray(start, end) {
      assert.equal(start, address >>> 0)
      assert.equal(end, (address >>> 0) + 4)
      return new Uint8Array(4)
    }}, wasmMemory: {},
    WebAssembly: {Module: function() {}, Instance: function() { this.exports = {kernel} }},
  })
  assert.equal(compile(address, 4), 0)
  Module._polyKernelCache[0].launch(address)
  assert.deepEqual(received, expected)
}

;(async () => {
  await checkReadbackStaging()
  for (const address of [1024, 0x80000400, 0x80000400 | 0]) {
    await checkHostCopies(address)
    for (const timed of [false, true]) await checkWebgpu(address, timed)
    await checkWebgpu(address, true, true)
    await checkWebgpu(address, true, false, 128)
    await checkWebgpu(address, false, false, 128)
    await checkWebgpu(address, true, true, 128)
    await checkWebgpu(address, true, false, -1)
    await checkWebgpu(address, true, true, -1)
    await checkWebgpu(address, true, true, -1, 200, 3)
    await checkWebgpu(address, true, false, -1, 0)
    await checkWebgpu(address, true, false, -1, 1)
    await checkWebgpu(address, true, false, -1, 10000)
    for (let count = 0; count <= 9; count++) checkWasm(address, count)
  }
  console.log('Wasm addresses, timing and readback staging: 85 passed')
})().catch(error => { console.error(error); process.exitCode = 1 })
