'use strict'

const { init, Tensor } = require('../src/index')
const path = require('path')

async function main() {
  await init()

  // Get Module reference for direct access
  const Module = require(path.resolve(__dirname, '..', '..', '..', 'build', 'polygrad.js'))
  const M = await Module()

  const ctx = M._poly_ctx_new()
  const N = 10000

  // 1. Benchmark raw FFI call: poly_buffer_f32
  let t0 = performance.now()
  for (let i = 0; i < N; i++) {
    M._poly_buffer_f32(ctx, BigInt(3))
  }
  let dt = performance.now() - t0
  console.log(`poly_buffer_f32 (raw):   ${(dt/N*1000).toFixed(1)}μs/call (${N} calls in ${dt.toFixed(1)}ms)`)

  // 2. Benchmark cwrap'ed FFI call
  const cwrap_buf = M.cwrap('poly_buffer_f32', 'number', ['number', 'number'])
  t0 = performance.now()
  for (let i = 0; i < N; i++) {
    M._poly_buffer_f32(ctx, BigInt(3))
  }
  dt = performance.now() - t0
  console.log(`poly_buffer_f32 (cwrap): ${(dt/N*1000).toFixed(1)}μs/call`)

  // 3. Benchmark Module.getValue vs HEAP32
  const ptr = M._malloc(4)
  M.HEAP32[ptr >> 2] = 42

  t0 = performance.now()
  for (let i = 0; i < N; i++) {
    M.getValue(ptr, 'i32')
  }
  dt = performance.now() - t0
  console.log(`Module.getValue:        ${(dt/N*1000).toFixed(1)}μs/call`)

  t0 = performance.now()
  for (let i = 0; i < N; i++) {
    M.HEAP32[ptr >> 2]
  }
  dt = performance.now() - t0
  console.log(`HEAP32[ptr>>2]:         ${(dt/N*1000).toFixed(1)}μs/call`)

  // 4. Benchmark hashBytes
  const bytes = new Uint8Array(500)
  for (let i = 0; i < 500; i++) bytes[i] = i & 0xFF
  t0 = performance.now()
  for (let i = 0; i < N; i++) {
    let h = 0x811c9dc5
    for (let j = 0; j < bytes.length; j++) {
      h ^= bytes[j]
      h = Math.imul(h, 0x01000193)
    }
    h >>> 0
  }
  dt = performance.now() - t0
  console.log(`hashBytes(500):         ${(dt/N*1000).toFixed(1)}μs/call`)

  // 5. Benchmark WebAssembly.Instance.exports.kernel call
  // Create a simple kernel
  const buf = M._poly_buffer_f32(ctx, BigInt(3))
  const buf2 = M._poly_buffer_f32(ctx, BigInt(3))
  const OPS_ADD = 1 // approximate
  const opCount = M._poly_op_count()
  let ADD_OP = -1
  const cwrapName = M.cwrap('poly_op_name', 'string', ['number'])
  for (let i = 0; i < opCount; i++) {
    if (cwrapName(i) === 'ADD') { ADD_OP = i; break }
  }
  const addUop = M._poly_alu2(ctx, ADD_OP, buf, buf2)
  const outBuf = M._poly_buffer_f32(ctx, BigInt(3))
  const store = M._poly_store_val(ctx, outBuf, addUop)
  const sink = M._poly_sink1(ctx, store)

  const lenPtr = M._malloc(4)
  const nBufsPtr = M._malloc(4)
  const wasmPtr = M._poly_render_kernel_wasm(ctx, sink, lenPtr, nBufsPtr)
  const wasmLen = M.HEAP32[lenPtr >> 2]
  const nBufs = M.HEAP32[nBufsPtr >> 2]
  const wasmBytes = new Uint8Array(wasmLen)
  wasmBytes.set(M.HEAPU8.subarray(wasmPtr, wasmPtr + wasmLen))
  M._free(wasmPtr)

  const memory = new WebAssembly.Memory({ initial: 1 })
  const module = new WebAssembly.Module(wasmBytes)
  const instance = new WebAssembly.Instance(module, {
    env: { memory },
    math: { exp2f: x => Math.pow(2,x), log2f: x => Math.log2(x), sinf: x => Math.sin(x) }
  })

  const memView = new Float32Array(memory.buffer)
  memView.set([1,2,3], 0)
  memView.set([4,5,6], 3)

  t0 = performance.now()
  for (let i = 0; i < N; i++) {
    instance.exports.kernel(0, 12, 24)
  }
  dt = performance.now() - t0
  console.log(`kernel() call:          ${(dt/N*1000).toFixed(1)}μs/call`)

  // 6. Benchmark Float32Array.set (3 elements)
  const src = new Float32Array([1,2,3])
  const dst = new Float32Array(memory.buffer)
  t0 = performance.now()
  for (let i = 0; i < N; i++) {
    dst.set(src, 0)
  }
  dt = performance.now() - t0
  console.log(`Float32Array.set(3):    ${(dt/N*1000).toFixed(1)}μs/call`)

  // 7. Benchmark new Float32Array(3)
  t0 = performance.now()
  for (let i = 0; i < N; i++) {
    new Float32Array(3)
  }
  dt = performance.now() - t0
  console.log(`new Float32Array(3):    ${(dt/N*1000).toFixed(1)}μs/call`)

  // 8. Benchmark new Map + set + get
  t0 = performance.now()
  for (let i = 0; i < N; i++) {
    const m = new Map()
    m.set(1, src)
    m.set(2, src)
    m.set(3, src)
    m.get(1)
    m.get(2)
    m.get(3)
  }
  dt = performance.now() - t0
  console.log(`new Map(3 set+get):     ${(dt/N*1000).toFixed(1)}μs/call`)

  // 9. Benchmark poly_render_kernel_wasm (cached)
  t0 = performance.now()
  for (let i = 0; i < N; i++) {
    const wp = M._poly_render_kernel_wasm(ctx, sink, lenPtr, nBufsPtr)
    M._free(wp)
  }
  dt = performance.now() - t0
  console.log(`render_kernel (cached): ${(dt/N*1000).toFixed(1)}μs/call`)

  // 10. Benchmark poly_kernel_buf × 3
  t0 = performance.now()
  for (let i = 0; i < N; i++) {
    M._poly_kernel_buf(ctx, 0)
    M._poly_kernel_buf(ctx, 1)
    M._poly_kernel_buf(ctx, 2)
  }
  dt = performance.now() - t0
  console.log(`poly_kernel_buf × 3:    ${(dt/N*1000).toFixed(1)}μs/call`)

  M._poly_ctx_destroy(ctx)
  console.log('\nDone.')
}

main().catch(e => { console.error(e); process.exit(1) })
