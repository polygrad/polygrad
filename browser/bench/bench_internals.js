'use strict'

const path = require('path')
const PolygradModuleFactory = require(path.resolve(__dirname, '..', '..', '..', 'build', 'polygrad.js'))

async function main() {
  const M = await PolygradModuleFactory()
  const ctx = M._poly_ctx_new()

  const opCount = M._poly_op_count()
  const cwrapName = M.cwrap('poly_op_name', 'string', ['number'])
  const OPS = {}
  for (let i = 0; i < opCount; i++) {
    const name = cwrapName(i)
    if (name) OPS[name] = i
  }

  const lenPtr = M._malloc(4)
  const nBufsPtr = M._malloc(4)

  const mathImports = {
    exp2f: x => Math.pow(2, x), log2f: x => Math.log2(x), sinf: x => Math.sin(x)
  }

  function hashBytes(bytes) {
    let h = 0x811c9dc5
    for (let i = 0; i < bytes.length; i++) { h ^= bytes[i]; h = Math.imul(h, 0x01000193) }
    return h >>> 0
  }

  const execCache = new Map()
  const steps = 1000

  // Simulate one training step's backward pass
  // loss = sum((x - t)^2), grad w.r.t. x
  const target = new Float32Array([3, 1, 4])
  let params = new Float32Array([0, 0, 0])

  // Warmup: do 3 steps to populate caches
  for (let w = 0; w < 3; w++) {
    const xBuf = M._poly_buffer_f32(ctx, BigInt(3))
    const tBuf = M._poly_buffer_f32(ctx, BigInt(3))
    const sub = M._poly_alu2(ctx, OPS.SUB, xBuf, tBuf)
    const mul = M._poly_alu2(ctx, OPS.MUL, sub, sub)
    const axisPtr = M._malloc(8)
    M.HEAP32[axisPtr >> 2] = 0; M.HEAP32[(axisPtr >> 2) + 1] = 0
    const red = M._poly_reduce_axis(ctx, OPS.ADD, mul, axisPtr, 1)
    M._free(axisPtr)

    const gradUop = M._poly_grad(ctx, red, xBuf)
    const gradBuf = M._poly_buffer_f32(ctx, BigInt(3))
    const store = M._poly_store_val(ctx, gradBuf, gradUop)
    const sink = M._poly_sink1(ctx, store)

    const wp = M._poly_render_kernel_wasm(ctx, sink, lenPtr, nBufsPtr)
    const wl = M.HEAP32[lenPtr >> 2]
    const nb = M.HEAP32[nBufsPtr >> 2]
    const wb = new Uint8Array(wl)
    wb.set(M.HEAPU8.subarray(wp, wp + wl))
    M._free(wp)

    const key = hashBytes(wb)
    if (!execCache.has(key)) {
      const offsets = []
      let totalBytes = 0
      for (let i = 0; i < nb; i++) {
        offsets.push(totalBytes)
        totalBytes += 3 * 4  // all buffers are 3 floats
      }
      const pages = Math.max(1, Math.ceil(totalBytes / 65536))
      const memory = new WebAssembly.Memory({ initial: pages })
      const module = new WebAssembly.Module(wb)
      const instance = new WebAssembly.Instance(module, { env: { memory }, math: mathImports })
      execCache.set(key, { instance, memory, offsets, nb })
    }
  }

  // Now benchmark individual pieces of a training step (cached path)
  let t_graph = 0, t_grad = 0, t_render = 0, t_kernel_buf = 0
  let t_hash = 0, t_cache = 0, t_copy_in = 0, t_exec = 0, t_copy_out = 0, t_alloc = 0

  for (let s = 0; s < steps; s++) {
    // Graph construction
    let t0 = performance.now()
    const xBuf = M._poly_buffer_f32(ctx, BigInt(3))
    const tBuf = M._poly_buffer_f32(ctx, BigInt(3))
    const sub = M._poly_alu2(ctx, OPS.SUB, xBuf, tBuf)
    const mul = M._poly_alu2(ctx, OPS.MUL, sub, sub)
    const axisPtr = M._malloc(8)
    M.HEAP32[axisPtr >> 2] = 0; M.HEAP32[(axisPtr >> 2) + 1] = 0
    const red = M._poly_reduce_axis(ctx, OPS.ADD, mul, axisPtr, 1)
    M._free(axisPtr)
    t_graph += performance.now() - t0

    // poly_grad
    t0 = performance.now()
    const gradUop = M._poly_grad(ctx, red, xBuf)
    t_grad += performance.now() - t0

    // Alloc + store + sink
    t0 = performance.now()
    const gradBuf = M._poly_buffer_f32(ctx, BigInt(3))
    const store = M._poly_store_val(ctx, gradBuf, gradUop)
    const sink = M._poly_sink1(ctx, store)
    t_alloc += performance.now() - t0

    // Render
    t0 = performance.now()
    const wp = M._poly_render_kernel_wasm(ctx, sink, lenPtr, nBufsPtr)
    const wl = M.HEAP32[lenPtr >> 2]
    const nb = M.HEAP32[nBufsPtr >> 2]
    t_render += performance.now() - t0

    // kernel_buf
    t0 = performance.now()
    const bufs = new Array(nb)
    for (let i = 0; i < nb; i++) bufs[i] = M._poly_kernel_buf(ctx, i)
    t_kernel_buf += performance.now() - t0

    // hash
    t0 = performance.now()
    const wb = new Uint8Array(M.HEAPU8.buffer, wp, wl)
    const key = hashBytes(wb)
    t_hash += performance.now() - t0

    // cache lookup
    t0 = performance.now()
    const exec = execCache.get(key)
    t_cache += performance.now() - t0

    M._free(wp)

    // copy in
    t0 = performance.now()
    const mv = new Float32Array(exec.memory.buffer)
    mv.set(params, exec.offsets[0] / 4)
    mv.set(target, exec.offsets[1] / 4)
    mv.set(new Float32Array(3), exec.offsets[2] / 4)  // output zeros
    t_copy_in += performance.now() - t0

    // execute
    t0 = performance.now()
    exec.instance.exports.kernel(...exec.offsets)
    t_exec += performance.now() - t0

    // copy out
    t0 = performance.now()
    const result = new Float32Array(3)
    const ov = new Float32Array(exec.memory.buffer)
    result.set(ov.subarray(exec.offsets[0] / 4, exec.offsets[0] / 4 + 3))
    t_copy_out += performance.now() - t0

    // Update params
    for (let j = 0; j < 3; j++) params[j] -= 0.1 * result[j]
  }

  const us = (ms) => (ms / steps * 1000).toFixed(1)
  console.log(`Per-step breakdown (${steps} steps):`)
  console.log(`  graph construction: ${us(t_graph)}μs`)
  console.log(`  poly_grad:           ${us(t_grad)}μs`)
  console.log(`  alloc+store+sink:   ${us(t_alloc)}μs`)
  console.log(`  render (cached):    ${us(t_render)}μs`)
  console.log(`  kernel_buf × ${3}:    ${us(t_kernel_buf)}μs`)
  console.log(`  hashBytes:          ${us(t_hash)}μs`)
  console.log(`  cache lookup:       ${us(t_cache)}μs`)
  console.log(`  copy in:            ${us(t_copy_in)}μs`)
  console.log(`  kernel exec:        ${us(t_exec)}μs`)
  console.log(`  copy out:           ${us(t_copy_out)}μs`)
  const total = t_graph + t_grad + t_alloc + t_render + t_kernel_buf + t_hash + t_cache + t_copy_in + t_exec + t_copy_out
  console.log(`  TOTAL:              ${us(total)}μs`)
  console.log(`  Total time:         ${total.toFixed(1)}ms for ${steps} steps`)

  M._poly_ctx_destroy(ctx)
}

main().catch(e => { console.error(e); process.exit(1) })
