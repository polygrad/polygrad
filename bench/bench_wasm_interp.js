/**
 * bench_wasm_interp.js -- Benchmark WASM JIT vs WASM interpreter
 *
 * Compares two execution paths inside the Emscripten build:
 *   WASM_JIT: linearize -> render WASM bytes -> WebAssembly.Module (V8 JIT compiles to native)
 *   INTERP:   linearize -> walk UOps in C interpreter loop (compiled to WASM by Emscripten)
 *
 * Run: node bench/bench_wasm_interp.js
 */

'use strict'

const polygrad = require('../js/src/index')

async function main() {
  const pg = await polygrad.create({ target: 'wasm' })
  const inst = pg._backend.instance

  const sizes = [64, 1024, 16384]
  const iters = 50

  console.log('WASM Backend Benchmark: WASM_JIT vs INTERP (inside Emscripten)')
  console.log('================================================================\n')

  // PolyDeviceId values from exec_plan.h
  const INTERP = 2
  const WASM_JIT = 4

  for (const n of sizes) {
    const spec = JSON.stringify({
      layers: [n, Math.min(n, 64), 1],
      activation: 'relu',
      loss: 'mse',
      seed: 42
    })

    const inputData = new Float32Array(n)
    for (let i = 0; i < n; i++) inputData[i] = Math.random()
    const names = ['input']
    const arrays = [inputData]

    // --- WASM JIT (default) ---
    const instJit = inst.mlp(spec)
    if (!instJit) { console.log(`  MLP(${n}) JIT creation failed`); continue }
    inst.setDevice(instJit, WASM_JIT)
    inst.forward(instJit, names, arrays) // warmup (compiles kernels)

    let t0 = performance.now()
    for (let i = 0; i < iters; i++) {
      inst.forward(instJit, names, arrays)
    }
    const jitUs = ((performance.now() - t0) / iters) * 1000
    inst.free(instJit)

    // --- INTERP ---
    const instInterp = inst.mlp(spec)
    if (!instInterp) { console.log(`  MLP(${n}) INTERP creation failed`); continue }
    inst.setDevice(instInterp, INTERP)
    inst.forward(instInterp, names, arrays) // warmup

    t0 = performance.now()
    for (let i = 0; i < iters; i++) {
      inst.forward(instInterp, names, arrays)
    }
    const interpUs = ((performance.now() - t0) / iters) * 1000
    inst.free(instInterp)

    const ratio = interpUs / (jitUs > 0 ? jitUs : 1)
    console.log(`  MLP(in=${n}):  WASM_JIT ${jitUs.toFixed(0)} us  |  INTERP ${interpUs.toFixed(0)} us  |  ratio ${ratio.toFixed(1)}x`)
  }
}

main().catch(console.error)
