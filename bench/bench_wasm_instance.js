/**
 * bench_wasm_instance.js -- Benchmark WASM JIT instance execution
 *
 * Compares WASM JIT forward pass timing for different model sizes.
 * Run: node bench/bench_wasm_instance.js
 */

'use strict'

const polygrad = require('../js/src/index')

async function main() {
  const pg = await polygrad.create({ target: 'wasm' })

  const sizes = [64, 1024, 16384]
  const iters = 200

  console.log('WASM JIT Instance Benchmark')
  console.log('===========================\n')

  for (const n of sizes) {
    // Build a simple MLP: n inputs -> n hidden -> 1 output
    const spec = JSON.stringify({
      layers: [n, Math.min(n, 64), 1],
      activation: 'relu',
      loss: 'mse',
      seed: 42
    })

    const inst = pg._backend.instance.mlp(spec)
    if (!inst) { console.log(`  MLP(${n}) creation failed`); continue }

    // Build input/target data
    const inputData = new Float32Array(n)
    const targetData = new Float32Array(1)
    for (let i = 0; i < n; i++) inputData[i] = Math.random()
    targetData[0] = 1.0

    // Forward pass timing
    const names = ['input']
    const arrays = [inputData]

    // Warmup
    pg._backend.instance.forward(inst, names, arrays)

    const t0 = performance.now()
    for (let i = 0; i < iters; i++) {
      pg._backend.instance.forward(inst, names, arrays)
    }
    const elapsed = (performance.now() - t0) / iters

    console.log(`  MLP(in=${n}) forward: ${elapsed.toFixed(3)} ms (${(elapsed * 1000).toFixed(0)} us)`)

    pg._backend.instance.free(inst)
  }

  pg.destroy()
}

main().catch(console.error)
