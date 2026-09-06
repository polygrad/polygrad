'use strict'

const polygrad = require('../js/src/index')

function check(actual, expected) {
  if (!Number.isFinite(actual) || Math.abs(actual - expected) > 1e-4 * Math.max(1, Math.abs(expected)))
    throw new Error(`MLP output ${actual} does not match reference ${expected}`)
}

async function runBenchmark(core = 'native') {
  const pg = polygrad.create({ core, device: core === 'native' ? 'cpu' : 'wasm' })
  const sizes = [64, 1024, 16384]
  const iterations = 200
  try {
    for (const n of sizes) {
      const hidden = Math.min(n, 64)
      const model = pg.models.MLP({ layers: [n, hidden, 1], bias: false, activation: 'relu',
        loss: 'none', batch_size: 1, seed: 42 })
      try {
        const x = Float32Array.from({ length: n }, (_, i) => (i % 17 - 8) / 17)
        const w0 = model.readBuffer('layers.0.weight')
        const w1 = model.readBuffer('layers.1.weight')
        let expected = 0
        for (let h = 0; h < hidden; h++) {
          let sum = 0
          for (let i = 0; i < n; i++) sum += w0[h*n+i] * x[i]
          expected += Math.max(sum, 0) * w1[h]
        }
        const io = { x }
        check(model.forward(io).output[0], expected)
        for (let i = 0; i < 10; i++) model.forward(io)
        const start = performance.now()
        for (let i = 0; i < iterations; i++) model.forward(io)
        const us = (performance.now() - start) * 1000 / iterations
        check(model.forward(io).output[0], expected)
        console.log(`${core} Model MLP(${n},${hidden},1): ${us.toFixed(3)} us/call (${iterations} iterations)`)
      } finally { model.dispose() }
    }
  } finally { pg.dispose() }
}

if (require.main === module) runBenchmark().catch(error => { console.error(error); process.exitCode = 1 })
module.exports = { runBenchmark }
