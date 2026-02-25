'use strict'

const { init, Tensor } = require('../src/index')

function benchTraining(steps, warmupRuns) {
  const target = [3, 1, 4]
  const lr = 0.1

  // Warmup runs (populate caches)
  for (let w = 0; w < warmupRuns; w++) {
    let params = new Float32Array([0, 0, 0])
    for (let i = 0; i < steps; i++) {
      const x = new Tensor(params, { requiresGrad: true })
      const t = new Tensor(new Float32Array(target))
      const diff = x.sub(t)
      const sq = diff.mul(diff)
      const loss = sq.sum()
      loss.backward()
      const grad = x.grad.toArray()
      for (let j = 0; j < params.length; j++) params[j] -= lr * grad[j]
    }
  }

  // Timed run
  let params = new Float32Array([0, 0, 0])
  const stepTimes = []
  const t0 = performance.now()

  for (let i = 0; i < steps; i++) {
    const st = performance.now()
    const x = new Tensor(params, { requiresGrad: true })
    const t = new Tensor(new Float32Array(target))
    const diff = x.sub(t)
    const sq = diff.mul(diff)
    const loss = sq.sum()
    loss.backward()
    const grad = x.grad.toArray()
    const lossVal = loss.item()
    for (let j = 0; j < params.length; j++) params[j] -= lr * grad[j]
    stepTimes.push(performance.now() - st)
  }

  const total = performance.now() - t0
  const avgStep = stepTimes.reduce((a, b) => a + b, 0) / stepTimes.length

  // Compute final loss
  const xFinal = new Tensor(params)
  const tFinal = new Tensor(new Float32Array(target))
  const diffFinal = xFinal.sub(tFinal)
  const sqFinal = diffFinal.mul(diffFinal)
  const lossFinal = sqFinal.sum()
  const finalLoss = lossFinal.item()

  return { total, avgStep, finalLoss, stepTimes }
}

async function main() {
  console.log('Initializing WASM module...')
  await init()
  console.log('WASM module ready.\n')

  const steps = 30
  const warmupRuns = 2
  const trials = 5

  console.log(`Training benchmark: minimize sum((x-[3,1,4])^2), ${steps} steps, lr=0.1`)
  console.log(`Warmup: ${warmupRuns} full runs, then ${trials} timed trials\n`)

  const results = []
  for (let t = 0; t < trials; t++) {
    const r = benchTraining(steps, t === 0 ? warmupRuns : 0)
    results.push(r)
    console.log(`  Trial ${t + 1}: ${r.total.toFixed(2)}ms total, ${r.avgStep.toFixed(3)}ms/step, loss=${r.finalLoss.toExponential(2)}`)
  }

  const best = results.reduce((a, b) => a.total < b.total ? a : b)
  const median = [...results].sort((a, b) => a.total - b.total)[Math.floor(trials / 2)]

  console.log(`\nBest:   ${best.total.toFixed(2)}ms total, ${best.avgStep.toFixed(3)}ms/step`)
  console.log(`Median: ${median.total.toFixed(2)}ms total, ${median.avgStep.toFixed(3)}ms/step`)
  console.log(`\nTarget: < 2ms total (tf.js baseline: ~7ms)`)
}

main().catch(e => { console.error(e); process.exit(1) })
