#!/usr/bin/env node
/**
 * bench_ratios.js -- WASM vs native ratio benchmarks via instance API.
 *
 * Reports wasm_time / native_time ratios that are stable across hardware.
 * Uses the instance API (compile once, execute many times) which measures
 * actual kernel execution quality, not scheduling overhead.
 *
 * Usage: node bench/bench_ratios.js [--json-file <path>]
 */

'use strict'

const path = require('path')
const fs = require('fs')
const polygrad = require('../js/src/index')

const WARMUP = 5
const ITERS = { 64: 200, 1024: 200, 16384: 50 }

function medianTime(fn, iters) {
  const times = []
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now()
    fn()
    times.push((performance.now() - t0) * 1000) // ms -> us
  }
  times.sort((a, b) => a - b)
  return times[Math.floor(times.length / 2)]
}

function benchMlpForward(pg, inputSize, iters) {
  const spec = JSON.stringify({
    layers: [inputSize, Math.min(inputSize, 64), 1],
    activation: 'relu',
    loss: 'mse',
    seed: 42
  })
  const inst = pg._backend.instance.mlp(spec)
  if (!inst) return null

  const input = new Float32Array(inputSize)
  for (let i = 0; i < inputSize; i++) input[i] = Math.random()

  for (let i = 0; i < WARMUP; i++) {
    pg._backend.instance.forward(inst, ['input'], [input])
  }
  const us = medianTime(() => {
    pg._backend.instance.forward(inst, ['input'], [input])
  }, iters)

  pg._backend.instance.free(inst)
  return us
}

function benchMlpTrain(pg, inputSize, iters) {
  const spec = JSON.stringify({
    layers: [inputSize, Math.min(inputSize, 64), 1],
    activation: 'relu',
    loss: 'mse',
    seed: 42
  })
  const inst = pg._backend.instance.mlp(spec)
  if (!inst) return null

  // SGD optimizer (kind=1)
  pg._backend.instance.setOptimizer(inst, 1, 0.01, 0.9, 0.999, 1e-8, 0)

  const input = new Float32Array(inputSize)
  const target = new Float32Array(1)
  for (let i = 0; i < inputSize; i++) input[i] = Math.random()
  target[0] = 1.0

  for (let i = 0; i < WARMUP; i++) {
    pg._backend.instance.trainStep(inst, ['input', 'target'], [input, target])
  }
  const us = medianTime(() => {
    pg._backend.instance.trainStep(inst, ['input', 'target'], [input, target])
  }, iters)

  pg._backend.instance.free(inst)
  return us
}

function benchBundleRoundtrip(pg, inputSize, iters) {
  const spec = JSON.stringify({
    layers: [inputSize, Math.min(inputSize, 64), 1],
    activation: 'relu',
    loss: 'mse',
    seed: 42
  })
  const inst = pg._backend.instance.mlp(spec)
  if (!inst) return null

  // Save bundle, reload, forward -- tests the full portable artifact path
  const bundle = pg._backend.instance.saveBundle(inst)
  pg._backend.instance.free(inst)
  if (!bundle) return null

  const inst2 = pg._backend.instance.fromBundle(bundle)
  if (!inst2) return null

  const input = new Float32Array(inputSize)
  for (let i = 0; i < inputSize; i++) input[i] = Math.random()

  for (let i = 0; i < WARMUP; i++) {
    pg._backend.instance.forward(inst2, ['input'], [input])
  }
  const us = medianTime(() => {
    pg._backend.instance.forward(inst2, ['input'], [input])
  }, iters)

  pg._backend.instance.free(inst2)
  return us
}

async function main() {
  const jsonFileIdx = process.argv.indexOf('--json-file')
  const jsonFile = jsonFileIdx >= 0 ? process.argv[jsonFileIdx + 1] : null

  let pgNative, pgWasm
  try {
    pgNative = await polygrad.create({ target: 'native' })
  } catch (e) {
    console.error('  Native backend unavailable:', e.message)
  }
  try {
    pgWasm = await polygrad.create({ target: 'wasm' })
  } catch (e) {
    console.error('  WASM backend unavailable:', e.message)
  }

  if (!pgNative && !pgWasm) {
    console.error('No backends available')
    process.exit(1)
  }

  const results = {}

  console.error('\n  JS ratio benchmark (WASM / native)')
  console.error('  ===================================\n')

  const workloads = [
    ['mlp_forward', benchMlpForward, [64, 1024, 16384]],
    ['mlp_train', benchMlpTrain, [64, 1024]],
    ['bundle_forward', benchBundleRoundtrip, [64, 1024]],
  ]

  for (const [name, fn, sizes] of workloads) {
    for (const n of sizes) {
      const key = `${name}_${n}`
      const iters = ITERS[n] || 50

      let nativeUs = null, wasmUs = null
      if (pgNative) {
        try { nativeUs = fn(pgNative, n, iters) } catch (e) {
          console.error(`  ${key} native error: ${e.message}`)
        }
      }
      if (pgWasm) {
        try { wasmUs = fn(pgWasm, n, iters) } catch (e) {
          console.error(`  ${key} wasm error: ${e.message}`)
        }
      }

      const ratio = (nativeUs && wasmUs) ? +(wasmUs / nativeUs).toFixed(3) : null
      results[key] = {
        native_us: nativeUs ? +nativeUs.toFixed(2) : null,
        wasm_us: wasmUs ? +wasmUs.toFixed(2) : null,
        wasm_native_ratio: ratio,
        iters
      }

      const rStr = ratio ? `${ratio.toFixed(2)}x` : 'n/a'
      const nStr = nativeUs ? `${nativeUs.toFixed(0)}us` : 'n/a'
      const wStr = wasmUs ? `${wasmUs.toFixed(0)}us` : 'n/a'
      console.error(`  ${key.padEnd(24)} native=${nStr.padStart(8)} wasm=${wStr.padStart(8)} ratio=${rStr}`)
    }
  }

  console.error('')

  // Output JSON
  if (jsonFile) {
    let existing = {}
    try { existing = JSON.parse(fs.readFileSync(jsonFile, 'utf8')) } catch (e) { /* new file */ }
    existing.js = results
    fs.writeFileSync(jsonFile, JSON.stringify(existing, null, 2) + '\n')
    console.error(`  Merged JS results into ${jsonFile}\n`)
  } else {
    process.stdout.write(JSON.stringify({ backend: 'js', results }, null, 2) + '\n')
  }
}

main().catch(e => { console.error(e); process.exit(1) })
