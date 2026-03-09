'use strict'

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed')
}

function assertClose(actual, expected, tol = 1e-4) {
  if (actual.length !== expected.length) {
    throw new Error(`length mismatch: ${actual.length} vs ${expected.length}`)
  }
  for (let i = 0; i < actual.length; i++) {
    if (Math.abs(actual[i] - expected[i]) > tol) {
      throw new Error(`mismatch at [${i}]: ${actual[i]} vs ${expected[i]}`)
    }
  }
}

async function runInstanceTests(pg) {
  const Instance = pg.Instance
  let passed = 0
  let failed = 0

  if (!pg.supportsInstance) {
    console.log('\n== Instance ==')
    console.log('  [SKIP] target does not expose PolyInstance runtime yet')
    return { passed: 0, failed: 0 }
  }

  async function test(name, fn) {
    try {
      await fn()
      console.log(`  [PASS] ${name}`)
      passed++
    } catch (e) {
      console.log(`  [FAIL] ${name}: ${e.message}`)
      failed++
    }
  }

  console.log('\n== Instance ==')

  await test('mlp create + param enumeration', async () => {
    const inst = Instance.mlp({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      assert(inst.paramCount === 4, `expected 4 params, got ${inst.paramCount}`)
      assert(inst.paramName(0) === 'layers.0.weight', 'unexpected first param name')
      assert(JSON.stringify(inst.paramShape(0)) === JSON.stringify([4, 2]), 'unexpected first param shape')
    } finally {
      inst.dispose()
    }
  })

  await test('mlp forward produces output', async () => {
    const inst = Instance.mlp({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      const outputs = inst.forward({ x: new Float32Array([1, 2]) })
      assert(outputs.output instanceof Float32Array, 'output should be Float32Array')
      assert(outputs.output.length === 1, `expected output length 1, got ${outputs.output.length}`)
      assert(Number.isFinite(outputs.output[0]), 'output should be finite')
    } finally {
      inst.dispose()
    }
  })

  await test('mlp train step decreases loss', async () => {
    const inst = Instance.mlp({
      layers: [2, 1],
      activation: 'none',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_SGD, 0.05)
      const x = new Float32Array([1, 2])
      const y = new Float32Array([5])
      let first = null
      let last = null
      for (let step = 0; step < 50; step++) {
        last = inst.trainStep({ x, y })
        if (first == null) first = last
      }
      assert(last < first, `expected loss to decrease (${first} -> ${last})`)
    } finally {
      inst.dispose()
    }
  })

  await test('weights export/import round trip', async () => {
    const spec = {
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    }
    const inst1 = Instance.mlp(spec)
    const inst2 = Instance.mlp({ ...spec, seed: 99 })
    try {
      const original = inst1.paramData(0)
      const different = inst2.paramData(0)
      let anyDiff = false
      for (let i = 0; i < original.length; i++) {
        if (original[i] !== different[i]) {
          anyDiff = true
          break
        }
      }
      assert(anyDiff, 'different seed should change weights')

      const weights = inst1.exportWeights()
      assert(weights instanceof Uint8Array && weights.length > 0, 'expected non-empty weights export')
      inst2.importWeights(weights)
      assertClose(inst2.paramData(0), original)
    } finally {
      inst1.dispose()
      inst2.dispose()
    }
  })

  await test('ir export/fromIR round trip', async () => {
    const inst1 = Instance.mlp({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      const ir = inst1.exportIR()
      const weights = inst1.exportWeights()
      const inst2 = Instance.fromIR(ir, weights)
      try {
        const out1 = inst1.forward({ x: new Float32Array([1, 2]) }).output
        const out2 = inst2.forward({ x: new Float32Array([1, 2]) }).output
        assertClose(out2, out1)
      } finally {
        inst2.dispose()
      }
    } finally {
      inst1.dispose()
    }
  })

  await test('tabm and nam builders are available', async () => {
    const tabm = Instance.tabm({
      layers: [2, 4, 1],
      activation: 'relu',
      loss: 'mse',
      batch_size: 1,
      seed: 42,
      n_ensemble: 4
    })
    const nam = Instance.nam({
      n_features: 2,
      hidden_sizes: [4],
      activation: 'relu',
      n_outputs: 1,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      const tabmOut = tabm.forward({ x: new Float32Array([1, 2]) })
      const namOut = nam.forward({ x: new Float32Array([1, 2]) })
      assert(tabmOut.output instanceof Float32Array, 'tabm output missing')
      assert(namOut.output instanceof Float32Array, 'nam output missing')
    } finally {
      tabm.dispose()
      nam.dispose()
    }
  })

  console.log(`\nInstance tests: ${passed} passed, ${failed} failed`)
  return { passed, failed }
}

module.exports = { runInstanceTests }
