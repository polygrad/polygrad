'use strict'

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed')
}

function assertClose(actual, expected, tol = 1e-4) {
  if (actual.length !== expected.length) {
    throw new Error(`length mismatch: ${actual.length} vs ${expected.length}`)
  }
  for (let i = 0; i < actual.length; i++) {
    if (Number.isNaN(expected[i])) {
      if (!Number.isNaN(actual[i])) {
        throw new Error(`mismatch at [${i}]: ${actual[i]} vs ${expected[i]}`)
      }
      continue
    }
    const diff = Math.abs(actual[i] - expected[i])
    if (!Number.isFinite(diff) || diff > tol) {
      throw new Error(`mismatch at [${i}]: ${actual[i]} vs ${expected[i]}`)
    }
  }
}

function safetensorNames(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const headerLen = Number(view.getBigUint64(0, true))
  const headerBytes = bytes.subarray(8, 8 + headerLen)
  const header = JSON.parse(new TextDecoder().decode(headerBytes))
  return new Set(Object.keys(header).filter(k => k !== '__metadata__'))
}

function testFilterFor(pg) {
  if (pg && pg.testFilter) return String(pg.testFilter)
  if (typeof globalThis !== 'undefined' && globalThis.__POLY_TEST_FILTER) {
    return String(globalThis.__POLY_TEST_FILTER)
  }
  if (typeof process !== 'undefined' && process.env && process.env.POLY_TEST_FILTER) {
    return String(process.env.POLY_TEST_FILTER)
  }
  return ''
}

async function checkTypedIntegerInput(pg, Instance) {
  const x = pg.Tensor.empty([3], { dtype: 'int32' })
  const outTensor = x.cast('float32')
  const inst = await Instance.fromTensors({
    inputs: { typed_x: x },
    outputs: { typed_out: outTensor }
  })
  try {
    const output = await inst.forward({ typed_x: new Int32Array([0, 1, 2]) })
    assertClose(output.typed_out, [0, 1, 2], 0)

    let rejected = false
    try {
      await inst.forward({ typed_x: new Float32Array([0, 1, 2]) })
    } catch (e) {
      rejected = /forward failed/.test(String(e && e.message))
    }
    assert(rejected, 'float32 bytes must not bind to an int32 Instance input')
  } finally {
    inst.dispose()
  }
}

async function checkModuleDeviceMap(pg, Instance) {
  const Tensor = pg.Tensor
  const x = Tensor.empty([2])
  const webgpu = String(pg.device).toLowerCase() === 'webgpu'
  const w0 = webgpu ? null : new Tensor([3, 4], { dtype: 'float32', requiresGrad: true })
  const w1 = webgpu ? null : new Tensor([2, 3], { dtype: 'float32', requiresGrad: true })
  const hidden = webgpu ? x.add(3) : x.add(w0)
  const output = webgpu ? hidden.mul(2) : hidden.mul(w1)
  const inst = await Instance.fromTensors({
    inputs: { x },
    outputs: { output },
    params: webgpu ? null : { 'layers.0.weight': w0, 'layers.1.weight': w1 },
    modules: [
      { name: 'layers.0', inputs: [x], output: hidden },
      { name: 'layers.1', inputs: [hidden], output }
    ]
  })
  const first = String(pg.device).toUpperCase()
  const second = first === 'INTERP' ? 'WASM' : 'INTERP'
  const place = async map => {
    if (webgpu) await inst.setDeviceMapAsync(map)
    else inst.setDeviceMap(map)
  }
  const forward = input => webgpu
    ? inst.forwardAsync(input) : inst.forward(input)
  const expected = webgpu ? [8, 10] : [8, 18]

  try {
    await place({ 'layers.0': first, 'layers.1': second })
    let result = await forward({ x: new Float32Array([1, 2]) })
    assertClose(result.output, expected)

    let rejected = false
    try {
      await place({ 'layers.0': first })
    } catch (err) {
      rejected = /incomplete|device map/.test(String(err.message || err))
    }
    assert(rejected, 'expected incomplete device map to fail')

    await place({ 'layers.0': second, 'layers.1': first })
    result = await forward({ x: new Float32Array([1, 2]) })
    assertClose(result.output, expected)
  } finally {
    inst.dispose()
  }
}

async function runInstanceTests(pg) {
  const Instance = pg.Instance
  const { MLP, TabM, NAM } = pg.models
  const testFilter = testFilterFor(pg)
  let passed = 0
  let failed = 0

  if (!pg.supportsInstance) {
    console.log('\n== Instance ==')
    console.log('  [SKIP] core does not expose PolyInstance runtime yet')
    return { passed: 0, failed: 0 }
  }

  async function test(name, fn) {
    if (testFilter && !name.includes(testFilter)) return
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

  await test('typed integer input preserves bytes and rejects float binding', async () => {
    await checkTypedIntegerInput(pg, Instance)
  })

  await test('module device map places exact Tensor cuts atomically', async () => {
    await checkModuleDeviceMap(pg, Instance)
  })

  await test('model-family constructors are not Instance methods', async () => {
    assert(typeof Instance.mlp === 'undefined', 'Instance.mlp should not exist')
    assert(typeof MLP === 'function', 'pg.models.MLP should exist')
  })

  await test('mlp create + param enumeration', async () => {
    const inst = MLP({
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

  await test('param trainability freezes optimizer updates', async () => {
    const inst = MLP({
      layers: [2, 1],
      activation: 'none',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      assert(inst.paramTrainable(0) === true, 'weight should default trainable')
      assert(inst.paramTrainable(1) === true, 'bias should default trainable')
      inst.setParamTrainable(0, false)
      assert(inst.paramTrainable(0) === false, 'weight should be frozen')

      const weightBefore = await inst.paramData(0)
      const biasBefore = await inst.paramData(1)
      inst.setOptimizer(pg.OPTIM_SGD, 0.05)
      const x = new Float32Array([1, 2])
      const y = new Float32Array([5])
      for (let step = 0; step < 10; step++) {
        const loss = await inst.trainStep({ x, y })
        assert(Number.isFinite(loss), `loss should be finite, got ${loss}`)
      }

      assertClose(await inst.paramData(0), weightBefore)
      let biasChanged = false
      const biasAfter = await inst.paramData(1)
      for (let i = 0; i < biasAfter.length; i++) {
        if (biasAfter[i] !== biasBefore[i]) biasChanged = true
      }
      assert(biasChanged, 'unfrozen bias should update')
    } finally {
      inst.dispose()
    }
  })

  await test('param trainability survives IR round trip', async () => {
    const inst1 = MLP({
      layers: [2, 1],
      activation: 'none',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      inst1.setParamTrainable(0, false)
      const inst2 = Instance.fromIR(inst1.exportIR(), await inst1.exportWeights())
      try {
        assert(inst2.paramTrainable(0) === false, 'frozen flag should round trip')
        assert(inst2.paramTrainable(1) === true, 'unfrozen flag should round trip')
      } finally {
        inst2.dispose()
      }
    } finally {
      inst1.dispose()
    }
  })

  await test('mlp forward produces output', async () => {
    const inst = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      const outputs = await inst.forward({ x: new Float32Array([1, 2]) })
      assert(outputs.output instanceof Float32Array, 'output should be Float32Array')
      assert(outputs.output.length === 1, `expected output length 1, got ${outputs.output.length}`)
      assert(Number.isFinite(outputs.output[0]), 'output should be finite')
    } finally {
      inst.dispose()
    }
  })

  await test('mlp train step decreases loss', async () => {
    const inst = MLP({
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
        last = await inst.trainStep({ x, y })
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
    const inst1 = MLP(spec)
    const inst2 = MLP({ ...spec, seed: 99 })
    try {
      const original = await inst1.paramData(0)
      const different = await inst2.paramData(0)
      let anyDiff = false
      for (let i = 0; i < original.length; i++) {
        if (original[i] !== different[i]) {
          anyDiff = true
          break
        }
      }
      assert(anyDiff, 'different seed should change weights')

      const weights = await inst1.exportWeights()
      assert(weights instanceof Uint8Array && weights.length > 0, 'expected non-empty weights export')
      inst2.importWeights(weights)
      assertClose(await inst2.paramData(0), original)
    } finally {
      inst1.dispose()
      inst2.dispose()
    }
  })

  await test('ir export/fromIR round trip', async () => {
    const inst1 = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      const ir = inst1.exportIR()
      const weights = await inst1.exportWeights()
      const inst2 = Instance.fromIR(ir, weights)
      try {
        const out1 = (await inst1.forward({ x: new Float32Array([1, 2]) })).output
        const out2 = (await inst2.forward({ x: new Float32Array([1, 2]) })).output
        assertClose(out2, out1)
      } finally {
        inst2.dispose()
      }
    } finally {
      inst1.dispose()
    }
  })

  await test('mlp batch_size=32 forward produces correct shape', async () => {
    const inst = MLP({
      layers: [4, 8, 3],
      activation: 'relu',
      bias: true,
      loss: 'cross_entropy',
      batch_size: 32,
      seed: 42
    })
    try {
      const x = new Float32Array(32 * 4)
      for (let i = 0; i < x.length; i++) x[i] = Math.random()
      const outputs = await inst.forward({ x })
      assert(outputs.output instanceof Float32Array, 'output should be Float32Array')
      assert(outputs.output.length === 32 * 3,
        `expected output length ${32 * 3}, got ${outputs.output.length}`)
      for (let i = 0; i < outputs.output.length; i++) {
        assert(Number.isFinite(outputs.output[i]),
          `output[${i}] should be finite, got ${outputs.output[i]}`)
      }
    } finally {
      inst.dispose()
    }
  })

  // Known bug: batch_size>1 cross_entropy backward has shape mismatch in codegen
  // Reproduces on CPU too (not WASM-specific). See PLAN.md P0.
  await test('mlp batch_size=32 train step decreases loss (P0)', async () => {
    const inst = MLP({
      layers: [4, 8, 3],
      activation: 'relu',
      bias: true,
      loss: 'cross_entropy',
      batch_size: 32,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_SGD, 0.01)
      const x = new Float32Array(32 * 4)
      const y = new Float32Array(32 * 3)
      for (let i = 0; i < x.length; i++) x[i] = (i % 7) * 0.1
      for (let i = 0; i < 32; i++) y[i * 3 + (i % 3)] = 1.0
      let first = null
      let last = null
      for (let step = 0; step < 30; step++) {
        last = await inst.trainStep({ x, y })
        if (first == null) first = last
      }
      assert(Number.isFinite(first), `first loss should be finite, got ${first}`)
      assert(Number.isFinite(last), `last loss should be finite, got ${last}`)
      assert(last < first, `expected loss to decrease (${first} -> ${last})`)
    } finally {
      inst.dispose()
    }
  })

  await test('tabm and nam builders are available', async () => {
    const tabm = TabM({
      layers: [2, 4, 1],
      activation: 'relu',
      loss: 'mse',
      batch_size: 1,
      seed: 42,
      n_ensemble: 4
    })
    const nam = NAM({
      n_features: 2,
      hidden_sizes: [4],
      activation: 'relu',
      n_outputs: 1,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      const tabmOut = await tabm.forward({ x: new Float32Array([1, 2]) })
      const namOut = await nam.forward({ x: new Float32Array([1, 2]) })
      assert(tabmOut.output instanceof Float32Array, 'tabm output missing')
      assert(namOut.output instanceof Float32Array, 'nam output missing')
    } finally {
      tabm.dispose()
      nam.dispose()
    }
  })

  await test('mlp train step with Adam', async () => {
    const inst = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_ADAM, 0.01)
      const x = new Float32Array([1, 2])
      const y = new Float32Array([3])
      let first = null
      let last = null
      for (let step = 0; step < 50; step++) {
        last = await inst.trainStep({ x, y })
        if (first == null) first = last
      }
      assert(last < first, `expected loss to decrease (${first} -> ${last})`)
    } finally {
      inst.dispose()
    }
  })

  await test('mlp train step with SGD momentum creates named state', async () => {
    const inst = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_SGD, 0.01, 0.9, 0.999, 1e-8, 0.0, 0.9)
      const x = new Float32Array([1, 2])
      const y = new Float32Array([3])
      const loss = await inst.trainStep({ x, y })
      assert(Number.isFinite(loss), `loss should be finite, got ${loss}`)
      const bi = inst.findBuf('optim.sgd.b.layers.0.weight')
      assert(bi >= 0, 'missing SGD momentum state buffer')
      const b = await inst.bufData(bi)
      assert(Array.from(b).some(v => Math.abs(v) > 0), 'momentum state should update')
      const defaultNames = safetensorNames(await inst.exportWeights())
      assert(defaultNames.has('optim.sgd.b.layers.0.weight'), 'default export should include optimizer state')
      const modelOnlyNames = safetensorNames(await inst.exportWeights({ includeOptimizer: false }))
      assert(modelOnlyNames.has('layers.0.weight'), 'model-only export should include params')
      assert(!modelOnlyNames.has('optim.sgd.b.layers.0.weight'), 'model-only export should exclude optimizer state')
    } finally {
      inst.dispose()
    }
  })

  await test('mlp batch_size=4 mse train', async () => {
    const inst = MLP({
      layers: [2, 4, 2],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 4,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_SGD, 0.01)
      const x = new Float32Array(4 * 2).fill(0.5)
      const y = new Float32Array(4 * 2).fill(0.3)
      let first = null
      let last = null
      for (let step = 0; step < 50; step++) {
        last = await inst.trainStep({ x, y })
        if (first == null) first = last
      }
      assert(Number.isFinite(first), `first loss should be finite, got ${first}`)
      assert(Number.isFinite(last), `last loss should be finite, got ${last}`)
      assert(last < first, `expected loss to decrease (${first} -> ${last})`)
    } finally {
      inst.dispose()
    }
  })

  await test('mlp 100-step convergence', async () => {
    const inst = MLP({
      layers: [2, 8, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_SGD, 0.01)
      const x = new Float32Array([1, 2])
      const y = new Float32Array([5])
      let first = null
      let last = null
      for (let step = 0; step < 100; step++) {
        last = await inst.trainStep({ x, y })
        if (first == null) first = last
      }
      assert(last < first * 0.1, `expected >90% loss reduction (${first} -> ${last})`)
    } finally {
      inst.dispose()
    }
  })

  console.log(`\nInstance tests: ${passed} passed, ${failed} failed`)
  return { passed, failed }
}

async function runInstanceSmokeTests(pg) {
  const Instance = pg.Instance
  const { MLP, TabM, NAM } = pg.models
  const testFilter = testFilterFor(pg)
  let passed = 0
  let failed = 0

  if (!pg.supportsInstance) {
    console.log('\n== Instance ==')
    console.log('  [SKIP] core does not expose PolyInstance runtime yet')
    return { passed: 0, failed: 0 }
  }

  async function test(name, fn) {
    if (testFilter && !name.includes(testFilter)) return
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

  await test('typed integer input preserves bytes and rejects float binding', async () => {
    await checkTypedIntegerInput(pg, Instance)
  })

  await test('webgpu module device map places exact Tensor cuts atomically', async () => {
    await checkModuleDeviceMap(pg, Instance)
  })

  await test('webgpu mlp forward smoke', async () => {
    const inst = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      const outputs = await inst.forward({ x: new Float32Array([1, 2]) })
      assert(outputs.output instanceof Float32Array, 'output should be Float32Array')
      assert(outputs.output.length === 1, `expected output length 1, got ${outputs.output.length}`)
      assert(Number.isFinite(outputs.output[0]), 'output should be finite')
    } finally {
      inst.dispose()
    }
  })

  console.log(`\nInstance smoke tests: ${passed} passed, ${failed} failed`)
  return { passed, failed }
}

module.exports = { runInstanceTests, runInstanceSmokeTests }
