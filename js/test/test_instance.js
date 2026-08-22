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
      rejected = /call\('forward'\) failed/.test(String(e && e.message))
    }
    assert(rejected, 'float32 bytes must not bind to an int32 Instance input')
  } finally {
    inst.dispose()
  }
}

async function checkCallSignatureAndSelectedOutputs(pg, Instance) {
  const x = pg.Tensor.empty([2])
  const y = pg.Tensor.empty([2])
  const inst = await Instance.fromTensors({
    inputs: { x, y },
    outputs: { plus: x.add(y), minus: x.sub(y) },
    entrypoints: [
      { name: 'plus_ep', inputs: ['x', 'y'], outputs: ['plus'] },
      { name: 'minus_ep', inputs: ['x', 'y'], outputs: ['minus'] }
    ]
  })
  const webgpu = String(pg.device).toLowerCase() === 'webgpu'
  const call = (entrypoint, io) => webgpu
    ? inst.callAsync(entrypoint, io) : inst.call(entrypoint, io)
  try {
    const io = {
      x: new Float32Array([5, 7]),
      y: new Float32Array([2, 3])
    }
    const plus = await call('plus_ep', io)
    const minus = await call('minus_ep', io)
    assert(Object.keys(plus).join(',') === 'plus', 'plus_ep returned undeclared outputs')
    assert(Object.keys(minus).join(',') === 'minus', 'minus_ep returned undeclared outputs')
    assertClose(plus.plus, [7, 10], 0)
    assertClose(minus.minus, [3, 4], 0)

    let rejected = false
    try {
      await call('plus_ep', { x: new Float32Array([9, 9]) })
    } catch (err) {
      rejected = /call\('plus_ep'\) failed/.test(String(err && err.message))
    }
    assert(rejected, 'missing required input must not reuse stale Instance bytes')
  } finally {
    if (webgpu) await inst.dispose()
    else inst.dispose()
  }

  let invalidParamRejected = false
  try {
    const unexpected = await Instance.fromTensors({
      inputs: { x }, outputs: { output: x.add(1) }, params: { bad: {} }
    })
    unexpected.dispose()
  } catch (err) {
    invalidParamRejected = /not a Tensor/.test(String(err && err.message))
  }
  assert(invalidParamRejected, 'invalid parameter must not be silently filtered')
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
  // Pinned Device opens only runtime/ops_* implementations (device.py:15-35).
  // Native Polygrad likewise has no WASM backend; Emscripten does. Keep the
  // test cross-device on every core without treating a graph spelling as an
  // executable runtime merely because it is a valid DEVICE name.
  const second = first === 'INTERP' ? (pg.core === 'native' ? 'CPU' : 'WASM') : 'INTERP'
  const place = async map => {
    if (webgpu) await inst.setDeviceMapAsync(map)
    else inst.setDeviceMap(map)
  }
  const forward = input => webgpu
    ? inst.forwardAsync(input) : inst.forward(input)
  const expected = webgpu ? [8, 10] : [8, 18]
  const present = (value, label) => {
    assert(value != null, `${label} returned null`)
    return value
  }
  const assertOptionalBytesEqual = (actual, expectedBytes, label) => {
    assert((actual == null) === (expectedBytes == null), `${label} presence changed`)
    if (actual != null) assertClose(actual, expectedBytes, 0)
  }

  try {
    const irBefore = present(inst.exportIR(), 'initial IR export')
    const weightsBefore = await inst.exportWeights()
    await place({ 'layers.0': first, 'layers.1': second })
    let result = await forward({ x: new Float32Array([1, 2]) })
    assertClose(present(result.output, 'first placed output'), expected)
    assertClose(present(inst.exportIR(), 'first placed IR export'), irBefore, 0)
    assertOptionalBytesEqual(await inst.exportWeights(), weightsBefore, 'first placed weight export')

    let rejected = false
    try {
      await place({ 'layers.0': first })
    } catch (err) {
      rejected = /incomplete|device map/.test(String(err.message || err))
    }
    assert(rejected, 'expected incomplete device map to fail')

    await place({ 'layers.0': second, 'layers.1': first })
    result = await forward({ x: new Float32Array([1, 2]) })
    assertClose(present(result.output, 'replacement placed output'), expected)
    const irAfter = present(inst.exportIR(), 'replacement IR export')
    const weightsAfter = await inst.exportWeights()
    assertClose(irAfter, irBefore, 0)
    assertOptionalBytesEqual(weightsAfter, weightsBefore, 'replacement weight export')

    const restored = Instance.fromIR(irAfter, weightsAfter)
    try {
      const restoredResult = webgpu
        ? await restored.forwardAsync({ x: new Float32Array([1, 2]) })
        : await restored.forward({ x: new Float32Array([1, 2]) })
      assertClose(present(restoredResult.output, 'restored output'), expected)
    } finally {
      if (webgpu) await restored.dispose()
      else restored.dispose()
    }
  } finally {
    if (webgpu) await inst.dispose()
    else inst.dispose()
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

  await test('generic call validates signature and returns selected outputs', async () => {
    await checkCallSignatureAndSelectedOutputs(pg, Instance)
  })

  await test('scalar rank8 and shared multi-output round trip', async () => {
    const scalarX = pg.Tensor.empty([])
    const scalarW = pg.Tensor.full([], 3, { dtype: 'float32', requiresGrad: true })
    const scalar = await Instance.fromTensors({
      inputs: { x: scalarX }, outputs: { output: scalarX.mul(scalarW) },
      params: { w: scalarW }
    })
    let scalarRestored = null
    try {
      scalarRestored = Instance.fromIR(scalar.exportIR(), await scalar.exportWeights())
      const result = await scalarRestored.forward({ x: new Float32Array([2]) })
      assertClose(result.output, [6], 0)
      assert(
        JSON.stringify(scalarRestored.bufShape(scalarRestored.findBuf('output'))) === '[]',
        'scalar output shape must remain []'
      )
    } finally {
      if (scalarRestored) scalarRestored.dispose()
      scalar.dispose()
    }

    const shape = [1, 1, 1, 1, 1, 1, 1, 1]
    const x = pg.Tensor.empty(shape)
    const w = pg.Tensor.ones(shape, { requiresGrad: true })
    const shared = x.add(w)
    const source = await Instance.fromTensors({
      inputs: { x },
      outputs: { plus: shared.add(1), minus: shared.sub(1) },
      params: { w }
    })
    let restored = null
    try {
      restored = Instance.fromIR(source.exportIR(), await source.exportWeights())
      const result = await restored.forward({ x: new Float32Array([2]) })
      assertClose(result.plus, [4], 0)
      assertClose(result.minus, [2], 0)
      assert(
        JSON.stringify(restored.bufShape(restored.findBuf('plus'))) === JSON.stringify(shape),
        'rank-8 output shape was not preserved'
      )
    } finally {
      if (restored) restored.dispose()
      source.dispose()
    }
  })

  await test('duplicate ABI storage alias fails closed like TinyJit', async () => {
    const x = pg.Tensor.empty([2])
    let error = null
    try {
      const unexpected = await Instance.fromTensors({
        inputs: { a: x, b: x }, outputs: { output: x.add(x) }
      })
      unexpected.dispose()
    } catch (err) {
      error = err
    }
    assert(error, 'duplicate ABI storage must be rejected')
  })

  await test('dynamic input alias with persistent state fails closed', async () => {
    const x = pg.Tensor.empty([2])
    let error = null
    try {
      const unexpected = await Instance.fromTensors({
        inputs: { x }, outputs: { output: x.add(x) }, params: { w: x }
      })
      unexpected.dispose()
    } catch (err) {
      error = err
    }
    assert(error, 'dynamic input must not alias persistent state')
  })

  await test('output alias of dynamic input round trips', async () => {
    const x = pg.Tensor.empty([2])
    const source = await Instance.fromTensors({ inputs: { x }, outputs: { output: x } })
    let restored = null
    try {
      restored = Instance.fromIR(source.exportIR())
      const value = new Float32Array([3, 4])
      assertClose((await source.forward({ x: value })).output, value, 0)
      assertClose((await restored.forward({ x: value })).output, value, 0)
    } finally {
      if (restored) restored.dispose()
      source.dispose()
    }
  })

  await test('named partial view state fails closed', async () => {
    const base = new pg.Tensor([1, 2, 3, 4], { dtype: 'float32', requiresGrad: true })
    const view = base.shrink([[1, 3]])
    const x = pg.Tensor.empty([2])
    let error = null
    try {
      const unexpected = await Instance.fromTensors({
        inputs: { x }, outputs: { output: x.add(view) }, params: { base, view }
      })
      unexpected.dispose()
    } catch (err) {
      error = err
    }
    assert(error, 'named partial view storage must be rejected')
  })

  await test('input-dependent named state effect fails closed', async () => {
    const x = pg.Tensor.empty([1])
    const w = new pg.Tensor([1], { dtype: 'float32' })
    const output = w.assign(w.add(x))
    let error = null
    try {
      const unexpected = await Instance.fromTensors({
        inputs: { x }, outputs: { output }, params: { w }
      })
      unexpected.dispose()
    } catch (err) {
      error = err
    }
    assert(error, 'input-dependent named state effect must be rejected')
  })

  await test('stochastic output requires named RNG state', async () => {
    pg.Tensor.manual_seed(123)
    const x = pg.Tensor.empty([2])
    let error = null
    try {
      const unexpected = await Instance.fromTensors({
        inputs: { x }, outputs: { output: x.add(pg.Tensor.rand(2)) }
      })
      unexpected.dispose()
    } catch (err) {
      error = err
    }
    assert(error, 'stochastic output without named RNG state must be rejected')
  })

  await test('state traversal preserves diamond aliases and stops cycles', async () => {
    const shared = new pg.Tensor([1, 2], { dtype: 'float32' })
    const root = { left: { weight: shared }, right: { weight: shared } }
    root.self = root
    const state = pg.nn.getStateDict(root)
    assert(
      JSON.stringify(Object.keys(state)) === JSON.stringify(['left.weight', 'right.weight']),
      `unexpected state paths: ${Object.keys(state)}`
    )
    assert(state['left.weight'] === state['right.weight'], 'alias paths must retain one Tensor')
    const params = pg.nn.getParameters(root)
    assert(params.length === 2, `unexpected parameter count: ${params.length}`)
    assert(params[0] === shared && params[1] === shared, 'parameter aliases must match state paths')
  })

  await test('float16 Instance state preserves exact storage bits', async () => {
    const w = new pg.Tensor([1.5, -2], { dtype: 'float16', requiresGrad: true })
    const x = pg.Tensor.empty([2], { dtype: 'float16' })
    const inst = await Instance.fromTensors({
      inputs: { x },
      outputs: { output: x.add(w) },
      params: { w }
    })
    try {
      assert(inst.paramDtype(0) === 'float16', `unexpected dtype ${inst.paramDtype(0)}`)
      const raw = await inst.paramData(0)
      assert(raw instanceof Uint16Array, `expected Uint16Array, got ${raw.constructor.name}`)
      assert(raw.length === 2 && raw[0] === 0x3e00 && raw[1] === 0xc000,
        `unexpected float16 bits: ${Array.from(raw)}`)
      const restored = Instance.fromIR(inst.exportIR(), await inst.exportWeights())
      try {
        assert(restored.paramDtype(0) === 'float16', 'restored dtype must remain float16')
        const restoredRaw = await restored.paramData(0)
        assert(restoredRaw instanceof Uint16Array,
          'restored float16 state must remain raw Uint16Array')
        assert(restoredRaw[0] === 0x3e00 && restoredRaw[1] === 0xc000,
          `unexpected restored bits: ${Array.from(restoredRaw)}`)
      } finally {
        restored.dispose()
      }
    } finally {
      inst.dispose()
    }
  })

  await test('typed Instance state round trips exact storage bytes', async () => {
    const cases = [
      ['float64', [1.25, -2.5], Float64Array],
      ['int32', [1, -2], Int32Array],
      ['uint8', [1, 255], Uint8Array],
      ['bool', [true, false], Uint8Array],
      ['bfloat16', [1.5, -2], Uint16Array]
    ]
    for (const [dtype, values, ArrayType] of cases) {
      const w = new pg.Tensor(values, { dtype })
      const x = pg.Tensor.empty([2], { dtype })
      const source = await Instance.fromTensors({
        inputs: { x }, outputs: { output: x }, params: { w }
      })
      try {
        const restored = Instance.fromIR(source.exportIR(), await source.exportWeights())
        try {
          const before = await source.paramData(0)
          const after = await restored.paramData(0)
          assert(before instanceof ArrayType,
            `${dtype}: expected ${ArrayType.name}, got ${before.constructor.name}`)
          assert(after instanceof ArrayType,
            `${dtype}: restored ${after.constructor.name}`)
          const beforeBytes = new Uint8Array(before.buffer, before.byteOffset, before.byteLength)
          const afterBytes = new Uint8Array(after.buffer, after.byteOffset, after.byteLength)
          assertClose(afterBytes, beforeBytes, 0)
        } finally {
          restored.dispose()
        }
      } finally {
        source.dispose()
      }
    }
  })

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

  await test('Adam checkpoint resumes the uninterrupted training trajectory', async () => {
    const spec = {
      layers: [2, 1], activation: 'none', bias: false,
      loss: 'mse', batch_size: 1, seed: 7
    }
    const source = MLP(spec)
    let restored = null
    try {
      source.setOptimizer(pg.OPTIM_ADAM, 0.05)
      const io = { x: new Float32Array([1, 2]), y: new Float32Array([3]) }
      for (let i = 0; i < 3; i++) await source.trainStep(io)
      restored = Instance.fromIR(source.exportIR(), await source.exportWeights())
      restored.setOptimizer(pg.OPTIM_ADAM, 0.05)

      const sourceLoss = await source.trainStep(io)
      const restoredLoss = await restored.trainStep(io)
      assert(sourceLoss === restoredLoss,
        `restored loss ${restoredLoss} != uninterrupted ${sourceLoss}`)
      assertClose(await restored.paramData(0), await source.paramData(0), 0)
      for (const name of [
        'optim.adam.b1_t', 'optim.adam.b2_t',
        'optim.adam.m.layers.0.weight', 'optim.adam.v.layers.0.weight'
      ]) {
        const sourceIndex = source.findBuf(name)
        const restoredIndex = restored.findBuf(name)
        assert(sourceIndex >= 0 && restoredIndex >= 0, `missing ${name}`)
        assertClose(
          await restored.bufData(restoredIndex), await source.bufData(sourceIndex), 0
        )
      }
    } finally {
      if (restored) restored.dispose()
      source.dispose()
    }
  })

  await test('stochastic named state requires checkpoint for portable activation', async () => {
    pg.Tensor.manual_seed(11)
    const w = pg.Tensor.rand(2, { requiresGrad: true })
    const x = pg.Tensor.empty([2])
    const source = await Instance.fromTensors({
      inputs: { x }, outputs: { output: x.mul(w) }, params: { w }
    })
    let restored = null
    try {
      const ir = source.exportIR()
      const weights = await source.exportWeights()
      let rejected = false
      try {
        const unexpected = Instance.fromIR(ir)
        unexpected.dispose()
      } catch (err) {
        rejected = true
      }
      assert(rejected, 'fresh stochastic activation must fail without checkpoint bytes')
      restored = Instance.fromIR(ir, weights)
      const input = new Float32Array([2, 3])
      const sourceOut = await source.forward({ x: input })
      const restoredOut = await restored.forward({ x: input })
      assertClose(restoredOut.output, sourceOut.output, 0)
    } finally {
      if (restored) restored.dispose()
      source.dispose()
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

  await test('generic call validates signature and returns selected outputs', async () => {
    await checkCallSignatureAndSelectedOutputs(pg, Instance)
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
