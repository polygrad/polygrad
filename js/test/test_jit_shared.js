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

async function assertThrowsAsync(fn, expected) {
  try {
    await fn()
  } catch (err) {
    if (expected && !String(err.message || err).includes(expected)) {
      throw new Error(`expected error containing "${expected}", got "${err.message || err}"`)
    }
    return
  }
  throw new Error('expected function to throw')
}

async function runJitTests(pg) {
  const Tensor = pg.Tensor
  let passed = 0
  let failed = 0

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

  console.log('\n== JIT ==')

  await test('captures on second call and replays with new inputs', async () => {
    const f = pg.jit((x) => x.add(1).mul(2))
    const a = new Tensor(new Float32Array([1, 2, 3]))
    const b = new Tensor(new Float32Array([10, 20, 30]))

    assertClose(await (await f(a)).toArray(), [4, 6, 8])
    assert(f.scheduleCount === 0, 'first call should not capture')
    assertClose(await (await f(a)).toArray(), [4, 6, 8])
    assert(f.scheduleCount === 1, `expected one captured schedule, got ${f.scheduleCount}`)
    assert(f.schedule_count === f.scheduleCount, 'schedule_count alias should match scheduleCount')
    assertClose(await (await f(b)).toArray(), [22, 42, 62])

    f.reset()
    assert(f.scheduleCount === 0, 'reset should release captured schedules')
  })

  await test('supports decorator-style options', async () => {
    const f = pg.jit({ prune: true })((x) => x.add(3))
    const x = new Tensor(new Float32Array([1, 2]))
    assertClose(await (await f(x)).toArray(), [4, 5])
    assertClose(await (await f(x)).toArray(), [4, 5])
    assert(f.schedule_count === 1, 'decorator-style jit should capture on second call')
    f.dispose()
  })

  await test('rejects duplicate input buffers', async () => {
    const f = pg.jit((x, y) => x.add(y))
    const x = new Tensor(new Float32Array([1, 2]))
    await assertThrowsAsync(() => f(x, x), 'duplicate inputs')
  })

  await test('rejects static shape mismatch after capture', async () => {
    const f = pg.jit((x) => x.add(1))
    const x = new Tensor(new Float32Array([1, 2, 3]))
    const y = new Tensor(new Float32Array([1, 2, 3, 4]))
    assertClose(await (await f(x)).toArray(), [2, 3, 4])
    assertClose(await (await f(x)).toArray(), [2, 3, 4])
    await assertThrowsAsync(() => f(y), 'args mismatch')
  })

  await test('jit matmul 64x64 captures and replays', async () => {
    const n = 64
    const aData = new Float32Array(n * n)
    const bData = new Float32Array(n * n)
    for (let i = 0; i < n * n; i++) {
      aData[i] = ((i * 13) % 37 - 18) * 0.03125
      bData[i] = ((i * 17) % 41 - 20) * 0.015625
    }

    const a = new Tensor(aData).reshape(n, n)
    const b = new Tensor(bData).reshape(n, n)
    const f = pg.jit((x, y) => x.matmul(y))

    await (await f(a, b)).realize()
    const out = await (await f(a, b)).toArray()
    assert(f.scheduleCount === 1, `expected one captured matmul schedule, got ${f.scheduleCount}`)

    const checks = [[0, 0], [3, 5], [17, 23], [63, 63]]
    for (const [row, col] of checks) {
      let expected = 0
      for (let k = 0; k < n; k++) expected += aData[row * n + k] * bData[k * n + col]
      assertClose([out[row * n + col]], [expected], 1e-4)
    }
    f.dispose()
  })

  await test('jit matmul A@B nonmultiple k tail captures and replays', async () => {
    const m = 8, n = 16, kDim = 5
    const aData = new Float32Array(m * kDim)
    const bData = new Float32Array(kDim * n)
    for (let i = 0; i < aData.length; i++) aData[i] = ((i * 7) % 19 - 9) * 0.0625
    for (let i = 0; i < bData.length; i++) bData[i] = ((i * 11) % 23 - 11) * 0.03125

    const a = new Tensor(aData).reshape(m, kDim)
    const b = new Tensor(bData).reshape(kDim, n)
    const f = pg.jit((x, y) => x.matmul(y))

    await (await f(a, b)).realize()
    const out = await (await f(a, b)).toArray()
    assert(f.scheduleCount === 1, `expected one captured tail matmul schedule, got ${f.scheduleCount}`)

    const checks = [[0, 0], [2, 7], [7, 15]]
    for (const [row, col] of checks) {
      let expected = 0
      for (let kk = 0; kk < kDim; kk++) expected += aData[row * kDim + kk] * bData[kk * n + col]
      assertClose([out[row * n + col]], [expected], 1e-4)
    }
    f.dispose()
  })

  await test('jit matmul 64x64 transposed rhs captures and replays', async () => {
    const n = 64
    const aData = new Float32Array(n * n)
    const bData = new Float32Array(n * n)
    for (let i = 0; i < n * n; i++) {
      aData[i] = ((i * 11) % 43 - 21) * 0.03125
      bData[i] = ((i * 19) % 47 - 23) * 0.015625
    }

    const a = new Tensor(aData).reshape(n, n)
    const b = new Tensor(bData).reshape(n, n)
    const f = pg.jit((x, y) => x.matmul(y.permute(1, 0)))

    await (await f(a, b)).realize()
    const out = await (await f(a, b)).toArray()
    assert(f.scheduleCount === 1, `expected one captured transposed matmul schedule, got ${f.scheduleCount}`)

    const checks = [[0, 0], [3, 5], [17, 23], [63, 63]]
    for (const [row, col] of checks) {
      let expected = 0
      for (let k = 0; k < n; k++) expected += aData[row * n + k] * bData[col * n + k]
      assertClose([out[row * n + col]], [expected], 1e-4)
    }
    f.dispose()
  })

  await test('jit fused relu between two transposed matmuls', async () => {
    const tokens = 32, d = 128, hidden = 256, outDim = 128
    const xData = new Float32Array(tokens * d)
    const w1Data = new Float32Array(hidden * d)
    const b1Data = new Float32Array(hidden)
    const w2Data = new Float32Array(outDim * hidden)
    for (let i = 0; i < xData.length; i++) xData[i] = ((i * 17 + 13) % 101 - 50) / 504
    for (let i = 0; i < w1Data.length; i++) w1Data[i] = ((i * 19 + 7) % 103 - 51) / 611
    for (let i = 0; i < b1Data.length; i++) b1Data[i] = ((i * 23 + 5) % 31 - 15) / 257
    for (let i = 0; i < w2Data.length; i++) w2Data[i] = ((i * 29 + 11) % 107 - 53) / 733

    const x = new Tensor(xData).reshape(tokens, d)
    const w1 = new Tensor(w1Data).reshape(hidden, d)
    const b1 = new Tensor(b1Data).reshape(1, hidden)
    const w2 = new Tensor(w2Data).reshape(outDim, hidden)
    const f = pg.jit((x, w1, b1, w2) =>
      x.matmul(w1.permute(1, 0)).add(b1).relu().matmul(w2.permute(1, 0)))

    await (await f(x, w1, b1, w2)).realize()
    const out = await (await f(x, w1, b1, w2)).toArray()
    assert(f.scheduleCount === 1, `expected one captured fused MLP schedule, got ${f.scheduleCount}`)

    const checks = [[0, 0], [3, 5], [17, 23], [31, 127]]
    for (const [row, col] of checks) {
      let expected = 0
      for (let h = 0; h < hidden; h++) {
        let act = b1Data[h]
        for (let k = 0; k < d; k++) act += xData[row * d + k] * w1Data[h * d + k]
        if (act < 0) act = 0
        expected += act * w2Data[col * hidden + h]
      }
      assertClose([out[row * outDim + col]], [expected], 2e-3)
    }
    f.dispose()
  })

  return { passed, failed }
}

module.exports = { runJitTests }
