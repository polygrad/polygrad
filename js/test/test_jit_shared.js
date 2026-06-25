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

  return { passed, failed }
}

module.exports = { runJitTests }
