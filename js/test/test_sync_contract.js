'use strict'

function isPromiseLike(value) {
  return value && typeof value.then === 'function'
}

function assertNotPromise(name, value) {
  if (isPromiseLike(value)) throw new Error(`${name} returned a Promise`)
  return value
}

function assertClose(actual, expected, tol = 1e-4) {
  if (actual.length !== expected.length) throw new Error(`length mismatch: ${actual.length} vs ${expected.length}`)
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i])
    if (!Number.isFinite(diff) || diff > tol) throw new Error(`mismatch at [${i}]: ${actual[i]} vs ${expected[i]}`)
  }
}

async function runSyncContractTests(polygrad, pg, createOpts) {
  console.log('\n== Sync API contract ==')
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

  await test('create and dispose are synchronous', () => {
    const tmp = assertNotPromise('polygrad.create', polygrad.create(createOpts))
    assertNotPromise('pg.dispose', tmp.dispose())
  })

  await test('realize and readback are synchronous on sync runtimes', () => {
    const Tensor = pg.Tensor
    const y = new Tensor([1, 2, 3]).mul(2).add(1)
    assertNotPromise('Tensor.realize', y.realize())
    assertClose(assertNotPromise('Tensor.toArray', y.toArray()), [3, 5, 7])
    assertClose(assertNotPromise('Tensor.toTypedArray', y.toTypedArray()), [3, 5, 7])
    const a = new Tensor([1, 2, 3]).add(1)
    const b = new Tensor([1, 2, 3]).mul(3)
    const pair = assertNotPromise('Tensor.toTypedArrays', Tensor.toTypedArrays(a, b))
    assertClose(pair[0], [2, 3, 4])
    assertClose(pair[1], [3, 6, 9])
  })

  await test('jit calls and compile wrapper are synchronous on sync runtimes', () => {
    const Tensor = pg.Tensor
    const f = pg.jit((x) => x.add(1).realize())
    try {
      assertClose(assertNotPromise('jit first call', f(new Tensor([1, 2, 3]))).toArray(), [2, 3, 4])
      assertClose(assertNotPromise('jit capture call', f(new Tensor([4, 5, 6]))).toArray(), [5, 6, 7])
      assertClose(assertNotPromise('jit replay call', f(new Tensor([7, 8, 9]))).toArray(), [8, 9, 10])
    } finally {
      f.dispose()
    }

    const compiled = assertNotPromise('pg.compile', pg.compile(
      (x) => x.mul(2).realize(),
      [new Tensor([1, 2, 3])]
    ))
    try {
      const out = assertNotPromise('compiled.run', compiled.run([new Tensor([4, 5, 6])]))
      assertClose(assertNotPromise('compiled output toArray', out.toArray()), [8, 10, 12])
      assertNotPromise('compiled.dispose', compiled.dispose())
    } finally {
      compiled.dispose()
    }
  })

  return { passed, failed }
}

module.exports = { runSyncContractTests, assertNotPromise }
