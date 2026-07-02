'use strict'

const polygrad = require('..')
const { runTests } = require('./test_shared')
const { runInstanceTests } = require('./test_instance_shared')
const { runJitTests } = require('./test_jit_shared')
const { runOptimTests } = require('./test_optim_shared')
const { runModelTests } = require('./test_model_shared')

function assertClose(actual, expected, tol = 1e-4) {
  if (actual.length !== expected.length) throw new Error(`length mismatch: ${actual.length} vs ${expected.length}`)
  for (let i = 0; i < actual.length; i++) {
    if (Math.abs(actual[i] - expected[i]) > tol) {
      throw new Error(`mismatch at [${i}]: ${actual[i]} vs ${expected[i]}`)
    }
  }
}

async function runWasmOwnershipTests() {
  console.log('\n== WASM ownership ==')
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

  await test('typed-array input survives jit replay and runtime disposal', async () => {
    const data = new Float32Array([1, 2, 3, 4])
    let pg = await polygrad.create({ core: 'wasm' })
    let Tensor = pg.Tensor
    let x = new Tensor(data)
    const f = pg.jit((a) => a.add(1).mul(2))
    assertClose(await (await f(x)).toArray(), [4, 6, 8, 10])
    assertClose(await (await f(x)).toArray(), [4, 6, 8, 10])
    f.dispose()
    await pg.dispose()

    pg = await polygrad.create({ core: 'wasm' })
    Tensor = pg.Tensor
    x = new Tensor(data)
    assertClose(await x.toArray(), [1, 2, 3, 4])
    assertClose(await x.add(3).toArray(), [4, 5, 6, 7])
    await pg.dispose()
  })

  return { passed, failed }
}

async function runWasmInterpTests() {
  console.log('\n== WASM interp ==')
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

  await test('realized expression assign keeps interp placement', async () => {
    const pg = await polygrad.create({ core: 'wasm', device: 'interp' })
    try {
      const Tensor = pg.Tensor
      const x = await new Tensor([1]).add(1).realize()
      const xBuffer = x.uop.buffer.key
      x.assign(new Tensor([9]))
      await x.realize()
      if (x.uop.buffer.key !== xBuffer) {
        throw new Error('realized assign did not reuse current buffer')
      }
      assertClose(await x.toArray(), [9])
    } finally {
      await pg.dispose()
    }
  })

  return { passed, failed }
}

async function main() {
  const pg = await polygrad.create({ core: 'wasm' })
  try {
    const tensorResult = await runTests(pg)
    const instanceResult = await runInstanceTests(pg)
    const jitResult = await runJitTests(pg)
    const optimResult = await runOptimTests(pg)
    const modelResult = await runModelTests(pg)
    await pg.dispose()
    const ownershipResult = await runWasmOwnershipTests()
    const interpResult = await runWasmInterpTests()
    const failed = tensorResult.failed + instanceResult.failed + jitResult.failed +
      optimResult.failed + modelResult.failed + ownershipResult.failed + interpResult.failed
    if (failed > 0) process.exit(1)
  } catch (e) {
    try { await pg.dispose() } catch (_) {}
    throw e
  }
}

main().catch(e => { console.error(e); process.exit(1) })
