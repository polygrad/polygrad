'use strict'

const polygrad = require('..')
const { runTensorTests } = require('./test_tensor')
const { runInstanceTests } = require('./test_instance')
const { runJitTests } = require('./test_jit')
const { runOptimTests } = require('./test_optim')
const { runModelTests } = require('./test_model')
const { runSyncContractTests } = require('./test_sync_contract')

function assertClose(actual, expected, tol = 1e-4) {
  if (actual.length !== expected.length) throw new Error(`length mismatch: ${actual.length} vs ${expected.length}`)
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

  await test('explicit runtimes keep host buffers independent across dispose cycles', async () => {
    const pgA = await polygrad.create({ core: 'wasm' })
    const pgB = await polygrad.create({ core: 'wasm' })
    try {
      const a = await new pgA.Tensor(new Float32Array([1, 2, 3]), { shape: [3] }).realize()
      const b = await new pgB.Tensor(new Float32Array([10, 20, 30]), { shape: [3] }).realize()
      assertClose(await a.add(1).toArray(), [2, 3, 4])
      assertClose(await b.add(1).toArray(), [11, 21, 31])
      await pgA.dispose()
      assertClose(await b.mul(2).toArray(), [20, 40, 60])
      b.copyFrom(new Float32Array([7, 8, 9]))
      assertClose(await b.add(1).toArray(), [8, 9, 10])
    } finally {
      try { await pgB.dispose() } catch (_) {}
    }

    const pgC = await polygrad.create({ core: 'wasm' })
    try {
      const c = await new pgC.Tensor(new Float32Array([4, 5, 6]), { shape: [3] }).realize()
      assertClose(await c.add(1).toArray(), [5, 6, 7])
    } finally {
      await pgC.dispose()
    }
  })

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

  await test('gradient bridge reacquires heap views after memory growth', async () => {
    const pg = await polygrad.create({ core: 'wasm' })
    let grownPtr = 0
    try {
      const Tensor = pg.Tensor
      const x = new Tensor([1, 2, 3], { requiresGrad: true })
      const loss = x.detach().sum()
      const core = pg._core
      const Module = core.Module
      const originalGradMany = Module._poly_grad_many_ex
      const beforeBytes = core.heapU8().buffer.byteLength

      Module._poly_grad_many_ex = (...args) => {
        const rc = originalGradMany(...args)
        grownPtr = Module._malloc(beforeBytes)
        return rc
      }

      let result
      try {
        result = core.ffi.poly_grad_many(
          core.ctx, loss.uopLogical.raw, 0, [x.uopLogical.raw]
        )
      } finally {
        Module._poly_grad_many_ex = originalGradMany
      }

      const afterBytes = core.heapU8().buffer.byteLength
      if (afterBytes <= beforeBytes) throw new Error('forced allocation did not grow WASM memory')
      if (!result || result.grads.length !== 1 || !result.grads[0]) {
        throw new Error('gradient bridge lost the zero-gradient UOp after memory growth')
      }
      if (result.present.length !== 1 || result.present[0] !== false) {
        throw new Error('gradient bridge changed an absent gradient to present after memory growth')
      }
    } finally {
      if (grownPtr && pg._core) pg._core.Module._free(grownPtr)
      await pg.dispose()
    }
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
    const syncResult = await runSyncContractTests(polygrad, pg, { core: 'wasm' })
    const tensorResult = await runTensorTests(pg)
    const instanceResult = await runInstanceTests(pg)
    const jitResult = await runJitTests(pg)
    const optimResult = await runOptimTests(pg)
    const modelResult = await runModelTests(pg)
    await pg.dispose()
    const ownershipResult = await runWasmOwnershipTests()
    const interpResult = await runWasmInterpTests()
    const failed = syncResult.failed + tensorResult.failed + instanceResult.failed + jitResult.failed +
      optimResult.failed + modelResult.failed + ownershipResult.failed + interpResult.failed
    if (failed > 0) process.exit(1)
  } catch (e) {
    try { await pg.dispose() } catch (_) {}
    throw e
  }
}

main().catch(e => { console.error(e); process.exit(1) })
