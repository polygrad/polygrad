'use strict'

const polygrad = require('..')
const { checkLogicalRuntimeOption, runTensorTests } = require('./test_tensor')
const { runModelRuntimeTests } = require('./test_model_runtime')
const { runJitTests } = require('./test_jit')
const { runOptimTests } = require('./test_optim')
const { runModelTests } = require('./test_model')
const { runSyncContractTests } = require('./test_sync_contract')
const { PolyRuntime } = require('../src/runtime')
const { createBoundModelClass } = require('../src/model')

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

  function lifecycleRuntime(core) {
    const rt = Object.create(PolyRuntime.prototype)
    rt._core = core
    rt._lifetime = {
      alive: true,
      core,
      asyncHost: Boolean(core.caps && core.caps.core === 'wasm' && core.caps.device === 'webgpu')
    }
    rt._closing = false
    rt._disposePromise = null
    rt._activeAsync = 0
    rt._asyncDrain = []
    rt.jit = { disposeAll() {} }
    return rt
  }

  await test('async runtime disposal waits for admitted work and rejects later work', async () => {
    const events = []
    const rt = lifecycleRuntime({
      caps: { core: 'wasm', device: 'webgpu' },
      destroy() { events.push('destroy') }
    })
    const release = rt._beginAsync()
    const disposed = rt.dispose()
    let rejected = false
    try { await rt._withAsync(() => events.push('late')) } catch (err) {
      rejected = /disposed/.test(String(err && err.message))
    }
    if (!rejected) throw new Error('runtime admitted work after close started')
    if (events.length) throw new Error(`runtime destroyed before admitted work settled: ${events}`)
    events.push('work')
    release()
    await disposed
    if (events.join(',') !== 'work,destroy') {
      throw new Error(`unexpected runtime teardown order: ${events}`)
    }
    if (rt.dispose() !== disposed) throw new Error('runtime disposal is not idempotent')
  })

  await test('stats rejects collection while async core work is suspended', async () => {
    const rt = lifecycleRuntime({
      caps: { core: 'wasm', device: 'webgpu' },
      ffi: { poly_ctx_stats() { return {} } },
      ctx: 1
    })
    rt.Tensor = { _disposeAll() {} }
    rt.uop = { _disposeAll() {} }
    const release = rt._beginAsync()
    let rejected = false
    try { rt.stats() } catch (err) {
      rejected = /active async work/.test(String(err && err.message))
    }
    release()
    if (!rejected) throw new Error('stats collected residency during active async work')
  })

  await test('async Model disposal follows its admitted forward on the core queue', async () => {
    const events = []
    let tail = Promise.resolve()
    const core = {
      caps: { core: 'wasm', device: 'webgpu' },
      enqueueAsync(fn) {
        const run = tail.then(fn, fn)
        tail = run.catch(() => {})
        return run
      },
      model: {
        call(handle, entrypoint) {
          if (entrypoint !== 'forward') throw new Error(`unexpected entrypoint ${entrypoint}`)
          events.push(`forward:${handle}`)
          return Promise.resolve(0)
        },
        free(handle) { events.push(`free:${handle}`) },
        bufCount() { return 0 },
        entrypointOutputCount() { return 0 },
        entrypointOutputName() { return null }
      }
    }
    const rt = lifecycleRuntime(core)
    const Model = createBoundModelClass(rt)
    const inst = new Model(123)
    const forward = inst.forwardAsync({})
    const disposed = inst.dispose()
    await Promise.all([forward, disposed])
    if (events.join(',') !== 'forward:123,free:123') {
      throw new Error(`unexpected Model teardown order: ${events}`)
    }
    let rejected = false
    try { await inst.forwardAsync({}) } catch (err) {
      rejected = /disposed/.test(String(err && err.message))
    }
    if (!rejected) throw new Error('disposed Model admitted later work')
    if (inst.dispose() !== disposed) throw new Error('Model disposal is not idempotent')
  })

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

  await test('Tensor(UOp) rejects a different WASM runtime context', async () => {
    const pgA = await polygrad.create({ core: 'wasm' })
    const pgB = await polygrad.create({ core: 'wasm' })
    try {
      const local = new pgA.Tensor([-2, 0, 3])
      const source = new pgB.Tensor([-2, 0, 3])
      let threw = false
      try {
        new pgA.Tensor(source.uop)
      } catch (e) {
        threw = String(e.message || e).includes('same Polygrad context')
      }
      if (!threw) throw new Error('cross-runtime Tensor(UOp) should be rejected')
      threw = false
      try {
        new pgA.Tensor(local.uop, { _ctx: pgB._core.ctx })
      } catch (e) {
        threw = String(e.message || e).includes('same Polygrad context')
      }
      if (!threw) throw new Error('Tensor(UOp) should reject a foreign _ctx override')
    } finally {
      await pgA.dispose()
      await pgB.dispose()
    }
  })

  await test('Tensor(UOp) rejects a different async WASM runtime', async () => {
    const pgA = await polygrad.createAsync({ core: 'wasm' })
    const pgB = await polygrad.createAsync({ core: 'wasm' })
    try {
      const source = new pgB.Tensor([-2, 0, 3])
      let threw = false
      try {
        new pgA.Tensor(source.uop)
      } catch (e) {
        threw = String(e.message || e).includes('same Polygrad context')
      }
      if (!threw) throw new Error('cross-runtime async Tensor(UOp) should be rejected')
    } finally {
      await pgA.dispose()
      await pgB.dispose()
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
      const x = new Tensor([1, 2, 3], { dtype: 'float32' })
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
  await checkLogicalRuntimeOption(polygrad, 'wasm')
  const pg = await polygrad.create({ core: 'wasm' })
  try {
    const syncResult = await runSyncContractTests(polygrad, pg, { core: 'wasm' })
    const tensorResult = await runTensorTests(pg, () => polygrad.create({ core: 'wasm', device: pg.device }))
    const instanceResult = await runModelRuntimeTests(pg)
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
