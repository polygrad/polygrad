'use strict'

const assert = require('assert')
const polygrad = require('..')
const { checkLogicalRuntimeOption, runTensorTests } = require('./test_tensor')
const { runModelRuntimeTests } = require('./test_model_runtime')
const { runJitTests } = require('./test_jit')
const { runOptimTests } = require('./test_optim')
const { runModelTests } = require('./test_model')
const { runSyncContractTests } = require('./test_sync_contract')

async function expectNativeDeviceReject(device) {
  let threw = false
  try {
    const pg = await polygrad.create({ core: 'native', device })
    await pg.dispose()
  } catch (e) {
    threw = String(e.message || e).includes('unsupported native device') ||
      String(e.message || e).includes('unknown native device')
  }
  assert(threw, `expected native device '${device}' to be rejected`)
}

async function runNativeDeviceSelectionSmoke() {
  const pg = await polygrad.create({ core: 'native', device: 'interp' })
  try {
    assert.strictEqual(pg.core, 'native')
    assert.strictEqual(pg.device, 'interp')
    assert(pg.canRun({ op: 'add', dtype: 'float32', shape: [4] }))
    const y = await new pg.Tensor([1, 2, 3]).add(1).realize()
    assert.deepStrictEqual(Array.from(await y.toArray()), [2, 3, 4])
  } finally {
    await pg.dispose()
  }
  await expectNativeDeviceReject('bogus')
  await expectNativeDeviceReject('webgpu')
}

async function runNativeCrossContextEinsumReject() {
  const pgA = await polygrad.create({ core: 'native', device: 'cpu' })
  const pgB = await polygrad.create({ core: 'native', device: 'cpu' })
  try {
    const a = new pgA.Tensor([1, 2, 3])
    const b = new pgB.Tensor([4, 5, 6])
    assert.throws(
      () => pgA.Tensor.einsum('i,i->', a, b),
      /same Polygrad context/
    )
  } finally {
    await pgA.dispose()
    await pgB.dispose()
  }
}

async function runNativeTensorUOpContextOwnership() {
  const pgA = await polygrad.create({ core: 'native', device: 'cpu' })
  const pgB = await polygrad.create({ core: 'native', device: 'cpu' })
  try {
    const source = new pgA.Tensor([-2, 0, 3])
    const wrapped = new pgA.Tensor(source.uop)
    assert.strictEqual(wrapped._ctx, source._ctx)
    assert.strictEqual(wrapped.constLike(1)._ctx, wrapped._ctx)
    assert.deepStrictEqual(Array.from(await wrapped.sign().toArray()), [-1, 0, 1])
    assert.throws(
      () => new pgA.Tensor(source.uop, { _ctx: pgB._core.ctx }),
      /same Polygrad context/
    )

    const foreign = new pgB.Tensor([-2, 0, 3])
    assert.throws(
      () => new pgA.Tensor(foreign.uop),
      /same Polygrad context/
    )
  } finally {
    await pgA.dispose()
    await pgB.dispose()
  }
}

async function main() {
  await checkLogicalRuntimeOption(polygrad, 'native')
  await runNativeDeviceSelectionSmoke()
  await runNativeCrossContextEinsumReject()
  await runNativeTensorUOpContextOwnership()
  const pg = await polygrad.create({ core: 'native' })
  const syncResult = await runSyncContractTests(polygrad, pg, { core: 'native' })
  const tensorResult = await runTensorTests(pg)
  const instanceResult = await runModelRuntimeTests(pg)
  const jitResult = await runJitTests(pg)
  const optimResult = await runOptimTests(pg)
  const modelResult = await runModelTests(pg)
  await pg.dispose()
  const failed = syncResult.failed + tensorResult.failed + instanceResult.failed + jitResult.failed +
    optimResult.failed + modelResult.failed
  if (failed > 0) process.exit(1)
}

main().catch(e => { console.error(e); process.exit(1) })
