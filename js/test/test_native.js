'use strict'

const assert = require('assert')
const polygrad = require('..')
const { runTensorTests } = require('./test_tensor')
const { runInstanceTests } = require('./test_instance')
const { runJitTests } = require('./test_jit')
const { runOptimTests } = require('./test_optim')
const { runModelTests } = require('./test_model')

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

async function main() {
  await runNativeDeviceSelectionSmoke()
  const pg = await polygrad.create({ core: 'native' })
  const tensorResult = await runTensorTests(pg)
  const instanceResult = await runInstanceTests(pg)
  const jitResult = await runJitTests(pg)
  const optimResult = await runOptimTests(pg)
  const modelResult = await runModelTests(pg)
  await pg.dispose()
  const failed = tensorResult.failed + instanceResult.failed + jitResult.failed +
    optimResult.failed + modelResult.failed
  if (failed > 0) process.exit(1)
}

main().catch(e => { console.error(e); process.exit(1) })
