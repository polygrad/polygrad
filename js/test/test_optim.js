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

async function runOptimTests(pg) {
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

  console.log('\n== Optimizers ==')

  await test('nn.optim API surface mirrors tinygrad names', async () => {
    assert(typeof pg.nn.optim.Optimizer === 'function', 'missing Optimizer')
    assert(typeof pg.nn.optim.OptimizerGroup === 'function', 'missing OptimizerGroup')
    assert(typeof pg.nn.optim.SGD === 'function', 'missing SGD')
    assert(typeof pg.nn.optim.Adam === 'function', 'missing Adam')
    assert(typeof pg.nn.optim.AdamW === 'function', 'missing AdamW')
  })

  await test('SGD standalone step uses shared optimizer graph builder', async () => {
    const p = new Tensor([1], { requiresGrad: true })
    p._grad = new Tensor([2])
    const opt = new pg.nn.optim.SGD([p], { lr: 0.1 })
    await opt.step()
    assertClose(await p.toArray(), [0.8])
    opt.zeroGrad()
    assert(p.grad == null, 'zeroGrad should clear parameter gradients')
  })

  await test('SGD momentum state is part of scheduled effects', async () => {
    const p = new Tensor([1], { requiresGrad: true })
    p._grad = new Tensor([2])
    const opt = new pg.nn.optim.SGD([p], { lr: 0.1, momentum: 0.9 })
    const scheduled = opt.scheduleStep()
    assert(scheduled.includes(p), 'scheduled effects should include parameter')
    assert(scheduled.includes(opt.b[0]), 'scheduled effects should include momentum buffer')
    await scheduled[0].realize(...scheduled.slice(1))
    assertClose(await p.toArray(), [0.8])
    assertClose(await opt.b[0].toArray(), [2])
  })

  await test('Adam updates beta-power and moment state in graph', async () => {
    const p = new Tensor([1], { requiresGrad: true })
    p._grad = new Tensor([1])
    const opt = new pg.nn.optim.Adam([p], { lr: 0.1 })
    const scheduled = opt.scheduleStep()
    assert(scheduled.includes(opt.m[0]), 'scheduled effects should include Adam m')
    assert(scheduled.includes(opt.v[0]), 'scheduled effects should include Adam v')
    assert(scheduled.includes(opt.b1_t), 'scheduled effects should include Adam beta1 power')
    assert(scheduled.includes(opt.b2_t), 'scheduled effects should include Adam beta2 power')
    await scheduled[0].realize(...scheduled.slice(1))
    assertClose(await opt.m[0].toArray(), [0.1])
    assertClose(await opt.v[0].toArray(), [0.001], 1e-6)
    assertClose(await opt.b1_t.toArray(), [0.9])
    assertClose(await opt.b2_t.toArray(), [0.999])
    assertClose(await p.toArray(), [0.9], 1e-4)
  })

  await test('AdamW weight decay is handled by shared update builder', async () => {
    const p = new Tensor([1], { requiresGrad: true })
    p._grad = new Tensor([0])
    const opt = new pg.nn.optim.AdamW([p], { lr: 0.1, weightDecay: 0.01 })
    await opt.step()
    assertClose(await p.toArray(), [0.999], 1e-4)
  })

  console.log(`\nOptimizer tests: ${passed} passed, ${failed} failed`)
  return { passed, failed }
}

module.exports = { runOptimTests }
