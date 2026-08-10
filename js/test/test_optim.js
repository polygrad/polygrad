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
    const p = new Tensor([1], { dtype: 'float32', requiresGrad: true })
    p._grad = new Tensor([2], { dtype: 'float32' })
    const opt = new pg.nn.optim.SGD([p], { lr: 0.1 })
    await opt.step()
    assertClose(await p.toArray(), [0.8])
    opt.zeroGrad()
    assert(p.grad == null, 'zeroGrad should clear parameter gradients')
  })

  await test('SGD momentum state is part of scheduled effects', async () => {
    const p = new Tensor([1], { dtype: 'float32', requiresGrad: true })
    p._grad = new Tensor([2], { dtype: 'float32' })
    const opt = new pg.nn.optim.SGD([p], { lr: 0.1, momentum: 0.9 })
    const scheduled = opt.scheduleStep()
    assert(scheduled.length === 2, 'SGD momentum should schedule state and parameter')
    assert(scheduled[0] === opt.b[0], 'momentum state should be scheduled first')
    assert(scheduled[1] === p, 'parameter should be scheduled after momentum state')
    await scheduled[0].realize(...scheduled.slice(1))
    assertClose(await p.toArray(), [0.8])
    assertClose(await opt.b[0].toArray(), [2])
  })

  await test('SGD momentum commits lazy backward gradient before view state assign', async () => {
    const p = await new Tensor([1], { dtype: 'float32', requiresGrad: true }).realize()
    const x = await new Tensor([1], { dtype: 'float32' }).realize()
    const opt = new pg.nn.optim.SGD([p], {
      lr: 0.02, momentum: 0.85, nesterov: true, weightDecay: 0, fused: false
    })
    const loss = p.mul(x).sum()
    opt.zeroGrad()
    await loss.backward()
    const scheduled = opt.scheduleStep()
    await scheduled[0].realize(...scheduled.slice(1))
    assertClose(await opt.b[0].toArray(), [1])
    assertClose(await p.toArray(), [0.963])
  })

  await test('SGD vector momentum state is writable and elementwise', async () => {
    const p = await new Tensor([1, 2], { dtype: 'float32', requiresGrad: true }).realize()
    p._grad = new Tensor([0.25, -0.5])
    const opt = new pg.nn.optim.SGD([p], {
      lr: 0.1, momentum: 0.9, nesterov: true, weightDecay: 0.1, fused: false
    })
    const scheduled = opt.scheduleStep()
    assert(scheduled[0] === opt.b[0], 'momentum state should be scheduled first')
    assert(scheduled[1] === p, 'parameter should depend on scheduled momentum state')
    await scheduled[0].realize(...scheduled.slice(1))
    assertClose(await opt.b[0].toArray(), [0.35, -0.3])
    assertClose(await p.toArray(), [0.9335, 2.057])
  })

  await test('Adam updates beta-power and moment state in graph', async () => {
    const p = new Tensor([1], { dtype: 'float32', requiresGrad: true })
    p._grad = new Tensor([1], { dtype: 'float32' })
    const opt = new pg.nn.optim.Adam([p], { lr: 0.1 })
    const scheduled = opt.scheduleStep()
    assert(scheduled.length === 5, 'Adam should schedule four state tensors and one parameter')
    assert(scheduled[0] === opt.b1_t, 'Adam beta1 power should be scheduled first')
    assert(scheduled[1] === opt.b2_t, 'Adam beta2 power should be scheduled second')
    assert(scheduled[2] === opt.m[0], 'Adam first moment should precede parameters')
    assert(scheduled[3] === opt.v[0], 'Adam second moment should precede parameters')
    assert(scheduled[4] === p, 'Adam parameter should be scheduled after state')
    await scheduled[0].realize(...scheduled.slice(1))
    assertClose(await opt.m[0].toArray(), [0.1])
    assertClose(await opt.v[0].toArray(), [0.001], 1e-6)
    assertClose(await opt.b1_t.toArray(), [0.9])
    assertClose(await opt.b2_t.toArray(), [0.999])
    assertClose(await p.toArray(), [0.9], 1e-4)
  })

  await test('AdamW weight decay is handled by shared update builder', async () => {
    const p = new Tensor([1], { dtype: 'float32', requiresGrad: true })
    p._grad = new Tensor([0], { dtype: 'float32' })
    const opt = new pg.nn.optim.AdamW([p], { lr: 0.1, weightDecay: 0.01 })
    await opt.step()
    assertClose(await p.toArray(), [0.999], 1e-4)
  })

  await test('SGD Adam and AdamW graphs read the current LR Tensor', async () => {
    const cases = [
      ['SGD', () => {
        const p = new Tensor([1], { dtype: 'float32', requiresGrad: true })
        p._grad = new Tensor([1], { dtype: 'float32' })
        return [p, new pg.nn.optim.SGD([p], { lr: 0.1 }), 0.8]
      }],
      ['Adam', () => {
        const p = new Tensor([1], { dtype: 'float32', requiresGrad: true })
        p._grad = new Tensor([1], { dtype: 'float32' })
        return [p, new pg.nn.optim.Adam([p], { lr: 0.1 }), 0.8]
      }],
      ['AdamW', () => {
        const p = new Tensor([1], { dtype: 'float32', requiresGrad: true })
        p._grad = new Tensor([0], { dtype: 'float32' })
        return [p, new pg.nn.optim.AdamW([p], { lr: 0.1, weightDecay: 0.01 }), 0.998]
      }]
    ]
    for (const [name, makeCase] of cases) {
      const [p, opt, expected] = makeCase()
      assert(opt.lr instanceof Tensor, `${name} LR must be a Tensor`)
      const scheduled = opt.scheduleStep()
      opt.lr.assign([0.2])
      await opt.lr.realize()
      await scheduled[0].realize(...scheduled.slice(1))
      assertClose(await p.toArray(), [expected], 1e-4)
    }
  })

  console.log(`\nOptimizer tests: ${passed} passed, ${failed} failed`)
  return { passed, failed }
}

module.exports = { runOptimTests }
