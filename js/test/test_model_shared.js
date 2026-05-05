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

async function runModelTests(pg) {
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

  console.log('\n== nn.Model export ==')

  await test('functional model exports selected forward entrypoint', async () => {
    const w = new Tensor([[2], [3]], { requiresGrad: true })
    await w.realize()
    const x = pg.nn.Input('js_export_x', { shape: [1, 2] })
    const y = x.dot(w)
    const model = new pg.nn.Model({
      inputs: { js_export_x: x },
      outputs: { js_export_output: y },
      params: { js_export_w: w }
    })
    const inst = await model.export()
    assert(inst.paramCount === 1, 'expected one param')
    assert(inst.paramName(0) === 'js_export_w', 'param name mismatch')
    const out = inst.forward({ js_export_x: new Float32Array([10, 20]) })
    assertClose(out.js_export_output, [80])
  })

  await test('trace keeps tinygrad-style plain object', async () => {
    class LinearNet {
      constructor() {
        this.weight = new Tensor([[4], [5]], { requiresGrad: true })
      }
      call(x) { return x.dot(this.weight) }
    }
    const net = new LinearNet()
    await net.weight.realize()
    const x = pg.nn.Input('js_trace_x', { shape: [1, 2] })
    const model = pg.nn.trace(net, {
      inputs: { js_trace_x: x }
    })
    const inst = await model.export()
    const out = inst.forward({ js_trace_x: new Float32Array([2, 3]) })
    assertClose(out.output, [23])
  })

  await test('model.fit uses instance training path', async () => {
    const w = new Tensor([[1]], { requiresGrad: true })
    await w.realize()
    const x = pg.nn.Input('js_fit_x', { shape: [1, 1] })
    const y = pg.nn.Target('js_fit_y', { shape: [1, 1] })
    const pred = x.dot(w)
    const loss = pred.sub(y).square().mean()
    const model = new pg.nn.Model({
      inputs: { js_fit_x: x },
      targets: { js_fit_y: y },
      outputs: { js_fit_out: pred },
      losses: { loss },
      params: { js_fit_w: w }
    })
    const losses = await model.fit({
      js_fit_x: new Float32Array([1]),
      js_fit_y: new Float32Array([3])
    }, { epochs: 4, optimizer: 'sgd', lr: 0.1 })
    assert(losses.length === 4, 'expected four losses')
    assert(losses[losses.length - 1] < losses[0], 'expected loss to decrease')
  })

  console.log(`\nModel tests: ${passed} passed, ${failed} failed`)
  return { passed, failed }
}

module.exports = { runModelTests }
