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

  console.log('\n== Instance.fromTensors export ==')

  await test('functional model exports selected forward entrypoint', async () => {
    const w = new Tensor([[2], [3]], { requiresGrad: true })
    await w.realize()
    const x = pg.Tensor.empty([1, 2])
    const y = x.dot(w)
    const ffi = pg._core.ffi
    const reach = (a, b) => Boolean(
      ffi.poly_uop_reachable(pg._core.ctx, a && a.raw ? a.raw : a, b && b.raw ? b.raw : b)
    )
    assert(reach(y.uopLogical, w.uopLogical), 'logical output should reach logical param')
    if (String(w.uopLogical.key) !== String(w.uop.key)) {
      assert(!reach(y.uopLogical, w.uop), 'logical output should not reach placed param root')
      assert(y.uopPhysical, 'physical output should exist for placed param root')
      assert(reach(y.uopPhysical, w.uop), 'physical output should reach placed param root')
    }
    const inst = await pg.Instance.fromTensors({
      inputs: { js_export_x: x },
      outputs: { js_export_output: y },
      params: { js_export_w: w }
    })
    assert(inst.paramCount === 1, 'expected one param')
    assert(inst.paramName(0) === 'js_export_w', 'param name mismatch')
    const out = await inst.forward({ js_export_x: new Float32Array([10, 20]) })
    assertClose(out.js_export_output, [80])
  })

  await test('fromTensors uses instance-local bindings', async () => {
    const count = pg._core && pg._core.ffi && pg._core.ffi.poly_ctx_named_count
    assert(typeof count === 'function', 'poly_ctx_named_count unavailable')
    const w = new Tensor([[2]], { requiresGrad: true })
    await w.realize()
    const x = pg.Tensor.empty([1, 1])
    const y = x.dot(w)
    const before = count(pg._core.ctx)
    const inst = await pg.Instance.fromTensors({
      inputs: { local_x: x },
      outputs: { local_y: y },
      params: { local_w: w }
    })
    assert(count(pg._core.ctx) === before, 'fromTensors mutated ctx named registry')
    assert(inst.paramName(0) === 'local_w', 'param name mismatch')
    const out = await inst.forward({ local_x: new Float32Array([3]) })
    assertClose(out.local_y, [6])
  })

  await test('fromBindings primitive uses instance-local bindings', async () => {
    const count = pg._core && pg._core.ffi && pg._core.ffi.poly_ctx_named_count
    assert(typeof count === 'function', 'poly_ctx_named_count unavailable')
    const w = new Tensor([[7]], { requiresGrad: true })
    await w.realize()
    const x = pg.Tensor.empty([1, 1])
    const y = x.dot(w)
    const before = count(pg._core.ctx)
    const inst = pg.Instance.fromBindings([
      { name: 'bind_x', role: 'input', tensor: x },
      { name: 'bind_w', role: 'state', tensor: w },
      { name: 'bind_y', role: 'output', tensor: y }
    ], [
      { name: 'forward', inputs: ['bind_x'], outputs: ['bind_y'] }
    ])
    assert(count(pg._core.ctx) === before, 'fromBindings mutated ctx named registry')
    assert(inst.paramName(0) === 'bind_w', 'param name mismatch')
    const out = await inst.forward({ bind_x: new Float32Array([3]) })
    assertClose(out.bind_y, [21])
  })

  await test('fromBindings uses tensor requiresGrad for trainability', async () => {
    const w = new Tensor([[7]], { requiresGrad: false })
    await w.realize()
    const x = pg.Tensor.empty([1, 1])
    const y = x.dot(w)
    const inst = pg.Instance.fromBindings([
      { name: 'x', role: 'input', tensor: x },
      { name: 'w', role: 'state', tensor: w },
      { name: 'y', role: 'output', tensor: y }
    ], [
      { name: 'forward', inputs: ['x'], outputs: ['y'] }
    ])
    assert(inst.paramTrainable(0) === false, 'state tensor should be frozen when requiresGrad is false')
  })

  await test('fromTensors keeps tinygrad-style plain object', async () => {
    class LinearNet {
      constructor() {
        this.weight = new Tensor([[4], [5]], { requiresGrad: true })
      }
      call(x) { return x.dot(this.weight) }
    }
    const net = new LinearNet()
    await net.weight.realize()
    const x = pg.Tensor.empty([1, 2])
    const outTensor = net.call(x)
    const inst = await pg.Instance.fromTensors({
      inputs: { js_trace_x: x },
      outputs: { output: outTensor },
      params: { weight: net.weight }
    })
    const out = await inst.forward({ js_trace_x: new Float32Array([2, 3]) })
    assertClose(out.output, [23])
  })

  await test('constructor state names survive IR round trip', async () => {
    const w = new Tensor([[2], [3]], { requiresGrad: true })
    await w.realize()
    const x = pg.Tensor.empty([1, 2])
    const logits = x.dot(w)
    const inst = new pg.Instance({
      inputs: { x },
      state: { 'layers.0.weight': w },
      outputs: { logits },
      entrypoints: [
        { name: 'forward', inputs: ['x'], outputs: ['logits'] }
      ]
    })
    assert(inst.paramCount === 1, 'expected one param')
    assert(inst.paramName(0) === 'layers.0.weight', 'param name mismatch')
    const out = await inst.forward({ x: new Float32Array([10, 20]) })
    assertClose(out.logits, [80])

    const inst2 = pg.Instance.fromIR(inst.exportIR(), await inst.exportWeights())
    assert(inst2.paramCount === 1, 'expected one reloaded param')
    assert(inst2.paramName(0) === 'layers.0.weight', 'reloaded param name mismatch')
    const out2 = await inst2.forward({ x: new Float32Array([10, 20]) })
    assertClose(out2.logits, [80])
  })

  await test('Instance.fit uses instance training path', async () => {
    const w = new Tensor([[1]], { requiresGrad: true })
    await w.realize()
    const x = pg.Tensor.empty([1, 1])
    const y = pg.Tensor.empty([1, 1])
    const pred = x.dot(w)
    const loss = pred.sub(y).square().mean()
    const inst = await pg.Instance.fromTensors({
      inputs: { js_fit_x: x },
      targets: { js_fit_y: y },
      outputs: { js_fit_out: pred },
      losses: { loss },
      params: { js_fit_w: w }
    })
    const losses = await inst.fit({
      js_fit_x: new Float32Array([1]),
      js_fit_y: new Float32Array([3])
    }, { epochs: 4, optimizer: 'sgd', lr: 0.1 })
    assert(losses.length === 4, 'expected four losses')
    assert(losses[losses.length - 1] < losses[0], 'expected loss to decrease')
  })

  console.log(`\nInstance graph tests: ${passed} passed, ${failed} failed`)
  return { passed, failed }
}

module.exports = { runModelTests }
