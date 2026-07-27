/**
 * Shared test suite for the polygrad Tensor class.
 * Target-agnostic: runs against WASM or native bindings.
 *
 * Usage: require this module and call runTensorTests(pg).
 */

'use strict'

function assertClose(arr, expected, tol) {
  if (tol === undefined) tol = 1e-4
  if (arr.length !== expected.length) {
    throw new Error(`Length mismatch: ${arr.length} vs ${expected.length}`)
  }
  for (let i = 0; i < arr.length; i++) {
    if (Number.isNaN(expected[i])) {
      if (!Number.isNaN(arr[i])) {
        throw new Error(`Mismatch at [${i}]: ${arr[i]} vs ${expected[i]}`)
      }
      continue
    }
    const diff = Math.abs(arr[i] - expected[i])
    if (!Number.isFinite(diff) || diff > tol) {
      throw new Error(`Mismatch at [${i}]: ${arr[i]} vs ${expected[i]}`)
    }
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed')
}

function assertShape(actual, expected) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`shape mismatch: got [${actual}], expected [${expected}]`)
  }
}

async function runTensorTests(pg) {
  const Tensor = pg.Tensor
  const caps = pg.caps || {}
  const testFilter = (() => {
    if (pg && pg.testFilter) return String(pg.testFilter)
    if (typeof globalThis !== 'undefined' && globalThis.__POLY_TEST_FILTER) {
      return String(globalThis.__POLY_TEST_FILTER)
    }
    if (typeof process !== 'undefined' && process.env && process.env.POLY_TEST_FILTER) {
      return String(process.env.POLY_TEST_FILTER)
    }
    return ''
  })()
  const supportsF16 = caps.f16 !== false
  const supportsF64 = caps.f64 !== false
  let passed = 0, failed = 0, skipped = 0

  async function test(name, fn) {
    if (testFilter && !name.includes(testFilter)) return
    try {
      await fn()
      console.log(`  [PASS] ${name}`)
      passed++
    } catch (e) {
      console.log(`  [FAIL] ${name}: ${e.message}`)
      failed++
    }
  }

  async function testIf(cond, name, fn) {
    if (!cond) {
      console.log(`  [SKIP] ${name}`)
      skipped++
      return
    }
    await test(name, fn)
  }

  console.log(`Core: ${pg.core}, device: ${pg.device}\n`)

  // -- Creation --
  console.log('-- Creation --')

  await test('from vector', async () => {
    const t = new Tensor([1, 2, 3])
    assertShape(t.shape, [3])
    assertClose(await t.toArray(), [1, 2, 3])
  })

  await test('from scalar', async () => {
    const t = new Tensor(42)
    const v = await t.item()
    assert(Math.abs(v - 42) < 1e-4, `Expected 42, got ${v}`)
  })

  await test('from 2D', async () => {
    const t = new Tensor([[1, 2], [3, 4]])
    assertShape(t.shape, [2, 2])
    assertClose(await t.toArray(), [1, 2, 3, 4])
  })

  await test('empty creates unrealized buffer placeholder', async () => {
    const t = Tensor.empty([2, 3])
    assertShape(t.shape, [2, 3])
    assert(t.uop.hasBufferIdentity(), 'empty should be backed by a BUFFER UOp')
    assert(t.uopPhysical, 'empty should have a physical root at construction')
    assert(
      t.uopLogical.buffer.src[0].key === t.uopPhysical.buffer.src[0].key,
      'logical and physical empty storage should share one UNIQUE'
    )
    assert(t.uopLogical.buffer.src.length === 1, 'logical BUFFER should stay device-free')
    assert(
      t.uopPhysical.buffer.src[1].op === pg._core.ops.DEVICE,
      'physical BUFFER should carry DEVICE'
    )
  })

  await test('movement is realized through recursive base', async () => {
    const source = await new Tensor([1, 2, 3, 4]).realize()
    const reshaped = source.reshape(2, 2)
    const view = reshaped.flatten().shrink([[1, 3]])

    assert(view.uop.op === pg._core.ops.SHRINK, 'expected a SHRINK view')
    assert(view.uop.base.key === source.uop.base.key, 'movement base should be recursive')
    assert(reshaped.uop.realized === null, 'RESHAPE is not directly realized')
    assert(reshaped.uop.isRealized, 'RESHAPE should be realized through its base')
    assert(view.uop.realized === null, 'movement UOp is not directly realized')
    assert(view.uop.isRealized, 'allocated recursive base should realize the movement view')
    assert(view.uop.is_realized, 'snake-case realization alias should match')
  })

  await test('clone is lazy separate and preserves state', async () => {
    const source = Tensor.empty([4], { dtype: 'float32', requiresGrad: true }).is_param_(false)
    source.copyFrom(new Float32Array([1, 2, 3, 4]))
    await source.sum().backward()

    const cloned = source.clone(pg.device)
    assert(cloned.uopLogical && cloned.uopLogical.src.length === 2, 'clone should be AFTER')
    assert(cloned.uopLogical.src[1].src.length === 2, 'clone effect should be STORE')
    assert(cloned.uopLogical.src[0].buffer.key !== source.uop.buffer.key, 'clone needs a separate buffer')
    assert(cloned.requiresGrad === true, 'clone should preserve requiresGrad')
    assert(cloned.isParam === false, 'clone should preserve isParam')
    assert(cloned.grad && cloned.grad.uopLogical.src.length === 2, 'clone should recursively clone grad')
    assert(
      cloned.grad.uopLogical.src[0].buffer.key !== source.grad.uopLogical.src[0].buffer.key,
      'cloned grad needs a separate buffer'
    )
    assertClose(await cloned.toArray(), [1, 2, 3, 4])
    assertClose(await cloned.grad.toArray(), [1, 1, 1, 1])
  })

  await test('clone preserves scalar shape across devices', async () => {
    const source = Tensor.full([], 3, { device: 'cpu' })
    const cloned = source.clone('interp')
    assertShape(cloned.shape, [])
    assert(cloned.device === 'INTERP', `expected INTERP, got ${cloned.device}`)
    assertClose(await cloned.toArray(), [3])
  })

  await test('static constructors preserve scalar shape', async () => {
    const tensors = [Tensor.zeros([]), Tensor.ones([]), Tensor.full([], 3)]
    for (const tensor of tensors) assertShape(tensor.shape, [])
    assertClose(await tensors[0].toArray(), [0])
    assertClose(await tensors[1].toArray(), [1])
    assertClose(await tensors[2].toArray(), [3])
  })

  await test('detach is a lazy graph boundary', async () => {
    const source = new Tensor([[1, 2], [3, 4]], { requiresGrad: true })
    const detached = source.detach()

    assertShape(detached.shape, source.shape)
    assert(detached.dtype === source.dtype, 'detach should preserve dtype')
    assert(detached.device === source.device, 'detach should preserve device')
    assert(detached.requiresGrad === false, 'detach should clear requiresGrad')
    assert(detached.uopLogical.op === pg._core.ops.DETACH, 'detach should create a DETACH UOp')
    assert(detached.uopLogical.src.length === 1, 'detach should retain a unary graph node')
    assert(
      detached.uopLogical.src[0].key === source.uopLogical.key,
      'logical detach should retain the logical source UOp'
    )
    if (detached.uopPhysical) {
      assert(detached.uopPhysical.op === pg._core.ops.DETACH, 'physical detach should retain DETACH')
      assert(
        detached.uopPhysical.src[0].key === source.uop.key,
        'physical detach should retain the current source UOp'
      )
    }

    await detached.sum().backward()
    assertClose(await source.grad.toArray(), [0, 0, 0, 0])
  })

  await test('backward clones deviceless grad and accumulates in place', async () => {
    const x = Tensor.empty([4], { dtype: 'float32', requiresGrad: true })
    const loss = x.sum()
    await loss.backward()
    const firstGrad = x.grad
    const firstRoot = firstGrad.uopLogical
    const firstBuffer = firstRoot.src[0].buffer.key
    assert(firstRoot.src.length === 2, 'first grad should be AFTER')

    await loss.backward()
    assert(x.grad === firstGrad, 'gradient accumulation should preserve Tensor identity')
    const secondRoot = x.grad.uopLogical
    assert(secondRoot.src[0].key === firstRoot.key, 'gradient effect root changed')
    assert(secondRoot.src[0].src[0].buffer.key === firstBuffer, 'gradient buffer identity changed')
    assertClose(await x.grad.toArray(), [2, 2, 2, 2])
  })

  await test('backward through clone reaches source', async () => {
    const source = Tensor.empty([4], { dtype: 'float32', requiresGrad: true })
    source.copyFrom(new Float32Array([1, 2, 3, 4]))
    const cloned = source.clone()
    await cloned.sum().backward()
    assertClose(await source.grad.toArray(), [1, 1, 1, 1])
    assertClose(await cloned.grad.toArray(), [1, 1, 1, 1])
  })

  await test('backward retains distinct wrappers sharing one UOp', async () => {
    const x = new Tensor([1, 2, 3, 4], { requiresGrad: true })
    const y = new Tensor(x.uopLogical, { requiresGrad: true })
    assert(x !== y, 'expected distinct Tensor wrappers')
    assert(x.uopLogical.key === y.uopLogical.key, 'expected one shared logical UOp')

    await x.sum().backward()
    assertClose(await x.grad.toArray(), [1, 1, 1, 1])
    assertClose(await y.grad.toArray(), [1, 1, 1, 1])
  })

  await test('static constructors accept tinygrad-style shape arrays', async () => {
    const z = Tensor.zeros([2, 3])
    const o = Tensor.ones([2, 3])
    const e = Tensor.empty([2, 3])
    assertShape(z.shape, [2, 3])
    assertShape(o.shape, [2, 3])
    assertShape(e.shape, [2, 3])
    assertClose(await z.toArray(), [0, 0, 0, 0, 0, 0])
    assertClose(await o.toArray(), [1, 1, 1, 1, 1, 1])
  })

  await test('arange follows tinygrad start stop order', async () => {
    assertClose(await Tensor.arange(6).toArray(), [0, 1, 2, 3, 4, 5])
    assertClose(await Tensor.arange(0, 6).toArray(), [0, 1, 2, 3, 4, 5])
    assertClose(await Tensor.arange(2, 8, 2).toArray(), [2, 4, 6])
  })

  await test('runtime exposes uop namespace', async () => {
    const t = new Tensor([[1, 2], [3, 4]])
    assert(pg.uop, 'runtime should expose pg.uop')
    assertShape(pg.uop.shape(t.uop), [2, 2])
    assert(pg.uop.dtype(t.uop) === 'float32', `expected float32, got ${pg.uop.dtype(t.uop)}`)
    assert(pg.uop.hasBufferIdentity(t.uop), 'host tensor should have buffer identity')
    assert(pg.uop.buffer(t.uop), 'pg.uop.buffer should return a UOp')
  })

  await test('customKernel executes UOp CALL body', async () => {
    function addKernel(c, a, b) {
      c = c.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(c.numel(), 0)
      return c.index(i).store(a.index(i).add(b.index(i))).end(i).sink()
    }
    const a = new Tensor([1, 2, 3, 4])
    const b = new Tensor([10, 20, 30, 40])
    const c = Tensor.empty([4], { dtype: 'float32' })
    const out = c.customKernel(a, b, addKernel)[0]
    assertClose(await out.toArray(), [11, 22, 33, 44])
  })

  await test('customKernel multi-output backward matches tinygrad pattern', async () => {
    let callbackCall = null
    function addmulKernel(c, d, a, b) {
      c = c.flatten(); d = d.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(c.numel(), 0)
      const storeC = c.index(i).store(a.index(i).add(b.index(i)))
      const storeD = d.index(i).store(a.index(i).mul(b.index(i)))
      return storeC.group(storeD).end(i).sink({ arg: new pg.uop.KernelInfo('addmul') })
    }
    function backwardAddmul(gradC, gradD, call) {
      callbackCall = call
      const [, , , a, b] = call.src
      const gradA = new Tensor(gradC).add(new Tensor(gradD).mul(new Tensor(b))).uop
      const gradB = new Tensor(gradC).add(new Tensor(gradD).mul(new Tensor(a))).uop
      return [null, null, gradA, gradB]
    }
    const aVals = [
      0.3, -1.2, 0.7, 2.1,
      -0.5, 1.4, -2.2, 0.9,
      1.1, -0.8, 2.4, -1.7,
      0.2, 0.6, -0.4, 1.8,
    ]
    const bVals = [
      1.2, 0.5, -0.3, 0.8,
      2.0, -1.1, 0.4, -0.7,
      0.9, 1.5, -2.5, 0.1,
      -1.3, 0.2, 1.7, -0.6,
    ]
    const aRef = new Tensor(aVals, { requiresGrad: true }).reshape(4, 4)
    const bRef = new Tensor(bVals, { requiresGrad: true }).reshape(4, 4)
    await aRef.add(bRef).sum().add(aRef.mul(bRef).sum()).backward()

    const a = new Tensor(aVals, { requiresGrad: true }).reshape(4, 4)
    const b = new Tensor(bVals, { requiresGrad: true }).reshape(4, 4)
    await a.realize(b)
    const aPhysical = a.uopPhysical.key
    const bPhysical = b.uopPhysical.key
    const [c, d] = Tensor.empty([4, 4]).customKernel(
      Tensor.empty([4, 4]), a, b, { fxn: addmulKernel, gradFxn: backwardAddmul }
    )
    await c.sum().add(d.sum()).backward()
    assert(callbackCall.src[3].key === aPhysical, 'callback must receive physical a CALL slot')
    assert(callbackCall.src[4].key === bPhysical, 'callback must receive physical b CALL slot')
    assertClose(await a.grad.toArray(), await aRef.grad.toArray(), 1e-4)
    assertClose(await b.grad.toArray(), await bRef.grad.toArray(), 1e-4)
  })

  await test('customKernel physical AFTER preserves data gradient', async () => {
    function identityKernel(x) {
      x = x.flatten()
      const i = pg.uop.range(x.numel(), 0)
      return x.index(i).store(x.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo('identity') })
    }
    function backwardIdentity(grad, call) {
      assert(call.src.length === 2, 'expected one-argument custom CALL in backward')
      return [null]
    }
    const x = Tensor.empty([4], { dtype: 'float32', requiresGrad: true })
    x.copyFrom(new Float32Array([1, 2, 3, 4]))
    const y = x.customKernel({ fxn: identityKernel, gradFxn: backwardIdentity })[0]
    assert(y.uopLogical && y.uopLogical.src.length === 2, 'expected logical AFTER')
    assert(y.uopPhysical && y.uopPhysical.src.length === 2, 'expected physical AFTER')
    assert(y.uopLogical.op === y.uopPhysical.op, 'logical and physical aliases must both be AFTER')
    await y.sum().backward()
    assertClose(await x.grad.toArray(), [1, 1, 1, 1])
    assertClose(await y.grad.toArray(), [1, 1, 1, 1])
  })

  await test('customKernel separates output and input gradient edges', async () => {
    function identityKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const i = pg.uop.range(out.numel(), 0)
      return out.index(i).store(x.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo('identity_grad_edges') })
    }
    function backwardIdentity(grad, call) {
      assert(call.src.length === 3, 'expected output and input custom CALL arguments')
      return [null, grad]
    }
    const out = Tensor.empty([4], { dtype: 'float32', requiresGrad: true })
    const x = new Tensor([1, 2, 3, 4], { requiresGrad: true })
    const y = out.customKernel(x, { fxn: identityKernel, gradFxn: backwardIdentity })[0]
    await y.sum().backward()

    assertClose(await out.grad.toArray(), [1, 1, 1, 1])
    assertClose(await x.grad.toArray(), [1, 1, 1, 1])
    assertClose(await y.grad.toArray(), [1, 1, 1, 1])
  })

  await test('customKernel duplicate output alias passes one accumulated upstream', async () => {
    function identityKernel(out0, out1, x) {
      out0 = out0.flatten(); out1 = out1.flatten(); x = x.flatten()
      const i = pg.uop.range(out0.numel(), 0)
      return out0.index(i).store(x.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo('duplicate_output_grad') })
    }
    const callbackCounts = []
    function backwardIdentity(...args) {
      const call = args.pop()
      callbackCounts.push(args.length)
      return [null, null, args[0]]
    }
    const out = Tensor.empty([4], { dtype: 'float32' })
    const x = new Tensor([1, 2, 3, 4], { requiresGrad: true })
    const [y0, y1] = out.customKernel(out, x, { fxn: identityKernel, gradFxn: backwardIdentity })
    assert(y0.uop.key === y1.uop.key, 'duplicate output aliases should share one AFTER')
    await y0.sum().add(y1.sum()).backward()

    assert(callbackCounts.length === 1 && callbackCounts[0] === 1, 'expected one accumulated callback upstream')
    assertClose(await x.grad.toArray(), [2, 2, 2, 2])
  })

  await test('customKernel without gradFxn rejects a needed input gradient', async () => {
    function identityKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const i = pg.uop.range(out.numel(), 0)
      return out.index(i).store(x.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo('missing_grad_fxn') })
    }
    const out = Tensor.empty([4], { dtype: 'float32' })
    const x = new Tensor([1, 2, 3, 4], { requiresGrad: true })
    const y = out.customKernel(x, identityKernel)[0]
    let threw = false
    try {
      await y.sum().backward()
    } catch (e) {
      threw = String(e.message || e).includes('expected TUPLE body for gradient, got Ops.SINK')
    }
    assert(threw, 'missing gradFxn should reject an opaque CALL input gradient')
    assert(x.grad === null, 'failed backward must not assign x.grad')
    assert(y.grad === null, 'failed backward must not assign y.grad')
  })

  await test('customKernel callback is inactive behind stop-gradient ops', async () => {
    function identityKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const i = pg.uop.range(out.numel(), 0)
      return out.index(i).store(x.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo('stopped_custom_grad') })
    }

    let out = Tensor.empty([4], { dtype: 'float32', requiresGrad: true })
    let x = new Tensor([1, 2, 3, 4], { requiresGrad: true })
    let y = out.customKernel(x, identityKernel)[0]
    await y.detach().sum().backward()
    assertClose(await x.grad.toArray(), [0, 0, 0, 0])
    assertClose(await y.grad.toArray(), [0, 0, 0, 0])

    const calls = []
    function backwardIdentity(grad, call) {
      calls.push(grad.op)
      return [null, new Tensor(grad).add(7).uop]
    }
    out = Tensor.empty([4], { dtype: 'float32', requiresGrad: true })
    x = new Tensor([1, 2, 3, 4], { requiresGrad: true })
    y = out.customKernel(x, { fxn: identityKernel, gradFxn: backwardIdentity })[0]
    await y.lt(0).cast('float32').sum().backward()
    assert(calls.length === 0, 'stop-gradient ops must not invoke custom callbacks')
    assertClose(await x.grad.toArray(), [0, 0, 0, 0])
    assertClose(await y.grad.toArray(), [0, 0, 0, 0])
  })

  await test('customKernel reuses buffers after input update', async () => {
    function addKernel(c, a, b) {
      c = c.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(c.numel(), 0)
      return c.index(i).store(a.index(i).add(b.index(i))).end(i).sink()
    }
    const a = Tensor.empty([4], { dtype: 'float32' })
    const b = new Tensor([10, 20, 30, 40])
    const c = Tensor.empty([4], { dtype: 'float32' })
    const runs = [
      [new Float32Array([1, 2, 3, 4]), [11, 22, 33, 44]],
      [new Float32Array([5, 6, 7, 8]), [15, 26, 37, 48]],
    ]
    for (const [vals, expected] of runs) {
      a.copyFrom(vals)
      const out = c.customKernel(a, b, addKernel)[0]
      assert(out.uopLogical, 'custom output should keep a logical root')
      await out.realize()
      assert(out.uopPhysical && out.uopPhysical.hasBufferIdentity(), 'custom output should realize to a buffer-backed root')
      assertClose(await out.toArray(), expected)
    }
  })

  await test('customKernel exposes UOp compare where and unary methods', async () => {
    function selectKernel(out, a, b) {
      out = out.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(out.numel(), 0)
      const av = a.index(i)
      const bv = b.index(i)
      const selected = av.lt(0).where(av.neg(), av.max(bv))
      return out.index(i).store(selected).end(i).sink()
    }
    const out = Tensor.empty([4], { dtype: 'float32' })
    const a = new Tensor([-3, 2, 5, -1])
    const b = new Tensor([1, 4, 3, 9])
    assertClose(await out.customKernel(a, b, selectKernel)[0].toArray(), [3, 4, 5, 1])
  })

  await test('customKernel rejects bool INDEX coordinate before codegen', async () => {
    function invalidIndexKernel(out) {
      out = out.flatten()
      const zero = pg.uop.constant(0)
      const gate = zero.lt(1)
      const bad = out.index(gate)
      assert(bad !== null, 'UOp.index(bool) construction should match tinygrad')
      return bad.store(out.index(zero)).sink()
    }
    const out = Tensor.empty([1], { dtype: 'float32' })
    let threw = false
    try {
      await out.customKernel(invalidIndexKernel)[0].toArray()
    } catch (e) {
      threw = true
    }
    assert(threw, 'invalid bool INDEX coordinate must fail before codegen')
  })

  await test('customKernel exposes tinygrad-style floor div and mod', async () => {
    function divKernel(out, x, y) {
      out = out.flatten(); x = x.flatten(); y = y.flatten()
      const i = pg.uop.range(x.numel(), 0)
      const q = x.index(i).floordiv(y.index(i))
      const r = x.index(i).floormod(y.index(i))
      return out.index(i).store(q).end(i).sink(out.index(i.add(x.numel())).store(r).end(i))
    }
    const x = new Tensor(new Int32Array([-7, -7, 7, 7, -1, 1, 0]), { dtype: 'int32' })
    const y = new Tensor(new Int32Array([3, -3, -3, 3, 4, -4, 3]), { dtype: 'int32' })
    const out = Tensor.empty([14], { dtype: 'int32' })
    assertClose(
      await out.customKernel(x, y, divKernel)[0].toArray(),
      [-3, 2, -3, 2, -1, -1, 0, 2, -1, -2, 1, 3, -3, 0]
    )
  })


  await test('customKernel numeric literals follow float operand dtype', async () => {
    function literalKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const i = pg.uop.range(x.numel(), 0)
      const xv = x.index(i)
      const sameAdd = xv.sub(1).div(xv.add(1))
      const sameMul = xv.mul(2).div(xv.mul(3))
      const s0 = out.index(i).store(sameAdd)
      const s1 = out.index(i.add(x.numel())).store(sameMul)
      return s0.end(i).sink(s1.end(i))
    }
    const xData = new Float32Array(16)
    for (let i = 0; i < xData.length; i++) xData[i] = i / 10 + 1
    const out = Tensor.empty([32], { dtype: 'float32' })
    const got = await out.customKernel(new Tensor(xData, { dtype: 'float32' }), literalKernel)[0].toArray()
    const expected = []
    for (const x of xData) expected.push((x - 1) / (x + 1))
    for (const x of xData) expected.push((x * 2) / (x * 3))
    assertClose(got, expected, 1e-6)
  })

  await test('customKernel descriptor branches keep indexed placeholders typed', async () => {
    function termKernel(out, x, fa, fb, op, p0, p1) {
      out = out.flatten(); x = x.flatten(); fa = fa.flatten(); fb = fb.flatten()
      op = op.flatten(); p0 = p0.flatten(); p1 = p1.flatten()
      const rows = 4
      const cols = 3
      const idx = pg.uop.range(out.numel(), 0)
      const c = idx.floordiv(rows)
      const r = idx.mod(rows)
      const a = x.index(r.mul(cols).add(fa.index(c)))
      const b = x.index(r.mul(cols).add(fb.index(c)))
      assert(pg.uop.dtype(a) === 'float32', `indexed placeholder should be float32, got ${pg.uop.dtype(a)}`)
      const ab = b.lt(0).where(b.neg(), b)
      const safeDen = ab.lt(1e-6).where(b.lt(0).where(-1e-6, 1e-6), b)
      const z = a.mul(p0.index(c)).add(p1.index(c))
      const opv = op.index(c)
      let v = a.add(b)
      v = opv.eq(1).where(a.sub(b), v)
      v = opv.eq(2).where(a.mul(b), v)
      v = opv.eq(3).where(a.div(safeDen), v)
      v = opv.eq(4).where(z.sin(), v)
      return out.index(idx).store(v).end(idx).sink()
    }
    const xData = new Float32Array([
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
      10, 11, 12,
    ])
    const fa = new Int32Array([0, 1])
    const fb = new Int32Array([1, 2])
    const op = new Int32Array([0, 1])
    const p0 = new Float32Array([1.25, -0.5])
    const p1 = new Float32Array([0.1, 0.2])
    const expected = []
    for (let c = 0; c < 2; c++) {
      for (let r = 0; r < 4; r++) {
        const a = xData[r * 3 + fa[c]]
        const b = xData[r * 3 + fb[c]]
        expected.push(c === 0 ? a + b : a - b)
      }
    }
    const out = Tensor.empty([8], { dtype: 'float32' })
    const got = out.customKernel(
      new Tensor(xData),
      new Tensor(fa, { dtype: 'int32' }),
      new Tensor(fb, { dtype: 'int32' }),
      new Tensor(op, { dtype: 'int32' }),
      new Tensor(p0),
      new Tensor(p1),
      termKernel
    )[0]
    assertClose(await got.toArray(), expected, 1e-4)
  })

  await test('jit captures customKernel and replays after input update', async () => {
    function addKernel(c, a, b) {
      c = c.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(c.numel(), 0)
      return c.index(i).store(a.index(i).add(b.index(i))).end(i).sink()
    }
    const f = pg.jit((a, b) => {
      const c = Tensor.empty([4], { dtype: 'float32' })
      return c.customKernel(a, b, addKernel)[0]
    })
    const a = Tensor.empty([4], { dtype: 'float32' })
    const b = new Tensor([10, 20, 30, 40])
    a.copyFrom(new Float32Array([1, 2, 3, 4]))
    assertClose(await (await f(a, b)).toArray(), [11, 22, 33, 44])
    assertClose(await (await f(a, b)).toArray(), [11, 22, 33, 44])
    assert(f.scheduleCount === 1, `expected one captured schedule, got ${f.scheduleCount}`)
    a.copyFrom(new Float32Array([5, 6, 7, 8]))
    assertClose(await (await f(a, b)).toArray(), [15, 26, 37, 48])
    assert(f.stats().replayCount === 1, 'third customKernel call should replay')
  })

  await test('compile captures customKernel and replays after input update', async () => {
    function addKernel(c, a, b) {
      c = c.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(c.numel(), 0)
      return c.index(i).store(a.index(i).add(b.index(i))).end(i).sink()
    }
    const a = Tensor.empty([4], { dtype: 'float32' })
    const b = new Tensor([10, 20, 30, 40])
    a.copyFrom(new Float32Array([1, 2, 3, 4]))
    const compiled = await pg.compile((x, y) => {
      const c = Tensor.empty([4], { dtype: 'float32' })
      return c.customKernel(x, y, addKernel)[0]
    }, [a, b])
    assert(compiled.scheduleCount === 1, `expected one captured schedule, got ${compiled.scheduleCount}`)
    a.copyFrom(new Float32Array([5, 6, 7, 8]))
    const out = await compiled.run([a, b])
    assertClose(await out.toArray(), [15, 26, 37, 48])
    assert(compiled.stats().runCount === 1, 'compiled customKernel run should update runCount')
    compiled.dispose()
  })

  await test('compile captures customKernel fused reduction and replays after input update', async () => {
    function summaryKernel(out, a, b) {
      out = out.flatten(); a = a.flatten(); b = b.flatten()
      const c = pg.uop.range(2, 0)
      const r = pg.uop.range(4, 1, pg.uop.AxisType.REDUCE)
      const offset = c.mul(4).add(r)
      const term = a.index(offset).mul(b.index(r))
      const sum = term.sum(r)
      return out.index(c).store(sum).end(c).sink()
    }
    const a = Tensor.empty([8], { dtype: 'float32' })
    const b = new Tensor([1, 2, 3, 4])
    a.copyFrom(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]))
    const compiled = await pg.compile((x, y) => {
      const out = Tensor.empty([2], { dtype: 'float32' })
      return out.customKernel(x, y, summaryKernel)[0]
    }, [a, b])
    assert(compiled.scheduleCount === 1, `expected one captured schedule, got ${compiled.scheduleCount}`)
    assertClose(await (await compiled.run([a, b])).toArray(), [30, 70])
    a.copyFrom(new Float32Array([2, 3, 4, 5, 6, 7, 8, 9]))
    assertClose(await (await compiled.run([a, b])).toArray(), [40, 80])
    assert(compiled.stats().runCount === 2, 'compiled custom reduction should replay twice')
    compiled.dispose()
  })

  await test('compile captures customKernel multi-output fused reductions', async () => {
    function summaryKernel(out0, out1, a, b) {
      out0 = out0.flatten(); out1 = out1.flatten(); a = a.flatten(); b = b.flatten()
      const c = pg.uop.range(2, 0)
      const r = pg.uop.range(4, 1, pg.uop.AxisType.REDUCE)
      const term = a.index(c.mul(4).add(r))
      const s0 = term.sum(r)
      const s1 = term.mul(b.index(r)).sum(r)
      const st0 = out0.index(c).store(s0)
      const st1 = out1.index(c).store(s1)
      return st0.end(c).sink(st1.end(c))
    }
    const a = Tensor.empty([8], { dtype: 'float32' })
    const b = new Tensor([1, 2, 3, 4])
    a.copyFrom(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]))
    const compiled = await pg.compile((x, y) => {
      const out0 = Tensor.empty([2], { dtype: 'float32' })
      const out1 = Tensor.empty([2], { dtype: 'float32' })
      const outs = out0.customKernel(out1, x, y, summaryKernel)
      return [outs[0], outs[1]]
    }, [a, b])
    const got0 = await compiled.run([a, b])
    assertClose(await got0[0].toArray(), [10, 26])
    assertClose(await got0[1].toArray(), [30, 70])
    a.copyFrom(new Float32Array([2, 3, 4, 5, 6, 7, 8, 9]))
    const got1 = await compiled.run([a, b])
    assertClose(await got1[0].toArray(), [14, 30])
    assertClose(await got1[1].toArray(), [40, 80])
    assert(compiled.stats().runCount === 2, 'compiled custom multi-output reduction should replay twice')
    compiled.dispose()
  })

  await test('compile captures grouped customKernel compact summary with intercept reductions', async () => {
    function summaryKernel(out, x, y) {
      out = out.flatten(); x = x.flatten(); y = y.flatten()
      const candidates = 2
      const rows = 128
      const c = pg.uop.range(candidates, 0)
      const r = pg.uop.range(rows, 1, pg.uop.AxisType.REDUCE)
      const one = pg.uop.constant(1.0)
      const xv = x.index(c.mul(rows).add(r))
      const yv = y.index(r)
      const stats = [
        one.sum(r),
        xv.sum(r),
        xv.mul(xv).sum(r),
        yv.sum(r),
        xv.mul(yv).sum(r),
      ]
      const stores = stats.map((s, stat) => out.index(c.add(stat * candidates)).store(s))
      return stores[0].group(...stores.slice(1)).end(c).sink()
    }
    const makeX = (shift = 0) => Float32Array.from({ length: 256 }, (_, i) => i + 1 + shift)
    const makeY = () => Float32Array.from({ length: 128 }, (_, i) => i + 1)
    const expected = (xv, yv) => {
      const out = new Float32Array(10)
      out[0] = out[1] = 128
      for (let c = 0; c < 2; c++) {
        for (let r = 0; r < 128; r++) {
          const xval = xv[c * 128 + r]
          const yval = yv[r]
          out[c + 2] += xval
          out[c + 4] += xval * xval
          out[c + 6] += yval
          out[c + 8] += xval * yval
        }
      }
      return Array.from(out)
    }
    const x = Tensor.empty([256], { dtype: 'float32' })
    const yData = makeY()
    const y = new Tensor(yData)
    const x0 = makeX(0)
    x.copyFrom(x0)
    const compiled = await pg.compile((tx, ty) => {
      const out = Tensor.empty([10], { dtype: 'float32' })
      return out.customKernel(tx, ty, summaryKernel)[0]
    }, [x, y])
    assertClose(await (await compiled.run([x, y])).toArray(), expected(x0, yData))
    const x1 = makeX(1)
    x.copyFrom(x1)
    assertClose(await (await compiled.run([x, y])).toArray(), expected(x1, yData))
    compiled.dispose()
  })

  await test('compile captures sym-style customKernel fused summary reductions', async () => {
    const candidates = 4
    const rows = 32
    const terms = 5
    const statsPerCandidate = 2 + 2 * terms + (terms * (terms + 1)) / 2
    function summaryKernel(out, x, y) {
      out = out.flatten(); x = x.flatten(); y = y.flatten()
      const c = pg.uop.range(candidates, 0)
      const r = pg.uop.range(rows, 1, pg.uop.AxisType.REDUCE)
      const one = pg.uop.constant(1.0)
      const yv = y.index(r)
      const termAt = (t) => x.index(c.mul(terms).add(t).mul(rows).add(r))
      const stats = [one.sum(r), yv.sum(r)]
      for (let t = 0; t < terms; t++) {
        const tv = termAt(t)
        stats.push(tv.sum(r))
        stats.push(tv.mul(yv).sum(r))
      }
      for (let i = 0; i < terms; i++) {
        const ti = termAt(i)
        for (let j = i; j < terms; j++) stats.push(ti.mul(termAt(j)).sum(r))
      }
      const stores = stats.map((s, stat) => out.index(c.add(stat * candidates)).store(s))
      return stores[0].group(...stores.slice(1)).end(c).sink()
    }
    const makeX = (shift = 0) => Float32Array.from(
      { length: candidates * terms * rows },
      (_, i) => Math.sin((i + shift) * 0.013) + Math.cos((i % 17) * 0.07) + 0.001 * i
    )
    const yData = Float32Array.from({ length: rows }, (_, i) => Math.cos(i * 0.05) - 0.25)
    const expected = (xv) => {
      const out = new Float32Array(candidates * statsPerCandidate)
      for (let c = 0; c < candidates; c++) {
        out[c] = rows
        for (let r = 0; r < rows; r++) out[c + candidates] += yData[r]
        let stat = 2
        for (let t = 0; t < terms; t++) {
          for (let r = 0; r < rows; r++) {
            const tv = xv[(c * terms + t) * rows + r]
            out[c + stat * candidates] += tv
            out[c + (stat + 1) * candidates] += tv * yData[r]
          }
          stat += 2
        }
        for (let i = 0; i < terms; i++) {
          for (let j = i; j < terms; j++) {
            for (let r = 0; r < rows; r++) {
              out[c + stat * candidates] += xv[(c * terms + i) * rows + r] * xv[(c * terms + j) * rows + r]
            }
            stat += 1
          }
        }
      }
      return Array.from(out)
    }
    const x = Tensor.empty([candidates * terms * rows], { dtype: 'float32' })
    const y = new Tensor(yData)
    const x0 = makeX(0)
    x.copyFrom(x0)
    const compiled = await pg.compile((tx, ty) => {
      const out = Tensor.empty([candidates * statsPerCandidate], { dtype: 'float32' })
      return out.customKernel(tx, ty, summaryKernel)[0]
    }, [x, y])
    assertClose(await (await compiled.run([x, y])).toArray(), expected(x0), 2e-3)
    const x1 = makeX(3)
    x.copyFrom(x1)
    assertClose(await (await compiled.run([x, y])).toArray(), expected(x1), 2e-3)
    compiled.dispose()
  })

  await test('compile captures customKernel tinygrad-style set accumulator reduction', async () => {
    function sumKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const candidates = 4
      const rows = 64
      const c = pg.uop.range(candidates, 0)
      const r = pg.uop.range(rows, 1, pg.uop.AxisType.REDUCE)
      let acc = out.index(c).set(0.0)
      acc = acc.index(c).set(acc.after(r).index(c).add(x.index(c.mul(rows).add(r))), r)
      return acc.end(c).sink(new pg.uop.KernelInfo({ name: 'custom_sum_4_64', opts_to_apply: [] }))
    }
    const xData = Float32Array.from({ length: 256 }, (_, i) => i + 1)
    const expected = []
    for (let c = 0; c < 4; c++) {
      let s = 0
      for (let r = 0; r < 64; r++) s += xData[c * 64 + r]
      expected.push(s)
    }
    const x = new Tensor(xData)
    const compiled = await pg.compile((tx) => {
      const out = Tensor.empty([4], { dtype: 'float32' })
      return out.customKernel(tx, sumKernel)[0]
    }, [x])
    assertClose(await (await compiled.run([x])).toArray(), expected)
    compiled.dispose()
  })

  await test('compile captures staged customKernel outputs feeding another customKernel', async () => {
    const candidates = 4
    const rows = 64
    function termKernel(t0, t1, x, y) {
      t0 = t0.flatten(); t1 = t1.flatten(); x = x.flatten(); y = y.flatten()
      const c = pg.uop.range(candidates, 0)
      const r = pg.uop.range(rows, 1)
      const idx = c.mul(rows).add(r)
      const xv = x.index(idx)
      const yv = y.index(r)
      const st0 = t0.index(idx).store(xv.add(yv))
      const st1 = t1.index(idx).store(xv.mul(yv))
      return st0.group(st1).end(c, r).sink(
        new pg.uop.KernelInfo({ name: 'custom_stage_terms_4_64', opts_to_apply: [] })
      )
    }
    function summaryKernel(out, t0, t1) {
      out = out.flatten(); t0 = t0.flatten(); t1 = t1.flatten()
      const c = pg.uop.range(candidates, 0)
      const r = pg.uop.range(rows, 1, pg.uop.AxisType.REDUCE)
      const idx = c.mul(rows).add(r)
      const s0 = t0.index(idx).sum(r)
      const s1 = t1.index(idx).sum(r)
      const st0 = out.index(c).store(s0)
      const st1 = out.index(c.add(candidates)).store(s1)
      return st0.group(st1).end(c).sink(
        new pg.uop.KernelInfo({ name: 'custom_stage_summary_4_64', opts_to_apply: [] })
      )
    }
    const xData = Float32Array.from({ length: candidates * rows }, (_, i) => Math.sin(i * 0.01) + i * 0.001)
    const yData = Float32Array.from({ length: rows }, (_, i) => Math.cos(i * 0.02) - 0.25)
    const expected = new Float32Array(candidates * 2)
    for (let c = 0; c < candidates; c++) {
      for (let r = 0; r < rows; r++) {
        const xv = xData[c * rows + r]
        const yv = yData[r]
        expected[c] += xv + yv
        expected[c + candidates] += xv * yv
      }
    }
    const x = new Tensor(xData)
    const y = new Tensor(yData)
    const compiled = await pg.compile((tx, ty) => {
      const t0 = Tensor.empty([candidates * rows], { dtype: 'float32' })
      const t1 = Tensor.empty([candidates * rows], { dtype: 'float32' })
      const staged = t0.customKernel(t1, tx, ty, termKernel)
      const out = Tensor.empty([candidates * 2], { dtype: 'float32' })
      return out.customKernel(staged[0], staged[1], summaryKernel)[0]
    }, [x, y])
    assertClose(await (await compiled.run([x, y])).toArray(), Array.from(expected), 1e-4)
    compiled.dispose()
  })

  await test('compiled customKernel consumer reads producer output after readback', async () => {
    const n = 1024
    function producerKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const i = pg.uop.range(out.numel(), 0)
      return out.index(i).store(x.index(i).mul(2).add(1)).end(i).sink(
        new pg.uop.KernelInfo({ name: 'custom_producer_readback_rebind', opts_to_apply: [] })
      )
    }
    function consumerKernel(out, y) {
      out = out.flatten(); y = y.flatten()
      const r = pg.uop.range(n, 0, pg.uop.AxisType.REDUCE)
      return out.index(0).store(y.index(r).sum(r)).sink(
        new pg.uop.KernelInfo({ name: 'custom_consumer_readback_rebind', opts_to_apply: [] })
      )
    }
    const x0 = Float32Array.from({ length: n }, (_, i) => i / 17)
    const x1 = Float32Array.from({ length: n }, (_, i) => 10 + i / 11)
    const x = new Tensor(x0, { dtype: 'float32' })
    const producer = await pg.compile((tx) => {
      const out = Tensor.empty([n], { dtype: 'float32' })
      return out.customKernel(tx, producerKernel)[0]
    }, [x])

    const firstY = await producer.run([x])
    await firstY.realize()
    const consumer = await pg.compile((ty) => {
      const out = Tensor.empty([1], { dtype: 'float32' })
      return out.customKernel(ty, consumerKernel)[0]
    }, [firstY])

    const first = await consumer.run([firstY])
    assertClose(await first.toArray(), [Array.from(x0).reduce((s, v) => s + v * 2 + 1, 0)], 1e-2)

    x.copyFrom(x1)
    const secondY = await producer.run([x])
    const second = await consumer.run([secondY])
    assertClose(await second.toArray(), [Array.from(x1).reduce((s, v) => s + v * 2 + 1, 0)], 1e-2)

    producer.dispose()
    consumer.dispose()
  })

  await test('runtime exposes conservative stats and capability checks', async () => {
    assert(typeof pg.stats === 'function', 'runtime should expose stats()')
    assert(typeof pg.canRun === 'function', 'runtime should expose canRun()')
    const stats = pg.stats()
    assert(stats.core === pg.core, 'runtime stats should include core')
    assert(stats.device === pg.device, 'runtime stats should include device')
    assert(stats.coreStats && typeof stats.coreStats.launchCount === 'number', 'runtime stats should include core counters')
    assert(typeof stats.coreStats.globalOps === 'number', 'runtime stats should include globalOps')
    assert(typeof stats.coreStats.globalMem === 'number', 'runtime stats should include globalMem')
    assert(typeof stats.coreStats.timeSumS === 'number', 'runtime stats should include timeSumS')
    assert(typeof stats.coreStats.kernelCount === 'number', 'runtime stats should include kernelCount')
    assert(typeof stats.coreStats.memUsed === 'number', 'runtime stats should include memUsed')
    assert(stats.jit && typeof stats.jit.liveCount === 'number', 'runtime stats should include jit live count')
    assert(typeof pg.resetCounters === 'function', 'runtime should expose resetCounters()')
    const liveMem = stats.coreStats.memUsed
    pg.resetCounters()
    const before = pg.stats().coreStats
    assert(before.globalOps === 0 && before.globalMem === 0 && before.kernelCount === 0,
      'resetCounters should clear execution counters')
    assert(before.memUsed === liveMem, 'resetCounters should preserve live memory')
    const x = Tensor.empty([3], { dtype: 'float32' })
    x.copyFrom(new Float32Array([1, 2, 3]))
    const t = await x.add(1).realize()
    assertClose(await t.toArray(), [2, 3, 4])
    const after = pg.stats().coreStats
    assert(after.bufferWriteBytes >= before.bufferWriteBytes + 12, 'stats should count host writes')
    assert(after.bufferReadBytes >= before.bufferReadBytes + 12, 'stats should count host reads')
    assert(after.launchCount >= before.launchCount + 1, 'stats should count backend launches')
    // Pinned tinygrad CPU:X86 computes estimates after ISA register allocation;
    // this exact add is therefore 0/0/1 there while value backends are 3/24/1.
    const expectedOps = pg.device === 'x86' ? 0 : 3
    const expectedMem = pg.device === 'x86' ? 0 : 24
    assert(after.globalOps === expectedOps, `expected ${expectedOps} global ops, got ${after.globalOps}`)
    assert(after.globalMem === expectedMem, `expected ${expectedMem} global memory bytes, got ${after.globalMem}`)
    assert(after.kernelCount === 1, `expected one tracked call, got ${after.kernelCount}`)
    assert(pg.canRun({ dtype: 'float32' }), 'float32 should be supported by every current runtime')
    if (pg.caps.f64 === false) {
      assert(!pg.canRun({ dtype: 'float64' }), 'canRun should reject f64 when caps.f64 is false')
    }
    assert(pg.canRun({ op: 'add', dtype: 'float32', shape: [4] }), 'canRun should probe add')
    assert(
      pg.canRun({ op: 'matmul', dtype: 'float32', shapes: [[2, 3], [3, 4]] }),
      'canRun should probe matmul shapes'
    )
    assert(pg.canRun({ op: 'gather', dtype: 'float32', shape: [2, 3] }), 'canRun should probe gather')
    assert(pg.canRun({ op: 'sort', dtype: 'float32', shape: [2, 3] }), 'canRun should probe sort')
    assert(pg.canRun({ op: 'argsort', dtype: 'float32', shape: [2, 3] }), 'canRun should probe argsort')
    assert(pg.canRun({ op: 'topk', dtype: 'float32', shape: [2, 3] }), 'canRun should probe topk')
    let threw = false
    try {
      pg.canRun({ shape: [4], dtype: 'float32' })
    } catch (e) {
      threw = String(e.message || e).includes('require an op')
    }
    assert(threw, 'shape-only canRun queries should fail explicitly')
  })

  await test('runtime compile wrapper warms capture and replays', async () => {
    assert(typeof pg.compile === 'function', 'runtime should expose pg.compile')
    const sample = new Tensor(new Float32Array([1, 2, 3]))
    const compiled = await pg.compile((x) => x.add(1), [sample])
    assert(compiled.scheduleCount === 1, `expected one captured schedule, got ${compiled.scheduleCount}`)
    const out = await compiled.run([new Tensor(new Float32Array([10, 20, 30]))])
    assertClose(await out.toArray(), [11, 21, 31])
    const stats = compiled.stats()
    assert(stats.captureRuns === 2, 'compile should perform two setup runs')
    assert(stats.runCount === 1, 'compiled run should update runCount')
    compiled.dispose()
  })

  await test('toTypedArray aliases flat typed readback', async () => {
    const t = new Tensor([1, 2, 3])
    const arr = await t.toTypedArray()
    assert(arr instanceof Float32Array, `expected Float32Array, got ${arr.constructor.name}`)
    assertClose(arr, [1, 2, 3])
  })

  await test('toTypedArrays batches flat typed readback', async () => {
    const t = new Tensor([1, 2, 3])
    const outs = await Tensor.toTypedArrays(t.add(1), t.mul(2))
    assert(Array.isArray(outs) && outs.length === 2, 'expected two output arrays')
    assert(outs[0] instanceof Float32Array, `expected Float32Array, got ${outs[0].constructor.name}`)
    assertClose(outs[0], [2, 3, 4])
    assertClose(outs[1], [2, 4, 6])

    const i32 = new Tensor(new Int32Array([1, 2, 3]), { dtype: 'int32' })
    const empty = Tensor.zeros([0])
    assertShape(empty.shape, [0])
    const more = await Tensor.toTypedArrays([i32, empty])
    assert(more[0] instanceof Int32Array, `expected Int32Array, got ${more[0].constructor.name}`)
    assert(more[1] instanceof Float32Array && more[1].length === 0, 'expected empty Float32Array')
    assertClose(more[0], [1, 2, 3])
  })

  await test('empty rejects named tensor keyword like tinygrad', async () => {
    let threw = false
    try {
      Tensor.empty([2, 3], { name: 'z' })
    } catch (_) {
      threw = true
    }
    assert(threw, 'expected Tensor.empty(..., {name}) to reject')
  })

  // -- Elementwise --
  console.log('\n-- Elementwise --')

  await test('add', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([4, 5, 6])
    assertClose(await a.add(b).toArray(), [5, 7, 9])
  })

  await test('sub', async () => {
    const a = new Tensor([10, 20, 30])
    const b = new Tensor([1, 2, 3])
    assertClose(await a.sub(b).toArray(), [9, 18, 27])
  })

  await test('mul', async () => {
    const a = new Tensor([2, 3, 4])
    const b = new Tensor([5, 6, 7])
    assertClose(await a.mul(b).toArray(), [10, 18, 28])
  })

  await test('div', async () => {
    const a = new Tensor([10, 20, 30])
    const b = new Tensor([2, 4, 5])
    assertClose(await a.div(b).toArray(), [5, 5, 6])
  })

  await test('neg', async () => {
    const a = new Tensor([1, -2, 3])
    assertClose(await a.neg().toArray(), [-1, 2, -3])
  })

  await test('scalar add', async () => {
    const a = new Tensor([1, 2, 3])
    assertClose(await a.add(10).toArray(), [11, 12, 13])
  })

  await test('scalar mul', async () => {
    const a = new Tensor([1, 2, 3])
    assertClose(await a.mul(3).toArray(), [3, 6, 9])
  })

  await test('chain: (a + 2) * b', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([4, 5, 6])
    assertClose(await a.add(2).mul(b).toArray(), [12, 20, 30])
  })

  // -- Composed math --
  console.log('\n-- Composed math --')

  await test('exp', async () => {
    const a = new Tensor([0, 1])
    const arr = await a.exp().toArray()
    assertClose(arr, [1, Math.E], 1e-3)
  })

  await test('log', async () => {
    const a = new Tensor([1, Math.E])
    const arr = await a.log().toArray()
    assertClose(arr, [0, 1], 1e-3)
  })

  await test('sqrt', async () => {
    const a = new Tensor([1, 4, 9, 16])
    assertClose(await a.sqrt().toArray(), [1, 2, 3, 4])
  })

  await test('abs', async () => {
    const a = new Tensor([-1, 2, -3])
    assertClose(await a.abs().toArray(), [1, 2, 3])
  })

  await test('square', async () => {
    const a = new Tensor([2, 3, 4])
    assertClose(await a.square().toArray(), [4, 9, 16])
  })

  await test('sigmoid', async () => {
    const a = new Tensor([0])
    const arr = await a.sigmoid().toArray()
    assertClose(arr, [0.5], 1e-3)
  })

  // -- Activations --
  console.log('\n-- Activations --')

  await test('relu', async () => {
    const a = new Tensor([-1, 0, 1, 2])
    assertClose(await a.relu().toArray(), [0, 0, 1, 2])
  })

  await test('gelu', async () => {
    const a = new Tensor([0])
    const arr = await a.gelu().toArray()
    assert(Math.abs(arr[0]) < 0.01, `gelu(0) should be ~0, got ${arr[0]}`)
  })

  await test('silu', async () => {
    const a = new Tensor([0])
    const arr = await a.silu().toArray()
    assert(Math.abs(arr[0]) < 0.01, `silu(0) should be ~0, got ${arr[0]}`)
  })

  // -- Comparisons --
  console.log('\n-- Comparisons --')

  await test('eq', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([1, 5, 3])
    assertClose(await a.eq(b).toArray(), [1, 0, 1])
  })

  await test('gt', async () => {
    const a = new Tensor([1, 5, 3])
    const b = new Tensor([2, 3, 3])
    assertClose(await a.gt(b).toArray(), [0, 1, 0])
  })

  await test('where with optimized-away middle input keeps param slots', async () => {
    const idx = Tensor.arange(2)
    const out = idx.ge(0).where(new Tensor([1, 3]), new Tensor([7, 8]))
    assertClose(await out.toArray(), [1, 3])
  })

  await test('maximum', async () => {
    const a = new Tensor([1, 5, 3])
    const b = new Tensor([2, 3, 4])
    assertClose(await a.maximum(b).toArray(), [2, 5, 4])
  })

  await test('clamp', async () => {
    const a = new Tensor([1, 5, 3])
    assertClose(await a.clamp(2, 4).toArray(), [2, 4, 3])
  })

  // -- Movement --
  console.log('\n-- Movement --')

  await test('reshape', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6]).reshape(2, 3)
    assertShape(t.shape, [2, 3])
    assertClose(await t.toArray(), [1, 2, 3, 4, 5, 6])
  })

  await test('reshape inference validates cardinality', async () => {
    assertShape(Tensor.empty(6).reshape(2, -1).shape, [2, 3])
    assertShape(Tensor.empty(0).reshape(-1, 3).shape, [0, 3])
    assertShape(Tensor.empty(0).reshape(1, 0).shape, [1, 0])
    assertShape(Tensor.empty(2, 3).reshape(null, 3).shape, [2, 3])

    for (const [shape, target, message] of [
      [[3072], [-1, 3073], 'size mismatch'],
      [[5], [2, -1], 'size mismatch'],
      [[6], [-1, -1], 'only one dimension can be inferred'],
      [[0], [0, -1], 'division by zero']
    ]) {
      let error = null
      try { Tensor.empty(...shape).reshape(...target) } catch (e) { error = e }
      assert(error && error.message.includes(message), `expected ${message}, got ${error && error.message}`)
    }
  })

  await test('flip', async () => {
    const t = new Tensor([1, 2, 3])
    assertClose(await t.flip(0).toArray(), [3, 2, 1])
  })

  await test('permute', async () => {
    const t = new Tensor([[1, 2, 3], [4, 5, 6]])  // [2,3]
    const p = t.permute(1, 0)  // [3,2]
    assertShape(p.shape, [3, 2])
    assertClose(await p.toArray(), [1, 4, 2, 5, 3, 6])
  })

  await test('expand negative and null keep original dim like tinygrad', async () => {
    const x = Tensor.arange(2, { dtype: 'int32' }).reshape(2, 1, 1, 1)
    const y = x.expand(-1, 3, 4, null)
    assertShape(y.shape, [2, 3, 4, 1])
    assertClose(Array.from(await y.toArray()).slice(0, 4), [0, 0, 0, 0])
    assertClose(Array.from(await y.toArray()).slice(12, 16), [1, 1, 1, 1])

    const img = Tensor.arange(2 * 3 * 34 * 34).reshape(2, 3, 34, 34)
    const lowX = Tensor.randint(2, { low: 0, high: 2 }).reshape(2, 1, 1, 1)
    const idxX = Tensor.arange(32, { dtype: 'int32' }).reshape(1, 1, 1, 32)
    const cropIdx = lowX.add(idxX).expand(-1, 3, img.shape[2], -1)
    assertShape(cropIdx.shape, [2, 3, 34, 32])
    assertShape(img.gather(-1, cropIdx).shape, [2, 3, 34, 32])
  })

  await test('pad', async () => {
    const t = new Tensor([1, 2, 3])
    const p = t.pad([[1, 1]])
    assertShape(p.shape, [5])
    assertClose(await p.toArray(), [0, 1, 2, 3, 0])

    const x = Tensor.arange(9, { dtype: 'float32' }).reshape(1, 1, 3, 3)
    const flat = x.pad([1, 0, 0, 1])
    assertShape(flat.shape, [1, 1, 4, 4])
    assertClose(await flat.toArray(), [0, 0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 0])

    const moved = Tensor.arange(12).reshape(3, 4).pad([[-1, 2], [1, -1]])
    assertShape(moved.shape, [4, 4])
    assertClose(await moved.toArray(), [0, 4, 5, 6, 0, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0])
  })

  await test('pad readback preserves non-float dtype', async () => {
    const p = new Tensor([1, 2, 3], { dtype: 'int32' }).pad([[1, 1]])
    assert(p.dtype === 'int32', `expected int32, got ${p.dtype}`)
    const arr = await p.toArray()
    assert(arr.constructor.name === 'Int32Array', `expected Int32Array, got ${arr.constructor.name}`)
    assertClose(arr, [0, 1, 2, 3, 0])
  })

  await test('pad readback preserves byte dtype', async () => {
    const p = new Tensor([1, 0, 1], { dtype: 'uint8' }).pad([[1, 1]])
    assert(p.dtype === 'uint8', `expected uint8, got ${p.dtype}`)
    const arr = await p.toArray()
    assert(arr instanceof Uint8Array, `expected Uint8Array-compatible view, got ${arr.constructor.name}`)
    assertClose(arr, [0, 1, 0, 1, 0])
  })

  await test('gather matches tinygrad probe', async () => {
    const t = new Tensor([[1, 2], [3, 4]])
    const idx = new Tensor(new Int32Array([0, 0, 1, 0]), { dtype: 'int32' }).reshape(2, 2)
    const out = t.gather(1, idx)
    assertShape(out.shape, [2, 2])
    assertClose(await out.toArray(), [1, 1, 4, 3])
    const x3 = Tensor.arange(24).reshape(2, 3, 4)
    const idx3 = new Tensor(new Int32Array([0, 2, 1, 0, 2, 1, 0, 2]), { dtype: 'int32' }).reshape(2, 2, 2)
    const out3 = x3.gather(1, idx3)
    assertShape(out3.shape, [2, 2, 2])
    assertClose(await out3.toArray(), [0, 9, 4, 1, 20, 17, 12, 21])
  })

  await test('tensor row indexing matches tinygrad probe', async () => {
    const idx = new Tensor(new Int32Array([-1, 0, 2]), { dtype: 'int32' })
    const out = Tensor.arange(12).reshape(3, 4).getitem(idx)
    assertShape(out.shape, [3, 4])
    assertClose(await out.toArray(), [8, 9, 10, 11, 0, 1, 2, 3, 8, 9, 10, 11])
  })

  await test('scatter matches tinygrad probe', async () => {
    const base = Tensor.zeros(3, 5)
    const idx0 = new Tensor(new Int32Array([0, 1, 2, 0]), { dtype: 'int32' }).reshape(1, 4)
    const src0 = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(2, 5)
    assertClose(await base.scatter(0, idx0, src0).toArray(), [1, 0, 0, 4, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0])

    const idx1 = new Tensor(new Int32Array([0, 1, 2, 0, 1, 4, 2, 3, 4]), { dtype: 'int32' }).reshape(3, 3)
    const src1 = new Tensor([1, 2, 3, 6, 7, 8, 9, 10, 11]).reshape(3, 3)
    assertClose(await base.scatter(1, idx1, src1).toArray(), [1, 2, 3, 0, 0, 6, 7, 0, 0, 8, 0, 0, 9, 10, 11])

    const dupIdx = new Tensor(new Int32Array([1, 1, 2]), { dtype: 'int32' }).reshape(1, 3)
    const dupSrc = new Tensor([7, 9, 8]).reshape(1, 3)
    assertClose(await new Tensor([[0, 0, 0, 0]]).scatter(1, dupIdx, dupSrc).toArray(), [0, 9, 8, 0])

    const scalarIdx = new Tensor(new Int32Array([2, 3]), { dtype: 'int32' }).reshape(2, 1)
    assertClose(await Tensor.full([2, 4], 2).scatter(1, scalarIdx, 1.23, 'add').toArray(), [2, 2, 3.23, 2, 2, 2, 2, 3.23])
    assertClose(await Tensor.full([2, 4], 2).scatter(1, scalarIdx, 1.23, 'multiply').toArray(), [2, 2, 2.46, 2, 2, 2, 2, 2.46])

    let threw = false
    try { base.scatter(1, idx1, src1, 'sum') } catch (e) { threw = true }
    assert(threw, 'expected invalid scatter reduce string to throw')
    threw = false
    try { base.scatter(1, idx1, src1, 'add') } catch (e) { threw = true }
    assert(threw, 'expected tensor src with scatter reduce arg to throw')
  })

  await test('scatterReduce matches tinygrad probe', async () => {
    const base = new Tensor([[1, 2, 3, 4, 5]])
    const idx = new Tensor(new Int32Array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), { dtype: 'int32' }).reshape(1, 10)
    const src = new Tensor([[1, 6, 2, 7, 3, 8, 4, 9, 5, 10]])
    assertClose(await base.scatterReduce(1, idx, src, 'sum').toArray(), [8, 11, 14, 17, 20])
    assertClose(await base.scatter_reduce(1, idx, src, 'prod').toArray(), [6, 28, 72, 144, 250])
    assertClose(await base.scatterReduce(1, idx, src, 'mean', false).toArray(), [3.5, 4.5, 5.5, 6.5, 7.5])
    const extremeBase = new Tensor([[-10, 20, 0, 5, 10]])
    assertClose(await extremeBase.scatterReduce(1, idx, src, 'amax').toArray(), [6, 20, 8, 9, 10])
    assertClose(await extremeBase.scatterReduce(1, idx, src, 'amin').toArray(), [-10, 2, 0, 4, 5])
    let threw = false
    try { base.scatterReduce(1, idx, src, 'max') } catch (e) { threw = true }
    assert(threw, 'expected invalid scatterReduce reduction to throw')
  })

  // -- Step slicing --
  console.log('\n-- Step slicing --')

  await test('step2 1d', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6, 7, 8])
    const r = t.getitem({ start: 0, stop: 8, step: 2 })
    assertShape(r.shape, [4])
    assertClose(await r.toArray(), [1, 3, 5, 7])
  })

  await test('step3 1d', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    const r = t.getitem({ start: 0, stop: 9, step: 3 })
    assertShape(r.shape, [3])
    assertClose(await r.toArray(), [1, 4, 7])
  })

  await test('step2 with start stop', async () => {
    const t = new Tensor([0, 1, 2, 3, 4, 5, 6, 7])
    const r = t.getitem({ start: 1, stop: 7, step: 2 })
    assertShape(r.shape, [3])
    assertClose(await r.toArray(), [1, 3, 5])
  })

  await test('step non-divisible', async () => {
    const t = new Tensor([0, 1, 2, 3, 4, 5, 6])
    const r = t.getitem({ start: 0, stop: 7, step: 3 })
    assertShape(r.shape, [3])
    assertClose(await r.toArray(), [0, 3, 6])
  })

  await test('negative step (reverse)', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6])
    const r = t.getitem({ step: -1 })
    assertClose(await r.toArray(), [6, 5, 4, 3, 2, 1])
  })

  await test('negative step2', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6])
    const r = t.getitem({ step: -2 })
    assertShape(r.shape, [3])
    assertClose(await r.toArray(), [6, 4, 2])
  })

  await test('step 2d axis0', async () => {
    const t = new Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    const r = t.getitem({ start: 0, stop: 4, step: 2 })
    assertShape(r.shape, [2, 3])
    assertClose(await r.toArray(), [0, 1, 2, 6, 7, 8])
  })

  await test('step 2d both axes', async () => {
    const t = new Tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    const r = t.getitem({ start: 0, stop: 4, step: 2 }, { start: 0, stop: 4, step: 2 })
    assertShape(r.shape, [2, 2])
    assertClose(await r.toArray(), [0, 2, 8, 10])
  })

  // -- Cast --
  console.log('\n-- Cast --')

  await testIf(supportsF64, 'cast float32 to float64', async () => {
    const t = new Tensor([1, 2, 3])
    const r = t.cast('float64')
    assert(r.dtype === 'float64', `expected float64, got ${r.dtype}`)
    const arr = await r.toArray()
    console.log('DEBUG cast f32->f64', Array.from(arr))
    assertClose(arr, [1, 2, 3])
  })

  await testIf(supportsF64, 'cast float64 to float32', async () => {
    const t = new Tensor([1.5, 2.5, 3.5], { dtype: 'float64' })
    const r = t.cast('float32')
    assert(r.dtype === 'float32', `expected float32, got ${r.dtype}`)
    const arr = await r.toArray()
    console.log('DEBUG cast f64->f32', Array.from(arr))
    assertClose(arr, [1.5, 2.5, 3.5])
  })

  await test('cast no-op same dtype', async () => {
    const t = new Tensor([1, 2, 3])
    const r = t.cast('float32')
    assertClose(await r.toArray(), [1, 2, 3])
  })

  await testIf(supportsF16, 'half and double convenience', async () => {
    const t = new Tensor([1, 2, 3])
    const h = t.half()
    assert(h.dtype === 'float16', `expected float16, got ${h.dtype}`)
    assertClose(await h.toArray(), [1, 2, 3])
    if (supportsF64) {
      const d = t.double()
      assert(d.dtype === 'float64', `expected float64, got ${d.dtype}`)
      assertClose(await d.toArray(), [1, 2, 3])
    }
  })

  await testIf(supportsF64, 'cast then compute', async () => {
    const t = new Tensor([1, 2, 3]).cast('float64')
    const r = t.add(new Tensor([10, 20, 30], { dtype: 'float64' }))
    assertClose(await r.toArray(), [11, 22, 33])
  })

  // -- Triu/Tril --
  console.log('\n-- Triu/Tril --')

  await test('triu 2d', async () => {
    const t = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    const r = t.triu()
    const arr = await r.toArray()
    console.log('DEBUG triu', Array.from(arr))
    assertClose(arr, [1, 2, 3, 0, 5, 6, 0, 0, 9])
  })

  await test('tril 2d', async () => {
    const t = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    const r = t.tril()
    const arr = await r.toArray()
    console.log('DEBUG tril', Array.from(arr))
    assertClose(arr, [1, 0, 0, 4, 5, 0, 7, 8, 9])
  })

  await test('triu with diagonal', async () => {
    const t = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    const r = t.triu(1)
    const arr = await r.toArray()
    console.log('DEBUG triu diag', Array.from(arr))
    assertClose(arr, [0, 2, 3, 0, 0, 6, 0, 0, 0])
  })

  await test('triu/tril batched last two dims', async () => {
    const data = [
      [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
      [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    ]
    const t = new Tensor(data)
    const upper = t.triu()
    const lower = t.tril(1)
    assertShape(upper.shape, [2, 3, 3])
    assertShape(lower.shape, [2, 3, 3])
    assertClose(await upper.toArray(), [1, 2, 3, 0, 5, 6, 0, 0, 9, 10, 11, 12, 0, 14, 15, 0, 0, 18])
    assertClose(await lower.toArray(), [1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18])

    const z = Tensor.zeros(5, 0, 3)
    assertShape(z.triu().shape, [5, 0, 3])
    assertShape(z.tril().shape, [5, 0, 3])
    assertClose(await z.triu().toArray(), [])
  })

  // -- Reduction --
  console.log('\n-- Reduction --')

  await test('sum all', async () => {
    const t = new Tensor([1, 2, 3])
    const v = await t.sum().item()
    assert(Math.abs(v - 6) < 1e-4, `Expected 6, got ${v}`)
  })

  await test('sum axis', async () => {
    const t = new Tensor([[1, 2], [3, 4]])
    const s = t.sum(1)
    assertShape(s.shape, [2])
    assertClose(await s.toArray(), [3, 7])
  })

  await test('mean', async () => {
    const t = new Tensor([2, 4, 6])
    const v = await t.mean().item()
    assert(Math.abs(v - 4) < 1e-4, `Expected 4, got ${v}`)
  })

  await test('max', async () => {
    const t = new Tensor([[1, 5], [3, 2]])
    const m = t.max(1)
    assertClose(await m.toArray(), [5, 3])
  })

  await test('argmax matches tinygrad probe', async () => {
    const t = new Tensor([[1.2, 0.5, 1.2], [2.2, 1.9, 0.0]])
    assertClose(await t.argmax(1).toArray(), [0, 0])
    assertShape(t.argmax(1, true).shape, [2, 1])
    assertClose(await t.argmax(1, true).toArray(), [0, 0])
    const flat = await t.argmax().item()
    assert(flat === 3, `expected flattened argmax 3, got ${flat}`)
  })

  await test('sort argsort topk match tinygrad probe', async () => {
    const x = new Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]])
    let pair = x.sort(1, false)
    assertShape(pair[0].shape, [2, 5])
    assertShape(pair[1].shape, [2, 5])
    assertClose(await pair[0].toArray(), [0.1, 0.5, 1.2, 2.1, 3.4, 0.3, 0.8, 1.9, 2.2, 4.5])
    assertClose(await pair[1].toArray(), [0, 1, 2, 4, 3, 2, 4, 1, 0, 3])

    pair = x.sort(1, true)
    assertClose(await pair[0].toArray(), [3.4, 2.1, 1.2, 0.5, 0.1, 4.5, 2.2, 1.9, 0.8, 0.3])
    assertClose(await pair[1].toArray(), [3, 4, 2, 1, 0, 3, 0, 1, 4, 2])

    pair = x.topk(2, 1)
    assertShape(pair[0].shape, [2, 2])
    assertShape(pair[1].shape, [2, 2])
    assertClose(await pair[0].toArray(), [3.4, 2.1, 4.5, 2.2])
    assertClose(await pair[1].toArray(), [3, 4, 3, 0])

    pair = x.topk(2, 1, false)
    assertClose(await pair[0].toArray(), [0.1, 0.5, 0.3, 0.8])
    assertClose(await pair[1].toArray(), [0, 1, 2, 4])

    const t = new Tensor([[2, 3, 4, 1], [1, 4, 3, 2]])
    assertClose(await t.argsort().toArray(), [3, 0, 1, 2, 0, 3, 2, 1])
  })

  await test('topk tie order and errors match tinygrad probe', async () => {
    const tie = new Tensor([[1.0, 1.0, 0.0, 1.0]])
    const pair = tie.topk(3, 1)
    assertClose(await pair[0].toArray(), [1.0, 1.0, 1.0])
    assertClose(await pair[1].toArray(), [0, 1, 3])

    let threw = false
    try {
      new Tensor([[0.1, 0.2]]).topk(6, 1)
    } catch (_) {
      threw = true
    }
    assert(threw, 'expected topk k out of range to throw')

    threw = false
    try {
      new Tensor([[0.1, 0.2]]).topk(1, 1, true, false)
    } catch (_) {
      threw = true
    }
    assert(threw, 'expected topk sorted_=false to throw')
  })

  await test('softmax', async () => {
    const t = new Tensor([1, 2, 3])
    const arr = await (await t.softmax()).toArray()
    const sum = arr[0] + arr[1] + arr[2]
    assert(Math.abs(sum - 1.0) < 1e-4, `Softmax sum should be 1, got ${sum}`)
  })

  await test('fusion fuzzer smoke patterns', async () => {
    const movement = new Tensor([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11]
    ]).pad([[1, 0], [0, 1]]).shrink([[1, 4], [1, 5]])
      .mul(0.25).add(1).sum(1)
    assertShape(movement.shape, [3])
    assertClose(await movement.toArray(), [5.5, 8.5, 11.5], 2e-4)

    const x = new Tensor([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
      [12, 13, 14, 15]
    ])
    const whereReduce = Tensor.full([4, 4], 7).gt(x).where(x, Tensor.full([4, 4], -2)).sum(0)
    assertShape(whereReduce.shape, [4])
    assertClose(await whereReduce.toArray(), [0, 2, 4, -3], 2e-4)

    const base = new Tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]
    ])
    const noBarrier = base.add(1)
    assertClose(await noBarrier.square().add(noBarrier).sum(0).toArray(), [108, 144, 186], 2e-4)

    const barrier = base.add(1)
    await barrier.realize()
    assertClose(await barrier.square().add(barrier).sum(0).toArray(), [108, 144, 186], 2e-4)

    const [q, r] = new Tensor([[1, 2], [-1, 3], [0.5, 4]]).qr()
    assertClose(await q.dot(r).toArray(), [1, 2, -1, 3, 0.5, 4], 2e-3)

    const [qb, rb] = new Tensor([[1, 2], [-1, 3], [0.5, 4]]).qr()
    await qb.realize(rb)
    assertClose(await qb.dot(rb).toArray(), [1, 2, -1, 3, 0.5, 4], 2e-3)

    const chol = new Tensor([[4, 2], [2, 5]]).cholesky()
    assertClose(
      await chol.choleskySolve(new Tensor([[1, 2], [3, 4]])).toArray(),
      [-0.0625, 0.125, 0.625, 0.75],
      3e-4
    )

    const cholBarrier = new Tensor([[4, 2], [2, 5]]).cholesky()
    await cholBarrier.realize()
    assertClose(
      await cholBarrier.choleskySolve(new Tensor([[1, 2], [3, 4]])).toArray(),
      [-0.0625, 0.125, 0.625, 0.75],
      3e-4
    )

    assertClose(
      await new Tensor([[0, 2], [1, 3]]).solve(new Tensor([4, 5])).toArray(),
      [-1, 2],
      4e-4
    )

    assertClose(
      await new Tensor([[1, 0], [1, 1], [1, 2]]).lstsq(new Tensor([1, 2, 3])).toArray(),
      [1, 1],
      6e-4
    )
  })

  await test('matmul', async () => {
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([[5, 6], [7, 8]])
    assertClose(await a.dot(b).toArray(), [19, 22, 43, 50])
  })

  await test('matmul shape mismatch throws', async () => {
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([[1, 2, 3]])
    let ok = false
    try {
      const c = a.dot(b)
      ok = c.shape.length === 0  // native: returns empty-shape result
    } catch (e) { ok = true }    // wasm: throws immediately
    assert(ok, 'expected dot to fail on shape mismatch')
  })

  await test('matmul broadcast batch', async () => {
    const a = new Tensor([
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]]
    ])
    const b = new Tensor([
      [[1, 10], [100, 1000]]
    ])
    const out = a.dot(b)
    assertShape(out.shape, [2, 2, 2])
    assertClose(await out.toArray(), [201, 2010, 403, 4030, 605, 6050, 807, 8070])
  })

  await test('matmul broadcast mismatch throws', async () => {
    const a = new Tensor([
      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    ])
    const b = new Tensor(new Array(5).fill(0).map(() => [
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0]
    ]))
    let ok = false
    try {
      const c = a.dot(b)
      ok = c.shape.length === 0
    } catch (e) { ok = true }
    assert(ok, 'expected dot to fail on broadcast-mismatched shapes')
  })

  await test('qr matches tinygrad probe', async () => {
    const cases = [
      { arr: [[1, 2], [3, 4]], q: [2, 2], r: [2, 2], flat: [1, 2, 3, 4] },
      { arr: [[1, 2], [3, 4], [5, 6]], q: [3, 3], r: [3, 2], flat: [1, 2, 3, 4, 5, 6] },
      { arr: [[1, 2, 3], [4, 5, 6]], q: [2, 2], r: [2, 3], flat: [1, 2, 3, 4, 5, 6] },
      { arr: [[0, 1], [0, 2]], q: [2, 2], r: [2, 2], flat: [0, 1, 0, 2] },
      {
        arr: [[[1, 2], [3, 4]], [[2, 0], [0, 2]]],
        q: [2, 2, 2], r: [2, 2, 2], flat: [1, 2, 3, 4, 2, 0, 0, 2]
      },
      {
        arr: [[[1, 2], [3, 4], [5, 6]], [[2, 1], [0, 3], [4, 5]]],
        q: [2, 3, 3], r: [2, 3, 2], flat: [1, 2, 3, 4, 5, 6, 2, 1, 0, 3, 4, 5]
      },
      {
        arr: [[[1, 2, 3], [4, 5, 6]], [[2, 1, 0], [0, 3, 4]]],
        q: [2, 2, 2], r: [2, 2, 3], flat: [1, 2, 3, 4, 5, 6, 2, 1, 0, 0, 3, 4]
      },
    ]
    for (const c of cases) {
      const pair = new Tensor(c.arr).qr()
      assertShape(pair[0].shape, c.q)
      assertShape(pair[1].shape, c.r)
      assertClose(await pair[0].dot(pair[1]).toArray(), c.flat, 2e-3)
      const qVals = Array.from(await pair[0].toArray())
      const rVals = Array.from(await pair[1].toArray())
      const vals = qVals.concat(rVals)
      for (const v of vals) assert(Number.isFinite(v), `expected finite QR value, got ${v}`)
    }
  })

  await test('qr reduced and r modes match reference shapes', async () => {
    const cases = [
      { arr: [[1, 2], [3, 4], [5, 6]], q: [3, 2], r: [2, 2], flat: [1, 2, 3, 4, 5, 6] },
      { arr: [[1, 2, 3], [4, 5, 6]], q: [2, 2], r: [2, 3], flat: [1, 2, 3, 4, 5, 6] },
      {
        arr: [[[1, 2], [3, 4], [5, 6]], [[2, 1], [0, 3], [4, 5]]],
        q: [2, 3, 2], r: [2, 2, 2], flat: [1, 2, 3, 4, 5, 6, 2, 1, 0, 3, 4, 5]
      },
    ]
    for (const c of cases) {
      const pair = new Tensor(c.arr).qr('reduced')
      assertShape(pair[0].shape, c.q)
      assertShape(pair[1].shape, c.r)
      assertClose(await pair[0].dot(pair[1]).toArray(), c.flat, 2e-3)
      const rOnly = new Tensor(c.arr).qr('r')
      assertShape(rOnly.shape, c.r)
    }
    let ok = false
    try { new Tensor([[1, 2], [3, 4]]).qr('raw') } catch (e) { ok = true }
    assert(ok, 'expected invalid QR mode to fail')
  })

  await test('triangularSolve matches numpy torch probe', async () => {
    const lower = [[2, 0, 0], [1, 3, 0], [-2, 0.5, 4]]
    const upper = [[2, -1, 0.5], [0, 3, 2], [0, 0, 4]]
    const lowerUnit = [[5, 0, 0], [1, 7, 0], [-2, 0.5, 9]]
    const bVec = [2, 7, 9]
    const bMat = [[2, 1], [7, 2], [9, 3]]
    const lowerBatch = [
      [[2, 0, 0], [1, 3, 0], [-2, 0.5, 4]],
      [[3, 0, 0], [1, 4, 0], [-2, 0.5, 5]]
    ]
    const bBatch = [
      [[2, 1], [7, 2], [9, 3]],
      [[3, 2], [8, 3], [10, 4]]
    ]
    const cases = [
      { a: lower, b: bVec, opts: {}, shape: [3], out: [1, 2, 2.5] },
      { a: lower, b: bMat, opts: {}, shape: [3, 2], out: [1, 0.5, 2, 0.5, 2.5, 0.9375] },
      {
        a: upper, b: bMat, opts: { upper: true }, shape: [3, 2],
        out: [0.8541666865, 0.3958333433, 0.8333333135, 0.1666666716, 2.25, 0.75]
      },
      {
        a: lower, b: bMat, opts: { transposeA: true }, shape: [3, 2],
        out: [2.2708332539, 0.9791666865, 1.9583333731, 0.5416666865, 2.25, 0.75]
      },
      {
        a: upper, b: bMat, opts: { upper: true, transposeA: true }, shape: [3, 2],
        out: [1, 0.5, 2.6666667461, 0.8333333135, 0.7916666865, 0.2708333433]
      },
      {
        a: lowerUnit, b: bMat, opts: { unitDiagonal: true }, shape: [3, 2],
        out: [2, 1, 5, 1, 10.5, 4.5]
      },
      {
        a: lowerBatch, b: bBatch, opts: {}, shape: [2, 3, 2],
        out: [1, 0.5, 2, 0.5, 2.5, 0.9375, 1, 0.6666666865, 1.75, 0.5833333135, 2.2249999046, 1.0083333254]
      },
      {
        a: lowerBatch, b: bVec, opts: {}, shape: [2, 3],
        out: [1, 2, 2.5, 0.6666666865, 1.5833333731, 1.9083333015]
      },
    ]
    for (const c of cases) {
      const x = new Tensor(c.a).triangularSolve(new Tensor(c.b), c.opts)
      assertShape(x.shape, c.shape)
      assertClose(await x.toArray(), c.out, 2e-4)
    }
  })

  await test('cholesky matches numpy torch probe', async () => {
    const cases = [
      { a: [[4]], shape: [1, 1], out: [2] },
      { a: [[4, 2], [2, 5]], shape: [2, 2], out: [2, 0, 1, 2] },
      {
        a: [[6, 2, 1], [2, 5, 2], [1, 2, 4]],
        shape: [3, 3],
        out: [2.4494898319, 0, 0, 0.8164966106, 2.0816659927, 0, 0.4082483053, 0.8006407619, 1.7867029905]
      },
      {
        a: [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]],
        shape: [4, 4],
        out: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2]
      },
      {
        a: [[[4, 2], [2, 5]], [[9, 3], [3, 2]]],
        shape: [2, 2, 2],
        out: [2, 0, 1, 2, 3, 0, 1, 1]
      },
    ]
    for (const c of cases) {
      const l = new Tensor(c.a).cholesky()
      assertShape(l.shape, c.shape)
      assertClose(await l.toArray(), c.out, 2e-4)
    }
    const u = new Tensor([[4, 2], [2, 5]]).cholesky({ upper: true })
    assertShape(u.shape, [2, 2])
    assertClose(await u.toArray(), [2, 1, 0, 2], 2e-4)
  })

  await test('choleskySolve matches torch probe', async () => {
    const a = [[4, 2], [2, 5]]
    const b = [[1, 2], [3, 4]]
    for (const upper of [false, true]) {
      const f = new Tensor(a).cholesky({ upper })
      const x = f.choleskySolve(new Tensor(b), { upper })
      assertShape(x.shape, [2, 2])
      assertClose(await x.toArray(), [-0.0625, 0.125, 0.625, 0.75], 2e-4)
    }
    const ab = [a, [[9, 3], [3, 2]]]
    const bbVec = [[1, 3], [2, 4]]
    for (const upper of [false, true]) {
      const f = new Tensor(ab).cholesky({ upper })
      const x = f.choleskySolve(new Tensor(bbVec), { upper })
      assertShape(x.shape, [2, 2])
      assertClose(await x.toArray(), [-0.0625, 0.625, -0.8888889, 3.3333333], 5e-4)

      const xb = f.choleskySolve(new Tensor([1, 4]), { upper })
      assertShape(xb.shape, [2, 2])
      assertClose(await xb.toArray(), [-0.1875, 0.875, -1.1111112, 3.6666667], 6e-4)
    }
  })

  await test('solve matches numpy torch probe', async () => {
    const a = [[2, 1], [1, 3]]
    const bVec = [1, 4]
    const bMat = [[1, 2], [3, 4]]
    const xVec = new Tensor(a).solve(new Tensor(bVec))
    assertShape(xVec.shape, [2])
    assertClose(await xVec.toArray(), [-0.2, 1.4], 3e-4)

    const xMat = new Tensor(a).solve(new Tensor(bMat))
    assertShape(xMat.shape, [2, 2])
    assertClose(await xMat.toArray(), [0, 0.4, 1, 1.2], 3e-4)

    const pivotA = [[0, 2], [1, 3]]
    const pivotBVec = [4, 5]
    const pivotBMat = [[4, 1], [5, 2]]
    const xpVec = new Tensor(pivotA).solve(new Tensor(pivotBVec))
    assertShape(xpVec.shape, [2])
    assertClose(await xpVec.toArray(), [-1, 2], 3e-4)

    const xpMat = new Tensor(pivotA).solve(new Tensor(pivotBMat))
    assertShape(xpMat.shape, [2, 2])
    assertClose(await xpMat.toArray(), [-1, 0.5, 2, 0.5], 3e-4)

    const ab = [a, [[3, 1], [1, 4]]]
    const bb = [bMat, [[2, 3], [4, 5]]]
    const xb = new Tensor(ab).solve(new Tensor(bb))
    assertShape(xb.shape, [2, 2, 2])
    assertClose(await xb.toArray(), [0, 0.4, 1, 1.2, 0.3636363745, 0.6363636255, 0.9090909362, 1.0909091234], 3e-4)

    const bbVec = [bVec, [2, 5]]
    const xbVec = new Tensor(ab).solve(new Tensor(bbVec))
    assertShape(xbVec.shape, [2, 2])
    assertClose(await xbVec.toArray(), [-0.2, 1.4, 0.27272728, 1.1818182], 4e-4)

    const pivotAB = [pivotA, [[3, 1], [0, 2]]]
    const pivotBBVec = [pivotBVec, [7, 4]]
    const xpbVec = new Tensor(pivotAB).solve(new Tensor(pivotBBVec))
    assertShape(xpbVec.shape, [2, 2])
    assertClose(await xpbVec.toArray(), [-1, 2, 1.6666667, 2], 4e-4)

    const xbBroadcastVec = new Tensor(ab).solve(new Tensor(bVec))
    assertShape(xbBroadcastVec.shape, [2, 2])
    assertClose(await xbBroadcastVec.toArray(), [-0.2, 1.4, 0, 1], 4e-4)

    const xbSingletonMatrix = new Tensor(ab).solve(new Tensor([bMat]))
    assertShape(xbSingletonMatrix.shape, [2, 2, 2])
    assertClose(await xbSingletonMatrix.toArray(), [0, 0.4, 1, 1.2, 0.09090909, 0.36363637, 0.7272727, 0.9090909], 5e-4)

    let ok = false
    try {
      new Tensor([[1, 1, 1], [1, 1, 1]]).solve(new Tensor([1, 1]))
    } catch (e) { ok = true }
    assert(ok, 'expected solve to reject non-square A')
  })

  await test('lstsq matches numpy torch probe', async () => {
    const a = [[1, 0], [1, 1], [1, 2]]
    const bVec = [1, 2, 2.5]
    const bMat = [[1, 0.5], [2, 1], [2.5, 1.5]]
    const xVec = new Tensor(a).lstsq(new Tensor(bVec))
    assertShape(xVec.shape, [2])
    assertClose(await xVec.toArray(), [1.0833334, 0.75], 5e-4)

    const xMat = new Tensor(a).lstsq(new Tensor(bMat))
    assertShape(xMat.shape, [2, 2])
    assertClose(await xMat.toArray(), [1.0833334, 0.5, 0.75, 0.5], 5e-4)

    const ab = [a, [[1, 0], [1, 1.5], [1, 3]]]
    const bb = [bMat, [[1.25, 0.75], [2.25, 1.25], [2.75, 1.75]]]
    const xb = new Tensor(ab).lstsq(new Tensor(bb))
    assertShape(xb.shape, [2, 2, 2])
    assertClose(await xb.toArray(), [1.0833334, 0.5, 0.75, 0.5, 1.3333334, 0.75, 0.5, 0.33333334], 6e-4)

    const bbVec = [bVec, [1.25, 2.25, 2.75]]
    const xbVec = new Tensor(ab).lstsq(new Tensor(bbVec))
    assertShape(xbVec.shape, [2, 2])
    assertClose(await xbVec.toArray(), [1.0833334, 0.75, 1.3333334, 0.5], 6e-4)

    const xbBroadcastVec = new Tensor(ab).lstsq(new Tensor(bVec))
    assertShape(xbBroadcastVec.shape, [2, 2])
    assertClose(await xbBroadcastVec.toArray(), [1.0833334, 0.75, 1.0833334, 0.5], 6e-4)

    const wide = [[1, 2, 0], [0, 1, 1]]
    const wideVec = new Tensor(wide).lstsq(new Tensor([1, 2]))
    assertShape(wideVec.shape, [3])
    assertClose(await wideVec.toArray(), [-0.33333334, 0.6666667, 1.3333334], 8e-4)

    const wideMat = new Tensor(wide).lstsq(new Tensor([[1, 3], [2, 4]]))
    assertShape(wideMat.shape, [3, 2])
    assertClose(
      await wideMat.toArray(),
      [-0.33333334, -0.33333334, 0.6666667, 1.6666666, 1.3333334, 2.3333333],
      1e-3
    )

    const rankSquareVec = new Tensor([[1, 1], [2, 2]]).lstsq(new Tensor([3, 6]))
    assertShape(rankSquareVec.shape, [2])
    assertClose(await rankSquareVec.toArray(), [1.5, 1.5], 2e-3)

    const rankSquareMat = new Tensor([[1, 1], [2, 2]]).lstsq(new Tensor([[3, 1], [6, 2]]))
    assertShape(rankSquareMat.shape, [2, 2])
    assertClose(await rankSquareMat.toArray(), [1.5, 0.5, 1.5, 0.5], 2e-3)

    const rankTall = new Tensor([[1, 1], [2, 2], [3, 3]]).lstsq(new Tensor([1, 2, 3]))
    assertShape(rankTall.shape, [2])
    assertClose(await rankTall.toArray(), [0.5, 0.5], 2e-3)

    const rankWide = new Tensor([[1, 1, 0], [2, 2, 0]]).lstsq(new Tensor([3, 6]))
    assertShape(rankWide.shape, [3])
    assertClose(await rankWide.toArray(), [1.5, 1.5, 0], 2e-3)

    let ok = false
    try {
      new Tensor([[1, 1, 1], [1, 1, 1]]).lstsq(new Tensor([1, 1, 1]))
    } catch (e) { ok = true }
    assert(ok, 'expected lstsq to reject incompatible RHS rows')
  })

  await test('crossEntropy with sparse targets', async () => {
    const logits = new Tensor([[0, 0, 0], [0, 0, 0]])
    const target = new Tensor([0, 2])
    const loss = await logits.crossEntropy(target)
    assertShape(loss.shape, [])
    assertClose(await loss.toArray(), [Math.log(3)])
  })

  await test('crossEntropy with dense targets', async () => {
    const logits = new Tensor([[0, 0, 0], [0, 0, 0]])
    const target = new Tensor([[1, 0, 0], [0, 0, 1]])
    const loss = await logits.crossEntropy(target)
    assertShape(loss.shape, [])
    assertClose(await loss.toArray(), [Math.log(3)])
  })

  await test('crossEntropy with sparse targets on non-last axis', async () => {
    const logits = new Tensor([
      [[0, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0]]
    ])
    const target = new Tensor([
      [0, 2],
      [1, 0]
    ])
    const loss = await logits.crossEntropy(target, -2)
    assertShape(loss.shape, [])
    assertClose(await loss.toArray(), [Math.log(3)])
  })

  await test('crossEntropy default matches tinygrad class axis', async () => {
    const logits = new Tensor([
      [[0, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0]]
    ])
    const target = new Tensor([
      [0, 2],
      [1, 0]
    ])
    const loss = await logits.crossEntropy(target)
    assertShape(loss.shape, [])
    assertClose(await loss.toArray(), [Math.log(3)])
  })

  await test('crossEntropy with dense targets on non-last axis', async () => {
    const logits = new Tensor([
      [[0, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0]]
    ])
    const target = new Tensor([
      [[1, 0], [0, 0], [0, 1]],
      [[0, 1], [1, 0], [0, 0]]
    ])
    const loss = await logits.crossEntropy(target, 1)
    assertShape(loss.shape, [])
    assertClose(await loss.toArray(), [Math.log(3)])
  })

  await test('crossEntropy shape mismatch throws', async () => {
    const logits = new Tensor([[0, 0, 0], [0, 0, 0]])
    const target = new Tensor([[1, 0], [0, 1]])
    let ok = false
    try {
      const loss = await logits.crossEntropy(target)
      ok = loss.shape.length === 0
    } catch (e) { ok = true }
    assert(ok, 'expected crossEntropy to fail on shape mismatch')
  })

  // -- Autograd --
  console.log('\n-- Autograd --')

  await test('grad: mul sum', async () => {
    const a = new Tensor([1, 2, 3], { requiresGrad: true })
    const b = new Tensor([4, 5, 6])
    const loss = a.mul(b).sum()
    await loss.backward()
    assert(a.grad, 'grad is null')
    assertClose(await a.grad.toArray(), [4, 5, 6])
  })

  await test('grad: neg sum', async () => {
    const a = new Tensor([1, 2, 3], { requiresGrad: true })
    const loss = a.neg().sum()
    await loss.backward()
    assert(a.grad, 'grad is null')
    assertClose(await a.grad.toArray(), [-1, -1, -1])
  })

  await test('grad: matmul backward', async () => {
    const W = new Tensor([[1, 2], [3, 4]], { requiresGrad: true })
    const x = new Tensor([[1, 0]])
    const loss = x.dot(W).sum()
    await loss.backward()
    assert(W.grad, 'W.grad is null')
    // dL/dW = x^T @ ones = [[1,1],[0,0]]
    assertClose(await W.grad.toArray(), [1, 1, 0, 0])
  })

  await test('grad: relu backward', async () => {
    const a = new Tensor([-1, 2, -3, 4], { requiresGrad: true })
    const loss = a.relu().sum()
    await loss.backward()
    assert(a.grad, 'grad is null')
    // relu grad: 0 where input<=0, 1 where input>0
    assertClose(await a.grad.toArray(), [0, 1, 0, 1])
  })

  await test('grad: chain backward', async () => {
    const a = new Tensor([1, 2, 3], { requiresGrad: true })
    const loss = a.mul(a).sum()  // d/da(a^2) = 2a
    await loss.backward()
    assert(a.grad, 'grad is null')
    assertClose(await a.grad.toArray(), [2, 4, 6])
  })

  await test('grad: backward uses current physical value after copyFrom', async () => {
    const weight = new Tensor([1]).mul(2)
    await weight.realize()
    weight.requiresGrad = true
    const physicalBuffer = weight.uopPhysical.buffer.raw
    pg._core.ffi.poly_buffer_ensure_device_allocated(
      weight._ctx, physicalBuffer, pg._core.deviceIds[weight.device.toLowerCase()]
    )
    pg._core.ffi.poly_buffer_write(weight._ctx, physicalBuffer, new Float32Array([3]))
    const loss = weight.square().sum()
    await loss.backward()
    assert(weight.grad, 'weight.grad is null')
    assertClose(await weight.grad.toArray(), [6])
  })

  // -- Assign --
  console.log('\n-- Assign --')

  await test('assign basic', async () => {
    const a = new Tensor([1, 2, 3])
    a.assign(a.add(10))
    await a.realize()
    assertClose(await a.toArray(), [11, 12, 13])
  })

  await test('assign rejects device mismatch', async () => {
    const a = new Tensor([1], { device: 'cpu' })
    const v = new Tensor([5], { device: 'cpu' }).to('cuda')
    let ok = false
    try {
      a.assign(v)
    } catch (e) {
      ok = /assign device mismatch CPU != CUDA/.test(String(e && e.message ? e.message : e))
    }
    assert(ok, 'expected assign device mismatch CPU != CUDA')
  })

  await test('assign rejects dtype mismatch', async () => {
    const a = new Tensor([1], { dtype: 'float32' })
    const mismatchDtype = supportsF64 ? 'float64' : 'int32'
    const v = new Tensor([5], { dtype: mismatchDtype })
    let ok = false
    try {
      a.assign(v)
    } catch (e) {
      ok = new RegExp(`assign dtype mismatch float32 != ${mismatchDtype}`).test(
        String(e && e.message ? e.message : e)
      )
    }
    assert(ok, `expected assign dtype mismatch float32 != ${mismatchDtype}`)
  })

  await test('assign to same-device place keeps place target', async () => {
    const a = new Tensor([1], { device: 'cpu' }).to('cuda')
    const v = new Tensor([5], { device: 'cpu' }).to('cuda')
    assert(a.assign(v) === a, 'assign should return self')
    assert(a.device === 'CUDA', 'assign should keep CUDA placement')
    assert(!a.uop.hasBufferIdentity(), 'assign before realize should be an effect graph')
  })

  await test('assign realized targets reuse current buffer', async () => {
    const a = new Tensor([1])
    await a.realize()
    const aBuffer = a.uop.buffer.key

    // assign() creates an effect graph first. Realizing that effect writes the
    // existing target storage and returns to the same current buffer root.
    a.assign(new Tensor([5]))
    assert(!a.uop.hasBufferIdentity(), 'assign before realize should be an effect graph')
    await a.realize()
    assert(a.uop.buffer.key === aBuffer, 'realized assign should reuse target buffer')
    assertClose(await a.toArray(), [5])

    const x = await new Tensor([1]).add(1).realize()
    const xBuffer = x.uop.buffer.key
    x.assign(new Tensor([9]))
    await x.realize()
    assert(x.uop.buffer.key === xBuffer, 'realized expression assign should reuse target buffer')
    assertClose(await x.toArray(), [9])
  })

  await test('shared-storage to copies and preserves source across assign', async () => {
    const source = await new Tensor([1, 2, 3], { device: 'cpu' }).realize()
    const sourceBuffer = source.uop.buffer.key
    const target = await source.to('interp').realize()
    const targetBuffer = target.uop.buffer.key

    assert(targetBuffer !== sourceBuffer, 'cross-device to should allocate a distinct target buffer')
    assertClose(await source.toArray(), [1, 2, 3])
    assertClose(await target.toArray(), [1, 2, 3])

    for (const values of [[9, 8, 7], [4, 5, 6]]) {
      target.assign(new Tensor(values, { device: 'interp' }))
      await target.realize()
      assert(target.uop.buffer.key === targetBuffer, 'assign should retain the copied target buffer')
      assertClose(await source.toArray(), [1, 2, 3])
      assertClose(await target.toArray(), values)
    }
  })

  await test('copyFrom preserves buffer identity and updates JIT replay input', async () => {
    const x = Tensor.empty([3], { dtype: 'float32' })
    const bufferKey = x.uop.buffer.key
    x.copyFrom(new Float32Array([1, 2, 3]))
    assert(x.uop.buffer.key === bufferKey, 'copyFrom should preserve input buffer identity')
    assertClose(await x.toArray(), [1, 2, 3])

    const f = pg.jit((a) => a.add(1).realize())
    assertClose(await (await f(x)).toArray(), [2, 3, 4])
    pg.resetCounters()
    const captured = await f(x)
    let counterStats = pg.stats().coreStats
    const expectedOps = pg.device === 'x86' ? 0 : 3
    const expectedMem = pg.device === 'x86' ? 0 : 24
    assert(counterStats.globalOps === expectedOps && counterStats.globalMem === expectedMem && counterStats.kernelCount === 1,
      'JIT capture execution should update counters exactly once')
    assertClose(await captured.toArray(), [2, 3, 4])
    assert(f.scheduleCount === 1, `expected one captured schedule, got ${f.scheduleCount}`)

    const replayBufferKey = x.uop.buffer.key
    x.updateFrom(new Float32Array([10, 20, 30]))
    assert(x.uop.buffer.key === replayBufferKey, 'updateFrom should preserve captured input buffer identity')
    pg.resetCounters()
    const replayed = await f(x)
    counterStats = pg.stats().coreStats
    assert(counterStats.globalOps === expectedOps && counterStats.globalMem === expectedMem && counterStats.kernelCount === 1,
      'JIT replay should update counters exactly once')
    assertClose(await replayed.toArray(), [11, 21, 31])
    f.dispose()
  })

  await test('pure movement view realize is zero call', async () => {
    const x = await Tensor.arange(8, { dtype: 'float32' }).realize()
    const out = x.reshape(2, 4).permute(1, 0)
    const memBefore = pg.stats().coreStats.memUsed

    pg.resetCounters()
    await out.realize()
    const stats = pg.stats().coreStats
    assert(
      stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
      `pure view realize should execute zero calls, got ${stats.globalOps}/${stats.globalMem}/${stats.kernelCount}`
    )
    assert(stats.memUsed === memBefore, 'pure view realize should not allocate storage')
    assertClose(await out.toArray(), [0, 4, 1, 5, 2, 6, 3, 7])
  })

  await test('realized contiguous and readback reuse current buffer identity', async () => {
    const source = await Tensor.arange(8, { dtype: 'float32' }).add(1).realize()
    const sourceCurrent = source.uop.key
    const sourceLogical = source.uopLogical.key
    assert(source.uopLogical.op === pg._core.ops.ADD, 'source logical root should retain ADD provenance')
    assert(source.uop.hasBufferIdentity(), 'realized source should have buffer identity')

    const out = source.contiguous()
    assert(out !== source, 'contiguous should return a new Tensor object')
    assert(out.uopLogical.op === pg._core.ops.CONTIGUOUS, 'logical result should retain CONTIGUOUS')
    assert(out.uopPhysical && out.uopPhysical.key === sourceCurrent,
      'physical result should reuse the exact current buffer')
    assert(out.uop.key === sourceCurrent, 'current result should reuse the exact current buffer')
    assert(source.uopLogical.key === sourceLogical, 'source logical provenance should be unchanged')

    pg.resetCounters()
    await out.realize()
    let stats = pg.stats().coreStats
    assert(stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
      `realized contiguous should execute zero calls, got ${stats.globalOps}/${stats.globalMem}/${stats.kernelCount}`)
    pg.resetCounters()
    assertClose(await out.toArray(), [1, 2, 3, 4, 5, 6, 7, 8])
    stats = pg.stats().coreStats
    assert(stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
      `realized readback should execute zero calls, got ${stats.globalOps}/${stats.globalMem}/${stats.kernelCount}`)

    const reshaped = source.reshape(2, 4)
    assert(reshaped.uop.hasBufferIdentity(), 'reshape of current buffer should retain identity')
    pg.resetCounters()
    assertClose(await reshaped.toArray(), [1, 2, 3, 4, 5, 6, 7, 8])
    stats = pg.stats().coreStats
    assert(stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
      'reshape readback should execute zero calls')

    const permuted = source.reshape(2, 4).permute(1, 0)
    pg.resetCounters()
    assertClose(await permuted.toArray(), [1, 5, 2, 6, 3, 7, 4, 8])
    assert(pg.stats().coreStats.kernelCount === 1, 'noncontiguous permute should materialize once')

    const casted = source.cast('int32')
    pg.resetCounters()
    assertClose(await casted.toArray(), [1, 2, 3, 4, 5, 6, 7, 8])
    assert(pg.stats().coreStats.kernelCount === 1, 'lazy cast should materialize once')
  })

  // -- Static constructors --
  console.log('\n-- Static constructors --')

  await test('zeros', async () => {
    const t = Tensor.zeros(3)
    assertClose(await t.toArray(), [0, 0, 0])
  })

  await test('ones', async () => {
    const t = Tensor.ones(2, 2)
    assertClose(await t.toArray(), [1, 1, 1, 1])
  })

  await test('full', async () => {
    const t = Tensor.full([3], 7)
    assertClose(await t.toArray(), [7, 7, 7])
  })

  await test('arange', async () => {
    const t = Tensor.arange(4)
    assertClose(await t.toArray(), [0, 1, 2, 3])
  })

  await test('eye', async () => {
    const t = Tensor.eye(2)
    assertClose(await t.toArray(), [1, 0, 0, 1])
  })

  // -- Lazy RNG --
  console.log('\n-- Lazy RNG --')

  await test('rand uniform [0,1)', async () => {
    Tensor.manual_seed(42)
    const t = Tensor.rand(100)
    const arr = await t.toArray()
    assert(arr.length === 100, `Expected 100 elements, got ${arr.length}`)
    let min = arr[0], max = arr[0]
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] < min) min = arr[i]
      if (arr[i] > max) max = arr[i]
    }
    assert(min >= 0 && max < 1, `Out of range: min=${min}, max=${max}`)
  })

  await test('randn gaussian', async () => {
    Tensor.manual_seed(99)
    const t = Tensor.randn(1000)
    const arr = await t.toArray()
    let sum = 0
    for (let i = 0; i < arr.length; i++) sum += arr[i]
    const mean = sum / arr.length
    assert(Math.abs(mean) < 0.3, `Mean too far from 0: ${mean}`)
  })

  await test('rand deterministic with seed', async () => {
    Tensor.manual_seed(1337)
    const a = await Tensor.rand(8).toArray()
    Tensor.manual_seed(1337)
    const b = await Tensor.rand(8).toArray()
    for (let i = 0; i < 8; i++) {
      assert(a[i] === b[i], `Not deterministic at [${i}]: ${a[i]} vs ${b[i]}`)
    }
  })

  // -- Float64 --
  console.log('\n-- Float64 --')

  // tinygrad gates backend dtype tests through is_dtype_supported. Polygrad's
  // WebGPU caps likewise advertise no f64 because WGSL has no f64 shader type.
  await testIf(supportsF64, 'f64: creation', async () => {
    const t = new Tensor([1.5, 2.5, 3.5], { dtype: 'float64' })
    assertClose(await t.toArray(), [1.5, 2.5, 3.5])
  })

  await testIf(supportsF64, 'f64: zeros', async () => {
    const t = Tensor.zeros(3, { dtype: 'float64' })
    assertClose(await t.toArray(), [0, 0, 0])
  })

  await testIf(supportsF64, 'f64: ones', async () => {
    const t = Tensor.ones(2, { dtype: 'float64' })
    assertClose(await t.toArray(), [1, 1])
  })

  await testIf(supportsF64, 'f64: add', async () => {
    const a = new Tensor([1, 2, 3], { dtype: 'float64' })
    const b = new Tensor([4, 5, 6], { dtype: 'float64' })
    const arr = await a.add(b).toArray()
    console.log('DEBUG f64 add', Array.from(arr))
    assertClose(arr, [5, 7, 9])
  })

  await testIf(supportsF64, 'f64: mul', async () => {
    const a = new Tensor([2, 3], { dtype: 'float64' })
    const b = new Tensor([4, 5], { dtype: 'float64' })
    const arr = await a.mul(b).toArray()
    console.log('DEBUG f64 mul', Array.from(arr))
    assertClose(arr, [8, 15])
  })

  await testIf(supportsF64, 'f64: sum', async () => {
    const t = new Tensor([1, 2, 3], { dtype: 'float64' })
    const v = await t.sum().item()
    console.log('DEBUG f64 sum', v)
    assert(Math.abs(v - 6) < 1e-10, `Expected 6, got ${v}`)
  })

  await testIf(supportsF64, 'f64: backward', async () => {
    const a = new Tensor([1, 2, 3], { dtype: 'float64', requiresGrad: true })
    const b = new Tensor([4, 5, 6], { dtype: 'float64' })
    const loss = a.mul(b).sum()
    await loss.backward()
    assert(a.grad, 'grad is null')
    assertClose(await a.grad.toArray(), [4, 5, 6])
  })

  await test('f64: default is f32', async () => {
    const a = new Tensor([1, 2, 3])
    assert(a.dtype === 'float32', `expected float32, got ${a.dtype}`)
  })

  // -- Kernel cache consistency --
  console.log('\n-- Cache consistency --')

  await test('repeated realize produces identical results', async () => {
    // Exercises the in-memory schedule cache + disk cache (native) or
    // WASM step plan cache. Same graph structure, same data = same result.
    const a = new Tensor([1, 2, 3, 4])
    const b = new Tensor([10, 20, 30, 40])
    const r1 = await a.add(b).toArray()
    const r2 = await (new Tensor([1, 2, 3, 4])).add(new Tensor([10, 20, 30, 40])).toArray()
    const r3 = await (new Tensor([1, 2, 3, 4])).add(new Tensor([10, 20, 30, 40])).toArray()
    assertClose(r1, [11, 22, 33, 44])
    assertClose(r2, [11, 22, 33, 44])
    assertClose(r3, [11, 22, 33, 44])
  })

  await test('realize retargets live tensors sharing lazy root', async () => {
    const a = new Tensor([1])
    await a.realize()
    const x1 = a.add(1)
    const x2 = a.add(1)
    assert(x1.uop.key === x2.uop.key, 'lazy roots should be hash-consed together')

    await x1.realize()

    assert(x1.uop.key === x2.uop.key, 'shared lazy root should retarget to one realized root')
    assert(x1.uop.buffer.key === x2.uop.buffer.key, 'shared lazy root should share realized buffer')
    assertClose(await x2.toArray(), [2])
  })

  await test('realize rewrites downstream live graph', async () => {
    const a = new Tensor([1])
    await a.realize()
    const x = a.add(1)
    const y = x.add(2)
    const oldYKey = y.uop.key

    await x.realize()

    assert(y.uop.key !== oldYKey, 'downstream graph should be rewritten through realized x')
    assertClose(await y.toArray(), [4])
  })

  await test('separate realizes do not alias buffers', async () => {
    const a = new Tensor([1])
    await a.realize()
    const y1 = await a.add(1).realize()
    const y2 = await a.add(1).realize()

    assert(y1.uop.buffer.key !== y2.uop.buffer.key, 'separate materializations should get separate buffers')
    assertClose(await y1.toArray(), [2])
    assertClose(await y2.toArray(), [2])
  })

  await test('to keeps separate realized same-logical occurrences', async () => {
    const a = new Tensor([1])
    await a.realize()
    const x1 = await a.add(1).realize()
    const x2 = await a.add(1).realize()
    assert(x1.uop.buffer.key !== x2.uop.buffer.key, 'separate materializations should stay distinct')

    // The preserved logical expression can be shared, but placement must see
    // the occurrence-specific realized source when .to(device) creates a fact.
    const x1Cuda = x1.to('cuda')
    const x2Cuda = x2.to('cuda')
    assert(x1Cuda.uop.op === pg._core.ops.COPY, 'x1.to(cuda) should be an eager COPY')
    assert(x2Cuda.uop.op === pg._core.ops.COPY, 'x2.to(cuda) should be an eager COPY')
    assert(x1Cuda.uop.buffer === null, 'an unrealized COPY should not claim buffer identity')
    assert(x2Cuda.uop.buffer === null, 'an unrealized COPY should not claim buffer identity')
    assert(
      x1Cuda.uop.src[0].buffer.key === x1.uop.buffer.key,
      'x1.to(cuda) COPY should point at x1 buffer'
    )
    assert(
      x2Cuda.uop.src[0].buffer.key === x2.uop.buffer.key,
      'x2.to(cuda) COPY should point at x2 buffer'
    )
    assert(
      x1Cuda.uop.src[0].buffer.key !== x2Cuda.uop.src[0].buffer.key,
      'to(cuda) COPY sources should not alias'
    )

    const y1 = x1Cuda.add(1)
    const y2 = x2Cuda.add(1)
    assert(y1.uop.key !== y2.uop.key, 'downstream graphs should use distinct occurrence sources')
  })

  await test('nested to keeps realized current and export logical separate', async () => {
    const x = await new Tensor([1]).add(1).realize()
    const xBase = x.to('cpu')
    const xCuda = xBase.to('cuda')
    const xCpu = xCuda.to('cpu')

    // Pinned Tensor.to (tensor.py:327-335) keeps both device moves as exact
    // current COPY occurrences. Polygrad additionally preserves the approved
    // portable logical twin.
    assert(xCuda.uop.op === pg._core.ops.COPY, 'CUDA move should be an eager COPY')
    assert(xCuda.uop.src[0].key === xBase.uop.key, 'CUDA COPY should use the CPU occurrence')
    assert(xCpu.uop.op === pg._core.ops.COPY, 'CPU roundtrip should be an eager COPY')
    assert(xCpu.uop.src[0].key === xCuda.uop.key, 'CPU COPY should use the CUDA occurrence')
    assert(xCpu.uop.key !== x.uop.key, 'roundtrip COPY should stay occurrence-distinct')
    assert(xCpu.uopPhysical.key === xCpu.uop.key, 'physical root should be the current COPY')
    assert(xCpu.uopLogical.key === x.uopLogical.key, 'nested to should preserve export logical root')

    const y = xCpu.add(1)
    assert(y.device === 'CPU', 'downstream value should keep selected CPU placement')
    assert(y.uop.key !== xCpu.uop.key, 'downstream value should build a new current graph')

    // Pinned Tensor.alu consumes ordered current roots (tensor.py:128-140).
    // Sharing one portable logical X must not collapse the second occurrence.
    const mixed = xBase.add(xCpu)
    assert(mixed.uop.op === pg._core.ops.ADD, 'mixed result should be ADD')
    assert(mixed.uop.src[0].key === xBase.uop.key, 'first ADD input should be CPU occurrence')
    assert(mixed.uop.src[1].key === xCpu.uop.key, 'second ADD input should be CPU COPY')
    assert(mixed.uop.src[1].src[0].key === xCuda.uop.key, 'CPU COPY should contain CUDA COPY')
    assert(
      mixed.uop.src[1].src[0].src[0].key === xBase.uop.key,
      'CUDA COPY should contain CPU occurrence'
    )
    assert(mixed.uopLogical.src[0].key === x.uopLogical.key, 'logical lhs should be X')
    assert(mixed.uopLogical.src[1].key === x.uopLogical.key, 'logical rhs should be X')
  })

  await test('repeated fused chain produces identical results', async () => {
    // Fused multi-op chain through cache: (a+b)*a-b
    const a = [2, 3, 4]
    const b = [1, 1, 1]
    const expected = [(2+1)*2-1, (3+1)*3-1, (4+1)*4-1]
    for (let i = 0; i < 3; i++) {
      const ta = new Tensor(a), tb = new Tensor(b)
      const r = await ta.add(tb).mul(ta).sub(tb).toArray()
      assertClose(r, expected)
    }
  })

  // -- Missing math ops --
  console.log('\n-- Missing math ops --')

  await test('pow integer', async () => {
    const t = new Tensor([2, 3, 4])
    const r = t.pow(2)
    assertClose(await r.toArray(), [4, 9, 16])
  })

  await test('pow float', async () => {
    const t = new Tensor([4, 9, 16])
    const r = t.pow(0.5)
    assertClose(await r.toArray(), [2, 3, 4])
  })

  await test('reciprocal', async () => {
    const t = new Tensor([2, 4, 5])
    const r = t.reciprocal()
    assertClose(await r.toArray(), [0.5, 0.25, 0.2])
  })

  await test('exp2', async () => {
    const t = new Tensor([0, 1, 2, 3])
    const r = t.exp2()
    assertClose(await r.toArray(), [1, 2, 4, 8])
  })

  await test('log2', async () => {
    const t = new Tensor([1, 2, 4, 8])
    const r = t.log2()
    assertClose(await r.toArray(), [0, 1, 2, 3])
  })

  await test('trunc', async () => {
    const t = new Tensor([1.7, -2.3, 3.9])
    const r = t.trunc()
    assertClose(await r.toArray(), [1, -2, 3])
  })

  // -- Aliases --
  console.log('\n-- Aliases --')

  await test('swish is silu', async () => {
    const t = new Tensor([1, 2, -1])
    assertClose(await t.swish().toArray(), await t.silu().toArray())
  })

  await test('view is reshape', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6])
    assertShape(t.view(2, 3).shape, [2, 3])
    assertClose(await t.view(2, 3).toArray(), [1, 2, 3, 4, 5, 6])
  })

  await test('matmul is dot', async () => {
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([[5, 6], [7, 8]])
    assertClose(await a.matmul(b).toArray(), await a.dot(b).toArray())
  })

  // -- Composed ops --
  console.log('\n-- Composed ops --')

  await test('conv2d stride and padding match reference', async () => {
    const xData = Float32Array.from({ length: 1 * 2 * 4 * 5 }, (_, i) => i / 7)
    const wData = Float32Array.from({ length: 3 * 2 * 2 * 3 }, (_, i) => (i - 5) / 11)
    const bData = Float32Array.from([0.5, -1.0, 2.0])
    const expected = new Float32Array(1 * 3 * 6 * 2)
    for (let oc = 0; oc < 3; oc++) {
      for (let oy = 0; oy < 6; oy++) {
        for (let ox = 0; ox < 2; ox++) {
          let acc = bData[oc]
          for (let ic = 0; ic < 2; ic++) {
            for (let ky = 0; ky < 2; ky++) {
              for (let kx = 0; kx < 3; kx++) {
                const iy = oy + ky - 2
                const ix = ox * 2 + kx - 1
                if (iy >= 0 && iy < 4 && ix >= 0 && ix < 5) {
                  acc += xData[((ic * 4 + iy) * 5) + ix] * wData[((oc * 2 + ic) * 2 + ky) * 3 + kx]
                }
              }
            }
          }
          expected[(oc * 6 + oy) * 2 + ox] = acc
        }
      }
    }
    const conv = new Tensor(xData).reshape(1, 2, 4, 5)
      .conv2d(new Tensor(wData).reshape(3, 2, 2, 3), new Tensor(bData), 1, [1, 2], 1, [1, 0, 2, 1])
    assertShape(conv.shape, [1, 3, 6, 2])
    assertClose(await conv.toArray(), expected, 1e-4)
  })

  await test('conv2d padded 3x3 4x4 devectorize regression', async () => {
    // tinygrad: arange(16).reshape(1,1,4,4).conv2d(ones(1,1,3,3), padding=1)
    // This crosses the 128-lane late-devectorize threshold.
    const padded = Tensor.arange(16, { dtype: 'float32' }).reshape(1, 1, 4, 4)
      .conv2d(Tensor.ones(1, 1, 3, 3), null, 1, 1, 1, 1)
    assertShape(padded.shape, [1, 1, 4, 4])
    assertClose(await padded.toArray(), [10, 18, 24, 18, 27, 45, 54, 39, 51, 81, 90, 63, 42, 66, 72, 50])
  })

  await test('max_pool2d padding matches reference', async () => {
    const pool = Tensor.arange(9, { dtype: 'float32' }).reshape(1, 1, 3, 3).max_pool2d(2, 1, 1, 1)
    assertShape(pool.shape, [1, 1, 4, 4])
    assertClose(await pool.toArray(), [0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 6, 7, 8, 8])
  })

  await test('batchnorm multi-axis matches reference', async () => {
    const bnX = Float32Array.from({ length: 2 * 3 * 4 * 5 }, (_, i) => i / 10)
    const mean = Float32Array.from({ length: 8 }, (_, i) => i / 20)
    const inv = Float32Array.from({ length: 8 }, () => 0.25)
    const weight = Float32Array.from({ length: 8 }, (_, i) => 0.5 + (0.7 * i) / 7)
    const bias = Float32Array.from({ length: 8 }, (_, i) => -0.3 + (0.7 * i) / 7)
    const bnExpected = new Float32Array(bnX.length)
    for (let n = 0; n < 2; n++) for (let c = 0; c < 3; c++) for (let h = 0; h < 4; h++) for (let w = 0; w < 5; w++) {
      const ki = n * 4 + h
      const oi = ((n * 3 + c) * 4 + h) * 5 + w
      bnExpected[oi] = ((bnX[oi] - mean[ki]) * weight[ki]) * inv[ki] + bias[ki]
    }
    const bn = new Tensor(bnX).reshape(2, 3, 4, 5).batchnorm(
      new Tensor(weight).reshape(2, 4), new Tensor(bias).reshape(2, 4),
      new Tensor(mean).reshape(2, 4), new Tensor(inv).reshape(2, 4), [0, 2]
    )
    assertClose(await bn.toArray(), bnExpected, 1e-5)
  })

  await test('nn Conv2d backward populates parameters', async () => {
    const mod = new pg.nn.Conv2d(3, 2, 3, { padding: 1 })
    const loss = mod.call(Tensor.randn(1, 3, 4, 4)).relu().mean()
    loss.backward()
    assert(mod.weight.grad !== null, 'Conv2d weight grad missing')
    assert(mod.bias.grad !== null, 'Conv2d bias grad missing')
  })

  await test('layernorm', async () => {
    const t = new Tensor([[1, 2, 3], [4, 5, 6]])
    const r = await t.layernorm()
    const arr = await r.toArray()
    // Each row should have mean ~0 and std ~1
    const row0 = arr.slice(0, 3)
    const mean0 = row0.reduce((a, b) => a + b) / 3
    assert(Math.abs(mean0) < 1e-4, `Row 0 mean should be ~0, got ${mean0}`)
  })

  await test('binaryCrossEntropy', async () => {
    const pred = new Tensor([0.9, 0.1, 0.8])
    const target = new Tensor([1, 0, 1])
    const loss = await pred.binaryCrossEntropy(target).item()
    // -mean(t*log(p) + (1-t)*log(1-p))
    const expected = -(Math.log(0.9) + Math.log(0.9) + Math.log(0.8)) / 3
    assert(Math.abs(loss - expected) < 1e-3, `Expected ~${expected}, got ${loss}`)
  })

  await test('cat 1d', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([4, 5, 6])
    const r = Tensor.cat(a, b)
    assertShape(r.shape, [6])
    assertClose(await r.toArray(), [1, 2, 3, 4, 5, 6])
  })

  await test('cat 2d axis0', async () => {
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([[5, 6]])
    const r = Tensor.cat(a, b, { dim: 0 })
    assertShape(r.shape, [3, 2])
    const arr = await r.toArray()
    console.log('DEBUG cat axis0', Array.from(arr))
    assertClose(arr, [1, 2, 3, 4, 5, 6])
  })

  await test('instance cat matches tinygrad binding', async () => {
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([[5, 6]])
    const c = new Tensor([[7, 8]])
    const r = a.cat(b, c, { dim: 0 })
    assertShape(r.shape, [4, 2])
    assertClose(await r.toArray(), [1, 2, 3, 4, 5, 6, 7, 8])
  })

  await test('stack', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([4, 5, 6])
    const r = Tensor.stack(a, b)
    assertShape(r.shape, [2, 3])
    const arr = await r.toArray()
    console.log('DEBUG stack', Array.from(arr))
    assertClose(arr, [1, 2, 3, 4, 5, 6])
  })

  await test('instance stack matches tinygrad binding', async () => {
    const a = new Tensor([1, 2])
    const b = new Tensor([3, 4])
    const r = a.stack(b, { dim: 0 })
    assertShape(r.shape, [2, 2])
    assertClose(await r.toArray(), [1, 2, 3, 4])
  })

  await test('repeat', async () => {
    const t = new Tensor([1, 2, 3])
    const r = t.repeat(3)
    assertShape(r.shape, [9])
    assertClose(await r.toArray(), [1, 2, 3, 1, 2, 3, 1, 2, 3])
  })

  await test('repeat readback preserves int32 dtype', async () => {
    const t = new Tensor([1, 2, 3], { dtype: 'int32' })
    const r = t.repeat(3)
    assert(r.dtype === 'int32', `expected int32, got ${r.dtype}`)
    const arr = await r.toArray()
    assert(arr.constructor.name === 'Int32Array', `expected Int32Array, got ${arr.constructor.name}`)
    assertClose(arr, [1, 2, 3, 1, 2, 3, 1, 2, 3])
  })

  await test('repeat 2d', async () => {
    const t = new Tensor([[1, 2], [3, 4]])
    const r = t.repeat(2, 3)
    assertShape(r.shape, [4, 6])
    assertClose(await r.toArray(), [
      1, 2, 1, 2, 1, 2,
      3, 4, 3, 4, 3, 4,
      1, 2, 1, 2, 1, 2,
      3, 4, 3, 4, 3, 4
    ])
  })

  console.log(`\nResults: ${passed} passed, ${failed} failed, ${skipped} skipped, ${passed + failed + skipped} total`)
  return { passed, failed, skipped }
}

module.exports = { runTensorTests }
