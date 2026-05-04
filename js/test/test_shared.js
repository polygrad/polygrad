/**
 * Shared test suite for the polygrad Tensor class.
 * Target-agnostic: runs against WASM or native bindings.
 *
 * Usage: require this module and call runTests(pg).
 */

'use strict'

function assertClose(arr, expected, tol) {
  if (tol === undefined) tol = 1e-4
  if (arr.length !== expected.length) {
    throw new Error(`Length mismatch: ${arr.length} vs ${expected.length}`)
  }
  for (let i = 0; i < arr.length; i++) {
    if (Math.abs(arr[i] - expected[i]) > tol) {
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

async function runTests(pg) {
  const Tensor = pg.Tensor
  const caps = pg.caps || {}
  const supportsF16 = caps.f16 !== false
  const supportsF64 = caps.f64 !== false
  let passed = 0, failed = 0, skipped = 0

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

  await test('pad', async () => {
    const t = new Tensor([1, 2, 3])
    const p = t.pad([[1, 1]])
    assertShape(p.shape, [5])
    assertClose(await p.toArray(), [0, 1, 2, 3, 0])
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

  await test('softmax', async () => {
    const t = new Tensor([1, 2, 3])
    const arr = await (await t.softmax()).toArray()
    const sum = arr[0] + arr[1] + arr[2]
    assert(Math.abs(sum - 1.0) < 1e-4, `Softmax sum should be 1, got ${sum}`)
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
    assert(x1Cuda.uop.buffer.key === x1.uop.buffer.key, 'x1.to(cuda) should point at x1 buffer')
    assert(x2Cuda.uop.buffer.key === x2.uop.buffer.key, 'x2.to(cuda) should point at x2 buffer')
    assert(x1Cuda.uop.buffer.key !== x2Cuda.uop.buffer.key, 'to(cuda) occurrences should not alias')

    const y1 = x1Cuda.add(1)
    const y2 = x2Cuda.add(1)
    assert(y1.uop.key !== y2.uop.key, 'downstream graphs should use distinct occurrence sources')
  })

  await test('nested to keeps realized current and export logical separate', async () => {
    const x = await new Tensor([1]).add(1).realize()
    const xCuda = x.to('cuda')
    const xCpu = xCuda.to('cpu')

    // Polygrad's portable logical UOp stays device-free, while current and
    // physical roots keep the realized buffer selected by this tensor
    // occurrence for later placement physicalization.
    assert(xCpu.uop.key === x.uop.key, 'nested to should preserve current realized buffer root')
    assert(xCpu.uopPhysical.key === x.uop.key, 'nested to should preserve physical buffer root')
    assert(xCpu.uopLogical.key === x.uopLogical.key, 'nested to should preserve export logical root')

    const y = xCpu.add(1)
    assert(y.device === 'CPU', 'downstream value should keep selected CPU placement')
    assert(y.uop.key !== xCpu.uop.key, 'downstream value should build a new current graph')
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

  await test('stack', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([4, 5, 6])
    const r = Tensor.stack(a, b)
    assertShape(r.shape, [2, 3])
    const arr = await r.toArray()
    console.log('DEBUG stack', Array.from(arr))
    assertClose(arr, [1, 2, 3, 4, 5, 6])
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

module.exports = { runTests }
