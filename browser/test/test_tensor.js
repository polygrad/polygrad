/**
 * Tests for polygrad-browser — WASM-based tensor computation.
 * Runs in Node.js using Emscripten's Node.js-compatible output.
 */

'use strict'

const { init, Tensor } = require('../src/index')

let passed = 0
let failed = 0

function approxEqual(a, b, tol = 1e-4) {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > tol) return false
  }
  return true
}

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

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed')
}

function assertArrayEq(actual, expected, msg) {
  if (!approxEqual(actual, expected)) {
    throw new Error(`${msg || 'array mismatch'}: got [${actual}], expected [${expected}]`)
  }
}

function assertShapeEq(actual, expected) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`shape mismatch: got [${actual}], expected [${expected}]`)
  }
}

async function main() {
  console.log('Initializing WASM module...')
  await init()
  console.log('WASM module ready.\n')

  console.log('── Creation ──')

  await test('from vector', async () => {
    const t = new Tensor([1, 2, 3])
    assert(t.shape.length === 1 && t.shape[0] === 3, 'shape')
    assertArrayEq(await t.toArray(), [1, 2, 3])
  })

  await test('from scalar', async () => {
    const t = new Tensor(42)
    assertArrayEq(await t.toArray(), [42])
  })

  await test('item', async () => {
    const t = new Tensor(42)
    assert(Math.abs(await t.item() - 42) < 1e-5, 'item')
  })

  console.log('\n── Element-wise ──')

  await test('add', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([4, 5, 6])
    assertArrayEq(await a.add(b).toArray(), [5, 7, 9])
  })

  await test('sub', async () => {
    const a = new Tensor([10, 20, 30])
    const b = new Tensor([1, 2, 3])
    assertArrayEq(await a.sub(b).toArray(), [9, 18, 27])
  })

  await test('mul', async () => {
    const a = new Tensor([2, 3, 4])
    const b = new Tensor([5, 6, 7])
    assertArrayEq(await a.mul(b).toArray(), [10, 18, 28])
  })

  await test('div', async () => {
    const a = new Tensor([10, 20, 30])
    const b = new Tensor([2, 4, 5])
    assertArrayEq(await a.div(b).toArray(), [5, 5, 6])
  })

  await test('neg', async () => {
    const a = new Tensor([1, -2, 3])
    assertArrayEq(await a.neg().toArray(), [-1, 2, -3])
  })

  await test('scalar add', async () => {
    const a = new Tensor([1, 2, 3])
    assertArrayEq(await a.add(2).toArray(), [3, 4, 5])
  })

  await test('scalar mul', async () => {
    const a = new Tensor([1, 2, 3])
    assertArrayEq(await a.mul(3).toArray(), [3, 6, 9])
  })

  await test('exp2', async () => {
    const a = new Tensor([0, 1, 2, 3])
    assertArrayEq(await a.exp2().toArray(), [1, 2, 4, 8])
  })

  await test('sqrt', async () => {
    const a = new Tensor([1, 4, 9, 16])
    assertArrayEq(await a.sqrt().toArray(), [1, 2, 3, 4])
  })

  await test('chain: (a + 2) * b', async () => {
    const a = new Tensor([1, 2, 3, 4])
    const b = new Tensor([0.5, 0.5, 0.5, 0.5])
    const c = a.add(new Tensor([2, 2, 2, 2])).mul(b)
    assertArrayEq(await c.toArray(), [1.5, 2, 2.5, 3])
  })

  await test('broadcast 2d+1d', async () => {
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([10, 20])
    assertArrayEq(await a.add(b).toArray(), [11, 22, 13, 24])
  })

  console.log('\n── Composed math ops ──')

  await test('exp', async () => {
    const a = new Tensor([0, 1])
    const r = await a.exp().toArray()
    assert(Math.abs(r[0] - 1) < 0.01, 'exp(0)')
    assert(Math.abs(r[1] - 2.7183) < 0.01, 'exp(1)')
  })

  await test('log', async () => {
    const a = new Tensor([1, Math.E])
    const r = await a.log().toArray()
    assert(Math.abs(r[0]) < 0.01, 'log(1)')
    assert(Math.abs(r[1] - 1) < 0.01, 'log(e)')
  })

  await test('sigmoid', async () => {
    assertArrayEq(await new Tensor([0]).sigmoid().toArray(), [0.5], 'sigmoid(0)')
  })

  await test('abs', async () => {
    assertArrayEq(await new Tensor([-2, 0, 3]).abs().toArray(), [2, 0, 3])
  })

  await test('square', async () => {
    assertArrayEq(await new Tensor([2, 3]).square().toArray(), [4, 9])
  })

  await test('pow', async () => {
    const a = new Tensor([2, 3, 4])
    const b = new Tensor([3, 2, 0.5])
    const r = await a.pow(b).toArray()
    assertArrayEq(r, [8, 9, 2])
  })

  console.log('\n── Activations ──')

  await test('relu', async () => {
    assertArrayEq(await new Tensor([-1, 0, 1, 2]).relu().toArray(), [0, 0, 1, 2])
  })

  await test('gelu', async () => {
    const r = await new Tensor([0]).gelu().toArray()
    assert(Math.abs(r[0]) < 0.01, 'gelu(0)')
  })

  await test('silu', async () => {
    const r = await new Tensor([0]).silu().toArray()
    assert(Math.abs(r[0]) < 0.01, 'silu(0)')
  })

  console.log('\n── Comparisons ──')

  await test('eq', async () => {
    assertArrayEq(await new Tensor([1, 2, 3]).eq(new Tensor([1, 3, 3])).toArray(), [1, 0, 1])
  })

  await test('gt', async () => {
    assertArrayEq(await new Tensor([1, 2, 3]).gt(new Tensor([0, 2, 4])).toArray(), [1, 0, 0])
  })

  await test('where with gt mask', async () => {
    const mask = new Tensor([3, 1, 4]).gt(new Tensor([2, 2, 2]))
    const r = mask.where(new Tensor([10, 20, 30]), new Tensor([0, 0, 0]))
    assertArrayEq(await r.toArray(), [10, 0, 30])
  })

  await test('maximum', async () => {
    assertArrayEq(await new Tensor([1, 5, 3]).maximum(new Tensor([2, 4, 3])).toArray(), [2, 5, 3])
  })

  await test('clamp', async () => {
    assertArrayEq(await new Tensor([0, 1, 2, 3, 4]).clamp(1, 3).toArray(), [1, 1, 2, 3, 3])
  })

  console.log('\n── Movement ──')

  await test('flip', async () => {
    const a = new Tensor([1, 2, 3, 4, 5])
    assertArrayEq(await a.flip(0).toArray(), [5, 4, 3, 2, 1])
  })

  await test('pad', async () => {
    const a = new Tensor([1, 2, 3])
    const b = a.pad([[1, 1]])
    assert(b.shape[0] === 5, 'padded shape')
    assertArrayEq(await b.toArray(), [0, 1, 2, 3, 0])
  })

  await test('transpose', async () => {
    assertArrayEq(await new Tensor([[1, 2], [3, 4]]).transpose().toArray(), [1, 3, 2, 4])
  })

  await test('squeeze', async () => {
    assertShapeEq(new Tensor([[1, 2, 3]]).squeeze(0).shape, [3])
  })

  await test('unsqueeze', async () => {
    assertShapeEq(new Tensor([1, 2, 3]).unsqueeze(0).shape, [1, 3])
  })

  await test('flatten', async () => {
    assertShapeEq(new Tensor([[1, 2], [3, 4]]).flatten().shape, [4])
  })

  await test('reshape -1', async () => {
    assertShapeEq(new Tensor([1, 2, 3, 4, 5, 6]).reshape(2, -1).shape, [2, 3])
  })

  console.log('\n── Reduction ──')

  await test('sum all', async () => {
    const a = new Tensor([1, 2, 3, 4])
    assert(Math.abs(await a.sum().item() - 10) < 1e-4, 'sum')
  })

  await test('sum axis', async () => {
    assertArrayEq(await new Tensor([[1, 2], [3, 4]]).sum(1).toArray(), [3, 7])
  })

  await test('sum neg axis', async () => {
    assertArrayEq(await new Tensor([[1, 2], [3, 4]]).sum(-1).toArray(), [3, 7])
  })

  await test('mean', async () => {
    assert(Math.abs(await new Tensor([1, 2, 3, 4]).mean().item() - 2.5) < 1e-4, 'mean')
  })

  await test('max axis', async () => {
    assertArrayEq(await new Tensor([[1, 4], [3, 2]]).max({axis: 1}).toArray(), [4, 3])
  })

  await test('softmax', async () => {
    const r = await new Tensor([[1, 2, 3]]).softmax().toArray()
    assertArrayEq(r, [0.0900, 0.2447, 0.6652], 'softmax')
  })

  await test('matmul', async () => {
    assertArrayEq(
      await new Tensor([[1, 2], [3, 4]]).matmul(new Tensor([[5, 6], [7, 8]])).toArray(),
      [19, 22, 43, 50]
    )
  })

  await test('multi-kernel: mean+div', async () => {
    // mean = sum/count, requires 2 kernels (reduce + elementwise)
    const r = await new Tensor([[1, 2], [3, 4]]).mean(1).toArray()
    assertArrayEq(r, [1.5, 3.5], 'mean axis=1')
  })

  console.log('\n── Autograd ──')

  await test('grad: mul sum', async () => {
    const x = new Tensor([1, 2, 3, 4], { requiresGrad: true })
    const loss = x.mul(x).sum()
    await loss.backward()
    assert(x.grad !== null, 'grad should not be null')
    assertArrayEq(await x.grad.toArray(), [2, 4, 6, 8])
  })

  await test('grad: neg sum', async () => {
    const x = new Tensor([1, 2, 3], { requiresGrad: true })
    const loss = x.neg().sum()
    await loss.backward()
    assertArrayEq(await x.grad.toArray(), [-1, -1, -1])
  })

  console.log('\n── Static constructors ──')

  await test('zeros', async () => {
    const t = Tensor.zeros(3)
    assertArrayEq(await t.toArray(), [0, 0, 0])
  })

  await test('ones', async () => {
    const t = Tensor.ones(4)
    assertArrayEq(await t.toArray(), [1, 1, 1, 1])
  })

  await test('arange', async () => {
    const t = Tensor.arange(5)
    assertArrayEq(await t.toArray(), [0, 1, 2, 3, 4])
  })

  await test('eye', async () => {
    assertArrayEq(await Tensor.eye(2).toArray(), [1, 0, 0, 1])
  })

  await test('rand shape', async () => {
    assertShapeEq(Tensor.rand(2, 3).shape, [2, 3])
  })

  await test('realize returns self', async () => {
    const x = new Tensor([1, 2, 3])
    const y = x.add(1).realize()
    assertArrayEq(await y.toArray(), [2, 3, 4])
  })

  // --- Float64 ---
  console.log('\n── Float64 ──')

  await test('f64: creation from array', async () => {
    const x = new Tensor([1.0, 2.0, 3.0], { dtype: 'float64' })
    assert(x.dtype === 'float64', `expected float64, got ${x.dtype}`)
    assertShapeEq(x.shape, [3])
    const arr = await x.toArray()
    assert(arr instanceof Float64Array, 'should be Float64Array')
    assertArrayEq(arr, [1, 2, 3])
  })

  await test('f64: creation 2D', async () => {
    const x = new Tensor([[1, 2], [3, 4]], { dtype: 'float64' })
    assert(x.dtype === 'float64', `expected float64, got ${x.dtype}`)
    assertShapeEq(x.shape, [2, 2])
    const arr = await x.toArray()
    assert(arr instanceof Float64Array, 'should be Float64Array')
    assertArrayEq(arr, [1, 2, 3, 4])
  })

  await test('f64: zeros', async () => {
    const x = Tensor.zeros(4, { dtype: 'float64' })
    assert(x.dtype === 'float64', `expected float64, got ${x.dtype}`)
    assertArrayEq(await x.toArray(), [0, 0, 0, 0])
  })

  await test('f64: ones', async () => {
    const x = Tensor.ones(3, { dtype: 'float64' })
    assert(x.dtype === 'float64', `expected float64, got ${x.dtype}`)
    assertArrayEq(await x.toArray(), [1, 1, 1])
  })

  await test('f64: full', async () => {
    const x = Tensor.full([2, 2], 7, { dtype: 'float64' })
    assert(x.dtype === 'float64', `expected float64, got ${x.dtype}`)
    assertArrayEq(await x.toArray(), [7, 7, 7, 7])
  })

  await test('f64: arange', async () => {
    const x = Tensor.arange(5, 0, 1, { dtype: 'float64' })
    assert(x.dtype === 'float64', `expected float64, got ${x.dtype}`)
    assertArrayEq(await x.toArray(), [0, 1, 2, 3, 4])
  })

  await test('f64: eye', async () => {
    const x = Tensor.eye(2, { dtype: 'float64' })
    assert(x.dtype === 'float64', `expected float64, got ${x.dtype}`)
    assertArrayEq(await x.toArray(), [1, 0, 0, 1])
  })

  await test('f64: add', async () => {
    const a = new Tensor([1.0, 2.0, 3.0], { dtype: 'float64' })
    const b = new Tensor([4.0, 5.0, 6.0], { dtype: 'float64' })
    const c = a.add(b)
    assert(c.dtype === 'float64', `expected float64, got ${c.dtype}`)
    const arr = await c.toArray()
    assert(arr instanceof Float64Array, 'should be Float64Array')
    assertArrayEq(arr, [5, 7, 9])
  })

  await test('f64: mul', async () => {
    const a = new Tensor([2.0, 3.0, 4.0], { dtype: 'float64' })
    const b = new Tensor([5.0, 6.0, 7.0], { dtype: 'float64' })
    const c = a.mul(b)
    assert(c.dtype === 'float64', `expected float64, got ${c.dtype}`)
    assertArrayEq(await c.toArray(), [10, 18, 28])
  })

  await test('f64: neg', async () => {
    const a = new Tensor([1.0, -2.0, 3.0], { dtype: 'float64' })
    const c = a.neg()
    assert(c.dtype === 'float64', `expected float64, got ${c.dtype}`)
    assertArrayEq(await c.toArray(), [-1, 2, -3])
  })

  await test('f64: scalar add', async () => {
    const a = new Tensor([1.0, 2.0, 3.0], { dtype: 'float64' })
    const c = a.add(10)
    assert(c.dtype === 'float64', `expected float64, got ${c.dtype}`)
    assertArrayEq(await c.toArray(), [11, 12, 13])
  })

  await test('f64: sum', async () => {
    const a = new Tensor([1.0, 2.0, 3.0, 4.0], { dtype: 'float64' })
    const s = await a.sum().item()
    assert(Math.abs(s - 10) < 1e-10, `expected 10, got ${s}`)
  })

  await test('f64: chain ops', async () => {
    const a = new Tensor([1.0, 2.0, 3.0], { dtype: 'float64' })
    const b = new Tensor([4.0, 5.0, 6.0], { dtype: 'float64' })
    const c = a.add(b).mul(a).sub(b)
    assert(c.dtype === 'float64', `expected float64, got ${c.dtype}`)
    assertArrayEq(await c.toArray(), [(1+4)*1-4, (2+5)*2-5, (3+6)*3-6])
  })

  await test('f64: reshape', async () => {
    const a = new Tensor([1, 2, 3, 4, 5, 6], { dtype: 'float64' })
    const b = a.reshape(2, 3)
    assert(b.dtype === 'float64', `expected float64, got ${b.dtype}`)
    assertShapeEq(b.shape, [2, 3])
    assertArrayEq(await b.toArray(), [1, 2, 3, 4, 5, 6])
  })

  await test('f64: default is f32', async () => {
    const a = new Tensor([1.0, 2.0])
    assert(a.dtype === 'float32', `expected float32, got ${a.dtype}`)
    const arr = await a.toArray()
    assert(arr instanceof Float32Array, 'default should be Float32Array')
  })

  await test('f64: repr includes float64', async () => {
    const x = new Tensor([1, 2, 3], { dtype: 'float64' })
    assert(x.toString().includes('float64'), 'repr should include float64')
  })

  await test('f64: backward', async () => {
    const a = new Tensor([1.0, 2.0, 3.0], { dtype: 'float64', requiresGrad: true })
    const b = new Tensor([4.0, 5.0, 6.0], { dtype: 'float64' })
    const loss = a.mul(b).sum()
    loss.backward()
    assert(a.grad !== null, 'grad should not be null')
    assertArrayEq(await a.grad.toArray(), [4, 5, 6])
  })

  // Summary
  console.log(`\nResults: ${passed} passed, ${failed} failed, ${passed + failed} total`)
  process.exit(failed > 0 ? 1 : 0)
}

main().catch(e => {
  console.error('Fatal:', e)
  process.exit(1)
})
