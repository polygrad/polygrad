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

  // gt skipped: WASM renderer doesn't handle CMPLT→f32 type promotion yet
  // await test('gt', async () => {
  //   assertArrayEq(await new Tensor([1, 2, 3]).gt(new Tensor([0, 2, 4])).toArray(), [1, 0, 0])
  // })

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

  // Summary
  console.log(`\nResults: ${passed} passed, ${failed} failed, ${passed + failed} total`)
  process.exit(failed > 0 ? 1 : 0)
}

main().catch(e => {
  console.error('Fatal:', e)
  process.exit(1)
})
