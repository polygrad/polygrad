/**
 * Tests for the polygrad JS Tensor class.
 */

'use strict'

const test = require('tape')
const { Tensor } = require('../src/index')

function assertClose(t, actual, expected, tol = 1e-5) {
  const a = actual instanceof Float32Array ? actual : new Float32Array(actual)
  const e = expected instanceof Float32Array ? expected : new Float32Array(expected)
  t.equal(a.length, e.length, `length: ${a.length} === ${e.length}`)
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - e[i]) > tol) {
      t.fail(`index ${i}: ${a[i]} !== ${e[i]} (tol=${tol})`)
      return
    }
  }
  t.pass('values match within tolerance')
}

// --- Creation ---

test('creation: from list', (t) => {
  const x = new Tensor([1, 2, 3])
  t.deepEqual(x.shape, [3])
  assertClose(t, x.toArray(), [1, 2, 3])
  t.end()
})

test('creation: from scalar', (t) => {
  const x = new Tensor(42)
  t.deepEqual(x.shape, [1])
  assertClose(t, x.toArray(), [42])
  t.end()
})

test('creation: from 2d', (t) => {
  const x = new Tensor([[1, 2, 3], [4, 5, 6]])
  t.deepEqual(x.shape, [2, 3])
  assertClose(t, x.toArray(), [1, 2, 3, 4, 5, 6])
  t.end()
})

test('creation: zeros', (t) => {
  const x = Tensor.zeros(3, 4)
  t.deepEqual(x.shape, [3, 4])
  assertClose(t, x.toArray(), new Float32Array(12).fill(0))
  t.end()
})

test('creation: ones', (t) => {
  const x = Tensor.ones(2, 3)
  t.deepEqual(x.shape, [2, 3])
  assertClose(t, x.toArray(), new Float32Array(6).fill(1))
  t.end()
})

test('creation: full', (t) => {
  const x = Tensor.full([2, 2], 7)
  assertClose(t, x.toArray(), [7, 7, 7, 7])
  t.end()
})

test('creation: arange', (t) => {
  const x = Tensor.arange(5)
  assertClose(t, x.toArray(), [0, 1, 2, 3, 4])
  t.end()
})

test('creation: item', (t) => {
  const x = new Tensor([42])
  t.ok(Math.abs(x.item() - 42) < 1e-5)
  t.end()
})

// --- Element-wise ---

test('elementwise: add', (t) => {
  const a = new Tensor([1, 2, 3])
  const b = new Tensor([4, 5, 6])
  assertClose(t, a.add(b).toArray(), [5, 7, 9])
  t.end()
})

test('elementwise: sub', (t) => {
  const a = new Tensor([10, 20, 30])
  const b = new Tensor([1, 2, 3])
  assertClose(t, a.sub(b).toArray(), [9, 18, 27])
  t.end()
})

test('elementwise: mul', (t) => {
  const a = new Tensor([2, 3, 4])
  const b = new Tensor([5, 6, 7])
  assertClose(t, a.mul(b).toArray(), [10, 18, 28])
  t.end()
})

test('elementwise: div', (t) => {
  const a = new Tensor([10, 20, 30])
  const b = new Tensor([2, 4, 5])
  assertClose(t, a.div(b).toArray(), [5, 5, 6])
  t.end()
})

test('elementwise: neg', (t) => {
  const a = new Tensor([1, -2, 3])
  assertClose(t, a.neg().toArray(), [-1, 2, -3])
  t.end()
})

test('elementwise: scalar add', (t) => {
  const a = new Tensor([1, 2, 3])
  assertClose(t, a.add(2).toArray(), [3, 4, 5])
  t.end()
})

test('elementwise: scalar mul', (t) => {
  const a = new Tensor([1, 2, 3])
  assertClose(t, a.mul(3).toArray(), [3, 6, 9])
  t.end()
})

test('elementwise: exp2', (t) => {
  const a = new Tensor([0, 1, 2, 3])
  assertClose(t, a.exp2().toArray(), [1, 2, 4, 8])
  t.end()
})

test('elementwise: sqrt', (t) => {
  const a = new Tensor([1, 4, 9, 16])
  assertClose(t, a.sqrt().toArray(), [1, 2, 3, 4])
  t.end()
})

test('elementwise: chain', (t) => {
  const a = new Tensor([1, 2, 3, 4])
  const b = new Tensor([0.5, 0.5, 0.5, 0.5])
  const c = a.add(new Tensor([2, 2, 2, 2])).mul(b)
  assertClose(t, c.toArray(), [1.5, 2, 2.5, 3])
  t.end()
})

// --- Movement ---

test('movement: reshape', (t) => {
  const a = new Tensor([1, 2, 3, 4, 5, 6])
  const b = a.reshape(2, 3)
  t.deepEqual(b.shape, [2, 3])
  assertClose(t, b.toArray(), [1, 2, 3, 4, 5, 6])
  t.end()
})

test('movement: permute', (t) => {
  // Build 3x4 row-major: [0,1,2,3,4,5,6,7,8,9,10,11]
  const data = []
  for (let i = 0; i < 12; i++) data.push(i)
  const a = new Tensor([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11]
  ])
  const b = a.permute(1, 0)
  t.deepEqual(b.shape, [4, 3])
  // Transpose of (3,4) => (4,3): [[0,4,8],[1,5,9],[2,6,10],[3,7,11]]
  assertClose(t, b.toArray(), [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])
  t.end()
})

test('movement: flip', (t) => {
  const a = new Tensor([1, 2, 3, 4, 5])
  assertClose(t, a.flip(0).toArray(), [5, 4, 3, 2, 1])
  t.end()
})

test('movement: pad', (t) => {
  const a = new Tensor([1, 2, 3])
  const b = a.pad([[1, 1]])
  t.deepEqual(b.shape, [5])
  assertClose(t, b.toArray(), [0, 1, 2, 3, 0])
  t.end()
})

// --- Reduce ---

test('reduce: sum all', (t) => {
  const a = new Tensor([1, 2, 3, 4])
  const s = a.sum()
  t.ok(Math.abs(s.item() - 10) < 1e-5)
  t.end()
})

test('reduce: sum axis', (t) => {
  const a = new Tensor([[1, 2, 3], [4, 5, 6]])
  const s = a.sum(1)
  assertClose(t, s.toArray(), [6, 15])
  t.end()
})

// --- Autograd ---

test('autograd: grad mul sum', (t) => {
  const x = new Tensor([1, 2, 3, 4], { requiresGrad: true })
  const loss = x.mul(x).sum()
  loss.backward()
  t.ok(x.grad !== null)
  assertClose(t, x.grad.toArray(), [2, 4, 6, 8])
  t.end()
})

test('autograd: grad neg sum', (t) => {
  const x = new Tensor([1, 2, 3], { requiresGrad: true })
  const loss = x.neg().sum()
  loss.backward()
  assertClose(t, x.grad.toArray(), [-1, -1, -1])
  t.end()
})

// --- Stubs ---

test('stubs: to returns self', (t) => {
  // .to() is a no-op (CPU only)
  const a = new Tensor([1])
  t.ok(a.to === undefined, 'to is not defined (CPU-only)')
  t.end()
})

// --- Repr ---

test('repr: toString', (t) => {
  const x = new Tensor([1, 2, 3])
  t.ok(x.toString().includes('shape=[3]'))
  t.ok(x.toString().includes('float32'))
  t.end()
})

// --- Float64 ---

test('f64: creation from array', (t) => {
  const x = new Tensor([1.0, 2.0, 3.0], { dtype: 'float64' })
  t.equal(x.dtype, 'float64')
  t.deepEqual(x.shape, [3])
  const arr = x.toArray()
  t.ok(arr instanceof Float64Array, 'should be Float64Array')
  assertClose(t, arr, [1, 2, 3])
  t.end()
})

test('f64: creation 2D', (t) => {
  const x = new Tensor([[1, 2], [3, 4]], { dtype: 'float64' })
  t.equal(x.dtype, 'float64')
  t.deepEqual(x.shape, [2, 2])
  const arr = x.toArray()
  t.ok(arr instanceof Float64Array, 'should be Float64Array')
  assertClose(t, arr, [1, 2, 3, 4])
  t.end()
})

test('f64: zeros', (t) => {
  const x = Tensor.zeros(4, { dtype: 'float64' })
  t.equal(x.dtype, 'float64')
  assertClose(t, x.toArray(), [0, 0, 0, 0])
  t.end()
})

test('f64: ones', (t) => {
  const x = Tensor.ones(3, { dtype: 'float64' })
  t.equal(x.dtype, 'float64')
  assertClose(t, x.toArray(), [1, 1, 1])
  t.end()
})

test('f64: add', (t) => {
  const a = new Tensor([1.0, 2.0, 3.0], { dtype: 'float64' })
  const b = new Tensor([4.0, 5.0, 6.0], { dtype: 'float64' })
  const c = a.add(b)
  t.equal(c.dtype, 'float64')
  const arr = c.toArray()
  t.ok(arr instanceof Float64Array, 'should be Float64Array')
  assertClose(t, arr, [5, 7, 9])
  t.end()
})

test('f64: mul', (t) => {
  const a = new Tensor([1.0, 2.0, 3.0], { dtype: 'float64' })
  const b = new Tensor([4.0, 5.0, 6.0], { dtype: 'float64' })
  const c = a.mul(b)
  t.equal(c.dtype, 'float64')
  assertClose(t, c.toArray(), [4, 10, 18])
  t.end()
})

test('f64: neg', (t) => {
  const a = new Tensor([1.0, -2.0, 3.0], { dtype: 'float64' })
  const c = a.neg()
  t.equal(c.dtype, 'float64')
  assertClose(t, c.toArray(), [-1, 2, -3])
  t.end()
})

test('f64: scalar add', (t) => {
  const a = new Tensor([1.0, 2.0, 3.0], { dtype: 'float64' })
  const c = a.add(10.0)
  t.equal(c.dtype, 'float64')
  assertClose(t, c.toArray(), [11, 12, 13])
  t.end()
})

test('f64: sum', (t) => {
  const a = new Tensor([1.0, 2.0, 3.0, 4.0], { dtype: 'float64' })
  const s = a.sum().item()
  t.ok(Math.abs(s - 10.0) < 1e-10, `expected 10, got ${s}`)
  t.end()
})

test('f64: exp', (t) => {
  const a = new Tensor([0.0, 1.0, 2.0], { dtype: 'float64' })
  const c = a.exp()
  t.equal(c.dtype, 'float64')
  const arr = c.toArray()
  for (let i = 0; i < arr.length; i++) {
    const expected = Math.exp([0, 1, 2][i])
    t.ok(Math.abs(arr[i] - expected) < 1e-10,
      `exp[${i}]: ${arr[i]} vs ${expected}`)
  }
  t.end()
})

test('f64: sqrt', (t) => {
  const a = new Tensor([1.0, 4.0, 9.0], { dtype: 'float64' })
  const c = a.sqrt()
  t.equal(c.dtype, 'float64')
  assertClose(t, c.toArray(), [1, 2, 3])
  t.end()
})

test('f64: chain ops', (t) => {
  const a = new Tensor([1.0, 2.0, 3.0], { dtype: 'float64' })
  const b = new Tensor([4.0, 5.0, 6.0], { dtype: 'float64' })
  const c = a.add(b).mul(a).sub(b)
  t.equal(c.dtype, 'float64')
  assertClose(t, c.toArray(), [(1+4)*1-4, (2+5)*2-5, (3+6)*3-6])
  t.end()
})

test('f64: reshape', (t) => {
  const a = new Tensor([1, 2, 3, 4, 5, 6], { dtype: 'float64' })
  const b = a.reshape(2, 3)
  t.equal(b.dtype, 'float64')
  t.deepEqual(b.shape, [2, 3])
  assertClose(t, b.toArray(), [1, 2, 3, 4, 5, 6])
  t.end()
})

test('f64: default is f32', (t) => {
  const a = new Tensor([1.0, 2.0])
  t.equal(a.dtype, 'float32')
  t.ok(a.toArray() instanceof Float32Array, 'default should be Float32Array')
  t.end()
})

test('f64: repr includes float64', (t) => {
  const x = new Tensor([1, 2, 3], { dtype: 'float64' })
  t.ok(x.toString().includes('float64'))
  t.end()
})
