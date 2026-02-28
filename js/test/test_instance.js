/**
 * Tests for the PolyInstance JS wrapper.
 */

'use strict'

const test = require('tape')
const { Instance, OPTIM_SGD, OPTIM_ADAM } = require('../src/instance')

// --- MLP Creation ---

test('instance: create simple MLP', (t) => {
  const inst = Instance.mlp({
    layers: [2, 4, 1], activation: 'relu',
    bias: true, loss: 'mse', batch_size: 1, seed: 42
  })
  t.equal(inst.paramCount, 4)
  t.equal(inst.paramName(0), 'layers.0.weight')
  t.deepEqual(inst.paramShape(0), [4, 2])
  inst.free()
  t.end()
})

test('instance: create no-bias MLP', (t) => {
  const inst = Instance.mlp({
    layers: [3, 2], activation: 'none',
    bias: false, loss: 'none', batch_size: 1, seed: 42
  })
  t.equal(inst.paramCount, 1)
  t.equal(inst.paramName(0), 'layers.0.weight')
  t.deepEqual(inst.paramShape(0), [2, 3])
  inst.free()
  t.end()
})

test('instance: deterministic init', (t) => {
  const spec = {
    layers: [2, 4, 1], activation: 'relu',
    bias: true, loss: 'mse', batch_size: 1, seed: 42
  }
  const i1 = Instance.mlp(spec)
  const i2 = Instance.mlp(spec)
  const d1 = i1.paramData(0)
  const d2 = i2.paramData(0)
  t.equal(d1.length, d2.length)
  for (let i = 0; i < d1.length; i++) {
    t.equal(d1[i], d2[i], `param[${i}] matches`)
  }
  i1.free()
  i2.free()
  t.end()
})

test('instance: null spec throws', (t) => {
  t.throws(() => Instance.mlp('{}'), /NULL pointer/)
  t.end()
})

// --- Forward ---

test('instance: forward produces output', (t) => {
  const inst = Instance.mlp({
    layers: [2, 4, 1], activation: 'relu',
    bias: true, loss: 'mse', batch_size: 1, seed: 42
  })
  const outputs = inst.forward({ x: new Float32Array([1.0, 2.0]) })
  t.ok(outputs.output, 'has output key')
  t.ok(isFinite(outputs.output[0]), 'output is finite')
  inst.free()
  t.end()
})

test('instance: forward deterministic', (t) => {
  const inst = Instance.mlp({
    layers: [2, 4, 1], activation: 'relu',
    bias: true, loss: 'mse', batch_size: 1, seed: 42
  })
  const out1 = inst.forward({ x: new Float32Array([1.0, 2.0]) })
  const out2 = inst.forward({ x: new Float32Array([1.0, 2.0]) })
  t.equal(out1.output[0], out2.output[0], 'same output')
  inst.free()
  t.end()
})

// --- Training ---

test('instance: train SGD', (t) => {
  const inst = Instance.mlp({
    layers: [2, 1], activation: 'none',
    bias: true, loss: 'mse', batch_size: 1, seed: 42
  })
  inst.setOptimizer(OPTIM_SGD, 0.05)
  const x = new Float32Array([1.0, 2.0])
  const y = new Float32Array([5.0])

  const losses = []
  for (let i = 0; i < 50; i++) {
    const loss = inst.trainStep({ x, y })
    t.ok(isFinite(loss), `step ${i} finite`)
    losses.push(loss)
  }

  t.ok(losses[losses.length - 1] < losses[0], 'loss decreased')
  inst.free()
  t.end()
})

test('instance: train multi-layer', (t) => {
  const inst = Instance.mlp({
    layers: [1, 4, 1], activation: 'relu',
    bias: true, loss: 'mse', batch_size: 1, seed: 42
  })
  inst.setOptimizer(OPTIM_SGD, 0.01)
  const x = new Float32Array([1.0])
  const y = new Float32Array([2.0])

  const losses = []
  for (let i = 0; i < 100; i++) {
    const loss = inst.trainStep({ x, y })
    t.ok(isFinite(loss), `step ${i} finite`)
    losses.push(loss)
  }

  t.ok(losses[losses.length - 1] < losses[0], 'loss decreased')
  inst.free()
  t.end()
})

// --- Weight I/O ---

test('instance: export/import weights round-trip', (t) => {
  const spec = {
    layers: [2, 4, 1], activation: 'relu',
    bias: true, loss: 'mse', batch_size: 1, seed: 42
  }
  const inst = Instance.mlp(spec)
  const originalW = inst.paramData(0).slice()

  const stBytes = inst.exportWeights()
  t.ok(stBytes, 'exported weights')
  t.ok(stBytes.length > 0, 'non-empty')

  const spec2 = Object.assign({}, spec, { seed: 99 })
  const inst2 = Instance.mlp(spec2)
  const differentW = inst2.paramData(0)
  let same = true
  for (let i = 0; i < originalW.length; i++) {
    if (originalW[i] !== differentW[i]) { same = false; break }
  }
  t.ok(!same, 'different seeds produce different weights')

  inst2.importWeights(stBytes)
  const importedW = inst2.paramData(0)
  for (let i = 0; i < originalW.length; i++) {
    t.equal(importedW[i], originalW[i], `weight[${i}] matches after import`)
  }

  inst.free()
  inst2.free()
  t.end()
})

// --- Buffer Enumeration ---

test('instance: buf roles', (t) => {
  const inst = Instance.mlp({
    layers: [2, 1], activation: 'none',
    bias: true, loss: 'mse', batch_size: 1, seed: 42
  })
  const roles = {}
  for (let i = 0; i < inst.bufCount; i++) {
    roles[inst.bufName(i)] = inst.bufRole(i)
  }
  t.ok('layers.0.weight' in roles, 'has weight buf')
  t.ok('x' in roles, 'has input buf')
  inst.free()
  t.end()
})

test('instance: findBuf', (t) => {
  const inst = Instance.mlp({
    layers: [2, 1], activation: 'none',
    bias: true, loss: 'mse', batch_size: 1, seed: 42
  })
  const idx = inst.findBuf('output')
  t.ok(idx >= 0, 'found output buf')
  t.equal(inst.bufName(idx), 'output')
  t.equal(inst.findBuf('nonexistent'), -1)
  inst.free()
  t.end()
})

// --- Param Iteration ---

test('instance: param enumeration', (t) => {
  const inst = Instance.mlp({
    layers: [2, 4, 1], activation: 'relu',
    bias: true, loss: 'mse', batch_size: 1, seed: 42
  })
  t.equal(inst.paramCount, 4)
  t.equal(inst.paramName(0), 'layers.0.weight')
  t.deepEqual(inst.paramShape(0), [4, 2])
  const data = inst.paramData(0)
  t.ok(data, 'param data not null')
  t.equal(data.length, 8, '4*2 = 8 elements')
  inst.free()
  t.end()
})
