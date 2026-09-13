'use strict'

const assert = require('node:assert/strict')
const path = require('node:path')
const { createRequire } = require('node:module')
const local = createRequire(path.join(process.cwd(), 'package.json'))
const resolved = local.resolve('polygrad')
const relative = path.relative(path.join(process.cwd(), 'node_modules', 'polygrad'), resolved)
assert(!relative.startsWith('..') && !path.isAbsolute(relative), resolved)
const pg = local('polygrad')
const expected = process.argv[2]
const rt = pg.create({ device: 'cpu' })
assert.equal(rt.core, expected)
assert.deepEqual(Array.from(new rt.Tensor([1, 2, 3]).mul(2).toArray()), [2, 4, 6])
const model = rt.models.Sequential({
  input: { name: 'x', shape: [1], dtype: 'float32' },
  layers: [{ name: 'copy', type: 'identity' }], output: 'prediction'
})
const bytes = model.saveBundle({ includeOptimizer: false })
model.dispose()
const restored = rt.Model.fromBundle(bytes)
rt.clearScheduleCache()
rt.collect()
assert.deepEqual(Array.from(restored.forward({ x: new Float32Array([7]) }).prediction), [7])
restored.dispose()
const Tensor = rt.Tensor
const cell = new rt.nn.LSTMCell(2, 2, {bias: false})
cell.weightIh = Tensor.zeros(8, 2)
cell.weightHh = Tensor.zeros(8, 2)
const [h, c] = cell.call(Tensor.ones(1, 2), [Tensor.zeros(1, 2), Tensor.ones(1, 2)])
assert.deepEqual(Array.from(c.toArray()), [.5, .5])
const hidden = Array.from(h.toArray())
assert.equal(hidden.length, 2)
assert(hidden.every(v => Math.abs(v - .23105858) < 1e-6))
const parameter = new Tensor([1, 2, 3, 4], {dtype: 'float32'}).reshape(2, 2)
const optimizer = new rt.nn.optim.Muon([parameter], {lr: .1, nsSteps: 2})
const previousTraining = Tensor.training
try {
  Tensor.training = true
  for (let i = 0; i < 2; i++) {
    parameter._grad = new Tensor([.1, -.2, .3, -.4]).reshape(2, 2)
    optimizer.step()
  }
  const expectedValues = [1.03743243, 2.11009645, 2.77558279, 4.05183554]
  const actual = Array.from(parameter.toArray())
  assert.equal(actual.length, expectedValues.length)
  assert(actual.every((v, i) => Math.abs(v - expectedValues[i]) < 2e-5))
} finally { Tensor.training = previousTraining }
rt.dispose()
console.log(JSON.stringify({ package: resolved, core: expected, tensor: true, modelBundle: true, lstm: true, muon: true, cacheClear: true }))
