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
assert.deepEqual(Array.from(restored.forward({ x: new Float32Array([7]) }).prediction), [7])
restored.dispose()
rt.dispose()
console.log(JSON.stringify({ package: resolved, core: expected, tensor: true, modelBundle: true }))
