'use strict'

// POLY_CORE=native|wasm node js/examples/linear_predict.js linear.pgb
const { Model, disposeDefault } = require('../src')

const model = Model.load(process.argv[2] || 'linear.pgb')
try {
  const { prediction } = model.forward({ x: new Float32Array([3, 4, 5, 6, 7]) })
  console.log(JSON.stringify(Array.from(prediction)))
} finally {
  model.dispose()
  disposeDefault()
}
