'use strict'

// Run from repo root:
//   POLY_CORE=native node js/examples/config_mlp_fit.js

const { create } = require('../src')

async function main() {
  const pg = await create({ core: process.env.POLY_CORE || 'auto' })
  const inst = pg.Instance.mlp({
    layers: [2, 4, 1],
    activation: 'relu',
    bias: true,
    loss: 'mse',
    batch_size: 1,
    seed: 42
  })

  const losses = inst.fit({
    x: new Float32Array([1, 2]),
    y: new Float32Array([4])
  }, { epochs: 12, optimizer: 'sgd', lr: 0.03 })

  console.log('loss', losses[0], '->', losses[losses.length - 1])
  console.log('output', Array.from(inst.forward({ x: new Float32Array([1, 2]) }).output))

  await pg.dispose()
}

main().catch(err => {
  console.error(err)
  process.exitCode = 1
})
