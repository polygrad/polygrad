'use strict'

// Run from repo root:
//   POLY_CORE=native node js/examples/nn_linear_model_fit.js

const { create } = require('../src')

async function main() {
  const pg = await create({ core: process.env.POLY_CORE || 'auto' })

  class Net {
    constructor() {
      this.fc = new pg.nn.Linear(2, 1)
      this.fc.weight = new pg.Tensor([[1, 1]], { requiresGrad: true })
      this.fc.bias = new pg.Tensor([0], { requiresGrad: true })
    }
    call(x) { return this.fc.call(x) }
  }

  const net = new Net()
  const x = pg.nn.Input('x', { shape: [1, 2] })
  const y = pg.nn.Target('y', { shape: [1, 1] })
  const model = pg.nn.trace(net, {
    inputs: { x },
    targets: { y },
    loss: (pred, target) => pred.sub(target).square().mean()
  })

  const losses = await model.fit({
    x: new Float32Array([1, 2]),
    y: new Float32Array([4])
  }, { epochs: 8, optimizer: 'sgd', lr: 0.03 })

  console.log('loss', losses[0], '->', losses[losses.length - 1])
  console.log('forward', Array.from(model.instance.forward({ x: new Float32Array([1, 2]) }).output))

  await pg.dispose()
}

main().catch(err => {
  console.error(err)
  process.exitCode = 1
})
