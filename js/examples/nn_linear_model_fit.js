'use strict'

// Run from repo root:
//   POLY_CORE=native node js/examples/nn_linear_model_fit.js

const { create } = require('../src')

async function main() {
  const pg = await create({ core: process.env.POLY_CORE || 'auto' })

  class Net {
    constructor() {
      this.fc = new pg.nn.Linear(2, 1)
      this.fc.weight = new pg.Tensor([[1, 1]])
      this.fc.bias = new pg.Tensor([0])
    }
    call(x) { return this.fc.call(x) }
  }

  const net = new Net()
  const x = pg.Tensor.empty([1, 2])
  const y = pg.Tensor.empty([1, 1])
  const pred = net.call(x)
  const loss = pred.sub(y).square().mean()
  const inst = await pg.Instance.fromTensors({
    inputs: { x },
    targets: { y },
    outputs: { output: pred },
    losses: { loss },
    params: { 'fc.weight': net.fc.weight, 'fc.bias': net.fc.bias }
  })

  const losses = await inst.fit({
    x: new Float32Array([1, 2]),
    y: new Float32Array([4])
  }, { epochs: 8, optimizer: 'sgd', lr: 0.03 })

  console.log('loss', losses[0], '->', losses[losses.length - 1])
  console.log('forward', Array.from(inst.forward({ x: new Float32Array([1, 2]) }).output))

  await pg.dispose()
}

main().catch(err => {
  console.error(err)
  process.exitCode = 1
})
