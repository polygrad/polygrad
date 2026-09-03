'use strict'

// Run from repo root:
//   POLY_CORE=native node js/examples/tinygrad_style_custom_loop_export.js

const { create } = require('../src')

async function main() {
  const pg = await create({ core: process.env.POLY_CORE || 'auto' })
  const Tensor = pg.Tensor

  class LinearNet {
    constructor() {
      this.weight = new Tensor([[1], [1]])
    }
    call(x) { return x.dot(this.weight) }
  }

  const model = new LinearNet()
  const opt = new pg.nn.optim.Adam([model.weight], { lr: 0.05 })
  const x = new Tensor([[1, 2]])
  const y = new Tensor([[4]])

  for (let i = 0; i < 6; i++) {
    opt.zeroGrad()
    const loss = model.call(x).sub(y).square().mean()
    await loss.backward()
    await opt.step()
  }

  const input = pg.Tensor.empty([1, 2])
  const output = model.call(input)
  const inst = await pg.Instance.fromTensors({
    inputs: { x: input },
    outputs: { output },
    params: { weight: model.weight }
  })
  const out = inst.forward({ x: new Float32Array([1, 2]) })
  console.log('forward', Array.from(out.output))
  console.log('bundle bytes', inst.saveBundle().length)

  await pg.dispose()
}

main().catch(err => {
  console.error(err)
  process.exitCode = 1
})
