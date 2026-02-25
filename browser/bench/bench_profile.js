'use strict'

const { init, Tensor } = require('../src/index')

function main() {
  const target = [3, 1, 4]
  const lr = 0.1
  const steps = 30

  // Warmup
  for (let w = 0; w < 3; w++) {
    let params = new Float32Array([0, 0, 0])
    for (let i = 0; i < steps; i++) {
      const x = new Tensor(params, { requiresGrad: true })
      const t = new Tensor(new Float32Array(target))
      const diff = x.sub(t)
      const sq = diff.mul(diff)
      const loss = sq.sum()
      loss.backward()
      const grad = x.grad.toArray()
      for (let j = 0; j < params.length; j++) params[j] -= lr * grad[j]
    }
  }

  // Profile one run with fine-grained timing
  let params = new Float32Array([0, 0, 0])
  let t_construct = 0, t_backward = 0, t_grad_read = 0, t_loss_read = 0, t_update = 0

  for (let i = 0; i < steps; i++) {
    let t0 = performance.now()
    const x = new Tensor(params, { requiresGrad: true })
    const t = new Tensor(new Float32Array(target))
    const diff = x.sub(t)
    const sq = diff.mul(diff)
    const loss = sq.sum()
    let t1 = performance.now()
    t_construct += t1 - t0

    t0 = performance.now()
    loss.backward()
    t1 = performance.now()
    t_backward += t1 - t0

    t0 = performance.now()
    const grad = x.grad.toArray()
    t1 = performance.now()
    t_grad_read += t1 - t0

    t0 = performance.now()
    const lossVal = loss.item()
    t1 = performance.now()
    t_loss_read += t1 - t0

    t0 = performance.now()
    for (let j = 0; j < params.length; j++) params[j] -= lr * grad[j]
    t1 = performance.now()
    t_update += t1 - t0
  }

  console.log(`Profile (${steps} steps):`)
  console.log(`  graph construction: ${t_construct.toFixed(2)}ms (${(t_construct/steps).toFixed(3)}ms/step)`)
  console.log(`  backward:           ${t_backward.toFixed(2)}ms (${(t_backward/steps).toFixed(3)}ms/step)`)
  console.log(`  grad read:          ${t_grad_read.toFixed(2)}ms (${(t_grad_read/steps).toFixed(3)}ms/step)`)
  console.log(`  loss read:          ${t_loss_read.toFixed(2)}ms (${(t_loss_read/steps).toFixed(3)}ms/step)`)
  console.log(`  param update:       ${t_update.toFixed(2)}ms (${(t_update/steps).toFixed(3)}ms/step)`)
  console.log(`  TOTAL:              ${(t_construct + t_backward + t_grad_read + t_loss_read + t_update).toFixed(2)}ms`)
}

init().then(main).catch(e => { console.error(e); process.exit(1) })
