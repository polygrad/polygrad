'use strict'

// Shared Node/browser check. The caller supplies bytes; no browser filesystem shim.
async function checkOnnxEncoder(pg, read) {
  const oracle = JSON.parse(new TextDecoder().decode(await read('oracle.json')))
  const models = []
  const errors = []
  try {
    models.push(pg.Model.fromONNX(await read('onnx/model.onnx'), { dimensions: oracle.dimensions }))
    // Also exercise a bundle authored by Python, then one authored by JavaScript.
    models.push(pg.Model.load(await read('model.pgb')))
    models.push(pg.Model.load(await models[0].saveAsync()))
    for (const model of models) {
      for (const item of oracle.cases) {
        const inputs = Object.fromEntries(Object.entries(item.inputs).map(([k,v]) => [k, BigInt64Array.from(v, BigInt)]))
        let output, submissions = 0
        const gpu = pg._core.Module && pg._core.Module.__polygradWebGpuState
        const queue = gpu && gpu.device && gpu.device.queue
        const submit = queue && queue.submit
        if (pg.device === 'webgpu' && !queue) throw new Error('WebGPU encoder has no initialized GPU queue')
        if (queue) queue.submit = function(...args) { submissions++; return submit.apply(this, args) }
        try { output = (await model.forwardAsync(inputs)).last_hidden_state }
        finally { if (queue) queue.submit = submit }
        if (pg.device === 'webgpu' && submissions === 0) throw new Error('encoder submitted no WebGPU work')
        if (output.length !== item.output.length) throw new Error('encoder output shape mismatch')
        let maxError = 0
        for (let i = 0; i < output.length; i++) {
          const error = Math.abs(output[i] - item.output[i])
          if (!(error <= 2e-5 + 2e-5 * Math.abs(item.output[i])))
            throw new Error(`encoder output[${i}]: ${output[i]} vs ${item.output[i]}`)
          maxError = Math.max(maxError, error)
        }
        errors.push(maxError)
      }
    }
    console.log('PASS: trained ONNX encoder, Python/JS bundles', JSON.stringify({core:pg.core, device:pg.device, maxErrors:errors}))
  } finally {
    for (const model of models.reverse()) await model.dispose()
  }
}

module.exports = { checkOnnxEncoder }
