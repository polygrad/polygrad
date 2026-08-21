'use strict'

const fs = require('fs')
const path = require('path')
const polygrad = require('..')

function readBytes(filename) {
  const data = fs.readFileSync(filename)
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
}

function writeBytes(filename, data) {
  fs.writeFileSync(filename, Buffer.from(data.buffer, data.byteOffset, data.byteLength))
}

async function main() {
  const [dir, core] = process.argv.slice(2)
  if (!dir || !['native', 'wasm'].includes(core)) {
    throw new Error('usage: node instance_interchange.js DIR native|wasm')
  }
  const pg = await polygrad.create({ core, device: core === 'native' ? 'cpu' : 'wasm' })
  try {
    const train = pg.Instance.fromIR(
      readBytes(path.join(dir, 'python-train.pgir')),
      readBytes(path.join(dir, 'python-train.safetensors'))
    )
    let trainLoss
    try {
      train.setOptimizer(pg.OPTIM_ADAM, 0.05)
      trainLoss = await train.trainStep({
        x: new Float32Array([1, 2]), y: new Float32Array([3])
      })
      writeBytes(
        path.join(dir, `javascript-${core}-resumed.safetensors`),
        await train.exportWeights()
      )
    } finally {
      train.dispose()
    }

    const inference = pg.models.MLP({
      layers: [2, 3, 1], activation: 'relu', bias: true,
      loss: 'none', batch_size: 1, seed: 19
    })
    let inferenceOutput
    try {
      inferenceOutput = Array.from(
        (await inference.forward({ x: new Float32Array([1.25, -0.5]) })).output
      )
      writeBytes(path.join(dir, `javascript-${core}-inference.pgir`), inference.exportIR())
      writeBytes(
        path.join(dir, `javascript-${core}-inference.safetensors`),
        await inference.exportWeights()
      )
    } finally {
      inference.dispose()
    }

    fs.writeFileSync(path.join(dir, `javascript-${core}-result.json`), JSON.stringify({
      core, trainLoss, inferenceOutput
    }, null, 2))
  } finally {
    await pg.dispose()
  }
}

main().catch(error => {
  console.error(error && error.stack ? error.stack : error)
  process.exit(1)
})
