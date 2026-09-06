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
    throw new Error('usage: node model_interchange.js DIR native|wasm')
  }
  const pg = await polygrad.create({ core, device: core === 'native' ? 'cpu' : 'wasm' })
  try {
    const graphInput = { x: new Float32Array([1, 2]) }
    const loadedGraph = pg.Model.fromBundle(readBytes(path.join(dir, 'python-graph.bundle')))
    try {
      const result = loadedGraph.forward(graphInput).prediction
      if (result[0] !== 28 || result[1] !== 61 || loadedGraph.paramCount !== 1)
        throw new Error('Python Graph factory artifact lost composition/sharing')
    } finally { loadedGraph.dispose() }
    const graph = pg.models.Graph(require('../../test/fixtures/model_definition.json'))
    try {
      graph.writeBuffer('modules.shared.weight', new Float32Array([1, 2, 3, 4]))
      writeBytes(path.join(dir, `javascript-${core}-graph.bundle`), graph.saveBundle({ includeOptimizer: false }))
    } finally { graph.dispose() }

    const custom = pg.Model.fromBundle(readBytes(path.join(dir, 'python-custom.bundle')))
    try {
      const output = custom.call('double', { x: new Float32Array([3]) })
      if (output.twice[0] !== 14) throw new Error('Python custom model lost graph/state')
      custom.writeBuffer('weight', new Float32Array([4]))
      if (custom.readBuffer('tied')[0] !== 4 || custom.forward({ x: new Float32Array([3]) }).prediction[0] !== 13)
        throw new Error('Python custom model lost alias or AUX state')
    } finally { custom.dispose() }

    const w = new pg.Tensor([2], { dtype: 'float32' })
    const offset = new pg.Tensor([1], { dtype: 'float32' }).is_param_(false)
    const authored = pg.Model.trace(({ x }) => {
      const prediction = x.mul(w).add(offset)
      return { prediction, twice: prediction.mul(2) }
    }, { inputs: { x: pg.Tensor.empty([1]) }, state: { weight: w, tied: w, offset }, entrypoints: [
      { name: 'forward', inputs: ['x'], outputs: ['prediction'] },
      { name: 'double', inputs: ['x'], outputs: ['twice'] }
    ] })
    try {
      writeBytes(path.join(dir, `javascript-${core}-custom.bundle`), authored.saveBundle({ includeOptimizer: false }))
    } finally { authored.dispose() }

    const train = pg.Model.fromIR(
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
