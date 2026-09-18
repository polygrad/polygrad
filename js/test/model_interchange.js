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
    const cases = require('../../test/fixtures/vision.json').cases
    for (const i of [0,1,2,4]) {
      const item = await require('./vision_fixture').expandVisionCase(cases[i])
      const bytes = readBytes(path.join(dir, `python-${item.name}.bundle`))
      const model = pg.Model.load(bytes)
      try {
        const inputs = Object.fromEntries(Object.entries(item.inputs).map(([k,v]) =>
          [k, k === 'input_ids' ? new Int32Array(v.flat(Infinity)) : new Float32Array(v.flat(Infinity))]))
        const outputs = await model.forwardAsync(inputs)
        for (const [name,v] of Object.entries(item.outputs)) {
          const expected = v.flat(Infinity), actual = outputs[name]
          if (actual.length !== expected.length || actual.some((x,j) => !Number.isFinite(x) || Math.abs(x-expected[j]) > 1e-4))
            throw new Error(`${item.name} ${name} changed across frontends`)
        }
        const saved = await model.saveAsync()
        if (saved.length !== bytes.length || saved.some((b,j) => b !== bytes[j])) throw new Error(`${item.name} noncanonical bundle`)
        writeBytes(path.join(dir, `javascript-${core}-${item.name}.bundle`), saved)
      } finally { await model.dispose() }
    }
    const qwenBytes = readBytes(path.join(dir, 'python-qwen.bundle'))
    const qwen = pg.Model.load(qwenBytes)
    try {
      const oracle = require('../../test/fixtures/qwen3.json')
      if (JSON.stringify(qwen.entrypoints()[0].inputs) !== '["x"]')
        throw new Error('Qwen artifact lost its token-only signature')
      const out = (await qwen.forwardAsync({ x: new Int32Array(oracle.tokens.flat()) })).output
      const expected = oracle.logits.flat(2)
      if (out.length !== expected.length || out.some((v, i) => !Number.isFinite(v) || Math.abs(v - expected[i]) > 3e-5))
        throw new Error('Qwen rotary state or logits changed across frontends')
      const saved = await qwen.saveAsync()
      if (saved.length !== qwenBytes.length || saved.some((b, i) => b !== qwenBytes[i]))
        throw new Error('Qwen bundle changed across frontends')
      writeBytes(path.join(dir, `javascript-${core}-qwen.bundle`), saved)
    } finally { await qwen.dispose() }
    const componentBytes = readBytes(path.join(dir,'python-components.bundle'))
    const components = pg.Model.load(componentBytes)
    try {
      const oracle = require('../../test/fixtures/model_components_expected.json')
      for (const item of [...oracle.cases].reverse()) {
        const out = await components.forwardAsync({tokens:new Int32Array(item.tokens.flat())})
        const expected = item.prediction.flat(2)
        if (out.prediction.length !== expected.length ||
            out.prediction.some((v,i)=>!Number.isFinite(v) || Math.abs(v-expected[i])>2e-5) ||
            Math.abs(out.mean[0]-item.mean)>2e-5)
          throw new Error('Python component artifact lost typed input, bound shape or values')
      }
      const saved = await components.saveAsync({includeOptimizer:false})
      if (saved.length !== componentBytes.length || saved.some((b,i)=>b!==componentBytes[i]))
        throw new Error('component bundle changed across frontends')
      writeBytes(path.join(dir,`javascript-${core}-components.bundle`),saved)
    } finally { await components.dispose() }
    const stateful = pg.Model.load(readBytes(path.join(dir,'python-stateful.bundle')))
    let statefulLoss
    try {
      stateful.setOptimizer('adam',0.01)
      statefulLoss = await stateful.trainStepAsync({x:new Float32Array(16).fill(1),y:new Float32Array(16)})
      writeBytes(path.join(dir,`javascript-${core}-stateful.bundle`),await stateful.saveAsync())
    } finally { await stateful.dispose() }
    const variable = pg.Model.load(readBytes(path.join(dir,'python-variable.bundle')))
    try {
      for (const n of [17,3,11]) {
        const x = Float32Array.from({length:n*2},(_,i)=>i)
        const result = variable.forward({x}).prediction
        if (result.length !== x.length || result.some((v,i)=>v!==x[i]*2))
          throw new Error('Python variable signature lost bound extent or values')
      }
    } finally { variable.dispose() }
    const n = pg.uop.variable('interchange_batch',1,32), bound = n.bind(17)
    const variableInput = pg.Tensor.empty([bound,2])
    const variableAuthor = new pg.Model(({x})=>({prediction:x.mul(2)}), {inputs:{x:variableInput}})
    try {
      writeBytes(path.join(dir,`javascript-${core}-variable.bundle`),variableAuthor.save({includeOptimizer:false}))
    } finally { variableAuthor.dispose(); variableInput.dispose(); bound.dispose(); n.dispose() }
    const recurrent = pg.Model.fromBundle(readBytes(path.join(dir, 'c-lstm.bundle')))
    try {
      const x = new Float32Array([1, 2])
      const first = recurrent.forward({x, h: new Float32Array(2), c: new Float32Array(2)})
      const second = recurrent.forward({x, h: first.hidden, c: first.cell_state})
      const expected = JSON.parse(fs.readFileSync(path.join(dir, 'c-lstm-expected.json'), 'utf8'))
      for (const name of ['hidden', 'cell_state']) {
        if (second[name].some((v, i) => Math.abs(v - expected[name][i]) > 1e-5))
          throw new Error(`C-built LSTM recurrent state mismatch: ${name}`)
      }
    } finally { recurrent.dispose() }
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
    const authored = pg.Model.fromCallable(({ x }) => {
      const prediction = x.mul(w).add(offset)
      return { prediction, twice: prediction.mul(2) }
    }, { inputs: { x: pg.Tensor.empty([1]) }, params: { weight: w, tied: w, offset }, entrypoints: [
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
      core, trainLoss, inferenceOutput, statefulLoss
    }, null, 2))
  } finally {
    await pg.dispose()
  }
}

main().catch(error => {
  console.error(error && error.stack ? error.stack : error)
  process.exit(1)
})
