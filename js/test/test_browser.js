'use strict'

const polygrad = require('..')
const { runTensorTests } = require('./test_tensor')
const { runModelRuntimeTests } = require('./test_model_runtime')

// Browser smoke: exercises the WASM-only path that browsers use
polygrad.create({ core: 'wasm' }).then(pg =>
  runTensorTests(pg).then(async tensorResult => {
    const instanceResult = await runModelRuntimeTests(pg)
    const failed = tensorResult.failed + instanceResult.failed
    if (failed > 0) process.exit(1)
  })
).catch(e => { console.error(e); process.exit(1) })
