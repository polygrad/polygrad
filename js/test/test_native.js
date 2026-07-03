'use strict'

const polygrad = require('..')
const { runTensorTests } = require('./test_tensor')
const { runInstanceTests } = require('./test_instance')
const { runJitTests } = require('./test_jit')
const { runOptimTests } = require('./test_optim')
const { runModelTests } = require('./test_model')

polygrad.create({ core: 'native' }).then(pg =>
  runTensorTests(pg).then(async tensorResult => {
    const instanceResult = await runInstanceTests(pg)
    const jitResult = await runJitTests(pg)
    const optimResult = await runOptimTests(pg)
    const modelResult = await runModelTests(pg)
    const failed = tensorResult.failed + instanceResult.failed + jitResult.failed +
      optimResult.failed + modelResult.failed
    if (failed > 0) process.exit(1)
  })
).catch(e => { console.error(e); process.exit(1) })
