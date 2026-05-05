'use strict'

const polygrad = require('..')
const { runTests } = require('./test_shared')
const { runInstanceTests } = require('./test_instance_shared')
const { runOptimTests } = require('./test_optim_shared')
const { runModelTests } = require('./test_model_shared')

polygrad.create({ core: 'native' }).then(pg =>
  runTests(pg).then(async tensorResult => {
    const instanceResult = await runInstanceTests(pg)
    const optimResult = await runOptimTests(pg)
    const modelResult = await runModelTests(pg)
    const failed = tensorResult.failed + instanceResult.failed + optimResult.failed + modelResult.failed
    if (failed > 0) process.exit(1)
  })
).catch(e => { console.error(e); process.exit(1) })
