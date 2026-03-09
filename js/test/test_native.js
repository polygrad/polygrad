'use strict'

const polygrad = require('..')
const { runTests } = require('./test_shared')
const { runInstanceTests } = require('./test_instance_shared')

polygrad.create({ target: 'native' }).then(pg =>
  runTests(pg).then(async tensorResult => {
    const instanceResult = await runInstanceTests(pg)
    const failed = tensorResult.failed + instanceResult.failed
    if (failed > 0) process.exit(1)
  })
).catch(e => { console.error(e); process.exit(1) })
