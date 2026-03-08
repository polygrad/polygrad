'use strict'

const polygrad = require('..')
const { runTests } = require('./test_shared')

polygrad.create({ backend: 'native' }).then(pg =>
  runTests(pg).then(({ failed }) => { if (failed > 0) process.exit(1) })
).catch(e => { console.error(e); process.exit(1) })
