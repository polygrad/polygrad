'use strict'

const polygrad = require('..')
const { runTests } = require('./test_shared')

// Browser smoke: exercises the WASM-only path that browsers use
polygrad.create({ backend: 'wasm' }).then(pg =>
  runTests(pg).then(({ failed }) => { if (failed > 0) process.exit(1) })
).catch(e => { console.error(e); process.exit(1) })
