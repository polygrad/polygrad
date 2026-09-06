'use strict'

require('./bench_native_model').runBenchmark('wasm').catch(error => {
  console.error(error)
  process.exitCode = 1
})
