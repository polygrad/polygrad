'use strict'

const assert = require('assert')
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads')

async function main() {
  if (isMainThread) {
    const core = process.argv[2] || 'native'
    // Repeat environment creation/teardown, not just simultaneous execution.
    for (const device of core === 'native' ? ['interp', 'cpu'] : ['interp', 'wasm']) {
      for (let round = 0; round < 2; round++) {
        await Promise.all(Array.from({ length: 4 }, () => new Promise((resolve, reject) => {
          const worker = new Worker(__filename, { workerData: { core, device } })
          let ready = false
          worker.on('message', message => { ready = message === 'passed' })
          worker.on('error', reject)
          worker.on('exit', code => code === 0 && ready ? resolve() : reject(new Error(`worker exit ${code}`)))
        })))
      }
    }
    console.log(`[PASS] ${core} isolated worker runtimes and teardown`)
    return
  }
  const pg = require('..')
  const runtimes = await Promise.all([0, 1].map(() => pg.create({ core: workerData.core, device: workerData.device })))
  try {
    for (let i = 1; i <= 20; i++) {
      for (const runtime of runtimes) {
        const x = new runtime.Tensor(new Float32Array(i).fill(i))
        const y = x.add(1)
        assert.deepStrictEqual(Array.from(await y.toArray()), Array(i).fill(i + 1))
        y.dispose(); x.dispose()
        runtime.clearScheduleCache(); runtime.collect()
      }
    }
  } finally {
    for (const runtime of runtimes) await runtime.dispose()
  }
  parentPort.postMessage('passed')
}

main().catch(error => { console.error(error); process.exitCode = 1 })
