'use strict'
const fs = require('node:fs/promises')
const path = require('node:path')
const { checkOnnxEncoder } = require('./check_onnx_encoder')

async function main() {
  const [core, directory] = process.argv.slice(2)
  if (!directory || !['native','wasm'].includes(core)) throw new Error('usage: test_onnx_encoder.js native|wasm DIRECTORY')
  const pg = require('..').create({core, device:'cpu'})
  try { await checkOnnxEncoder(pg, name => fs.readFile(path.join(directory, name))) }
  finally { await pg.dispose() }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
