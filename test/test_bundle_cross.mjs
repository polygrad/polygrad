/**
 * Cross-language bundle test: load Python-generated bundle in JS WASM.
 *
 * Prerequisites: run `python test/test_bundle_cross.py` to generate fixtures.
 */

import { readFileSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))

async function main() {
  // Load polygrad
  const polygrad = await (await import('../js/src/index.js')).default.create({ target: 'wasm' })

  // Load fixture
  const bundleBytes = readFileSync(join(__dirname, 'fixtures/mlp_cross.polybndl'))
  const sidecar = JSON.parse(readFileSync(join(__dirname, 'fixtures/mlp_cross.json'), 'utf8'))

  console.log('bundle size:', bundleBytes.length, 'bytes')
  console.log('expected output:', sidecar.output)

  // Load from bundle
  const inst = polygrad.Instance.fromBundle(new Uint8Array(bundleBytes))
  const pc = typeof inst.paramCount === 'function' ? inst.paramCount() : inst.paramCount
  const bc = typeof inst.bufCount === 'function' ? inst.bufCount() : inst.bufCount
  console.log('param_count:', pc)
  console.log('buf_count:', bc)

  assert(pc === sidecar.param_count,
    `param_count mismatch: ${pc} vs ${sidecar.param_count}`)
  assert(bc === sidecar.buf_count,
    `buf_count mismatch: ${bc} vs ${sidecar.buf_count}`)

  // Forward with same input
  const x = new Float32Array(sidecar.input)
  inst.forward({ x })

  // Read output
  const outIdx = findOutputBuf(inst)
  const outData = inst.bufData(outIdx)
  console.log('JS output:', Array.from(outData))

  // Compare with Python output
  const expectedOut = sidecar.output.output
  for (let i = 0; i < expectedOut.length; i++) {
    const diff = Math.abs(outData[i] - expectedOut[i])
    assert(diff < 1e-5,
      `output[${i}] mismatch: JS=${outData[i]} Python=${expectedOut[i]} diff=${diff}`)
  }

  console.log('cross-language bundle test: PASS')
  inst.free()
  process.exit(0)
}

function findOutputBuf(inst) {
  const n = typeof inst.bufCount === 'function' ? inst.bufCount() : inst.bufCount
  for (let i = 0; i < n; i++) {
    if (inst.bufRole(i) === 3) return i  // POLY_ROLE_OUTPUT = 3
  }
  throw new Error('no output buffer found')
}

function assert(cond, msg) {
  if (!cond) {
    console.error('FAIL:', msg)
    process.exit(1)
  }
}

main().catch(e => {
  console.error(e)
  process.exit(1)
})
