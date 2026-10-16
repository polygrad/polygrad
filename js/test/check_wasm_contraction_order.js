'use strict'

// Generic pinned order: four rounded products, left-associated horizontal
// addition, then accumulator addition. SIMD lanes must be independent outputs.
function checkContractionOrder(bytes, rows, width, columns) {
  const assert = require('node:assert/strict'), f = Math.fround
  const x = rows * columns, w = x + rows * width, b = w + columns * width
  const memory = new WebAssembly.Memory({ initial: Math.ceil((b + columns) * 4 / 65536) })
  const a = new Float32Array(memory.buffer), bits = new Uint32Array(memory.buffer)
  const kernel = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory }, math: {} }).exports.kernel
  const expected = new Float32Array(rows * columns)
  for (const pattern of ['varying', 'cancellation', 'zeros']) {
    for (let i = 0; i < rows * width; i++) a[x + i] = pattern === 'varying'
      ? Math.sin(i * .17) * .1 + Math.cos(i * .037)
      : pattern === 'zeros' ? -0 : [1e8, 1, -1e8, 1][i % 4]
    for (let i = 0; i < columns * width; i++) a[w + i] = pattern === 'varying'
      ? Math.cos(i * .13) * .3 : pattern === 'zeros' ? 1 : (i % width < width / 2 ? 1 : -1)
    for (let i = 0; i < columns; i++) a[b + i] = pattern === 'zeros' ? -0 : (i % 7 - 3) * .125
    for (let r = 0; r < rows; r++) for (let c = 0; c < columns; c++) {
      let acc = 0
      for (let k = 0; k < width; k += 4) {
        const p = j => f(a[x + r * width + k + j] * a[w + c * width + k + j])
        acc = f(acc + f(f(f(p(0) + p(1)) + p(2)) + p(3)))
      }
      expected[r * columns + c] = Math.max(f(acc + a[b + c]), 0)
    }
    a.fill(NaN, 0, rows * columns)
    kernel(0, x * 4, w * 4, b * 4)
    assert.deepEqual(bits.slice(0, expected.length), new Uint32Array(expected.buffer), pattern)
  }
}

if (require.main === module) {
  const fs = require('node:fs')
  checkContractionOrder(fs.readFileSync(process.argv[2]), ...process.argv.slice(3).map(Number))
}
module.exports = { checkContractionOrder }
