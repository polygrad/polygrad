'use strict'

// Production and direct generic lowering must keep the pinned accumulation
// order. Varying/cancellation inputs expose a hidden striped replacement.
async function checkReductionOrder(productionBytes, genericBytes, width) {
  const assert = (ok, message) => { if (!ok) throw new Error(message) }
  const f = Math.fround, rows = 2, outputOffset = rows * width * 4
  const memory = new WebAssembly.Memory({ initial: Math.ceil((outputOffset + rows * 4) / 65536) })
  const x = new Float32Array(memory.buffer, 0, rows * width)
  const output = new Float32Array(memory.buffer, outputOffset, rows)
  const functions = []
  for (const bytes of [productionBytes, genericBytes]) {
    const { instance } = await WebAssembly.instantiate(bytes, { env: { memory }, math: {} })
    functions.push(instance.exports.kernel)
  }
  for (const pattern of ['varying', 'cancellation', 'constant']) {
    for (let i = 0; i < x.length; i++) x[i] = pattern === 'varying'
      ? Math.sin(i * .17) * .1 + Math.cos(i * .037)
      : pattern === 'cancellation' ? [1e8, 1, -1e8, 1][i % 4] : -Math.LN2
    const expected = [[], []]
    for (let row = 0; row < rows; row++) {
      const lanes = [0, 0, 0, 0]
      let grouped = 0, scalar = 0, i = 0
      for (; i + 3 < width; i += 4) {
        const a = x.subarray(row * width + i, row * width + i + 4)
        for (let j = 0; j < 4; j++) {
          lanes[j] = f(lanes[j] + a[j])
          scalar = f(scalar + a[j])
        }
        grouped = f(grouped + f(f(f(a[0] + a[1]) + a[2]) + a[3]))
      }
      let striped = f(f(lanes[0] + lanes[2]) + f(lanes[1] + lanes[3]))
      for (; i < width; i++) {
        striped = f(striped + x[row * width + i])
        scalar = f(scalar + x[row * width + i])
      }
      expected[0].push(striped)
      // Pinned generic unrolling groups four terms for divisible widths;
      // these nondivisible kernels instead use a scalar loop.
      expected[1].push(width % 4 ? scalar : grouped)
    }
    if (pattern === 'cancellation') assert(expected[0][0] !== expected[1][0], 'insensitive order fixture')
    for (let mode = 0; mode < functions.length; mode++) {
      output.fill(NaN)
      functions[mode](outputOffset, 0)
      assert(output.every((value, row) => Object.is(value, expected[1][row])),
        `${pattern}/${width}/${mode}: ${Array.from(output)} != ${expected[1]}`)
    }
  }
}

if (require.main === module) {
  const fs = require('node:fs')
  checkReductionOrder(fs.readFileSync(process.argv[2]), fs.readFileSync(process.argv[3]), Number(process.argv[4]))
    .catch(error => { console.error(error); process.exitCode = 1 })
}
module.exports = { checkReductionOrder }
